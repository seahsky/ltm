"""DREAM, assembled: the per-step loop, the end-of-episode consolidation, and the knobs.

PR 6c of 6. Five PRs built the parts in isolation -- `agent/stm.py` (eq. 4-7),
`memory/consolidate.py` (eq. 8-13), `memory/longterm.py` (eq. 14-18),
`memory/retrieve.py` (eq. 19-24), `task/plan.py` (eq. 25-27) -- and none of them was
called by anything. This module is where they meet, and `task/runner.py` calls exactly two
functions from it per episode plus one per step.

**IT MIRRORS `memory_prior.MemoryContext`, DELIBERATELY.** That type is the tree's existing
answer to "a run must not assemble its own memory": the sweep driver builds the context, the
runner is handed it, and the audit records what the caller did rather than a branch the
runner took. `DreamContext` is the same shape for the same reason, and its knobs live on it
rather than on `RunConfig` for the reason `MemoryContext.k` does -- `LAYER_IMPORTS["config"]`
has no edge to `memory`, and ADR-0013 does not widen for a value only the caller uses.

**NO `RunConfig` FIELD, AND THAT IS WHY.** A `DreamConfig` on `RunConfig` would need
`PlanWeights` and `ImportanceWeights`, which live in `memory/` and `task/`. So the knobs are
here, `run_episode` takes `dream: Optional[DreamContext]`, and `None` is byte-identical to
every run made before this existed.

**THE COST THIS ADDS IS NOT COVERED BY CRITERION 7, AND SAYING SO IS THE POINT.** Criterion
7 audits `audio_render_s`, the per-step AUDIO bill -- the render, the convolution, the bed
mix, the guard. `f_v` and `f_u` on every step are a second forward pass each and sit
OUTSIDE that bracket, so a DREAM run could double its step cost with the gate still green.
`observe` therefore returns its own elapsed seconds, the runner accumulates them, and the
run's metrics carry `dream_step_s_mean` / `dream_step_s_worst`. A cost nothing measures is
a cost nobody finds.

PR #108 measured `f_v` at 11.6 ms mean over 20 real 480x640 frames. `f_u` PER STEP has
never been measured: the tree runs CLAP ONCE per episode, at the onset.
`tests/box/test_dream_box.py` is where that number comes from, and it is the first thing a
DREAM sweep needs to know.

**THE BELIEF `U_j` READS IS THE CONTROLLER'S, NOT THE MEMORY'S.** `runner.py` overrides the
investigate target with `memory_prior.target` once per episode, the first step after the
window closes. That override is a recalled prior rather than a prediction from the current
observation, and it fires exactly once -- so feeding it to `U_j` would put a single large
revision in whichever segment happened to hold it, in every memory-arm episode. That is the
same defect `consolidate.surprise`'s `None -> Xyz` rule already refuses, wearing different
clothes. The belief passed in is `decision.investigate_waypoint or investigate_probe`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Optional, Tuple

import numpy as np

from earshot.agent.stm import ShortTermMemory, StmEntry, fuse
from earshot.memory.consolidate import (
    ImportanceWeights,
    TrajectoryStep,
    importance,
    retain,
    segment_trajectory,
)
from earshot.memory.longterm import (
    ExperienceEntry,
    ExperienceStore,
    LongTermMemory,
    Outcome,
    PatternStore,
    abstract,
    trajectory_descriptor,
)
from earshot.memory.retrieve import RetrievedContext, query, retrieve
from earshot.memory.store import SemanticStore
from earshot.task.plan import PlanWeights
from earshot.types import Pose, Xyz

__all__ = [
    "DreamKnobs",
    "DreamContext",
    "DreamEpisode",
    "StepOutcome",
    "begin_episode",
    "observe",
    "consolidate_episode",
    "empty_memory",
    "key_spread",
]


@dataclass(frozen=True)
class DreamKnobs:
    """Every number `ICRA2027_Memory` leaves open, in one place, none with a default.

    Fifteen of them. That is not a design smell, it is the count of things the paper
    names and does not value, and collecting them here is what makes them auditable -- the
    alternative is fifteen defaults scattered across five modules, each of which would
    reach no artefact. Each field names the equation it belongs to.

    `max_retained` is the one that is NOT simply a number the paper omits: eq. 13 has no
    cap and this adds one. It is argued at `consolidate.retain` and in ADR-0024, and it
    sits here so it reaches the audit like every other choice.
    """

    # eq. 4 and 19: the short-term memory and the query built from it
    stm_horizon: int
    stm_decay: float
    present_weight: float
    # eq. 9: the segmentation
    coherence: float
    min_segment: int
    max_segment: int
    # eq. 10 and 13: what is worth keeping
    importance: ImportanceWeights
    eta: float
    max_retained: int
    # eq. 16: how often a regularity must recur
    min_support: int
    # eq. 20-23: the three retrievals and the weighting
    k_experience: int
    k_pattern: int
    k_knowledge: int
    temperature: float
    # eq. 26: the plan blend
    plan_weights: PlanWeights

    def as_metrics(self) -> Tuple[Tuple[str, float], ...]:
        """Every knob as a named number for the audit record, in a fixed order.

        A sweep that cannot say what its knobs were is a sweep whose result cannot be
        reproduced, and this repo has spent real box time on exactly that
        (`pilot-1` reported three dead arms over 120 episodes that were on disk).
        """
        return (
            ("dream_stm_horizon", float(self.stm_horizon)),
            ("dream_stm_decay", float(self.stm_decay)),
            ("dream_present_weight", float(self.present_weight)),
            ("dream_coherence", float(self.coherence)),
            ("dream_min_segment", float(self.min_segment)),
            ("dream_max_segment", float(self.max_segment)),
            ("dream_alpha", float(self.importance.alpha)),
            ("dream_beta", float(self.importance.beta)),
            ("dream_gamma", float(self.importance.gamma)),
            ("dream_eta", float(self.eta)),
            ("dream_max_retained", float(self.max_retained)),
            ("dream_min_support", float(self.min_support)),
            ("dream_k_experience", float(self.k_experience)),
            ("dream_k_pattern", float(self.k_pattern)),
            ("dream_k_knowledge", float(self.k_knowledge)),
            ("dream_temperature", float(self.temperature)),
            ("dream_lambda_plan", float(self.plan_weights.plan)),
            ("dream_lambda_memory", float(self.plan_weights.memory)),
            ("dream_lambda_feasibility", float(self.plan_weights.feasibility)),
        )


def key_spread(memory: LongTermMemory) -> Optional[Tuple[float, float, float]]:
    """`(min, mean, max)` pairwise cosine over `M^E`'s keys, or `None` below two rows.

    **THE PRECONDITION `omega_t` NEEDS, AND THE BOX SAID IT MAY NOT HOLD.** The first real
    measurement put ten `M^E` keys from one walk at 0.909-0.989 cosine of each other, and
    `omega^E` was flat at 0.4775-0.5055 as a direct consequence: a retrieval cannot
    discriminate between keys that are all the same key. `memory/retrieve.py`'s
    `test_a_pattern_over_near_identical_experiences_carries_no_information` predicted
    exactly that shape, and this is the number that says whether a real sweep escapes it.

    Lives here rather than in `runner.py` because that module imports no numpy on purpose
    (its own comment at the `Any` annotations says so), and a pairwise cosine is not worth
    changing that for.

    `None` below two rows: one key has no pairwise anything, and reporting 1.0 for it
    would read as maximal degeneracy rather than as no measurement.
    """
    keys = memory.experience.keys
    if len(keys) < 2:
        return None
    cosines = [
        float(np.dot(keys[a], keys[b]))
        for a in range(len(keys))
        for b in range(a + 1, len(keys))
    ]
    return (min(cosines), sum(cosines) / len(cosines), max(cosines))


def empty_memory() -> LongTermMemory:
    """`M^L` before the first episode: three empty levels.

    The honest starting state and NOT a special case anywhere downstream --
    `retrieve` returns `weights is None`, `plan.memory_consistency` returns
    `(0.0, False)`, and `plan.pick_plan` then picks exactly what `agent.scorer` picks
    (PR #113's proof: a constant added to every candidate cannot move an argmax). So the
    first episode of a DREAM run is the pre-DREAM agent, by construction rather than by
    a branch.
    """
    return LongTermMemory(
        experience=ExperienceStore(), pattern=PatternStore(), knowledge=SemanticStore()
    )


@dataclass(frozen=True)
class DreamContext:
    """What the sweep driver builds and hands to `run_episode`. Never assembled in a run.

    `memory` GROWS between episodes and is replaced rather than mutated:
    `consolidate_episode` returns a new `LongTermMemory` and `run()` rebinds the context.
    That is what lets the audit hold the memory as it stood at the START of each episode,
    which is the only state a retrieval can be reproduced against afterwards.

    `clip_encoder` is `f_v` (eq. 6). `f_u` is the CLAP encoder `run_episode` already takes,
    so it is NOT duplicated here: two encoder handles for the same model is two things that
    can disagree, and the store-versus-query hazard `audio.clap.audio_embedding`'s docstring
    names is exactly that shape.
    """

    knobs: DreamKnobs
    memory: LongTermMemory
    clip_encoder: Any

    @property
    def is_live(self) -> bool:
        """True when this context can encode at all. An absent `clip_encoder` is a wiring
        error rather than an empty memory: `M^L` starts empty on purpose and that is not a
        reason to skip the STM, which is reset per episode and never empty for long."""
        return self.clip_encoder is not None

    def with_memory(self, memory: LongTermMemory) -> "DreamContext":
        return replace(self, memory=memory)


@dataclass(frozen=True)
class DreamEpisode:
    """The per-episode state: `M^S_t` (eq. 4) and `tau` so far (eq. 8).

    Immutable, like everything it holds. `observe` returns a new one each step, so a caller
    that kept step 5's state still has step 5's history -- the property `agent/stm.py`
    argues for at length and the one a mutating implementation passes every other test
    without.
    """

    stm: ShortTermMemory
    trajectory: Tuple[TrajectoryStep, ...] = ()

    def __len__(self) -> int:
        return len(self.trajectory)


@dataclass(frozen=True)
class StepOutcome:
    """What one DREAM step produces: the retrieval, and what it cost to get there."""

    episode: DreamEpisode
    context: RetrievedContext
    seconds: float


def begin_episode(context: DreamContext) -> DreamEpisode:
    """A fresh `M^S` and an empty `tau`. What `run_episode` calls before its first step.

    The paper's one structural guarantee for short-term memory -- "STM is reset between
    navigation episodes" -- realised by building a new state rather than clearing one, so
    nothing can carry an episode's answer into the next even by accident.
    """
    return DreamEpisode(stm=ShortTermMemory(horizon=int(context.knobs.stm_horizon)))


def observe(
    episode: DreamEpisode,
    context: DreamContext,
    *,
    frame: Any,
    audio: np.ndarray,
    pose: Pose,
    prev_action: Optional[str],
    belief: Optional[Xyz],
) -> StepOutcome:
    """One step of eq. 5-7 and 19-24: encode, remember, query, retrieve.

    `audio` arrives already encoded because `run_episode` owns the CLAP encoder and the
    heard signal, and handing this function a waveform would make it the second place that
    decides what CLAP is fed -- the hazard `audio.clap.audio_embedding` exists to prevent.
    `frame` arrives raw because `f_v` has no such second caller.

    **Times itself, because criterion 7 does not.** See the module docstring: the audio
    bill the smoke gate audits does not include these two forward passes, so a DREAM run
    could double its step cost with every criterion green. The returned `seconds` is the
    only number that says otherwise.
    """
    started = time.perf_counter()
    from earshot.vlm.encode import visual_embedding

    visual = visual_embedding(frame, context.clip_encoder)
    entry = StmEntry(visual=visual, audio=audio, pose=pose, prev_action=prev_action)
    stm = episode.stm.push(entry)
    fused = fuse(visual, audio)
    step = TrajectoryStep(observation=fused, position=pose.position, belief=belief)
    q = query(
        fused,
        stm.context(decay=float(context.knobs.stm_decay)),
        present_weight=float(context.knobs.present_weight),
    )
    retrieved = retrieve(
        q,
        context.memory,
        k_experience=int(context.knobs.k_experience),
        k_pattern=int(context.knobs.k_pattern),
        k_knowledge=int(context.knobs.k_knowledge),
        temperature=float(context.knobs.temperature),
    )
    return StepOutcome(
        episode=DreamEpisode(stm=stm, trajectory=episode.trajectory + (step,)),
        context=retrieved,
        seconds=time.perf_counter() - started,
    )


def consolidate_episode(
    episode: DreamEpisode,
    context: DreamContext,
    *,
    sound_concept: str,
    target_concept: str,
    room_concept: Optional[str],
    reached: bool,
    final_gap_m: float,
) -> Tuple[LongTermMemory, Tuple[float, ...]]:
    """Eq. 8-16 at the end of an episode. Returns the new `M^L` and the `I_j` it scored.

    The scores come back rather than staying inside, for the reason `importance` and
    `retain` are separate functions at all: they are what the audit records and what tells
    a later sweep whether `eta` was set anywhere sensible.

    **`target` FOR `C_j` IS THE EPISODE'S OWN FINAL POSITION, NEVER THE SOURCE.** This is
    the line PR #110's whole argument was about: the memory is written BY the agent, so a
    consolidation scored against ground truth makes every later retrieval GT-privileged --
    the defect `audio-localization-stop-closed` recorded against the old energy-STOP. The
    true source is available in `run_episode` and is deliberately not passed here.

    **EVERY RETAINED SEGMENT INHERITS THE EPISODE'S OUTCOME AND CONCEPTS.** Eq. 15 stores
    `y_i` per experience and this episode has one outcome, so each segment carries it. That
    is the honest reading: a segment's contribution is a contribution to THIS episode's
    ending, which is what `C_j` already measures.

    Returns `context.memory` unchanged when the trajectory is too short to segment, rather
    than raising: a two-step episode is a real episode and not an error.
    """
    if len(episode.trajectory) < 2:
        return (context.memory, ())
    segments = segment_trajectory(
        episode.trajectory,
        coherence=float(context.knobs.coherence),
        min_length=int(context.knobs.min_segment),
        max_length=int(context.knobs.max_segment),
    )
    scores = importance(
        segments,
        target=episode.trajectory[-1].position,
        memory=context.memory.novelty_vectors(),
        weights=context.knobs.importance,
    )
    kept = retain(
        segments,
        scores,
        eta=float(context.knobs.eta),
        max_kept=int(context.knobs.max_retained),
    )
    if not kept:
        return (context.memory, tuple(scores))
    outcome = Outcome(reached=bool(reached), final_gap_m=float(final_gap_m))
    rows = []
    for segment in kept:
        rows.append(ExperienceEntry(
            context=segment.representation,
            trajectory=trajectory_descriptor([step.position for step in segment.steps]),
            target_concept=target_concept,
            outcome=outcome,
            sound_concept=sound_concept,
            room_concept=room_concept,
        ))
    grown = context.memory.experience.extend(rows)
    return (
        LongTermMemory(
            experience=grown,
            # `M^P` is REBUILT from the whole of `M^E` rather than appended to. `G` groups
            # and averages, so a pattern's signature depends on every member -- an
            # incremental update would have to re-derive each affected group anyway, and a
            # rebuild is the version that cannot drift from `abstract`'s own definition.
            pattern=abstract(grown, min_support=int(context.knobs.min_support)),
            knowledge=context.memory.knowledge,
        ),
        tuple(scores),
    )
