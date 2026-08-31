"""The sounding window as a value: when the source starts, and when it stops (ADR-0017).

Before this, ``playing = step >= t_anom`` (``runner.py:536``) was the whole mechanism and
nothing closed it -- the source sounded forever, so an agent could walk to a live cue.
ADR-0017 makes one episode "the source sounds for a bounded window, goes silent, and the
agent must reach it".

**The vocabulary is ``CONTEXT.md``'s and is binding.** The *sounding window* opens at
``t_anom`` and closes at the *offset step*; the *silent phase* is every step from the
offset step to the end of the episode. The offset step is the first step the source no
longer emits -- it is not the last sounding step -- and ``SoundingWindow.is_sounding``
below is where that is settled, because "closes at" is ambiguous in prose.

**The collision ``CONTEXT.md`` records, kept rather than resolved.** In the literature
"onset" is the step the SOURCE started. In this tree ``onset_step`` is the step the
AGENT's own RMS crossed ``onset_rms`` (``audio/onset.py``), which is a different event
and usually a later one. Nothing here renames it: every audit record on disk carries the
tree's meaning, and renaming the field would invalidate all of them. The step the source
starts is ``t_anom``; the step it stops is ``offset_step``.

**THE DEFAULT DURATION IS PROVISIONAL AND PENDING A DECISION THAT HAS NOT BEEN TAKEN.**
Four policies exist (``WindowPolicy``): ``CONTINUOUS`` (the control arm, no offset step
at all), ``FIXED_STEPS``, ``BUDGET_FRACTION`` and ``DRAWN``. The default is
``FIXED_STEPS`` at 60 steps, and **60 is `provenance: fake` -- there is no sweep behind
it**. It is chosen generously per ADR-0014: a short window's failure mode is not a harder
task but a silent one, since the agent never gets in earshot, ``onset.fired`` stays False
forever, the funnel caps at ``T_ANOM_REACHED``, CLAP is never handed a clip and nothing
raises. Sixty also clears the accumulator's ramp with room to spare.

The evidence that ANSWERS the open question is what the first sweep at this default
records, not an argument made here: the ``onset_step - t_anom`` distribution (how long a
fixed count has to outlast), and the ``heard_within_window`` rate (whether the agent
heard the source or only its reverb tail). Both land on the metrics bag and therefore in
``audit.json``. **Do not read 60 as a measurement.**

All three bounded defaults resolve to the SAME 60 steps at ``max_steps = 500`` -- 60,
``floor(0.12 * 500)``, ``mean(30, 90)`` -- on purpose. Switching policy at the defaults
changes the VARIANCE of the duration and not its level, so the first policy comparison is
not confounded by a level change riding along with it.

Python 3.9 (the SoundSpaces pin).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from earshot.audio.config import WindowPolicy

__all__ = ["WindowPolicy", "SoundingWindow", "plan_window"]


@dataclass(frozen=True)
class SoundingWindow:
    """One episode's window: when the source starts, when it stops, and under which rule.

    ``opens_at`` is ``t_anom``. ``offset_step`` is the **first silent step** -- the
    sounding phase is the half-open interval ``[opens_at, offset_step)``, so the source
    emits on ``offset_step - 1`` and not on ``offset_step``. ``None`` means the window
    never closes, which is ``WindowPolicy.CONTINUOUS`` and the pre-ADR-0017 behaviour.

    **That boundary is CONTEXT.md's, not this file's, and the two sentences it is read
    from need holding together.** *Offset step: the first step at which the source no
    longer emits.* *Silent phase: the steps after the offset step.* Taken alone the second
    would put the offset step outside the silent phase, which contradicts the first: a
    step on which the source does not emit is a silent step. The first sentence governs,
    so the offset step IS the first step of the silent phase and ``is_silent`` says so.
    It matters because SWS is adopted verbatim to stay cross-quotable and this decides
    one episode at the boundary -- an agent that reaches the source exactly on the offset
    step scores, because the source had already stopped when it got there. If CONTEXT.md
    is ever tightened the other way, ``metrics.sws_episode``'s ``>=`` moves with it.

    **``offset_step`` is deliberately NOT clamped to ``max_steps``.** An episode that
    ends before its offset step simply never enters a silent phase; that is a funnel
    fact about that episode, and it is exactly the fact SWS's denominator is built on.
    Clamping would manufacture a silent phase in an episode that never had one, and the
    record would then say the source stopped when it did not.

    Frozen, like ``OnsetState`` and ``ControllerState``: the runner owns the episode's
    one mutable slot, and a window that could be edited mid-episode is a window the
    record cannot describe.
    """

    opens_at: int
    offset_step: Optional[int]
    policy: WindowPolicy

    def is_sounding(self, step: int) -> bool:
        """Is the source emitting on this step? ``[opens_at, offset_step)``."""
        index = int(step)
        if index < self.opens_at:
            return False
        return self.offset_step is None or index < self.offset_step

    def is_silent(self, step: int) -> bool:
        """Is this step in the silent phase? Never true before the window opens.

        Not simply ``not is_sounding``: the steps before ``t_anom`` are not silence in
        the sense ADR-0017 means, they are the pre-onset phase §3.1 asserts the bed on.
        Collapsing the two would put the provenance invariant's steps and the silent
        phase's steps in one bucket, which is the confusion ``test_task_runner.py``'s
        ``not row.source_playing`` filter already has.
        """
        return int(step) >= self.opens_at and not self.is_sounding(step)

    @property
    def duration_steps(self) -> Optional[int]:
        """How many steps the source sounds for, or ``None`` when it never stops.

        The runner writes this into ``metrics["sounding_duration_steps"]`` rather than
        subtracting the two fields again at the call site. One definition of the window's
        length, in the type that owns its boundaries.
        """
        if self.offset_step is None:
            return None
        return int(self.offset_step) - int(self.opens_at)

    # **There is deliberately no `as_dict` here.** There was one, it never reached disk,
    # and it disagreed with the record that does: `report.audit.SoundingWindowRecord` is
    # what `run_episode` writes, and it carries the accumulator's measurements beside the
    # boundaries. Two serializers for one concept is how a reader ends up comparing a
    # field that exists in one of them, and ADR-0013 forbids `report` importing this
    # module anyway, so they could never have been checked against each other in
    # production code.


def plan_window(
    *,
    t_anom: int,
    max_steps: int,
    policy: WindowPolicy,
    sounding_steps: int,
    budget_fraction: float,
    draw_steps_range: Tuple[int, int],
    seed: int,
    episode_index: int,
) -> SoundingWindow:
    """Resolve one episode's window. Pure, keyword-only, and never touches a global RNG.

    Every policy's parameters are accepted on every call so the resolved window can be
    recorded beside the ones that were not used, but **only the chosen policy's
    parameters are validated**: a run that never draws must not be stopped by a draw
    range nobody reads.

    **The draw is a pure function of ``(seed, episode_index)`` and touches no global RNG
    and no shared stream.** Two reasons, both of which have cost this project a result.
    A global draw makes a red run unreproducible, which ``RunConfig.seed``'s own comment
    already calls not-evidence. And ``cfg.seed`` is consumed in sequence by the navmesh's
    pose draws, so a window drawn off the same stream would depend on how many
    calibration poses happened to be drawn before it -- and ``tools/episode_diff.py``
    pairs the SAME episode index across two sweeps, so a duration that is not a function
    of the index alone would put a different task on each side of the pair and quietly
    break the only test this apparatus has that can resolve a delta of a dozen episodes.

    ``np.random.default_rng([seed, episode_index])`` is verified on this box's numpy
    1.23.5: a list is accepted as entropy, the draw is deterministic across calls, it
    changes with the index, and exhausting the global RNG between two calls does not
    move it (``test_audio_window.py``).
    """
    opens_at = int(t_anom)
    if opens_at < 0:
        raise ValueError(
            "the window cannot open before the episode starts: t_anom is {}".format(
                t_anom
            )
        )
    budget = int(max_steps)
    if budget <= 0:
        raise ValueError(
            "max_steps must be positive to plan a window against, got {}".format(
                max_steps
            )
        )

    if policy is WindowPolicy.CONTINUOUS:
        return SoundingWindow(opens_at=opens_at, offset_step=None, policy=policy)

    if policy is WindowPolicy.FIXED_STEPS:
        duration = int(sounding_steps)
        if duration < 1:
            raise ValueError(
                "a sounding window of {} steps never sounds, so the offset step would "
                "be at or before t_anom {} -- FIXED_STEPS needs at least 1".format(
                    duration, opens_at
                )
            )
    elif policy is WindowPolicy.BUDGET_FRACTION:
        fraction = float(budget_fraction)
        if not (0.0 < fraction <= 1.0):
            raise ValueError(
                "budget_fraction must be in (0.0, 1.0], got {} against max_steps "
                "{}".format(budget_fraction, budget)
            )
        duration = max(1, int(math.floor(fraction * budget)))
    elif policy is WindowPolicy.DRAWN:
        lo, hi = (int(draw_steps_range[0]), int(draw_steps_range[1]))
        if lo < 1 or hi < lo:
            raise ValueError(
                "draw_steps_range must be (lo, hi) with 1 <= lo <= hi, got ({}, "
                "{})".format(lo, hi)
            )
        rng = np.random.default_rng([int(seed), int(episode_index)])
        duration = int(rng.integers(lo, hi + 1))
    else:
        raise ValueError("unhandled window policy {!r}".format(policy))

    return SoundingWindow(
        opens_at=opens_at, offset_step=opens_at + duration, policy=policy
    )
