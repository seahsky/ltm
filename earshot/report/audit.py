"""``EpisodeAudit`` — everything privileged or diagnostic. Task spec §5.2.

The other side of ADR-0013's boundary. What an analyst needs and the agent must never
have seen: the ground-truth source position, how far the STOP actually landed from it,
the ``sourceIsVisible()`` history §3.3 calls analyst-only, §3's provenance evidence, the
calibration gate's margin, the funnel stage, and the per-step audio render bill.

``AudioContextReport`` nests here, which is map requirement 1(c) satisfied by location
rather than by a third artefact.

## The two projections, and why they are not the audio types

ADR-0013's layer graph gives ``report`` exactly ``report``, ``audio.guard`` and
``types``. So this module cannot hold an ``audio.calibration.CalibrationResult`` or an
``audio.onset.OnsetState`` — and that turns out to be the right shape rather than a
constraint worked around, because §5.2 asks for **the separation margin and the
threshold in force**, not for the whole sweep.

The cost of a projection is drift: two dataclasses in two layers that quietly stop
agreeing. That is closed the way ticket 23 closed the frame convention — a *test* sits
outside the layer graph, so ``tests/mac/test_report_audit.py`` imports both and asserts
every projected name still exists on its source. A rename in ``audio/`` fails here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..audio.guard import AudioContextReport
from ..types import Xyz

__all__ = [
    "FunnelStage",
    "StepRecord",
    "OnsetRecord",
    "CalibrationRecord",
    "SoundingWindowRecord",
    "EpisodeAudit",
]


class FunnelStage(IntEnum):
    """Task spec §6's staged funnel, as an ordinal.

    ``IntEnum`` because the stages nest: an episode that resumed the primary task
    necessarily entered INVESTIGATE, so "reached stage N" is a comparison and the
    per-stage counts are ``sum(a.funnel_stage >= stage)``. A plain ``Enum`` would push
    that ordering into whichever module happened to aggregate, in a metric the spec
    warns about precisely because an aggregate has hidden the mechanism here before.

    **The denominator for the loop is stage 2** (§6). §2.5's audibility attrition then
    shows as the drop from stage 2 to stage 3 rather than being absorbed into a single
    fraction that mixes "never heard it" with "heard it and failed to reach it".
    """

    RUN = 1  # the episode ran at all
    T_ANOM_REACHED = 2  # survived to the step the anomaly starts — the denominator
    ONSET_FIRED = 3  # the agent's own threshold crossed (§2.5's attrition lands here)
    INVESTIGATE_ENTERED = 4  # diverted off the primary task
    SOURCE_REACHED = 5  # CHECK: peak-or-plateau plus visual confirm, or the oracle arm
    PRIMARY_RESUMED = 6  # back in SEARCH on the primary goal


@dataclass(frozen=True)
class StepRecord:
    """§3.2's per-step row, recorded at **every** step rather than windowed on the onset.

    ``action`` is the load-bearing one and the reason this is not just an RMS trace:
    ADR-0011 needs a rotation-driven rise in RMS to be distinguishable from a
    translation-driven one after the fact, and only the action taken separates them.

    ``source_is_visible`` is §3.3's analyst-only probe. It is recorded here and read by
    nothing in ``agent/`` — ``tests/mac/test_analyst_only.py`` holds that, because
    feeding it to the decision rule would plant a hidden oracle inside the arm ADR-0001
    built specifically to avoid one.

    ``audio_render_s`` is the per-step wall-clock §6 requires reporting on every run.
    Ticket 06's 27.2 ms at the ``cheap_preset`` is the measurement the whole
    live-every-step feasibility claim rests on, so it is audited per run rather than
    trusted from one sweep.

    ``collided`` and ``displacement_m`` are ticket 26's addition to §3.2's five, and they
    exist for one question the first box run could not answer: **did that forward move,
    or did it hit a wall?** The episode reported 110 forwards for 6.57 m of path with no
    way to tell which, because ``World.step`` returns habitat's collision flag and the
    runner discarded it. Path length cannot substitute — habitat slides an agent along a
    wall, so a collided forward displaces a little rather than nothing.

    ``position`` is the yield-1 addition, for the question *that* run could not answer:
    **was the detour converging when the budget cut it off?** Twelve of twenty episodes
    spent exactly ``investigate_max_steps`` and abandoned the investigation, and nothing
    on disk separated "the source was too far for 120 steps" from "the climb wandered".
    Displacement says the agent moved and ``measured_rms`` is only a proxy for where it
    moved *to*; a position plus the audit's ``source_xyz`` is the distance itself.

    It is recorded here and nowhere near ``AgentReport``: a position the agent may keep
    is its own pose, which §5.1 already carries as ``stopped_at_pose``. What makes this
    privileged is the *pairing* with ``source_xyz``, which is why the derived series
    lives on ``EpisodeAudit`` rather than on this row.

    ``realizable_action`` is what the **cue** said, and ``action`` is what the agent
    **did**. They are different since ticket 26 put a planner in the detour:
    ``realizable_investigate_step`` now names a probe point and the follower routes to
    it, so ``action`` is the follower's and the carried rule's own answer survived only
    on ``ControllerDecision.realizable_action``, which nothing persisted.

    That gap is why it is here. ``rising`` is exactly reconstructible from
    ``measured_rms``, so an analyst can recompute what the rule *should* have answered —
    but with no record of what it *did* answer there is nothing to check the
    reconstruction against, and an unvalidated model of the controller is not a
    measurement. Recording both makes the pair falsifiable: recomputed against recorded,
    on data that already exists. ``None`` on a step where the rule did not run (the
    oracle arm, or any step outside INVESTIGATE), which is a different claim from the
    rule having answered nothing.
    """

    step: int
    measured_rms: float
    lateral_sign: Optional[int] = None
    source_playing: bool = False
    source_is_visible: Optional[bool] = None
    action: Optional[str] = None
    audio_render_s: Optional[float] = None
    collided: Optional[bool] = None
    displacement_m: Optional[float] = None
    position: Optional[Xyz] = None
    realizable_action: Optional[str] = None
    # The navmesh route length from this pose to the source, which `position` paired with
    # `source_xyz` CANNOT give: that derivation is horizontal `xz` distance, and past a few
    # metres in a house the two come apart — another room is a short hop in `xz` and a long
    # walk around a wall, and the sound takes the walk. `eps-1` read its field profile on
    # the `xz` axis and found the gradient inverting beyond 5 m, which is either a real
    # inversion or the axis failing, and nothing on disk could separate them. It is filled
    # by the runner because only `sim/` can query a navmesh (ADR-0013); `None` on a record
    # written before this existed, which reads as *unknown* and never as zero.
    geodesic_to_source: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "step": int(self.step),
            "measured_rms": float(self.measured_rms),
            "geodesic_to_source": self.geodesic_to_source,
            "lateral_sign": self.lateral_sign,
            "source_playing": bool(self.source_playing),
            "source_is_visible": self.source_is_visible,
            "action": self.action,
            "audio_render_s": self.audio_render_s,
            "collided": self.collided,
            "displacement_m": self.displacement_m,
            "position": None if self.position is None else list(self.position.as_tuple()),
            "realizable_action": self.realizable_action,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StepRecord":
        position = data.get("position")
        return cls(
            step=int(data["step"]),
            measured_rms=float(data["measured_rms"]),
            lateral_sign=data.get("lateral_sign"),
            source_playing=bool(data.get("source_playing", False)),
            source_is_visible=data.get("source_is_visible"),
            action=data.get("action"),
            audio_render_s=data.get("audio_render_s"),
            collided=data.get("collided"),
            displacement_m=data.get("displacement_m"),
            # Absent on every record written before yield-1, and absent is not (0, 0, 0):
            # a run from before this field means the distance is unknown, which the
            # derived series reports as None rather than as a distance to the origin.
            position=None if position is None else Xyz.from_sequence(position),
            # Absent on every record written before this field landed, and absent reads
            # as "the rule's answer was not recorded", never as "the rule answered
            # nothing" — the plateau replay must be able to tell an unvalidatable old
            # record from a step where the carried rule genuinely did not run.
            realizable_action=data.get("realizable_action"),
            geodesic_to_source=data.get("geodesic_to_source"),
        )


@dataclass(frozen=True)
class OnsetRecord:
    """§3's provenance evidence, projected off ``audio.onset.OnsetState``.

    ``provenance_asserted`` is the field ticket 16 taught this map to want. The §3.1
    assertions **raise**, so an audit record that exists at all looks like proof they
    passed — unless they were never called, which is exactly the shape ticket 16 found
    in the log canary and ticket 13 found in the version-blind skip. Recording whether
    ``assert_provenance`` actually ran is what makes smoke criterion 4 ("provenance did
    not raise") checkable from the artefact rather than from the absence of a traceback.

    ``n_pre_onset_readings`` is the second half of the same idea, carried through from
    ``OnsetState``: with ``t_anom > 0`` and zero readings, §3.1's first invariant is
    *unverified*, not satisfied.

    ``t_anom`` is deliberately **not** here, though §3.1 states both invariants against
    it. It is a property of the episode as it was built, not of the onset as it was
    measured, and this type is a projection of ``OnsetState`` plus exactly one field the
    audit owns. It lives on ``EpisodeAudit`` beside ``source_xyz``, the builder's other
    per-episode decision.
    """

    onset_step: Optional[int] = None
    pre_onset_rms: Optional[float] = None
    n_pre_onset_readings: int = 0
    provenance_asserted: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "onset_step": self.onset_step,
            "pre_onset_rms": self.pre_onset_rms,
            "n_pre_onset_readings": int(self.n_pre_onset_readings),
            "provenance_asserted": bool(self.provenance_asserted),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OnsetRecord":
        return cls(
            onset_step=data.get("onset_step"),
            pre_onset_rms=data.get("pre_onset_rms"),
            n_pre_onset_readings=int(data.get("n_pre_onset_readings", 0)),
            provenance_asserted=bool(data.get("provenance_asserted", False)),
        )


@dataclass(frozen=True)
class CalibrationRecord:
    """§5.2's "calibration separation margin and the threshold in force".

    A projection of ``audio.calibration.CalibrationResult``, narrowed to what the audit
    needs: the derived threshold, the bed it was separated from, the gate number, and
    the evidence behind it. ``separation_db`` is the one to paste back — the pattern of
    ticket 13's EER 0.00 and the CapRL gate's separation figure.

    ``passed`` is always ``True`` on a real result, because ``calibrate_onset`` raises
    otherwise. It is carried so the serialised record says so explicitly rather than by
    absence, which is the same reason ``provenance_asserted`` exists above.

    **THREE NAMED SCATTER ARMS, because ``climb_eps``' input changed domain TWICE while
    the field name stood still.** They are three estimates of one sentence -- "the spread
    of the reading the climb compares" -- and that sentence stayed true across both
    changes, which is exactly what would have let the domain move under a stable name.

    - ``cue_render_scatter`` is what ``climb_eps`` reads since ADR-0019: the spread of
      successive CUE readouts, ``hop`` samples wide, which is the reading ``is_rising``
      actually compares.
    - ``clip_render_scatter`` is the ADR-0017 arm -- the same folds, read at the clip
      window's width -- and it costs nothing, because both readouts come off one loop.
    - ``single_render_scatter`` is the pre-ADR-0017 arm: independent whole-clip renders
      at the same pose. Measured 1.91x above the clip arm (3.490e-04 against 1.830e-04
      over 400 repeats at a held pose; 3.55x under a second noise model), because
      consecutive readouts share 80% of their samples.

    ``render_scatter`` was RENAMED to ``cue_render_scatter`` rather than redefined, and
    ``as_dict`` no longer emits the legacy key: every number on disk under it is a
    clip-loop estimate (post-ADR-0017) or a whole-clip estimate (before it), and a reader
    differencing across the change under one label would be subtracting two domains. A
    record carrying both the legacy key and a new one would let a reader pick the wrong
    one, so it carries only the new ones. ``from_dict`` maps the legacy key onto whichever
    arm it actually was -- see its own docstring for the disambiguator.

    ``None`` on any arm means it was not run, never that the arms agreed.

    **The phase block is the loop, summarised**, and it is what makes a bursty clip
    identifiable on disk. ``cue_phase_folds`` is how many distinct readings one held pose
    cycles through; ``cue_phase_crest`` and ``cue_phase_min_ratio`` are the median over
    the swept poses of ``max/level`` and ``min/level``; ``cue_phase_aggregation`` names
    the aggregation that produced ``onset_rms``, so the record states it rather than
    leaving it to be inferred. A crest of 2.24 with a min ratio of 0.0 is a source
    audible on one fold in five -- measured for a 0.6 s transient on a 5 s loop -- and
    nothing here is gated on it.
    """

    onset_rms: float
    bed_rms: float
    separation_db: float
    n_poses: int
    global_volume: float
    passed: bool = True
    cue_render_scatter: Optional[float] = None
    cue_scatter_repeats: int = 0
    clip_render_scatter: Optional[float] = None
    clip_scatter_repeats: int = 0
    single_render_scatter: Optional[float] = None
    single_render_repeats: int = 0
    cue_phase_folds: int = 0
    cue_phase_crest: Optional[float] = None
    cue_phase_min_ratio: Optional[float] = None
    cue_phase_aggregation: Optional[str] = None
    profile: Tuple[Tuple[float, float], ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            # The sweep's own (distance, rms) pairs — the field the climb has to climb.
            # The four percentiles below summarise the level and drop the distance axis,
            # and every question about whether a gradient EXISTS at a given range needs
            # the axis. Sixteen pairs, rendered anyway to place the threshold.
            "profile": [[float(d), float(r)] for d, r in self.profile],
            "onset_rms": float(self.onset_rms),
            "bed_rms": float(self.bed_rms),
            "separation_db": float(self.separation_db),
            "n_poses": int(self.n_poses),
            "global_volume": float(self.global_volume),
            "passed": bool(self.passed),
            # The climb's threshold, per episode, in the domain the climb reads. `None`
            # means it was not measured, and the run then used the pre-`detour-2` `1e-6`
            # — a distinction the record has to keep, because "the windowed rule ran" and
            # "the windowed rule ran against a real noise floor" are different claims
            # about the same run.
            "cue_render_scatter": (
                None
                if self.cue_render_scatter is None
                else float(self.cue_render_scatter)
            ),
            "cue_scatter_repeats": int(self.cue_scatter_repeats),
            # The same folds read at the clip window's width: the ADR-0017 arm, free.
            "clip_render_scatter": (
                None
                if self.clip_render_scatter is None
                else float(self.clip_render_scatter)
            ),
            "clip_scatter_repeats": int(self.clip_scatter_repeats),
            # The same pose, the pre-ADR-0017 estimator. Never read by `climb_eps`; the
            # ratio between the three is what a reader needs to compare this run's `eps`
            # with the ones `detour-2` and `eps-1` were tuned at.
            "single_render_scatter": (
                None
                if self.single_render_scatter is None
                else float(self.single_render_scatter)
            ),
            "single_render_repeats": int(self.single_render_repeats),
            # The loop, so a reading that is audible on one fold in five is identifiable
            # rather than merely suffered.
            "cue_phase_folds": int(self.cue_phase_folds),
            "cue_phase_crest": (
                None if self.cue_phase_crest is None else float(self.cue_phase_crest)
            ),
            "cue_phase_min_ratio": (
                None
                if self.cue_phase_min_ratio is None
                else float(self.cue_phase_min_ratio)
            ),
            "cue_phase_aggregation": self.cue_phase_aggregation,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CalibrationRecord":
        """Rebuild, mapping the legacy ``render_scatter`` key onto the arm it really was.

        **The legacy key means two different things in two eras and the disambiguator is
        ``single_render_scatter``'s presence.** That field was added by the same commit
        that made ``render_scatter`` the clip-loop estimate (ADR-0017), so:

        - ``render_scatter`` WITH ``single_render_scatter`` -- the record is post-ADR-0017
          and its ``render_scatter`` IS the clip-loop number, so it lands on
          ``clip_render_scatter``;
        - ``render_scatter`` WITHOUT it -- the record is pre-ADR-0017 (``detour-2``,
          ``eps-1``) and its ``render_scatter`` is the whole-clip number, so it lands on
          ``single_render_scatter``.

        Repeats follow their scatter. Guessing was rejected in favour of writing the rule
        down here, because the two eras' numbers differ by a measured 1.91x and the wrong
        mapping is a silently mispriced epsilon rather than an error.

        **The consequence, stated:** ``cue_render_scatter`` is ``None`` on every record
        written before ADR-0019, so ``climb_eps`` falls back to ``UNMEASURED_EPS`` on a
        replay of an old run. That is correct -- the agent never ran at a cue-domain
        epsilon -- and it is better than silently replaying at a foreign domain's number.
        """
        legacy = data.get("render_scatter")
        legacy_repeats = int(data.get("scatter_repeats", 0))
        single = data.get("single_render_scatter")
        single_repeats = int(data.get("single_render_repeats", 0))
        clip = data.get("clip_render_scatter")
        clip_repeats = int(data.get("clip_scatter_repeats", 0))
        if clip is None and legacy is not None:
            if single is None:
                single, single_repeats = legacy, legacy_repeats
            else:
                clip, clip_repeats = legacy, legacy_repeats
        return cls(
            onset_rms=float(data["onset_rms"]),
            bed_rms=float(data["bed_rms"]),
            separation_db=float(data["separation_db"]),
            n_poses=int(data["n_poses"]),
            global_volume=float(data["global_volume"]),
            passed=bool(data.get("passed", True)),
            cue_render_scatter=(
                None if data.get("cue_render_scatter") is None
                else float(data["cue_render_scatter"])
            ),
            cue_scatter_repeats=int(data.get("cue_scatter_repeats", 0)),
            clip_render_scatter=None if clip is None else float(clip),
            clip_scatter_repeats=clip_repeats,
            single_render_scatter=None if single is None else float(single),
            single_render_repeats=single_repeats,
            cue_phase_folds=int(data.get("cue_phase_folds", 0)),
            cue_phase_crest=(
                None if data.get("cue_phase_crest") is None
                else float(data["cue_phase_crest"])
            ),
            cue_phase_min_ratio=(
                None if data.get("cue_phase_min_ratio") is None
                else float(data["cue_phase_min_ratio"])
            ),
            cue_phase_aggregation=data.get("cue_phase_aggregation"),
            profile=tuple(
                (float(pair[0]), float(pair[1]))
                for pair in (data.get("profile") or ())
            ),
        )


@dataclass(frozen=True)
class SoundingWindowRecord:
    """ADR-0017's window and the accumulator that rendered it, as primitives.

    A projection with primitive fields, exactly as ``CalibrationRecord`` projects
    ``CalibrationResult``: ADR-0013 gives ``report`` only ``report``, ``audio.guard``
    and ``types``, so this module may not name ``audio.window.SoundingWindow`` or
    ``audio.tail.TailState``. ``tests/mac/test_report_audit.py`` imports both sides and
    pins the projection, which is where a rename in ``audio/`` fails.

    **Why each number is on the answer key rather than left to be inferred.**

    ``offset_step`` exists nowhere else on disk. The per-step ``source_playing`` trace
    shows WHEN the source stopped, not what the task ASKED for, and those are different
    claims -- a source that failed to stop leaves a trace that agrees with itself. That
    is the argument ``t_anom`` won above: a record carrying a number derived from a
    bound it does not state cannot be read a year later.

    ``max_ir_samples`` and ``n_buffer_grows`` are the accumulator's own measurement of
    the thing ``audio/spec.py`` deliberately does not cap. There is no ``maxIRLength``
    anywhere in this tree, so a buffer that truncated a wide IR would produce a quiet,
    plausible, wrong tail; recorded, a truncation is visible in the artefact instead of
    inferred from a level that looks a little low.

    ``analysis_window_samples`` KEEPS its meaning across ADR-0019: the buffer's read
    window, ``len(clip)``, the CLIP readout's width. It is **not what the controller
    reads** -- the cue readout is ``hop_samples`` wide, which is already a field above,
    which is why the split added no field for it.

    ``tail_steps`` KEEPS its name and its arithmetic and CHANGED ITS ROLE at ADR-0019. It
    is how long the CLIP readout takes to empty after the last sounding step --
    ``ceil((N + L - 1) / hop)``, so ``tail_steps - 1`` steps past the offset step -- and
    it is **not evidence that the room did any work**: ``audio/tail.py`` measures an
    anechoic 1-sample IR reproducing the same decay to within 1.1 points, because the read
    window (``N``) is always wider than the IR (``L``) in this tree. Read it as the
    analysis window emptying, which is what it mostly is; ``cue_tail_steps`` below is the
    number that IS evidence. Since the split it bounds what CLAP reads rather than what
    the agent reads, smoke criterion 4 no longer measures its fence post from it, and it
    remains ``runner.tail_is_active``'s clause. The name is kept because every audit.json
    on disk uses it and renaming a serialised field reinterprets every record ever
    written -- the argument ``onset_step`` already won in this file against the
    literature's meaning of "onset".

    ``cue_tail_steps`` is NEW at ADR-0019 and is the first number on this record that IS
    evidence the geometric acoustics did any work. ``ceil((hop + L - 1) / hop)``: how long
    the CUE readout -- the ``hop`` samples that arrived during one step, which is what the
    agent reads -- takes to reach exactly zero after the last sounding step. **1 means the
    IR fits inside a step** and the silent phase is an honest hard cut; greater than 1
    means the room outlives a step. Measured 3 at the box's numbers against a
    ``tail_steps`` of 7, and 1 for an anechoic 1-sample IR that leaves ``tail_steps``
    almost unchanged. Smoke criterion 4's fence post is measured from this. ``None`` on
    every record written before the split, which reads as unknown and never as 1.

    **There is no ``cue_ramp_steps`` field, deliberately.** The cue window is written
    whole by one sounding fold, so the cue ramp is the literal 1 (``tail.CUE_RAMP_STEPS``)
    -- and a record field a literal could replace is exactly the hole
    ``TestTheWindowRecordIsTheAccumulatorsOwnMeasurement`` exists to close.

    ``ramp_steps`` KEEPS its name and its arithmetic and ALSO changed role. It is
    ``tail_steps``' mirror at the other end -- ``ceil(N / hop)``, the folds the CLIP read
    window takes to fill. It used to be the correction ``onset_delay_steps`` could not be
    read without; since the split its consumer is the CLAP deferral, which it bounds at
    ``ramp_steps - 1`` steps. Measured CLIP readout over settled level at a fixed pose
    across the first five sounding steps: **0.441 0.629 0.772 0.891 0.997**. Under the old
    readout an agent already inside earshot when the window opens, whose settled level sits
    exactly at ``onset_rms``, crossed **4 steps late**; at 1.3x the threshold, 2 steps
    late; at 5x, on the first step. That bias is gone from the cue readout, and the curve
    is kept here labelled as the clip readout's so a reader of a pre-split run can still
    read that run.

    ``post_offset_audible_steps`` is the MEASURED half and the one a reader should
    believe: how many silent-phase steps the agent's own reading stayed outside
    ``pre_onset_rms_tol`` of the bed. Since ADR-0019 it is measured on the CUE trace, so
    it counts steps at which the ROOM was still audible and its values FALL -- that is the
    correction, not a regression. It can be far below ``cue_tail_steps`` -- zero, for a
    transient clip whose loop rang last four steps before the window closed -- and zero
    means the silent phase arrived as a hard cut. ``None`` is not measured, never 0.

    ``hop_samples`` and ``step_seconds`` are the invented unit (``AudioConfig.
    step_seconds``, ``provenance: fake``) that a window duration in STEPS has to be read
    through before it can be cross-quoted against SAVN-CE's 15 s mean.
    """

    opens_at: int
    offset_step: Optional[int] = None
    policy: Optional[str] = None
    step_seconds: Optional[float] = None
    hop_samples: Optional[int] = None
    analysis_window_samples: Optional[int] = None
    max_ir_samples: Optional[int] = None
    n_buffer_grows: int = 0
    tail_steps: Optional[int] = None
    ramp_steps: Optional[int] = None
    post_offset_audible_steps: Optional[int] = None
    cue_tail_steps: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "opens_at": int(self.opens_at),
            "offset_step": None if self.offset_step is None else int(self.offset_step),
            "policy": self.policy,
            "post_offset_audible_steps": (
                None
                if self.post_offset_audible_steps is None
                else int(self.post_offset_audible_steps)
            ),
            "step_seconds": (
                None if self.step_seconds is None else float(self.step_seconds)
            ),
            "hop_samples": None if self.hop_samples is None else int(self.hop_samples),
            "analysis_window_samples": (
                None
                if self.analysis_window_samples is None
                else int(self.analysis_window_samples)
            ),
            "max_ir_samples": (
                None if self.max_ir_samples is None else int(self.max_ir_samples)
            ),
            "n_buffer_grows": int(self.n_buffer_grows),
            "tail_steps": None if self.tail_steps is None else int(self.tail_steps),
            "cue_tail_steps": (
                None if self.cue_tail_steps is None else int(self.cue_tail_steps)
            ),
            "ramp_steps": None if self.ramp_steps is None else int(self.ramp_steps),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SoundingWindowRecord":
        def _int(key: str) -> Optional[int]:
            value = data.get(key)
            return None if value is None else int(value)

        return cls(
            opens_at=int(data.get("opens_at", 0)),
            offset_step=_int("offset_step"),
            policy=data.get("policy"),
            step_seconds=(
                None if data.get("step_seconds") is None
                else float(data["step_seconds"])
            ),
            hop_samples=_int("hop_samples"),
            analysis_window_samples=_int("analysis_window_samples"),
            max_ir_samples=_int("max_ir_samples"),
            n_buffer_grows=int(data.get("n_buffer_grows", 0)),
            tail_steps=_int("tail_steps"),
            cue_tail_steps=_int("cue_tail_steps"),
            ramp_steps=_int("ramp_steps"),
            post_offset_audible_steps=_int("post_offset_audible_steps"),
        )


@dataclass(frozen=True)
class EpisodeAudit:
    """The answer key. Never handed to a reviewer alongside the testimony.

    ``source_xyz`` and ``dist_at_stop`` are the two names ``test_report_boundary.py``
    watches: they live here and they must never appear in ``AgentReport``.

    ``localization_arm`` and ``detector_arm`` are recorded because "which arm ran" has
    to be readable off the artefact. The spec requires an identical *testimony* schema
    in both arms, which is precisely what makes the arm invisible there — so if it were
    not recorded here it would be recorded nowhere.

    ``t_anom`` is here for the same reason and was found the same way. It is derived per
    episode (``task.dataset.derive_t_anom``), so ``run_config`` holds ``None`` unless a
    run pinned one — and ``funnel_stage`` right below it is *computed* from ``t_anom``.
    A record carrying a number derived from a bound it does not state cannot be read a
    year later, which is what this whole type is for.
    """

    episode_index: int = 0
    scene_id: Optional[str] = None
    localization_arm: Optional[str] = None
    detector_arm: Optional[str] = None
    # ADR-0018's four ablation arms, for `localization_arm`'s reason and found the same
    # way: the run config is written once per RUN (`summary.json`) and the comparison
    # this tree keeps making is per EPISODE. `tools/episode_diff.py` pairs two sweeps by
    # episode index, and without these it can pair an episode that ran with the climb
    # against one that ran without it and call the difference a result.
    #
    # Held as the enum's `.value` string, never the enum: `metrics` is `Mapping[str,
    # float]` and every reader in the tree does `float(value)` on it, so an arm cannot
    # live there. `None` means the record predates the arms — which is NOT the same fact
    # as "the arm was off", and is why the default is None rather than `"live"`.
    climb_rule: Optional[str] = None
    lateral_cue: Optional[str] = None
    cast_policy: Optional[str] = None
    ir_policy: Optional[str] = None
    # ADR-0018's matrix cell, on the same terms as the four arms above and for the same
    # reason: `episode_diff` pairs by episode index, so a cell that is only in
    # `summary.json` lets a `heard_seen` episode be subtracted from an `unseen_unheard`
    # one. Held as `MemoryCondition.value`, never the enum, because `metrics` is
    # `Mapping[str, float]` and a string cannot live there.
    #
    # `None` means the record predates the field, which is NOT the same fact as
    # `MemoryCondition.NONE` ("this run had no memory arm"). The writer landed with the
    # field, so no record can carry a `None` that means the second thing.
    memory_condition: Optional[str] = None
    # What the memory actually said, once, at the step the room went quiet. Exactly one
    # of these is ever set on a run with a live memory arm: the category it recalled, or
    # the named reason it recalled nothing (`memory_prior.PriorMiss`). Both `None` means
    # the prior was never consulted -- the source never went silent while investigating,
    # or there is no memory arm -- and that is a third fact again.
    memory_prior_category: Optional[str] = None
    memory_prior_miss: Optional[str] = None
    source_xyz: Optional[Xyz] = None
    t_anom: Optional[int] = None
    # ADR-0017's window, beside the step it opens at. `None` on every record written
    # before the window existed, which reads as "unknown" rather than as "continuous".
    sounding_window: Optional["SoundingWindowRecord"] = None
    # The step the agent reached the SOUND SOURCE, which was recoverable from nothing.
    # The primary STOP is `len(steps) - 1`; the source reach was not written down at all
    # -- `InvestigationEvent.investigate_steps` is a RELATIVE count of INVESTIGATE ticks
    # and the ORACLE arm leaves no `realizable_action` trail to read it off. So SWS
    # could not have been computed from any artefact this tree wrote before this field.
    source_reached_step: Optional[int] = None
    dist_at_stop: Optional[float] = None
    funnel_stage: FunnelStage = FunnelStage.RUN
    onset: Optional[OnsetRecord] = None
    calibration: Optional[CalibrationRecord] = None
    audio_context: Optional[AudioContextReport] = None
    steps: Tuple[StepRecord, ...] = ()
    # provenance: runtime — whatever the run's metric layer computed. Held as a mapping
    # rather than as fields because §6 deliberately reports several numbers that are
    # NOT headlines (soft-SPL is computed and not led with, benchmark SPL is computed
    # and never cross-quoted from this map), and giving each a named field here would
    # invite exactly the cross-quoting ADR-0005/0006 ruled out.
    metrics: Mapping[str, float] = field(default_factory=dict)

    @property
    def source_is_visible_history(self) -> Tuple[Optional[bool], ...]:
        """§5.2's history, derived rather than stored.

        Two copies of the same series — a list here and the same values inside
        ``steps`` — is a drift trap with no upside, and the per-step row already has to
        carry it for §3.2. Deriving it also keeps it aligned with ``action`` by
        construction, which is the pairing an analyst actually reads.
        """
        return tuple(row.source_is_visible for row in self.steps)

    @property
    def geodesic_to_source_history(self) -> Tuple[Optional[float], ...]:
        """Navmesh route length from the agent to the source, per step. Recorded.

        Stored rather than derived, unlike the horizontal series below, because it cannot
        be derived here: a route needs the navmesh and the navmesh lives behind ADR-0013's
        boundary. ``None`` throughout on any run before the field existed, which is what
        makes a reader choose the horizontal axis explicitly rather than silently.
        """
        return tuple(row.geodesic_to_source for row in self.steps)

    @property
    def distance_to_source_history(self) -> Tuple[Optional[float], ...]:
        """Horizontal distance from the agent to the source, per step. Derived.

        The pairing that makes ``StepRecord.position`` privileged, kept on this side of
        ADR-0013's boundary and derived rather than stored for the same reason
        ``source_is_visible_history`` is: two copies of one series is a drift trap.

        ``None`` where either half is missing — a record from before the position field
        existed, or an episode with no source. A gap reads as unknown; substituting a
        number there is how an un-measured detour would come to look like a converging
        one.
        """
        source = self.source_xyz
        if source is None:
            return tuple(None for _ in self.steps)
        return tuple(
            None if row.position is None else row.position.horizontal_distance_to(source)
            for row in self.steps
        )

    @property
    def n_render_steps(self) -> int:
        """Steps with a recorded audio render. Smoke criterion 1's numerator.

        Criterion 1 is "render count equals step count exactly", and it is measured on
        the shared ``get_sensor_observations()`` call (ADR-0013) — so a step with no
        ``audio_render_s`` is a step whose audio did not render, which is the failure
        the criterion exists to catch.
        """
        return sum(1 for row in self.steps if row.audio_render_s is not None)

    def audio_render_summary(self) -> Dict[str, float]:
        """Per-step render wall-clock, as §6 requires reporting every run.

        Returns an empty mapping when nothing rendered, rather than zeros: a ceiling
        check against a fabricated 0.0 would pass criterion 7 on an episode that never
        rendered at all.
        """
        values = [row.audio_render_s for row in self.steps if row.audio_render_s is not None]
        if not values:
            return {}
        return {
            "n": float(len(values)),
            "mean_s": sum(values) / len(values),
            "max_s": max(values),
            "min_s": min(values),
            "total_s": sum(values),
        }

    def forward_summary(self) -> Dict[str, float]:
        """How many forwards hit a wall, and how far the rest actually went (ticket 26).

        The number that decides whether the realizable climb needs obstacle awareness or
        a bigger sub-budget. Restricted to ``move_forward`` because a turn displaces
        nothing by design and averaging it in would hide exactly the walls this measures.

        Empty rather than zeros when the episode took no forward, for
        ``audio_render_summary``'s reason: absent is not 0.0.
        """
        rows = [row for row in self.steps if row.action == "move_forward"]
        if not rows:
            return {}
        moved = [row.displacement_m for row in rows if row.displacement_m is not None]
        summary = {
            "n_forward": float(len(rows)),
            "n_collided": float(sum(1 for row in rows if row.collided)),
        }
        if moved:
            summary["total_displacement_m"] = sum(moved)
            summary["mean_displacement_m"] = sum(moved) / len(moved)
        return summary

    def as_dict(self) -> Dict[str, Any]:
        return {
            "episode_index": int(self.episode_index),
            "scene_id": self.scene_id,
            "localization_arm": self.localization_arm,
            "detector_arm": self.detector_arm,
            "climb_rule": self.climb_rule,
            "memory_condition": self.memory_condition,
            "memory_prior_category": self.memory_prior_category,
            "memory_prior_miss": self.memory_prior_miss,
            "lateral_cue": self.lateral_cue,
            "cast_policy": self.cast_policy,
            "ir_policy": self.ir_policy,
            "source_xyz": list(self.source_xyz.as_tuple()) if self.source_xyz else None,
            "t_anom": None if self.t_anom is None else int(self.t_anom),
            "sounding_window": (
                self.sounding_window.as_dict()
                if self.sounding_window is not None
                else None
            ),
            "source_reached_step": (
                None if self.source_reached_step is None else int(self.source_reached_step)
            ),
            "dist_at_stop": self.dist_at_stop,
            "funnel_stage": int(self.funnel_stage),
            "funnel_stage_name": self.funnel_stage.name,
            "onset": self.onset.as_dict() if self.onset is not None else None,
            "calibration": self.calibration.as_dict() if self.calibration is not None else None,
            "audio_context": (
                self.audio_context.as_dict() if self.audio_context is not None else None
            ),
            "steps": [row.as_dict() for row in self.steps],
            "source_is_visible_history": list(self.source_is_visible_history),
            "audio_render_summary": self.audio_render_summary(),
            "forward_summary": self.forward_summary(),
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EpisodeAudit":
        """The inverse. The derived keys are ignored rather than trusted.

        ``source_is_visible_history``, ``audio_render_summary`` and ``forward_summary``
        are serialised for an analyst reading the JSON directly, and re-derived from
        ``steps`` on the way back in. Reading them would let a hand-edited file disagree
        with its own step rows and have the disagreement survive a round trip.
        """
        source = data.get("source_xyz")
        t_anom = data.get("t_anom")
        window = data.get("sounding_window")
        reached = data.get("source_reached_step")
        return cls(
            episode_index=int(data.get("episode_index", 0)),
            scene_id=data.get("scene_id"),
            localization_arm=data.get("localization_arm"),
            detector_arm=data.get("detector_arm"),
            # `.get`, so every record written before the arms existed reads back None —
            # "this run predates the ablation arms", and not "the arm was off".
            climb_rule=data.get("climb_rule"),
            memory_condition=data.get("memory_condition"),
            memory_prior_category=data.get("memory_prior_category"),
            memory_prior_miss=data.get("memory_prior_miss"),
            lateral_cue=data.get("lateral_cue"),
            cast_policy=data.get("cast_policy"),
            ir_policy=data.get("ir_policy"),
            source_xyz=Xyz.from_sequence(source) if source is not None else None,
            t_anom=None if t_anom is None else int(t_anom),
            sounding_window=(
                SoundingWindowRecord.from_dict(window) if window is not None else None
            ),
            source_reached_step=None if reached is None else int(reached),
            dist_at_stop=data.get("dist_at_stop"),
            funnel_stage=FunnelStage(int(data.get("funnel_stage", FunnelStage.RUN))),
            onset=(
                OnsetRecord.from_dict(data["onset"]) if data.get("onset") is not None else None
            ),
            calibration=(
                CalibrationRecord.from_dict(data["calibration"])
                if data.get("calibration") is not None
                else None
            ),
            audio_context=(
                _audio_context_from_dict(data["audio_context"])
                if data.get("audio_context") is not None
                else None
            ),
            steps=tuple(StepRecord.from_dict(row) for row in data.get("steps", ())),
            metrics=dict(data.get("metrics", {})),
        )


def _audio_context_from_dict(data: Mapping[str, Any]) -> AudioContextReport:
    """Rebuild the guard's report from its own ``as_dict``.

    The inverse lives here rather than on ``AudioContextReport`` deliberately.
    ``audio/guard.py`` is a stdlib-only leaf carried verbatim from tickets 12 and 16 and
    verified against the real binary; the round trip is this layer's requirement, not
    the guard's, and the carried file stays untouched.
    """
    shape: Optional[Sequence[int]] = data.get("ir_shape")
    fatal: List[str] = list(data.get("fatal_log_lines", ()))
    return AudioContextReport(
        n_vertices=int(data.get("n_vertices", 0)),
        submitted_n_vertices=data.get("submitted_n_vertices"),
        obj_written=bool(data.get("obj_written", False)),
        ir_peak_abs=float(data.get("ir_peak_abs", 0.0)),
        ir_shape=tuple(shape) if shape is not None else None,
        ray_efficiency=data.get("ray_efficiency"),
        source_is_visible=data.get("source_is_visible"),
        log_chars=int(data.get("log_chars", 0)),
        stdout_chars=int(data.get("stdout_chars", 0)),
        stderr_chars=int(data.get("stderr_chars", 0)),
        log_canary_seen=bool(data.get("log_canary_seen", False)),
        rlr_engine_error=bool(data.get("rlr_engine_error", False)),
        fatal_log_lines=fatal,
        stdout_tail=data.get("stdout_tail", ""),
        stderr_tail=data.get("stderr_tail", ""),
    )
