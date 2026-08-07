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

    def as_dict(self) -> Dict[str, Any]:
        return {
            "step": int(self.step),
            "measured_rms": float(self.measured_rms),
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
    """

    onset_rms: float
    bed_rms: float
    separation_db: float
    n_poses: int
    global_volume: float
    passed: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "onset_rms": float(self.onset_rms),
            "bed_rms": float(self.bed_rms),
            "separation_db": float(self.separation_db),
            "n_poses": int(self.n_poses),
            "global_volume": float(self.global_volume),
            "passed": bool(self.passed),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CalibrationRecord":
        return cls(
            onset_rms=float(data["onset_rms"]),
            bed_rms=float(data["bed_rms"]),
            separation_db=float(data["separation_db"]),
            n_poses=int(data["n_poses"]),
            global_volume=float(data["global_volume"]),
            passed=bool(data.get("passed", True)),
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
    source_xyz: Optional[Xyz] = None
    t_anom: Optional[int] = None
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
            "source_xyz": list(self.source_xyz.as_tuple()) if self.source_xyz else None,
            "t_anom": None if self.t_anom is None else int(self.t_anom),
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
        return cls(
            episode_index=int(data.get("episode_index", 0)),
            scene_id=data.get("scene_id"),
            localization_arm=data.get("localization_arm"),
            detector_arm=data.get("detector_arm"),
            source_xyz=Xyz.from_sequence(source) if source is not None else None,
            t_anom=None if t_anom is None else int(t_anom),
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
