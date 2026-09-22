"""Does the cue fall while the agent stands still? The hold probe, rendered on the box.

    nrun bash earshot/tools/hold_probe.sh --tag hold-1
    python -m earshot.tools.hold_probe read runs/hold-1

**Why it exists.** The leg replay (PRs #149 to #156) found the cue falling along about
three quarters of sounding cast legs whichever way they walked. It found the detour's
first scan falling too: 184 standing first scans fell on 83.7%, by a median 11.1% per
loop. PR #156 read that as a trend in time. Three things stand between that reading and
a cause, and this probe renders a sequence for each:

- **Selection.** A scan the level rose through is cut by a surge and never read, so the
  recorded scans are the ones that did not rise. The probe renders the same poses and
  the same turns again, and nothing is cut.
- **Heading.** A scan's same-phase pairs are five turns apart, up to 150 degrees. Every
  turn follows the lateral sign, so the scan ends facing the source, and a binaural
  level depends on where the source is relative to the head. The probe holds the same
  position at a fixed heading beside the turning replay.
- **The renderer.** The preset ships ``temporalCoherence: 1``, which reuses paths from
  earlier frames, and nothing has A/B'd it against a leg or a scan. Every sequence is
  rendered once with it on and once with it off, each arm in its own process, and
  nothing else changes.

After the renderer, nothing can fall at a fixed pose: with one IR the cue settles by the
second fold (``audio/tail.py``), and the bed is one fixed array per episode. The probe
does not take that on trust. Its FROZEN arm folds the first held IR again at every hold
reading, which is the pipeline alone on a real IR.

What each pose renders
----------------------

A pose is one complete, static, sounding first scan from a ``full`` run: the population
``leg_replay.by_scan`` reads as "standing, first scan". One pose per episode: the same
episode in two runs of the same behaviour is one sample, not two, so a later run's scan
of an episode already posed is counted and left out. Its source is that episode's.
Every render goes through the run's own ``heard_step``, with the run's clip and bed at
the audio configuration the runs recorded in ``env_report.json``, and the source
sounding throughout, so every reading is settled before it is read.

1. ``rescan``: seated ``walk_in`` steps before the scan, at the heading rebuilt from the
   record, the agent takes the recorded actions through the scan's last turn. That is
   the run's own render sequence from the walk-in on. Its eight scan readings are read
   the way ``by_scan`` reads the recorded ones.
2. ``hold``: the same walk-in, then ``hold`` readings at the scan's first pose with no
   move and no turn.
3. ``teleport``: seated at the scan's first pose directly, then the same hold. This is
   the probe in its simplest form, with no history before the jump.

**Each kind renders in its own World** (``probe_scene``). With temporal coherence on,
what a render returns can depend on the renders before it. In one World the hold would
follow the rescan, which ended at the same pose, and the teleport would follow the hold
at that pose. Opened fresh per kind, every sequence at a pose follows only the same kind
of sequence at the previous pose, with a different source: the structure the run itself
has, one World per scene and ``set_source`` per episode.

**Two arms check the instrument** (ADR-0014), folded off the same renders. FORCED
scales the source's IR by ``FORCED_DECAY`` per scan or hold reading, a fall of known size
in the source that the readout must see; the bed is not scaled, so where the source is
weak the fall is diluted and the check is stricter. It rides on both the rescan and the
hold, the two sequences the branch reads. FROZEN folds the hold's first IR at every
hold reading.

**The walk-in is replayed by action, not by position**, and every position is checked
against the record at ``POSITION_TOLERANCE_M``. A pose whose replay leaves the recorded
path is not read, because its renders are no longer the run's. The heading is the one
thing the record does not carry. It is rebuilt from the walk-in's first forward that
moved one clean step, whose displacement IS the heading, and the recorded turns before
it. The position check tests that rebuild: the anchor forward only lands where the
record says if the seat heading and the turn arithmetic are both right.

How it reads, and the branch it decides
---------------------------------------

Each window's change is ``leg_replay.same_phase_change`` over its readings: pairs one
loop apart, so the loop cancels. The hold's first ``SCAN_READINGS`` readings span what a
scan spans. A population FALLS when the exact two-sided sign test on fell against rose
gives at most ``SIGN_ALPHA`` and the median change per loop is at most ``-MIN_CHANGE``.
RISES is the mirror, and anything else is FLAT. Fewer than ``MIN_POSES`` read poses is
UNREAD.

The branch, fixed before the first box run, read in order (``decide``). The hold, at a
fixed pose and heading, is read before anything the turns could hide:

- NOT_RUN: a population is UNREAD; the recorded scans do not fall at these poses; or a
  FORCED arm does not fall where its own sequence does not rise.
- PIPELINE: FROZEN is not FLAT in either arm.
- RENDERER: the hold falls with TC on and with TC off.
- PRESET: the hold falls with TC on and not with TC off.
- MIXED: the hold falls only with TC off, or rises in either arm; or, with the hold flat
  in both, the rescan falls only with TC off. No branch predicted any of these.
- HEADING: the hold is flat in both arms, and the rescan falls in both.
- PRESET: the hold is flat in both arms, and the rescan falls with TC on only.
- SELECTION: the hold is flat and the rescan does not fall, in both arms.

The teleport rows, the whole-hold rows and the IR level rows are printed and decide
nothing.

Python 3.9 (the SoundSpaces pin).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import pathlib
import statistics
import sys
from dataclasses import dataclass, replace
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from earshot.agent.controller import ACT_FORWARD, ACT_STOP, ACT_TURN_LEFT, ACT_TURN_RIGHT
from earshot.audio.bed import bed_signal
from earshot.audio.clips import as_binaural, rms
from earshot.audio.config import AudioConfig
from earshot.audio.tail import TailState, heard_step, hop_samples, open_tail, phase_folds
from earshot.config import IrPolicy
from earshot.report.artifacts import ENV_REPORT_NAME, episode_paths, read_audit, run_paths
from earshot.report.audit import EpisodeAudit
from earshot.tools.funnel_diff import two_sided_exact_binomial
from earshot.tools.leg_replay import (
    FULL_ARM,
    SCAN_READINGS,
    SOUNDING,
    Scan,
    episode_legs,
    same_phase_change,
)
from earshot.types import Xyz

__all__ = [
    "ARMS",
    "ARM_ORDER",
    "SEQUENCE_KINDS",
    "DEFAULT_WALK_IN",
    "DEFAULT_HOLD",
    "TURN_DEG",
    "STEP_M",
    "POSITION_TOLERANCE_M",
    "FORCED_DECAY",
    "TELEPORT_PREROLL",
    "SIGN_ALPHA",
    "MIN_CHANGE",
    "MIN_POSES",
    "FALLS",
    "RISES",
    "FLAT",
    "UNREAD",
    "NOT_RUN",
    "PIPELINE",
    "SELECTION",
    "RENDERER",
    "PRESET",
    "MIXED",
    "HEADING",
    "POSE_ROWS",
    "ProbePose",
    "RenderConfig",
    "Selection",
    "Trace",
    "PoseResult",
    "ProbeIO",
    "plan_pose",
    "select_poses",
    "audio_config_of",
    "trace_rescan",
    "trace_hold",
    "trace_teleport",
    "probe_scene",
    "yaw_rotation",
    "render_arm",
    "arm_header",
    "scene_payload",
    "trend",
    "decide",
    "readout",
    "read_tag",
    "format_readout",
    "main",
]

# The two arms, by the value of `temporalCoherence` each renders at. `tc1` is the shipped
# preset and comes first, so a crash in `tc0` still leaves the arm every run used.
ARMS: Dict[str, bool] = {"tc1": True, "tc0": False}
ARM_ORDER: Tuple[str, ...] = ("tc1", "tc0")

# The three sequences, each rendered in its own World (`probe_scene`).
SEQUENCE_KINDS: Tuple[str, ...] = ("rescan", "hold", "teleport")

# Steps of the run's own path replayed before the scan. Two whole loops at the shipped
# `phase_folds` of 5, and at least the three folds the cue readout takes to settle.
DEFAULT_WALK_IN = 10
MIN_WALK_IN = 3

# Readings held still. The task's `sounding_steps`: as long as the source ever sounds.
DEFAULT_HOLD = 60

# provenance: source — `sim.world.OBJECTNAV_HM3D.turn_angle_deg`, the embodiment every
# run used. Not imported, because `sim/world.py` imports habitat_sim and this module must
# load on a Mac. `render_arm` checks it against the live World before any render.
TURN_DEG = 30.0

# provenance: source — `OBJECTNAV_HM3D.step_size_m`. A forward that moved this far, give or
# take `CLEAN_STEP_TOL_M`, with no collision, moved along the agent's heading: habitat
# slides a collided agent along the wall, so only a clean step carries the heading.
STEP_M = 0.25
CLEAN_STEP_TOL_M = 0.01

# How far a replayed position may sit from the recorded one. `leg_replay`'s static test
# uses the same 1 cm, for the same reason: well above float32 round trips, well below a
# step.
POSITION_TOLERANCE_M = 0.01

# FORCED's fall per reading. 0.98 is about -9.6% per loop at a period of 5, the size of
# the fall the walking legs measured (-9.0%), so the check asks whether the readout sees
# a fall of the size it exists to explain, amid the real renders' own noise.
FORCED_DECAY = 0.98

# Extra folds of the teleport's first render before its first reading. The cue readout's
# first fold reads 0.9696 of settled (`tail.CUE_RAMP_STEPS`), which would add a rise to
# the first pair. No extra render: the renderer's history is still one render.
TELEPORT_PREROLL = 2

# The readout's thresholds, fixed before the first box run. `MIN_CHANGE` is under a third
# of the -11.1% per loop the standing first scans measured, so a FALLS is a fall that
# could explain a real part of it, and a sign test at 0.01 is not satisfied by a few
# poses.
SIGN_ALPHA = 0.01
MIN_CHANGE = 0.03
MIN_POSES = 20

FALLS = "FALLS"
RISES = "RISES"
FLAT = "FLAT"
UNREAD = "UNREAD"

NOT_RUN = "NOT_RUN"
PIPELINE = "PIPELINE"
SELECTION = "SELECTION"
RENDERER = "RENDERER"
PRESET = "PRESET"
MIXED = "MIXED"
HEADING = "HEADING"

# Why a first scan was not made a pose. Counted, never silent.
EXCLUDED_SHORT = "fewer than walk_in steps before the scan"
EXCLUDED_NO_HEADING = "no clean forward in the walk-in to rebuild the heading from"
EXCLUDED_DUPLICATE = "episode already posed from an earlier run"


# ----------------------------------------------------------------------
# the poses
# ----------------------------------------------------------------------


def _xyz(values: Sequence[float]) -> Xyz:
    return Xyz.from_sequence([float(v) for v in values])


@dataclass(frozen=True)
class ProbePose:
    """One recorded first scan, and everything needed to render it again.

    ``positions`` are the recorded positions from ``walk_in`` steps before the scan
    through its last reading, ``walk_in + SCAN_READINGS`` of them. ``actions`` are the
    actions the agent took at each of those steps but the last. ``seat_yaw`` is the
    heading at the first position and ``scan_yaw`` the heading at the scan's first
    reading, both rebuilt (``plan_pose``).
    """

    run: str
    scene: str
    episode: int
    start_step: int
    source_class: Optional[str]
    period: int
    walk_in: int
    source: Xyz
    seat_yaw: float
    scan_yaw: float
    positions: Tuple[Xyz, ...]
    actions: Tuple[Optional[str], ...]
    recorded_rms: Tuple[float, ...]
    recorded_change: float

    @property
    def key(self) -> Tuple[str, str, int, int]:
        return (self.run, self.scene, self.episode, self.start_step)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run": self.run,
            "scene": self.scene,
            "episode": self.episode,
            "start_step": self.start_step,
            "source_class": self.source_class,
            "period": self.period,
            "walk_in": self.walk_in,
            "source": list(self.source.as_tuple()),
            "seat_yaw": self.seat_yaw,
            "scan_yaw": self.scan_yaw,
            "positions": [list(p.as_tuple()) for p in self.positions],
            "actions": list(self.actions),
            "recorded_rms": list(self.recorded_rms),
            "recorded_change": self.recorded_change,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProbePose":
        return cls(
            run=str(data["run"]),
            scene=str(data["scene"]),
            episode=int(data["episode"]),
            start_step=int(data["start_step"]),
            source_class=None if data["source_class"] is None else str(data["source_class"]),
            period=int(data["period"]),
            walk_in=int(data["walk_in"]),
            source=_xyz(data["source"]),
            seat_yaw=float(data["seat_yaw"]),
            scan_yaw=float(data["scan_yaw"]),
            positions=tuple(_xyz(p) for p in data["positions"]),
            actions=tuple(None if a is None else str(a) for a in data["actions"]),
            recorded_rms=tuple(float(v) for v in data["recorded_rms"]),
            recorded_change=float(data["recorded_change"]),
        )


def _turn_rad(action: Optional[str]) -> float:
    """The heading change one action makes. A left turn is positive (``lateral.py``)."""
    if action == ACT_TURN_LEFT:
        return math.radians(TURN_DEG)
    if action == ACT_TURN_RIGHT:
        return -math.radians(TURN_DEG)
    return 0.0


def _horizontal(a: Xyz, b: Xyz) -> Tuple[float, float]:
    return b.x - a.x, b.z - a.z


def plan_pose(
    audit: EpisodeAudit, scan: Scan, *, walk_in: int
) -> Tuple[Optional[ProbePose], Optional[str]]:
    """The pose for one recorded scan, or why it cannot be one. Pure.

    The heading comes from the walk-in's first clean forward: habitat moves the agent
    along ``(-sin yaw, 0, -cos yaw)`` (the frame ``audio/lateral.py`` states), so a clean
    step's displacement gives the yaw at that step, and the recorded turns before it give
    the yaw at the seat.
    """
    if int(walk_in) < 1:
        raise ValueError("walk_in is at least one step, got {}".format(walk_in))
    rows = audit.steps
    index = {int(r.step): i for i, r in enumerate(rows)}
    start = index[int(scan.start_step)]
    first, end = start - int(walk_in), start + SCAN_READINGS
    if first < 0:
        return None, EXCLUDED_SHORT
    window = rows[first:end]
    if len(window) != int(walk_in) + SCAN_READINGS or any(
            int(r.step) != int(rows[first].step) + k for k, r in enumerate(window)):
        raise ValueError(
            "episode {} of {}/{}: the steps around scan {} are not consecutive. The "
            "runner records every step, so this is a writer fault".format(
                scan.episode, scan.run, scan.scene, scan.start_step))
    positions = [r.position for r in window]
    if any(p is None for p in positions):
        raise ValueError(
            "episode {} of {}/{} has a step with no position near scan {}. Every record "
            "since yield-1 carries one, so this is a writer fault".format(
                scan.episode, scan.run, scan.scene, scan.start_step))
    actions = [r.action for r in window[:-1]]
    if ACT_STOP in actions:
        raise ValueError(
            "episode {} of {}/{} recorded a STOP before scan {} ended. A scan opens on a "
            "plateau and a STOP ends the detour, so the scan cannot have completed".format(
                scan.episode, scan.run, scan.scene, scan.start_step))
    anchor = None
    for k in range(int(walk_in)):
        if actions[k] != ACT_FORWARD or window[k].collided:
            continue
        dx, dz = _horizontal(positions[k], positions[k + 1])
        if abs(math.hypot(dx, dz) - STEP_M) <= CLEAN_STEP_TOL_M:
            anchor = k
            break
    if anchor is None:
        return None, EXCLUDED_NO_HEADING
    dx, dz = _horizontal(positions[anchor], positions[anchor + 1])
    seat_yaw = math.atan2(-dx, -dz) - sum(_turn_rad(a) for a in actions[:anchor])
    scan_yaw = seat_yaw + sum(_turn_rad(a) for a in actions[: int(walk_in)])
    scan_rows = window[int(walk_in):]
    source = audit.source_xyz
    if source is None:
        raise ValueError(
            "episode {} of {}/{} records no source_xyz, and a pose with no source has "
            "nothing to render".format(scan.episode, scan.run, scan.scene))
    if scan.phase_folds is None or scan.change is None:
        raise ValueError(
            "scan {} of episode {} of {}/{} was not read: plan only the scans by_scan "
            "reads".format(scan.start_step, scan.episode, scan.run, scan.scene))
    return ProbePose(
        run=scan.run,
        scene=scan.scene,
        episode=int(scan.episode),
        start_step=int(scan.start_step),
        source_class=audit.source_class,
        period=int(scan.phase_folds),
        walk_in=int(walk_in),
        source=source,
        seat_yaw=seat_yaw,
        scan_yaw=scan_yaw,
        positions=tuple(positions),
        actions=tuple(actions),
        recorded_rms=tuple(float(r.measured_rms) for r in scan_rows),
        recorded_change=float(scan.change),
    ), None


@dataclass(frozen=True)
class RenderConfig:
    """What the runs rendered with, off each scene's ``env_report.json``.

    ``audio`` is the recorded ``AudioConfig`` as a mapping. The probe renders at it, with
    ``temporal_coherence`` the only field it changes.
    """

    audio: Mapping[str, Any]
    anomaly_class: str
    anomaly_clip: Optional[str]
    split: str

    def as_dict(self) -> Dict[str, Any]:
        return {"audio": dict(self.audio), "anomaly_class": self.anomaly_class,
                "anomaly_clip": self.anomaly_clip, "split": self.split}


@dataclass(frozen=True)
class Selection:
    """Every pose the runs give, what they rendered with, what was left out and why, or
    why the runs are refused."""

    poses: Tuple[ProbePose, ...]
    config: Optional[RenderConfig]
    n_first_scans: int
    excluded: Mapping[str, int]
    refusals: Tuple[str, ...]


def _label(path: pathlib.Path) -> str:
    """``oracle-2/full`` for ``runs/oracle-2/full``, as ``leg_replay`` names a run."""
    return "/".join(path.resolve().parts[-2:])


def _is_read_first_scan(scan: Scan) -> bool:
    """The scans ``by_scan`` reads as "standing, first scan"."""
    return (scan.first and scan.complete and bool(scan.static)
            and scan.sounding == SOUNDING and scan.change is not None)


def _render_config(scene_dir: pathlib.Path) -> Tuple[Optional[RenderConfig], Optional[str]]:
    """The scene's recorded render configuration, or why it cannot be used."""
    path = scene_dir / ENV_REPORT_NAME
    if not path.is_file():
        return None, (
            "{} is missing: without the run's recorded configuration the probe cannot "
            "render what the run rendered".format(path))
    run_config = json.loads(path.read_text(encoding="utf-8"))["run_config"]
    if run_config["ir_policy"] != IrPolicy.FULL.value:
        return None, (
            "{} ran ir_policy {!r}. The probe convolves the room's IR as the renderer "
            "returns it, which is {!r} alone".format(
                scene_dir, run_config["ir_policy"], IrPolicy.FULL.value))
    return RenderConfig(
        audio=dict(run_config["audio"]),
        anomaly_class=str(run_config["anomaly_class"]),
        anomaly_clip=run_config.get("anomaly_clip"),
        split=str(run_config["split"]),
    ), None


def select_poses(
    arm_dirs: Sequence[str], *, walk_in: int, scenes: Optional[Sequence[str]] = None
) -> Selection:
    """The poses under each ``full`` arm directory, ``<tag>/full``. Reads only.

    One pose per episode: the first run named wins, and a later run's scan of the same
    ``(scene, episode)`` is counted as a duplicate. Every scene must have rendered with
    one configuration, or the runs are refused.
    """
    from earshot.task.smoke import episode_indices

    wanted = None if not scenes else set(scenes)
    poses: List[ProbePose] = []
    posed: Dict[Tuple[str, int], str] = {}
    configs: Dict[str, List[str]] = {}
    config: Optional[RenderConfig] = None
    excluded: Dict[str, int] = {}
    refusals: List[str] = []
    n_first = 0

    def exclude(why: str) -> None:
        excluded[why] = excluded.get(why, 0) + 1

    for arm_dir in arm_dirs:
        root = pathlib.Path(arm_dir)
        label = _label(root)
        _, episodes_dir = run_paths(root)
        if episodes_dir.is_dir():
            refusals.append(
                "{} is a scene directory: it holds episodes itself. Pass the arm directory "
                "above it, <tag>/full.".format(root))
            continue
        if not root.is_dir():
            refusals.append("{} is not a directory".format(root))
            continue
        n_episodes = 0
        mismatched: Dict[Tuple[str, Optional[str]], int] = {}
        for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if wanted is not None and scene_dir.name not in wanted:
                continue
            scene_config, why = _render_config(scene_dir)
            if scene_config is None:
                refusals.append(str(why))
                continue
            config = config or scene_config
            configs.setdefault(json.dumps(scene_config.as_dict(), sort_keys=True), []).append(
                "{}/{}".format(label, scene_dir.name))
            for index in episode_indices(str(scene_dir)):
                _, audit_path = episode_paths(scene_dir, index)
                audit = read_audit(audit_path)
                n_episodes += 1
                for name, value in FULL_ARM:
                    got = getattr(audit, name)
                    if got != value:
                        mismatched[(name, got)] = mismatched.get((name, got), 0) + 1
                if audit.source_class not in (None, scene_config.anomaly_class):
                    refusals.append(
                        "{}/{} episode {} sounded {!r} and the run's clip is {!r}: one "
                        "probe renders one clip".format(
                            label, scene_dir.name, index, audit.source_class,
                            scene_config.anomaly_class))
                replay = episode_legs(audit, run=label, scene=scene_dir.name)
                for scan in replay.scans:
                    if not _is_read_first_scan(scan):
                        continue
                    n_first += 1
                    if (scene_dir.name, int(scan.episode)) in posed:
                        exclude(EXCLUDED_DUPLICATE)
                        continue
                    pose, why = plan_pose(audit, scan, walk_in=walk_in)
                    if pose is None:
                        exclude(str(why))
                        continue
                    posed[(scene_dir.name, int(scan.episode))] = label
                    poses.append(pose)
        if not n_episodes:
            refusals.append("no episode records under {}".format(root))
        for (name, got), count in sorted(mismatched.items(), key=lambda kv: str(kv[0])):
            refusals.append(
                "{} is not `full`'s rule: {} is {} on {} of {} episode(s). The scans are "
                "rebuilt from `full`'s cycle and no other".format(
                    label, name, "unrecorded" if got is None else repr(got), count,
                    n_episodes))
    if len(configs) > 1:
        refusals.append(
            "the scenes rendered with {} different configurations; one probe renders one. "
            "Groups: {}".format(len(configs), "; ".join(
                ", ".join(members[:3]) + (" ..." if len(members) > 3 else "")
                for members in configs.values())))
    return Selection(tuple(poses), config, n_first, dict(sorted(excluded.items())),
                     tuple(refusals))


def audio_config_of(recorded: Mapping[str, Any], *, temporal_coherence: bool) -> AudioConfig:
    """The runs' recorded ``AudioConfig``, with ``temporal_coherence`` set for one arm.

    A field the record carries and this tree no longer has is refused: the run rendered
    under a setting nobody can reproduce. A field the record lacks is a knob added after
    the run, and its default is by convention the behaviour before it existed.
    """
    fields = set(AudioConfig.__dataclass_fields__)
    unknown = sorted(set(recorded) - fields)
    if unknown:
        raise ValueError(
            "the runs recorded audio field(s) {} that AudioConfig no longer has, so the "
            "probe cannot render what they rendered".format(unknown))
    values = {k: tuple(v) if isinstance(v, list) else v for k, v in recorded.items()}
    return replace(AudioConfig(**values), temporal_coherence=bool(temporal_coherence))


# ----------------------------------------------------------------------
# the renders
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeIO:
    """The five things a sequence does to the world. The box wires them to a ``World``;
    a Mac test wires them to a fake renderer."""

    seat: Callable[[Xyz, float], None]
    render: Callable[[], Any]
    act: Callable[[str], None]
    where: Callable[[], Xyz]
    place_source: Callable[[Xyz], None]


@dataclass(frozen=True)
class Trace:
    """What one sequence heard. Every list has one entry per render, in order.

    ``cue`` is ``rms`` of the cue readout, the number the runner records as
    ``measured_rms``. ``ir_level`` is the root of the IR's summed energy over both ears:
    the renderer's own output, with no clip, no loop and no pipeline. ``forced`` is the
    power check on the rescan and the hold; ``frozen`` is the hold's pipeline check. Both
    are empty where a sequence has none. ``read_from`` is the first render the readout
    reads. ``diverged_at`` is the first render whose position left the record, and a
    trace with one is not read.
    """

    cue: Tuple[float, ...] = ()
    ir_level: Tuple[float, ...] = ()
    forced: Tuple[float, ...] = ()
    frozen: Tuple[float, ...] = ()
    read_from: int = 0
    diverged_at: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cue": list(self.cue),
            "ir_level": list(self.ir_level),
            "forced": list(self.forced),
            "frozen": list(self.frozen),
            "read_from": self.read_from,
            "diverged_at": self.diverged_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Trace":
        return cls(
            cue=tuple(float(v) for v in data["cue"]),
            ir_level=tuple(float(v) for v in data["ir_level"]),
            forced=tuple(float(v) for v in data["forced"]),
            frozen=tuple(float(v) for v in data["frozen"]),
            read_from=int(data["read_from"]),
            diverged_at=None if data["diverged_at"] is None else int(data["diverged_at"]),
        )


def _ir_level(impulse: np.ndarray) -> float:
    return float(math.sqrt(float(np.sum(np.square(impulse, dtype=np.float64)))))


def _fold(
    tail: TailState, impulse: np.ndarray, clip: np.ndarray, bed_cue: np.ndarray
) -> Tuple[TailState, float]:
    """One sounding fold through the run's own composition point, and its reading."""
    nxt, cue = heard_step(tail, ir=impulse, clip=clip, bed_cue=bed_cue, sounding=True)
    return nxt, float(rms(cue))


def _at(where: Xyz, recorded: Xyz) -> bool:
    return math.dist(where.as_tuple(), recorded.as_tuple()) <= POSITION_TOLERANCE_M


def _render(io: ProbeIO) -> np.ndarray:
    return as_binaural(io.render())


def _decay(reading: int) -> np.float32:
    """FORCED's scale at a scan or hold reading; 1 through the walk-in."""
    return np.float32(FORCED_DECAY ** max(0, int(reading)))


def trace_rescan(
    pose: ProbePose, io: ProbeIO, *, clip: np.ndarray, bed_cue: np.ndarray, hop: int
) -> Trace:
    """The walk-in and the scan, by the recorded actions, with every position checked."""
    io.seat(pose.positions[0], pose.seat_yaw)
    real = forced = open_tail(window=len(clip), hop=hop)
    cue: List[float] = []
    levels: List[float] = []
    forced_cue: List[float] = []
    for i, recorded in enumerate(pose.positions):
        if not _at(io.where(), recorded):
            return Trace(tuple(cue), tuple(levels), tuple(forced_cue),
                         read_from=pose.walk_in, diverged_at=i)
        impulse = _render(io)
        real, reading = _fold(real, impulse, clip, bed_cue)
        forced, forced_reading = _fold(
            forced, impulse * _decay(i - pose.walk_in), clip, bed_cue)
        cue.append(reading)
        levels.append(_ir_level(impulse))
        forced_cue.append(forced_reading)
        if i < len(pose.actions) and pose.actions[i] is not None:
            io.act(str(pose.actions[i]))
    return Trace(tuple(cue), tuple(levels), tuple(forced_cue), read_from=pose.walk_in)


def trace_hold(
    pose: ProbePose, io: ProbeIO, *, clip: np.ndarray, bed_cue: np.ndarray, hop: int,
    hold: int,
) -> Trace:
    """The walk-in, then ``hold`` readings at the scan's first pose, and the two checks.

    FORCED and FROZEN fold the same walk-in renders as the real accumulator, so all three
    are settled identically when the hold begins. From the hold on, FORCED folds each
    render scaled by ``FORCED_DECAY ** k`` and FROZEN folds the hold's first render.
    """
    io.seat(pose.positions[0], pose.seat_yaw)
    real = forced = frozen = open_tail(window=len(clip), hop=hop)
    cue: List[float] = []
    levels: List[float] = []
    forced_cue: List[float] = []
    frozen_cue: List[float] = []

    def diverged(at: int) -> Trace:
        return Trace(tuple(cue), tuple(levels), tuple(forced_cue), tuple(frozen_cue),
                     read_from=pose.walk_in, diverged_at=at)

    for i in range(pose.walk_in):
        if not _at(io.where(), pose.positions[i]):
            return diverged(i)
        impulse = _render(io)
        real, reading = _fold(real, impulse, clip, bed_cue)
        forced, forced_reading = _fold(forced, impulse, clip, bed_cue)
        frozen, frozen_reading = _fold(frozen, impulse, clip, bed_cue)
        cue.append(reading)
        levels.append(_ir_level(impulse))
        forced_cue.append(forced_reading)
        frozen_cue.append(frozen_reading)
        if pose.actions[i] is not None:
            io.act(str(pose.actions[i]))
    if not _at(io.where(), pose.positions[pose.walk_in]):
        return diverged(pose.walk_in)
    first: Optional[np.ndarray] = None
    for k in range(int(hold)):
        impulse = _render(io)
        if first is None:
            first = impulse
        real, reading = _fold(real, impulse, clip, bed_cue)
        forced, forced_reading = _fold(forced, impulse * _decay(k), clip, bed_cue)
        frozen, frozen_reading = _fold(frozen, first, clip, bed_cue)
        cue.append(reading)
        levels.append(_ir_level(impulse))
        forced_cue.append(forced_reading)
        frozen_cue.append(frozen_reading)
    return Trace(tuple(cue), tuple(levels), tuple(forced_cue), tuple(frozen_cue),
                 read_from=pose.walk_in)


def trace_teleport(
    pose: ProbePose, io: ProbeIO, *, clip: np.ndarray, bed_cue: np.ndarray, hop: int,
    hold: int,
) -> Trace:
    """Seated straight at the scan's first pose, then ``hold`` readings without moving."""
    io.seat(pose.positions[pose.walk_in], pose.scan_yaw)
    if not _at(io.where(), pose.positions[pose.walk_in]):
        return Trace(diverged_at=0)
    tail = open_tail(window=len(clip), hop=hop)
    cue: List[float] = []
    levels: List[float] = []
    for k in range(int(hold)):
        impulse = _render(io)
        if k == 0:
            for _ in range(TELEPORT_PREROLL):
                tail, _reading = _fold(tail, impulse, clip, bed_cue)
        tail, reading = _fold(tail, impulse, clip, bed_cue)
        cue.append(reading)
        levels.append(_ir_level(impulse))
    return Trace(tuple(cue), tuple(levels), read_from=0)


@dataclass(frozen=True)
class PoseResult:
    """One pose's three sequences in one arm."""

    pose: ProbePose
    rescan: Trace
    hold: Trace
    teleport: Trace

    @property
    def diverged(self) -> bool:
        return any(t.diverged_at is not None for t in (self.rescan, self.hold, self.teleport))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pose": self.pose.as_dict(),
            "rescan": self.rescan.as_dict(),
            "hold": self.hold.as_dict(),
            "teleport": self.teleport.as_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PoseResult":
        return cls(
            pose=ProbePose.from_dict(data["pose"]),
            rescan=Trace.from_dict(data["rescan"]),
            hold=Trace.from_dict(data["hold"]),
            teleport=Trace.from_dict(data["teleport"]),
        )


def probe_scene(
    poses: Sequence[ProbePose],
    fresh: Callable[[], ContextManager[ProbeIO]],
    *,
    clip: np.ndarray,
    bed_cue: np.ndarray,
    hop: int,
    hold: int,
    progress: Callable[[str], None] = print,
) -> List[PoseResult]:
    """Every pose in one scene, each sequence kind in a World of its own.

    ``fresh()`` opens a new World and closes it on exit. Within one, the poses follow each
    other with ``place_source`` between them, as the run's episodes did, so a sequence at
    a pose is only ever preceded by the same kind of sequence at another pose and source.
    """
    tracers: Dict[str, Callable[[ProbePose, ProbeIO], Trace]] = {
        "rescan": lambda pose, io: trace_rescan(
            pose, io, clip=clip, bed_cue=bed_cue, hop=hop),
        "hold": lambda pose, io: trace_hold(
            pose, io, clip=clip, bed_cue=bed_cue, hop=hop, hold=hold),
        "teleport": lambda pose, io: trace_teleport(
            pose, io, clip=clip, bed_cue=bed_cue, hop=hop, hold=hold),
    }
    traces: Dict[str, List[Trace]] = {}
    for kind in SEQUENCE_KINDS:
        with fresh() as io:
            traces[kind] = []
            for pose in poses:
                io.place_source(pose.source)
                trace = tracers[kind](pose, io)
                traces[kind].append(trace)
                progress("  {} episode {} step {}: {}".format(
                    kind, pose.episode, pose.start_step,
                    "rendered" if trace.diverged_at is None
                    else "DIVERGED at render {}".format(trace.diverged_at)))
    return [
        PoseResult(pose=pose, rescan=traces["rescan"][i], hold=traces["hold"][i],
                   teleport=traces["teleport"][i])
        for i, pose in enumerate(poses)
    ]


def yaw_rotation(yaw: float) -> Tuple[float, float, float, float]:
    """``[x, y, z, w]``, the dataset's order ``World.set_pose`` takes, for a yaw about +y."""
    return (0.0, math.sin(yaw / 2.0), 0.0, math.cos(yaw / 2.0))


def _wrapped(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def arm_header(
    arm: str, *, applied: Mapping[str, Any], config: RenderConfig, clip: str,
    clip_samples: int, hop: int, period: int, hold: int, n_poses: int,
) -> Dict[str, Any]:
    """``<tag>/<arm>/arm.json``: what the arm rendered with, read back from the spec."""
    return {
        "arm": arm,
        "temporal_coherence": ARMS[arm],
        "acoustics": {k: (v if isinstance(v, (int, float)) else str(v))
                      for k, v in applied.items()},
        "recorded": config.as_dict(),
        "clip": clip,
        "clip_samples": int(clip_samples),
        "hop": int(hop),
        "phase_folds": int(period),
        "hold": int(hold),
        "n_poses": int(n_poses),
    }


def scene_payload(
    label: str, error: Optional[str], results: Sequence[PoseResult]
) -> Dict[str, Any]:
    """``<tag>/<arm>/<scene>.json``: every pose rendered there, and the scene's error."""
    return {"scene": label, "error": error, "poses": [r.as_dict() for r in results]}


def _write(path: pathlib.Path, payload: Any) -> None:
    """Write once. ``x`` mode refuses a file that exists: one directory is one run."""
    with open(str(path), "x", encoding="utf-8") as sink:
        json.dump(payload, sink, indent=2, sort_keys=True)
        sink.write("\n")


def render_arm(
    *,
    out_dir: str,
    arm: str,
    selection: Selection,
    hold: int,
    data_root: str,
    progress: Callable[[str], None] = print,
) -> int:
    """Render every pose in one arm, three Worlds per scene. The box half.

    Habitat is imported inside, for ``runner.run``'s reason: ``sim/world.py`` imports
    habitat_sim, so a module-level import would make this file uncollectable on a Mac.
    Continue-on-failure at the scene grain, like ``prior_pass``: a scene that fails is
    written with its error and no poses, and the rest still render. Returns how many
    scenes rendered.
    """
    from earshot.audio.clips import load_anomaly_clip, resolve_anomaly_clip
    from earshot.audio.sensor import AudioSensorHandle
    from earshot.audio.spec import ACOUSTICS_PRESET, audio_sensor_spec
    from earshot.env_check import assert_env
    from earshot.sim.world import World, audio_spec_parts, camera_sensor_specs
    from earshot.task.episodes import find_scenes_dir, find_split_dir, load_scene

    if arm not in ARMS:
        raise ValueError("unknown arm {!r}; the arms are {}".format(arm, list(ARMS)))
    config = selection.config
    if config is None or not selection.poses:
        raise ValueError("nothing to render: the selection has no pose")
    arm_dir = pathlib.Path(out_dir) / arm
    if arm_dir.exists() and any(arm_dir.iterdir()):
        raise FileExistsError(
            "{} already holds files. One directory is one run: pick a fresh tag".format(
                arm_dir))

    env = assert_env(clap=False)
    progress(env.summary())
    audio_cfg = audio_config_of(config.audio, temporal_coherence=ARMS[arm])
    clip_path = resolve_anomaly_clip(
        config.anomaly_class, config.anomaly_clip, audio_cfg.clip_dir)
    if not clip_path:
        raise FileNotFoundError(
            "no staged clip for {!r} under {}: stage ESC-50 first".format(
                config.anomaly_class, audio_cfg.clip_dir))
    clip = load_anomaly_clip(clip_path, audio_cfg.sample_rate, audio_cfg.target_norm_rms_db)
    hop = hop_samples(step_seconds=audio_cfg.step_seconds, sample_rate=audio_cfg.sample_rate)
    period = phase_folds(window=len(clip), hop=hop)
    periods = sorted({p.period for p in selection.poses})
    if periods != [period]:
        raise ValueError(
            "the runs recorded a loop period of {} and this clip gives {}: it is not the "
            "clip the runs rendered".format(periods, period))
    bed_cue = bed_signal(hop, audio_cfg.bed_rms)

    spec, binaural = audio_spec_parts()
    audio_sensor_spec(spec, audio_cfg, binaural)
    applied = {key: getattr(spec.acousticsConfig, key) for key in ACOUSTICS_PRESET}
    if bool(applied["temporalCoherence"]) != ARMS[arm]:
        raise ValueError(
            "the spec reads temporalCoherence {!r} after configuration, and arm {} renders "
            "at {}".format(applied["temporalCoherence"], arm, ARMS[arm]))
    arm_dir.mkdir(parents=True, exist_ok=True)
    _write(arm_dir / "arm.json", arm_header(
        arm, applied=applied, config=config, clip=str(clip_path), clip_samples=len(clip),
        hop=hop, period=period, hold=hold, n_poses=len(selection.poses)))
    progress("arm {}: {}".format(arm, ", ".join(
        "{} {}".format(k, v) for k, v in sorted(applied.items()))))

    split_dir = find_split_dir(config.split, root=data_root)
    scenes_dir = find_scenes_dir(root=data_root)
    by_scene: Dict[str, List[ProbePose]] = {}
    for pose in selection.poses:
        by_scene.setdefault(pose.scene, []).append(pose)

    rendered = 0
    for label in sorted(by_scene):
        scene_poses = by_scene[label]
        progress("[scene] {}: {} pose(s)".format(label, len(scene_poses)))
        error: Optional[str] = None
        results: List[PoseResult] = []
        try:
            scene_path = load_scene(split_dir, label, scenes_dir=scenes_dir).scene_path

            @contextlib.contextmanager
            def fresh() -> Iterator[ProbeIO]:
                world = World(scene_path, list(camera_sensor_specs()) + [spec])
                try:
                    if float(world.agent_spec.turn_angle_deg) != TURN_DEG:
                        raise ValueError(
                            "the World turns {} degrees and the heading rebuild assumes "
                            "{}".format(world.agent_spec.turn_angle_deg, TURN_DEG))
                    # Seated before arming: the arming render rejects a silent IR.
                    world.set_pose(scene_poses[0].positions[0])
                    handle = AudioSensorHandle(
                        world.sensor_handle(str(spec.uuid)), world.observe,
                        scene_poses[0].source, uuid=str(spec.uuid))

                    def seat(position: Xyz, yaw: float) -> None:
                        world.set_pose(position, yaw_rotation(yaw))
                        got = world.pose().yaw_rad
                        if abs(_wrapped(got - yaw)) > 1e-3:
                            raise ValueError(
                                "seated at yaw {:.6f} and the World reads {:.6f}: the "
                                "rotation did not take as written".format(yaw, got))

                    yield ProbeIO(
                        seat=seat,
                        render=lambda: handle.audio_of(handle.observe()[0]),
                        act=lambda action: world.step(action),
                        where=lambda: world.pose().position,
                        place_source=handle.set_source,
                    )
                finally:
                    world.close()

            results = probe_scene(scene_poses, fresh, clip=clip, bed_cue=bed_cue, hop=hop,
                                  hold=hold, progress=progress)
        except Exception as exc:  # noqa: BLE001 -- one scene must not cost the rest
            error = "{}: {}".format(type(exc).__name__, exc)
            results = []
            progress("  WARN: {} failed ({}) -- continuing".format(label, error))
        _write(arm_dir / "{}.json".format(label), scene_payload(label, error, results))
        rendered += 0 if error else 1
    return rendered


# ----------------------------------------------------------------------
# the readout
# ----------------------------------------------------------------------


def trend(changes: Sequence[float]) -> Dict[str, Any]:
    """How a population of per-pose changes moved, and its verdict. Pure.

    A change of exactly zero is a tie and counts neither way in the sign test.
    """
    values = [float(c) for c in changes]
    fell = sum(1 for v in values if v < 0.0)
    rose = sum(1 for v in values if v > 0.0)
    median = statistics.median(values) if values else None
    p = two_sided_exact_binomial(fell, fell + rose)
    if len(values) < MIN_POSES or median is None:
        verdict = UNREAD
    elif p is not None and p <= SIGN_ALPHA and median <= -MIN_CHANGE:
        verdict = FALLS
    elif p is not None and p <= SIGN_ALPHA and median >= MIN_CHANGE:
        verdict = RISES
    else:
        verdict = FLAT
    return {"n": len(values), "fell": fell, "rose": rose, "median": median, "p": p,
            "verdict": verdict}


# Every row read per pose and per arm. The first five decide the branch.
POSE_ROWS: Tuple[str, ...] = (
    "rescan", "hold", "rescan_forced", "hold_forced", "frozen", "hold_whole", "teleport",
    "teleport_whole", "ir_rescan", "ir_hold", "ir_hold_whole", "ir_teleport",
)
_DECIDING = ("rescan", "hold", "rescan_forced", "hold_forced", "frozen")


def decide(verdicts: Mapping[str, Mapping[str, str]]) -> Tuple[str, str]:
    """The pre-registered branch, and why. Pure.

    ``verdicts`` maps ``recorded`` to one verdict under ``"run"``, and each row in
    ``_DECIDING`` to one verdict per arm.
    """
    needed = [("recorded", "run")] + [(row, arm) for row in _DECIDING for arm in ARM_ORDER]
    unread = ["{} {}".format(row, arm) for row, arm in needed
              if verdicts[row][arm] == UNREAD]
    if unread:
        return NOT_RUN, "fewer than {} poses read in: {}".format(MIN_POSES, ", ".join(unread))
    if verdicts["recorded"]["run"] != FALLS:
        return NOT_RUN, (
            "the recorded scans at these poses read {}, so there is no fall here to "
            "explain".format(verdicts["recorded"]["run"]))
    for sequence in ("rescan", "hold"):
        for arm in ARM_ORDER:
            forced = verdicts[sequence + "_forced"][arm]
            if verdicts[sequence][arm] != RISES and forced != FALLS:
                return NOT_RUN, (
                    "FORCED on the {} reads {} on {}: the readout did not see a {:.0%} "
                    "fall per loop on these renders, so a {} that does not fall cannot "
                    "be told from one it could not see".format(
                        sequence, forced, arm, 1.0 - FORCED_DECAY ** 5, sequence))
    moved = [arm for arm in ARM_ORDER if verdicts["frozen"][arm] != FLAT]
    if moved:
        return PIPELINE, (
            "the first held IR, folded again at every reading, reads {} on {}. The "
            "pipeline after the renderer moves the level at a fixed IR, which "
            "audio/tail.py's steady-state tests say it cannot. Find that defect before "
            "anything else: every cue since ADR-0019 reads through it".format(
                ", ".join(verdicts["frozen"][a] for a in moved), ", ".join(moved)))
    hold = {arm: verdicts["hold"][arm] for arm in ARM_ORDER}
    if hold["tc1"] == FALLS and hold["tc0"] == FALLS:
        return RENDERER, (
            "the level falls at a fixed pose and heading with temporalCoherence on and "
            "off, and the frozen IR holds. The renderer's output changes over repeated "
            "renders at one pose whatever the preset. The IR level rows say whether its "
            "energy falls with it")
    if hold["tc1"] == FALLS:
        return PRESET, (
            "the level falls at a fixed pose and heading with temporalCoherence on and "
            "not with it off. The preset makes the fall. Next: a full sweep with TC off "
            "beside `full`, then the leg replay re-read on it under a new "
            "pre-registration")
    if FALLS in hold.values() or RISES in hold.values():
        return MIXED, (
            "the hold reads {} with temporalCoherence on and {} with it off. No branch "
            "predicted that at a fixed pose and heading: read the table".format(
                hold["tc1"], hold["tc0"]))
    rescan = {arm: verdicts["rescan"][arm] for arm in ARM_ORDER}
    if rescan["tc1"] == FALLS and rescan["tc0"] == FALLS:
        return HEADING, (
            "the level is flat at a fixed heading in both arms, and the turning replay "
            "falls in both. The scan's standing fall is its turns: its same-phase pairs "
            "are five turns apart. It is no evidence of a trend in time. The walking legs "
            "keep their heading, so their fall is still open")
    if rescan["tc1"] == FALLS:
        return PRESET, (
            "the level is flat at a fixed heading in both arms, and the turning replay "
            "falls with temporalCoherence on and not with it off. The preset makes the "
            "fall under turning")
    if rescan["tc0"] == FALLS:
        return MIXED, (
            "the level is flat at a fixed heading, and the turning replay falls only with "
            "temporalCoherence off. No branch predicted that: read the table")
    return SELECTION, (
        "the recorded scans fall; at a fixed heading the level is flat, and the same poses "
        "and turns rendered again read {} with temporalCoherence on and {} with it off. "
        "The candidate named before the run is the surge cut: a scan the level rose "
        "through was never read. The other is a renderer that remembers further back "
        "than the walk-in, and --walk-in is the knob that tests it. The walking legs lost "
        "only 4.4% to surges, so their fall is still open".format(
            rescan["tc1"], rescan["tc0"]))


def _change(values: Sequence[float], start: int, length: int, lag: int) -> Optional[float]:
    window = list(values[start:start + length])
    if len(window) < length:
        return None
    return same_phase_change(window, lag=lag)


def _per_pose(result: PoseResult, hold: int) -> Dict[str, Optional[float]]:
    """Every row's change at one pose in one arm."""
    lag = result.pose.period
    rescan, held, jump = result.rescan, result.hold, result.teleport
    return {
        "rescan": _change(rescan.cue, rescan.read_from, SCAN_READINGS, lag),
        "hold": _change(held.cue, held.read_from, SCAN_READINGS, lag),
        "rescan_forced": _change(rescan.forced, rescan.read_from, SCAN_READINGS, lag),
        "hold_forced": _change(held.forced, held.read_from, SCAN_READINGS, lag),
        "frozen": _change(held.frozen, held.read_from, hold, lag),
        "hold_whole": _change(held.cue, held.read_from, hold, lag),
        "teleport": _change(jump.cue, jump.read_from, SCAN_READINGS, lag),
        "teleport_whole": _change(jump.cue, jump.read_from, hold, lag),
        "ir_rescan": _change(rescan.ir_level, rescan.read_from, SCAN_READINGS, lag),
        "ir_hold": _change(held.ir_level, held.read_from, SCAN_READINGS, lag),
        "ir_hold_whole": _change(held.ir_level, held.read_from, hold, lag),
        "ir_teleport": _change(jump.ir_level, jump.read_from, SCAN_READINGS, lag),
    }


def _loop_profile(results: Sequence[PoseResult], trace: str, series: str,
                  hold: int) -> List[Optional[float]]:
    """Median over poses of each loop's mean level over the first loop's. Pure."""
    by_loop: Dict[int, List[float]] = {}
    for result in results:
        chosen = getattr(result, trace)
        values = list(getattr(chosen, series))[chosen.read_from:chosen.read_from + hold]
        lag = result.pose.period
        loops = [values[j:j + lag] for j in range(0, len(values) - lag + 1, lag)]
        base = statistics.fmean(loops[0]) if loops else 0.0
        if base <= 0.0:
            continue
        for j, loop in enumerate(loops):
            by_loop.setdefault(j, []).append(statistics.fmean(loop) / base)
    return [statistics.median(by_loop[j]) if by_loop.get(j) else None
            for j in range(max(by_loop) + 1 if by_loop else 0)]


def readout(
    arms: Mapping[str, Mapping[Tuple[str, str, int, int], PoseResult]], *, hold: int
) -> Dict[str, Any]:
    """Pair the arms by pose, read every row, and decide. Pure."""
    keys = sorted(set.intersection(*(set(arms[a]) for a in ARM_ORDER)))
    diverged = {a: sum(1 for k in keys if arms[a][k].diverged) for a in ARM_ORDER}
    read = [k for k in keys if not any(arms[a][k].diverged for a in ARM_ORDER)]
    per_pose = {a: [_per_pose(arms[a][k], hold) for k in read] for a in ARM_ORDER}
    rows: Dict[str, Dict[str, Dict[str, Any]]] = {
        "recorded": {"run": trend([arms[ARM_ORDER[0]][k].pose.recorded_change
                                   for k in read])}}
    for row in POSE_ROWS:
        rows[row] = {a: trend([p[row] for p in per_pose[a] if p[row] is not None])
                     for a in ARM_ORDER}
    verdicts = {row: {col: cell["verdict"] for col, cell in cols.items()}
                for row, cols in rows.items()}
    branch, why = decide(verdicts)
    profiles = {
        "{} {} {}".format(trace, series, a): _loop_profile(
            [arms[a][k] for k in read], trace, series, hold)
        for trace, series in (("hold", "cue"), ("teleport", "cue"), ("hold", "ir_level"))
        for a in ARM_ORDER}
    return {
        "n_paired": len(keys),
        "only_in": {a: len(set(arms[a]) - set(keys)) for a in ARM_ORDER},
        "diverged": diverged,
        "n_read": len(read),
        "rows": rows,
        "profiles": profiles,
        "branch": branch,
        "why": why,
    }


def read_tag(tag_dir: str) -> Tuple[Dict[str, Dict[Tuple[str, str, int, int], PoseResult]],
                                    Dict[str, Any], List[str]]:
    """Every arm's results under ``<tag>/<arm>/``, their headers, and every scene error."""
    root = pathlib.Path(tag_dir)
    arms: Dict[str, Dict[Tuple[str, str, int, int], PoseResult]] = {}
    headers: Dict[str, Any] = {}
    errors: List[str] = []
    for arm in ARM_ORDER:
        arm_dir = root / arm
        header_path = arm_dir / "arm.json"
        if not header_path.is_file():
            raise FileNotFoundError(
                "{} has no arm.json: arm {} never started under this tag".format(arm_dir, arm))
        header = json.loads(header_path.read_text(encoding="utf-8"))
        if bool(header["temporal_coherence"]) != ARMS[arm]:
            raise ValueError(
                "{} says temporal_coherence {} and arm {} renders at {}".format(
                    header_path, header["temporal_coherence"], arm, ARMS[arm]))
        headers[arm] = header
        results: Dict[Tuple[str, str, int, int], PoseResult] = {}
        for path in sorted(arm_dir.glob("*.json")):
            if path.name == "arm.json":
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("error"):
                errors.append("{} {}: {}".format(arm, payload["scene"], payload["error"]))
            for entry in payload["poses"]:
                result = PoseResult.from_dict(entry)
                results[result.pose.key] = result
        arms[arm] = results
    holds = {int(h["hold"]) for h in headers.values()}
    if len(holds) != 1:
        raise ValueError("the arms held for different lengths: {}".format(sorted(holds)))
    return arms, dict(headers, hold=holds.pop()), errors


_ROWS = (
    ("recorded", "recorded first scan, the run's renders"),
    ("rescan", "rescan: the same walk-in and turns"),
    ("hold", "hold: fixed heading, first 8"),
    ("teleport", "teleport hold, first 8"),
    ("hold_whole", "hold: fixed heading, whole hold"),
    ("teleport_whole", "teleport hold, whole hold"),
    ("rescan_forced", "CHECK forced {:.0%}/loop fall, rescan"),
    ("hold_forced", "CHECK forced {:.0%}/loop fall, hold"),
    ("frozen", "CHECK frozen IR, whole hold"),
    ("ir_rescan", "IR level: rescan"),
    ("ir_hold", "IR level: hold, first 8"),
    ("ir_hold_whole", "IR level: hold, whole hold"),
    ("ir_teleport", "IR level: teleport, first 8"),
)
_ROW = "  {:<40} {:>4} {:>5} {:>7} {:>7} {:>9} {:>8}  {}"


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else "{:+.1%}".format(value)


def _share(part: int, whole: int) -> str:
    return "n/a" if not whole else "{:.1%}".format(float(part) / float(whole))


def format_readout(result: Mapping[str, Any], *, tag: str, errors: Sequence[str]) -> str:
    """The readout as text."""
    lines = [
        "hold probe  {}".format(tag),
        "  arms: tc1 = temporalCoherence on (the shipped preset), tc0 = off",
        "  poses in both arms: {}   only in tc1: {}   only in tc0: {}".format(
            result["n_paired"], result["only_in"]["tc1"], result["only_in"]["tc0"]),
        "  diverged from the record: tc1 {}, tc0 {}   READ: {}".format(
            result["diverged"]["tc1"], result["diverged"]["tc0"], result["n_read"]),
    ]
    for error in errors:
        lines.append("  SCENE ERROR {}".format(error))
    lines += ["", "  change per loop (pairs one loop apart)", _ROW.format(
        "", "arm", "read", "rose", "fell", "median", "sign p", "reads")]
    for key, label in _ROWS:
        cols = result["rows"].get(key)
        if cols is None:
            continue
        text = label.format(1.0 - FORCED_DECAY ** 5) if "{" in label else label
        for i, (arm, cell) in enumerate(cols.items()):
            n = cell["n"]
            lines.append(_ROW.format(
                text if i == 0 else "", arm, n, _share(cell["rose"], n),
                _share(cell["fell"], n), _pct(cell["median"]),
                "n/a" if cell["p"] is None else "{:.4f}".format(cell["p"]),
                cell["verdict"]))
    lines += ["", "  loop profile: median level per loop over the first loop"]
    for name, values in result["profiles"].items():
        lines.append("  {:<20} {}".format(name, " ".join(
            "  n/a" if v is None else "{:.3f}".format(v) for v in values)))
    lines += [
        "",
        "  BRANCH: {}".format(result["branch"]),
        "  {}".format(result["why"]),
        "",
        "  Pre-registered in hold_probe.py before the first box run. FALLS needs a sign-test",
        "  p <= {} and a median change <= -{:.0%} per loop over >= {} poses. The teleport,".format(
            SIGN_ALPHA, MIN_CHANGE, MIN_POSES),
        "  whole-hold and IR level rows decide nothing.",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# the command line
# ----------------------------------------------------------------------


def _selection_lines(selection: Selection) -> List[str]:
    lines = ["read first scans in the runs: {}   poses: {}".format(
        selection.n_first_scans, len(selection.poses))]
    for why, count in selection.excluded.items():
        lines.append("  excluded {}: {}".format(count, why))
    scenes: Dict[str, int] = {}
    for pose in selection.poses:
        scenes[pose.scene] = scenes.get(pose.scene, 0) + 1
    lines.append("  scenes: {}   poses per scene: {}".format(
        len(scenes), " ".join("{} {}".format(s, n) for s, n in sorted(scenes.items()))))
    if selection.config is not None:
        lines.append("  rendering as the runs did: clip {} ({}), split {}".format(
            selection.config.anomaly_class, selection.config.anomaly_clip or "staged",
            selection.config.split))
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m earshot.tools.hold_probe",
        description="Does the cue fall while the agent stands still? TC on against off.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("select", "render"):
        cmd = sub.add_parser(name)
        cmd.add_argument("runs", nargs="+",
                         help="`full` arm directories, <tag>/full; the first wins an episode")
        cmd.add_argument("--walk-in", type=int, default=DEFAULT_WALK_IN)
        cmd.add_argument("--scenes", default="", help="comma-separated scene labels")
    render = sub.choices["render"]
    render.add_argument("--arm", required=True, choices=list(ARMS))
    render.add_argument("--out", required=True, help="the tag directory, runs/<tag>")
    render.add_argument("--hold", type=int, default=DEFAULT_HOLD)
    render.add_argument("--data-root", default=".")
    read = sub.add_parser("read")
    read.add_argument("tag_dir")
    args = parser.parse_args(argv)

    if args.command == "read":
        arms, headers, errors = read_tag(args.tag_dir)
        result = readout(arms, hold=int(headers["hold"]))
        print(format_readout(result, tag=args.tag_dir, errors=errors))
        return 2 if result["branch"] == NOT_RUN else 0

    if int(args.walk_in) < MIN_WALK_IN:
        print("FATAL: --walk-in {} is under the {} folds the cue takes to settle".format(
            args.walk_in, MIN_WALK_IN))
        return 2
    selection = select_poses(
        args.runs, walk_in=int(args.walk_in),
        scenes=None if not args.scenes else args.scenes.split(","))
    for refusal in selection.refusals:
        print("REFUSED: {}".format(refusal))
    if selection.refusals:
        return 2
    print("\n".join(_selection_lines(selection)))
    if not selection.poses:
        print("FATAL: no pose to render")
        return 2
    if args.command == "select":
        return 0
    rendered = render_arm(
        out_dir=args.out, arm=args.arm, selection=selection, hold=int(args.hold),
        data_root=args.data_root)
    print("arm {}: {} of {} scene(s) rendered".format(
        args.arm, rendered, len({p.scene for p in selection.poses})))
    return 0 if rendered else 1


if __name__ == "__main__":
    sys.exit(main())
