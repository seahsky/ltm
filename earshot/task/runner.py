"""The episode loop: where the four layers stop being modules and start being a run.

``task/`` is the only layer ADR-0013's graph lets import everything, so this is the one
place the simulator, the audio sensor, the agent and the report meet.

**Three shapes worth naming before the code.**

*The loop takes its world and its sensor as objects, not as a bundle of callables.*
``audio/`` and ``agent/`` inject callables because the layer graph forbids them from
importing ``sim`` at all; ``task/`` has no such rule, so the injection here would buy
nothing but a dozen lambdas at every call site. What it does buy — a Mac-testable loop —
is bought instead by duck typing: ``run_episode`` names only the handful of methods
``World`` and ``AudioSensorHandle`` publish, and ``tests/mac/test_task_runner.py`` drives
the whole thing with two fakes. ``run()`` is the box-only half, and it is deliberately
thin: it constructs, it loops, it writes.

*Every model is constructed eagerly at startup* (map requirement 9). Ticket 15 measured
the full stack at 5.547 GiB against 31.73 GiB usable, so there is no lazy-loading seam to
build and the layout must not grow one — a model that appears at step 200 is a stall
inside the per-step audio budget criterion 7 audits.

*Nothing may write to fd 1 or 2 concurrently.* ``guarded_observe`` captures both
descriptors around every render. ADR-0013 narrowed this from ticket 18's original
requirement: ``capture_habitat_logs`` flushes Python's buffers on entry and exit, so an
interleaved in-thread ``print()`` between steps is safe. What is forbidden is a
*concurrent* writer — a background thread, a timer-driven progress bar, a subprocess that
inherited the descriptor, a logging handler flushed off-thread. This module therefore
prints progress and starts nothing.

**The calibration is per episode, and that is a reading of §2.3 rather than a departure
from it.** The spec says ``onset_rms`` is derived "at run start", from the anomaly's RMS
distribution across the audible band. That distribution is a property of *this episode's*
source placement and *this scene's* geometry, so with one positioned source per episode
(ADR-0009) the faithful form is one sweep per source. It is measurement either way — the
gate still fails on overlap and the correction is still ``globalVolume`` — but a run of
several episodes carries several thresholds, and each lands on its own audit record with
its own separation margin.

**The sweep's renders are outside the loop, and the record says so.** Smoke criterion 1
is "render count equals step count exactly". Arming the guard costs one render and the
calibration costs ``sweep_poses`` more, all before the first step, so the criterion is
checkable on ``n_renders_in_loop`` versus ``n_loop_steps`` — both recorded — rather than
on the simulator's lifetime counter, which would fail for a reason that is not a defect.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from earshot.agent.config import PlannerConfig
from earshot.agent.controller import (
    ACT_STOP,
    ControllerState,
    NavMode,
    is_diverting,
    step_controller,
)
from earshot.agent.detector import GoalDetector, OracleDetector
from earshot.agent.proposers import SOURCE_INVESTIGATE, Candidate, FrontierProposer
from earshot.agent.reachability import assert_pool, reachable_pool
from earshot.agent.scorer import pick_waypoint
from earshot.audio.bed import bed_signal, heard_signal
from earshot.audio.calibration import (
    CalibrationError,
    CalibrationResult,
    band_poses,
    calibrate_onset,
    sweep_anomaly_rms,
)
from earshot.audio.clap import heard_clip_for_clap, is_anomaly
from earshot.audio.clips import load_anomaly_clip, render_through_ir, resolve_anomaly_clip, rms
from earshot.audio.lateral import lateral_sign
from earshot.audio.normality import NullRoomLabeler, RoomLabeler, is_anomalous_here
from earshot.audio.onset import OnsetState, assert_provenance, observe_step
from earshot.config import Detector, Localization, RunConfig
from earshot.env_check import assert_env
from earshot.metrics import compute_benchmark_spl, compute_soft_spl
from earshot.report.agent import AgentReport
from earshot.report.artifacts import write_env_report, write_episode
from earshot.report.audit import (
    CalibrationRecord,
    EpisodeAudit,
    FunnelStage,
    OnsetRecord,
    StepRecord,
)
from earshot.task.dataset import AnomalyEpisode, build_anomaly_episodes
from earshot.task.episodes import available_scenes, find_scenes_dir, find_split_dir, load_scene
from earshot.types import NoRouteError, Pose, Xyz

__all__ = [
    "EpisodeResult",
    "RunSummary",
    "calibration_poses",
    "calibrate_episode",
    "make_detector",
    "run_episode",
    "run",
]

# How many navmesh draws the calibration sweep samples before matching them to the
# band's target distances. Generous against `AudioConfig.sweep_poses` (16): the draws are
# free — `random_navigable_point` touches no renderer — and the cost of too few is a
# sweep whose poses cluster at whatever distances happened to come up, which is the
# distribution the threshold is then placed inside.
CALIBRATION_DRAWS = 512

# How many recent loudness readings the greedy climb keeps. `realizable_investigate_step`
# reads the last two; the rest are kept because the per-step record already carries the
# full series and a short window makes the controller's input inspectable at a glance.
ENERGY_HISTORY = 8

# The candidate id the oracle arm's investigate divert is injected with.
# `FrontierProposer._emit` issues ids from 1, so 0 is never a proposed candidate and the
# divert is identifiable in the audit by its id as well as by its source.
DIVERT_CANDIDATE_ID = 0


@dataclass(frozen=True)
class EpisodeResult:
    """One episode's two artefacts, before they are written."""

    report: AgentReport
    audit: EpisodeAudit


@dataclass(frozen=True)
class RunSummary:
    """What a whole run reached, for the operator and for ticket 26's smoke."""

    run_dir: str
    scene_label: str
    n_episodes: int
    funnel: Dict[str, int]
    skipped: Tuple[Tuple[str, str], ...] = ()

    def summary(self) -> str:
        lines = [
            "run: {} episode(s) in {} -> {}".format(
                self.n_episodes, self.scene_label, self.run_dir
            ),
            "funnel (§6, denominator is stage 2):",
        ]
        for stage in FunnelStage:
            lines.append("  {}. {:<20} {}".format(
                int(stage), stage.name, self.funnel.get(stage.name, 0)
            ))
        for episode_id, reason in self.skipped:
            lines.append("  skipped {:<12} {}".format(episode_id, reason.splitlines()[0]))
        return "\n".join(lines)


# ----------------------------------------------------------------------
# §2.3's calibration, and the half ticket 25 owns
# ----------------------------------------------------------------------


def calibration_poses(
    world: Any,
    source: Xyz,
    band: Tuple[float, float],
    n_poses: int,
    *,
    n_draws: int = CALIBRATION_DRAWS,
) -> List[Xyz]:
    """Navigable positions spread across the band, by geodesic distance to the source.

    ``audio/calibration.band_poses`` returns **distances** and says why: turning one into
    a pose needs the navmesh, which lives behind ``sim/`` and reaches this layer. This is
    that other half.

    Draw a pile of navigable points, measure each one's geodesic distance to the source,
    and assign the closest match to each target distance without reusing a point. Greedy
    and deliberately not exact: the targets are where the sweep would *like* to sample,
    and a scene simply may not have navmesh at 8 m from the source. What matters is that
    the samples span the band rather than clustering, which is what stops the threshold
    being placed inside a distribution the episode never visits.

    Euclidean nearness is not the question — a point 2 m away through a wall is 12 m of
    walking and hears the source like a point 12 m away, so the geodesic is the axis the
    band is defined on.
    """
    targets = band_poses(band, n_poses)
    draws: List[Tuple[float, Xyz]] = []
    for _ in range(int(n_draws)):
        point = world.random_navigable_point()
        distance = world.geodesic_distance(point, [source])
        if distance is None or not math.isfinite(float(distance)):
            continue
        draws.append((float(distance), point))
    if not draws:
        raise CalibrationError(
            "no navigable point in this scene has a geodesic route to the source at {} "
            "over {} draws — the source is on a disconnected navmesh island, so no pose "
            "can hear it and no threshold can be placed".format(source, n_draws)
        )

    chosen: List[Xyz] = []
    used = set()
    for target in targets:
        best_index = None
        best_error = None
        for index, (distance, _point) in enumerate(draws):
            if index in used:
                continue
            error = abs(distance - target)
            if best_error is None or error < best_error:
                best_index, best_error = index, error
        if best_index is None:
            break
        used.add(best_index)
        chosen.append(draws[best_index][1])
    return chosen


def calibrate_episode(
    world: Any,
    handle: Any,
    source: Xyz,
    clip: Any,
    cfg: RunConfig,
) -> Tuple[CalibrationResult, List[Xyz]]:
    """Run §2.3's sweep against this episode's source and derive ``onset_rms``.

    Seats the agent at each swept pose and renders once, so the distribution is the
    anomaly's own received level — measured through ``clips.render_through_ir``, not off
    the IR's energy, because the threshold is applied to what the agent hears and an IR is
    not in that domain.

    **Leaves the agent wherever the last pose was.** The caller re-seats it at the episode
    start, which it has to do anyway; restoring it here would hide that the sweep moves
    the agent, and a caller that forgot would start its episode at a calibration pose with
    nothing saying so.
    """
    poses = calibration_poses(
        world, source, cfg.audio.audible_band_m, cfg.audio.sweep_poses
    )

    def render_at(position: Xyz) -> Any:
        world.set_pose(position)
        observation, _guard = handle.observe()
        return handle.audio_of(observation)

    samples = sweep_anomaly_rms(poses, render_at, clip)
    result = calibrate_onset(
        cfg.audio.bed_rms, samples, global_volume=cfg.audio.global_volume
    )
    return result, poses


# ----------------------------------------------------------------------
# the detector arm
# ----------------------------------------------------------------------


def make_detector(cfg: RunConfig, world: Any, anomaly_episode: AnomalyEpisode) -> GoalDetector:
    """The configured arm of ADR-0008's one seam, wired to this episode's two objects.

    The oracle's table is the primary goal's view points plus the anomaly source, keyed by
    object name — the two things ``detects()`` is ever asked about, since the controller's
    active goal is the primary category in SEARCH and the anomaly object in INVESTIGATE.

    **When the source is a same-category instance the two entries merge, and that is
    correct rather than convenient.** ``detects("chair")`` means "a chair is here", and a
    chair at the source is a chair. What it costs is instance discrimination during the
    detour: the visual confirm can fire at a different chair. The builder prefers a
    different category precisely to keep that rare, records ``same_category`` when it
    cannot, and this is where the consequence lands.
    """
    view_points: Dict[str, List[Xyz]] = {
        anomaly_episode.primary_category: [
            view_point.position for view_point in anomaly_episode.episode.view_points()
        ]
    }
    view_points.setdefault(anomaly_episode.source.anomaly_object, []).append(
        anomaly_episode.source.position
    )

    if cfg.detector is Detector.ORACLE:

        def distance_to(obj: str) -> Optional[float]:
            points = view_points.get(obj)
            if not points:
                return None
            return world.geodesic_distance(world.pose().position, points)

        return OracleDetector(distance_to, cfg.detector_config)

    raise RuntimeError(
        "Detector.CAPTION needs a grounding VLM, and `earshot/vlm.py` — the Qwen2-VL-2B "
        "connector ADR-0013's tree names — has not been built. `agent/detector."
        "CaptionDetector` ships live and takes an injected `Grounder`, so the missing "
        "piece is the connector rather than the arm; it belongs to R2, which is out of "
        "scope for this map (the smoke runs Detector.ORACLE, §8). Raising rather than "
        "substituting the oracle: an arm that silently ran the other arm would produce "
        "numbers labelled `caption` in every audit record."
    )


# ----------------------------------------------------------------------
# steering
# ----------------------------------------------------------------------


def _divert_candidate(target: Xyz, pose: Pose) -> Candidate:
    """The oracle arm's investigate waypoint, as a candidate for the pool.

    Injected rather than steered to directly, which is the seam ``agent/scorer.py``
    describes: the divert sorts ahead of everything by rank, so the interrupt is an
    override rather than a high score that a maximal frontier candidate could tie. Going
    around the pool would also skip ``reachability``'s navmesh filter, and a source the
    agent cannot route to is exactly the case that must not become a silent straight-line
    walk into a wall.
    """
    dx, dz = target.x - pose.position.x, target.z - pose.position.z
    return Candidate(
        candidate_id=DIVERT_CANDIDATE_ID,
        position=target,
        source=SOURCE_INVESTIGATE,
        distance_m=math.hypot(dx, dz),
        # The bearing is recomputed at the snapped position by `reachable_pool`, and the
        # scorer ignores it for a divert anyway (the rank puts it first). 0.0 is the
        # honest placeholder rather than a second derivation of the frame convention.
        bearing_rad=0.0,
        raw_score=1.0,
    )


def _choose_waypoint(
    proposer: FrontierProposer,
    pose: Pose,
    *,
    snap_point: Callable[[Xyz], Optional[Xyz]],
    geodesic: Callable[[Xyz, Xyz], Optional[float]],
    planner: PlannerConfig,
    divert: Optional[Candidate] = None,
) -> Tuple[Xyz, str, Dict[str, int]]:
    """Propose, filter on the navmesh, and pick one. Returns ``(waypoint, source, counters)``.

    Two stages because ADR-0008's invariant has two ways to be met. The frontier pool can
    be empty of *cells* — the proposer answers that itself with the compass fan — or full
    of cells the navmesh rejects wholesale, which is the occupancy-versus-navmesh
    disagreement the invariant exists for. Only the caller can answer the second, which is
    why ``compass_fan`` is public.
    """
    proposed = list(proposer.propose(pose))
    if divert is not None:
        proposed.insert(0, divert)
    report = reachable_pool(
        proposed, pose, snap_point=snap_point, geodesic=geodesic, cfg=planner
    )
    kept, counters = report.candidates, report.counters()
    if not kept:
        fan = list(proposer.compass_fan(pose))
        if divert is not None:
            fan.insert(0, divert)
        report = reachable_pool(
            fan, pose, snap_point=snap_point, geodesic=geodesic, cfg=planner
        )
        kept = assert_pool(report, stage="compass fan after the frontier pool emptied")
        counters = report.counters()
    scored = pick_waypoint(kept)
    return scored.candidate.position, scored.candidate.source, counters


# ----------------------------------------------------------------------
# the episode
# ----------------------------------------------------------------------


def run_episode(
    world: Any,
    handle: Any,
    anomaly_episode: AnomalyEpisode,
    cfg: RunConfig,
    *,
    clip: Any,
    detector: GoalDetector,
    index: int = 0,
    room_labeler: Optional[RoomLabeler] = None,
    clap_encoder: Optional[Any] = None,
    calibration: Optional[CalibrationResult] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> EpisodeResult:
    """One mission: the primary find-task, the interrupt, the detour, the resume.

    ``world`` and ``handle`` are duck-typed (see the module docstring). ``calibration``
    is passed in when the caller has already run the sweep — ``run()`` does, because it
    also wants the sweep's poses for the record — and derived here otherwise.

    The order inside a step is fixed and each part of it is load-bearing: render once
    (the shared observation, so RGB, depth and the IR are the same instant), mix the bed
    and measure, fold the onset, feed the map and the detector, ask the controller, apply
    what it decided, record the row. A detector observing a frame from a different render
    than the onset measured would make "peak-or-plateau plus visual confirm" a statement
    about two moments.
    """
    episode = anomaly_episode.episode
    source = anomaly_episode.source.position
    t_anom = int(anomaly_episode.t_anom)
    realizable = cfg.localization is Localization.REALIZABLE
    labeler = room_labeler if room_labeler is not None else NullRoomLabeler()
    say = progress if progress is not None else (lambda _message: None)

    goal_positions = [view_point.position for view_point in episode.view_points()]
    world.set_pose(episode.start_position, episode.start_rotation)
    start_pose = world.pose()
    start_end_distance = world.geodesic_distance(start_pose.position, goal_positions)

    if calibration is None:
        calibration, _poses = calibrate_episode(world, handle, source, clip, cfg)
        world.set_pose(episode.start_position, episode.start_rotation)
        start_pose = world.pose()

    bed = bed_signal(len(clip), cfg.audio.bed_rms)
    proposer = FrontierProposer(cfg=cfg.planner)
    proposer.reset(start_pose)
    follow = world.follower()
    state = ControllerState.for_episode(anomaly_episode.primary_category)
    onset = OnsetState()

    def geodesic(a: Xyz, b: Xyz) -> Optional[float]:
        return world.geodesic_distance(a, [b])

    # §2.5's one smoke exception: verify audibility at the episode's own start pose, once,
    # with a calibration render, so the smoke is deterministic. It is a measurement rather
    # than a screen — nothing is rejected on it, and it is recorded next to the threshold
    # it is compared against.
    observation, _guard = handle.observe()
    start_anomaly_rms = rms(render_through_ir(handle.audio_of(observation), clip))
    n_renders_before = int(getattr(world, "n_renders", 0))

    steps: List[StepRecord] = []
    energy: List[float] = []
    counters: Dict[str, int] = {}
    waypoint: Optional[Xyz] = None
    saved_waypoint: Optional[Xyz] = None
    room: Optional[str] = None
    anomaly_class: Optional[str] = None
    verdict: Optional[bool] = None
    classified = False
    entered_investigate = False
    path_len = 0.0
    n_actions = 0
    n_no_action = 0
    min_d2g: Optional[float] = None
    min_d2source = float("inf")
    stopped = False
    wall_clock_0 = time.perf_counter()

    for step in range(int(cfg.max_steps)):
        audio_t0 = time.perf_counter()
        observation, _guard = handle.observe()
        impulse = handle.audio_of(observation)
        playing = step >= t_anom
        heard = heard_signal(impulse, clip, bed, playing=playing)
        measured = rms(heard)
        # The whole per-step audio bill, not the render alone: criterion 7 audits what
        # live-every-step costs, and the convolution, the bed mix and the guard's two
        # tempfiles are all part of it now (ticket 06's 27.2 ms was the render only).
        audio_s = time.perf_counter() - audio_t0
        lateral = lateral_sign(heard)

        onset = observe_step(
            onset,
            step=step,
            measured_rms=measured,
            t_anom=t_anom,
            onset_rms=calibration.onset_rms,
            bed_rms=cfg.audio.bed_rms,
            tolerance=cfg.audio.pre_onset_rms_tol,
        )
        if onset.fired and not classified:
            classified = True
            room = labeler.label(observation.get("rgb"))
            if clap_encoder is not None:
                waveform, sample_rate = heard_clip_for_clap(heard, cfg.audio.sample_rate)
                fired, best_class, _scores = is_anomaly(waveform, sample_rate, clap_encoder)
                anomaly_class = best_class
                verdict = is_anomalous_here(fired, best_class, room)
            # With no CLAP the verdict stays None, which `step_controller` reads as
            # "nothing conditioned this, so any onset interrupts" — the smoke's case
            # (§4.3: one sound, the anomaly by construction), and honest about it: the
            # report's `anomaly_class` stays None rather than being copied off the
            # dataset, which would be the task telling the agent what it heard.
            say("  step {}: onset at RMS {:.6g} (threshold {:.6g})".format(
                step, measured, calibration.onset_rms
            ))

        pose = world.pose()
        depth = observation.get("depth")
        proposer.observe(depth, pose)
        detector.observe(rgb=observation.get("rgb"), depth=depth, pose=pose)
        energy.append(measured)
        del energy[:-ENERGY_HISTORY]

        diverting = is_diverting(state.mode)
        visual_confirm = bool(detector.detects(state.active_goal)) if diverting else False
        primary_reached = (
            False if diverting else bool(detector.detects(anomaly_episode.primary_category))
        )

        distance_to_goal = world.geodesic_distance(pose.position, goal_positions)
        if distance_to_goal is not None:
            min_d2g = distance_to_goal if min_d2g is None else min(min_d2g, distance_to_goal)
        min_d2source = min(min_d2source, pose.position.horizontal_distance_to(source))

        state, decision = step_controller(
            state,
            cfg.controller,
            onset_fired=onset.fired,
            is_anomaly=verdict,
            primary_goal_reached=primary_reached,
            realizable=realizable,
            # The realizable arm is never handed the coordinate at all. The controller
            # would ignore it, but "the arm does not have it" is a stronger property than
            # "the arm does not read it", and it costs one conditional.
            source_xyz=None if realizable else source,
            arrived_at_source=(
                False
                if realizable
                else pose.position.horizontal_distance_to(source)
                <= float(cfg.controller.investigate_arrive_radius_m)
            ),
            anomaly_class=anomaly_class,
            anomaly_object=anomaly_episode.source.anomaly_object,
            energy_history=energy,
            lateral_sign=lateral,
            visual_confirm=visual_confirm,
            pose=pose,
        )
        if state.mode is NavMode.INVESTIGATE or decision.mode is NavMode.INVESTIGATE:
            entered_investigate = True
        if decision.save_primary_state:
            saved_waypoint = waypoint
            say("  step {}: INVESTIGATE — primary state saved".format(step))
        if decision.restore_primary_state:
            waypoint = saved_waypoint
            say("  step {}: {} — primary state restored".format(step, decision.mode.name))
        if decision.force_requery:
            proposer.request_replan()

        action: Optional[str] = decision.realizable_action
        if decision.mode is NavMode.COMPLETE:
            # The primary STOP, which is a task decision and never reaches the simulator
            # (`sim.world.step` refuses it). Recorded as the action taken, because "the
            # episode ended here and why" is exactly what the per-step record is for.
            action = ACT_STOP
        elif action is None:
            divert = (
                _divert_candidate(decision.investigate_waypoint, pose)
                if decision.investigate_waypoint is not None
                else None
            )
            waypoint, action, step_counters = _steer(
                proposer,
                pose,
                waypoint,
                follow=follow,
                snap_point=world.snap_point,
                geodesic=geodesic,
                planner=cfg.planner,
                divert=divert,
            )
            for key, value in step_counters.items():
                counters[key] = counters.get(key, 0) + int(value)

        if action is not None and action != ACT_STOP:
            before = pose.position
            world.step(action)
            after = world.pose().position
            path_len += math.hypot(after.x - before.x, after.z - before.z)
            n_actions += 1
        elif action is None:
            n_no_action += 1

        steps.append(
            StepRecord(
                step=step,
                measured_rms=float(measured),
                lateral_sign=int(lateral),
                source_playing=bool(playing),
                source_is_visible=handle.source_is_visible(),
                action=action,
                audio_render_s=float(audio_s),
            )
        )

        if state.mode is NavMode.COMPLETE:
            stopped = True
            say("  step {}: primary goal reached — STOP".format(step))
            break

    wall_clock = time.perf_counter() - wall_clock_0
    n_renders_in_loop = int(getattr(world, "n_renders", 0)) - n_renders_before

    # §3.1, on the recorded state, before anything is written. It RAISES: an episode that
    # reached here with an impossible onset step or a drifted pre-onset reading is one an
    # analyst would otherwise quote, and both causes are silent-fabrication bugs.
    assert_provenance(
        onset,
        t_anom=t_anom,
        bed_rms=cfg.audio.bed_rms,
        tolerance=cfg.audio.pre_onset_rms_tol,
    )

    final_pose = world.pose()
    primary_dist_at_stop = (
        world.geodesic_distance(final_pose.position, goal_positions) if stopped else None
    )
    event = state.investigation_event
    stop_pose = event.stopped_at_pose if event is not None else None
    source_dist_at_stop = (
        stop_pose.position.horizontal_distance_to(source) if stop_pose is not None else None
    )

    report = AgentReport(
        primary_completed=state.mode in (NavMode.COMPLETE, NavMode.REPORTED),
        heard_at_step=onset.onset_step,
        room=room,
        anomaly_class=anomaly_class,
        stopped_at_pose=stop_pose,
        visual_confirm_object=event.visual_confirm_object if event is not None else None,
        investigate_aborted=state.investigate_aborted,
        resumed=state.resumed,
        n_benign_ignored=state.n_benign_ignored,
    )

    success_1m, spl_1m = compute_benchmark_spl(
        stopped=stopped,
        dist_at_stop=primary_dist_at_stop,
        geodesic_optimal=float(start_end_distance or 0.0),
        path_len_taken=path_len,
        success_radius=1.0,
    )
    _success_01, spl_01 = compute_benchmark_spl(
        stopped=stopped,
        dist_at_stop=primary_dist_at_stop,
        geodesic_optimal=float(start_end_distance or 0.0),
        path_len_taken=path_len,
        success_radius=0.1,
    )
    metrics: Dict[str, float] = {
        # Find-SR, both rings (§6): 1.0 m primary, 0.1 m the localization-bound
        # diagnostic. Both are STOP-gated.
        "find_sr_1m": float(bool(success_1m)),
        "find_sr_0_1m": float(bool(_success_01)),
        "benchmark_spl_1m": float(spl_1m),
        "benchmark_spl_0_1m": float(spl_01),
        # STOP-independent reach diagnostic, and never a success number (§6).
        "reach_1m": float(min_d2g is not None and min_d2g <= 1.0),
        "soft_spl": compute_soft_spl(
            dist_to_goal_final=world.geodesic_distance(final_pose.position, goal_positions),
            start_end_distance=float(start_end_distance or 0.0),
            path_len_taken=path_len,
        ),
        "start_end_distance_m": float(start_end_distance or 0.0),
        "path_len_m": float(path_len),
        "stopped": float(stopped),
        "n_loop_steps": float(len(steps)),
        "n_renders_in_loop": float(n_renders_in_loop),
        "n_sim_actions": float(n_actions),
        "n_no_action_steps": float(n_no_action),
        "wall_clock_s": float(wall_clock),
        # §2.5's smoke exception, as two numbers rather than a verdict.
        "start_pose_anomaly_rms": float(start_anomaly_rms),
        "start_pose_audible": float(start_anomaly_rms >= calibration.onset_rms),
        "source_separation_m": float(anomaly_episode.source.separation_m),
        "source_dy_m": float(anomaly_episode.source.height_difference_m),
        "source_same_category": float(anomaly_episode.source.same_category),
    }
    # Three measurements are recorded only when they exist, rather than as NaN or inf.
    # `json.dump` writes those as bare `NaN` / `Infinity`, which is not valid JSON and
    # which a strict reader rejects — and "unreachable" and "not measured" are different
    # facts from "the number happens to be missing", which is what an absent key says.
    if min_d2g is not None:
        metrics["min_d2g_m"] = float(min_d2g)
    if primary_dist_at_stop is not None:
        metrics["primary_dist_at_stop_m"] = float(primary_dist_at_stop)
    if math.isfinite(min_d2source):
        # §4.2's replacement for ADR-0001's asserted ~1 m ceiling: measured, per episode,
        # and pooled across a run into the distribution §6 asks for.
        metrics["min_d2source_m"] = float(min_d2source)
    metrics.update({key: float(value) for key, value in proposer.stats().items()})
    metrics.update({key: float(value) for key, value in counters.items()})

    audit = EpisodeAudit(
        episode_index=int(index),
        scene_id=episode.scene_id,
        localization_arm=cfg.localization.value,
        detector_arm=cfg.detector.value,
        source_xyz=source,
        dist_at_stop=source_dist_at_stop,
        funnel_stage=_funnel_stage(
            n_steps=len(steps),
            t_anom=t_anom,
            onset_fired=onset.fired,
            entered_investigate=entered_investigate,
            investigated=state.investigated,
            resumed=state.resumed,
        ),
        onset=OnsetRecord(
            onset_step=onset.onset_step,
            pre_onset_rms=onset.pre_onset_rms,
            n_pre_onset_readings=onset.n_pre_onset_readings,
            provenance_asserted=True,
        ),
        calibration=CalibrationRecord(
            onset_rms=calibration.onset_rms,
            bed_rms=calibration.bed_rms,
            separation_db=calibration.separation_db,
            n_poses=calibration.n_poses,
            global_volume=calibration.global_volume,
            passed=calibration.passed,
        ),
        audio_context=getattr(handle, "report", None),
        steps=tuple(steps),
        metrics=metrics,
    )
    return EpisodeResult(report=report, audit=audit)


def _steer(
    proposer: FrontierProposer,
    pose: Pose,
    waypoint: Optional[Xyz],
    *,
    follow: Callable[[Xyz], Optional[str]],
    snap_point: Callable[[Xyz], Optional[Xyz]],
    geodesic: Callable[[Xyz, Xyz], Optional[float]],
    planner: PlannerConfig,
    divert: Optional[Candidate],
) -> Tuple[Optional[Xyz], Optional[str], Dict[str, int]]:
    """The next action toward a waypoint, re-proposing once if the current one is spent.

    Two attempts, because the two ways a waypoint stops being answerable both resolve by
    picking another one: the follower reports arrival (``None``), or it reports no route
    (``NoRouteError``). A single attempt would spend a whole step standing still after
    every arrival.

    **After two attempts it returns ``None`` and the runner takes no simulator step.**
    That is deliberate: the old tree's answer here was a straight-line fallback, which is
    how "a waypoint was chosen" and "the agent got there" came apart with nothing in the
    code marking where. A recorded ``action: null`` is a visible symptom in the per-step
    record; an invented action is a trajectory that lies.
    """
    counters: Dict[str, int] = {}
    for _attempt in range(2):
        if waypoint is None or proposer.is_decision_step():
            waypoint, _source, counters = _choose_waypoint(
                proposer,
                pose,
                snap_point=snap_point,
                geodesic=geodesic,
                planner=planner,
                divert=divert,
            )
        try:
            action = follow(waypoint)
        except NoRouteError:
            proposer.request_replan()
            waypoint = None
            continue
        if action is not None:
            return waypoint, action, counters
        proposer.request_replan()
        waypoint = None
    return waypoint, None, counters


def _funnel_stage(
    *,
    n_steps: int,
    t_anom: int,
    onset_fired: bool,
    entered_investigate: bool,
    investigated: bool,
    resumed: bool,
) -> FunnelStage:
    """§6's staged funnel, as the highest stage this episode reached.

    The stages nest, so this is a ladder rather than a classification: an episode that
    resumed necessarily investigated. ``T_ANOM_REACHED`` is ``n_steps > t_anom`` because
    the step indices are zero-based — an episode of exactly ``t_anom`` steps ended on the
    step *before* the source started playing.
    """
    stage = FunnelStage.RUN
    if n_steps > int(t_anom):
        stage = FunnelStage.T_ANOM_REACHED
    if onset_fired:
        stage = FunnelStage.ONSET_FIRED
    if entered_investigate:
        stage = FunnelStage.INVESTIGATE_ENTERED
    if investigated:
        stage = FunnelStage.SOURCE_REACHED
    if resumed:
        stage = FunnelStage.PRIMARY_RESUMED
    return stage


# ----------------------------------------------------------------------
# the run
# ----------------------------------------------------------------------


def _pick_scene(split_dir: str, scenes_dir: str, wanted: str) -> Any:
    """The named scene, or the first one in the split whose mesh is on this machine.

    The search exists because the box carries a partial HM3D download and a run that
    fails on the fourth scene, forty minutes in, is a worse failure than one that starts
    somewhere it can.
    """
    labels = [wanted] if wanted else list(available_scenes(split_dir))
    tried = []
    for label in labels:
        dataset = load_scene(split_dir, label, scenes_dir=scenes_dir)
        if os.path.exists(dataset.scene_path):
            return dataset
        tried.append(dataset.scene_path)
    raise FileNotFoundError(
        "no scene mesh found under {} (tried {} scene(s); first few: {})".format(
            scenes_dir, len(tried), ", ".join(tried[:3])
        )
    )


def run(cfg: RunConfig, *, progress: Optional[Callable[[str], None]] = None) -> RunSummary:
    """Assert the environment, build the dataset, run the episodes, write the artefacts.

    The box-only half. Everything expensive is constructed once, before the first step:
    the clip, the models, the simulator, the audio sensor and its armed guard (map
    requirement 9).

    ``env_report.json`` carries the resolved **configuration** beside the resolved
    environment. One artefact, two kinds of fact, and both answer the same question about
    a run directory a year from now — which is the argument ``agent/config.py`` makes for
    ``DetectorConfig`` existing at all.
    """
    say = progress if progress is not None else print

    say("env_check: probing (clap={})".format(cfg.clap))
    env = assert_env(clap=cfg.clap)
    say(env.summary())

    split_dir = find_split_dir(cfg.split, root=cfg.data_root)
    scenes_dir = find_scenes_dir(root=cfg.data_root)
    dataset = _pick_scene(split_dir, scenes_dir, cfg.scene)
    build = build_anomaly_episodes(
        dataset,
        anomaly_class=cfg.anomaly_class,
        t_anom=cfg.t_anom,
        category=cfg.category,
        n_episodes=cfg.n_episodes,
        min_sep_m=cfg.min_source_sep_m,
        max_dy_m=cfg.max_source_dy_m,
    )
    say(build.summary())

    clip_path = resolve_anomaly_clip(cfg.anomaly_class, cfg.anomaly_clip, cfg.audio.clip_dir)
    clip = load_anomaly_clip(
        clip_path, cfg.audio.sample_rate, cfg.audio.target_norm_rms_db
    )
    say("anomaly clip: {} ({} samples at {} Hz)".format(
        clip_path, len(clip), cfg.audio.sample_rate
    ))

    clap_encoder = None
    if cfg.clap:
        from earshot.task.models import load_clap_encoder

        clap_encoder = load_clap_encoder()
        say("CLAP: loaded")

    write_env_report(
        cfg.run_dir,
        dict(env.as_dict(), run_config=cfg.as_dict(), scene=dataset.scene_label),
        overwrite=cfg.overwrite,
    )

    # Imported here rather than at module scope: `sim/world.py` imports habitat_sim, so
    # a top-level import would make every Mac test in this file's suite uncollectable —
    # the same reason `audio/clips.py` imports scipy inside its two callers.
    from earshot.audio.sensor import AudioSensorHandle
    from earshot.audio.spec import audio_sensor_spec
    from earshot.sim.world import World, audio_spec_parts, camera_sensor_specs

    spec, binaural = audio_spec_parts()
    audio_sensor_spec(spec, cfg.audio, binaural)
    audio_uuid = str(spec.uuid)

    funnel: Dict[str, int] = {stage.name: 0 for stage in FunnelStage}
    world = World(dataset.scene_path, list(camera_sensor_specs()) + [spec])
    try:
        world.seed_navmesh(cfg.seed)
        handle = None
        for index, anomaly_episode in enumerate(build.episodes):
            episode = anomaly_episode.episode
            say("episode {}: {} in {}, source at {} ({})".format(
                index,
                anomaly_episode.primary_category,
                episode.scene_label,
                anomaly_episode.source.anomaly_object,
                anomaly_episode.source.position,
            ))
            world.set_pose(episode.start_position, episode.start_rotation)
            if handle is None:
                handle = AudioSensorHandle(
                    world.sensor_handle(audio_uuid),
                    world.observe,
                    anomaly_episode.source.position,
                    uuid=audio_uuid,
                )
                say("  audio context armed: {} vertices, IR {}".format(
                    handle.report.n_vertices, handle.report.ir_shape
                ))
            else:
                handle.set_source(anomaly_episode.source.position)

            calibration, poses = calibrate_episode(
                world, handle, anomaly_episode.source.position, clip, cfg
            )
            say("  calibration: onset_rms {:.6g}, separation {:.2f} dB over {} pose(s)".format(
                calibration.onset_rms, calibration.separation_db, len(poses)
            ))

            result = run_episode(
                world,
                handle,
                anomaly_episode,
                cfg,
                clip=clip,
                detector=make_detector(cfg, world, anomaly_episode),
                index=index,
                clap_encoder=clap_encoder,
                calibration=calibration,
                progress=say,
            )
            write_episode(
                cfg.run_dir,
                index,
                result.report,
                result.audit,
                overwrite=cfg.overwrite,
            )
            for stage in FunnelStage:
                if result.audit.funnel_stage >= stage:
                    funnel[stage.name] += 1
            say("  funnel: {}   audio {}".format(
                result.audit.funnel_stage.name, result.audit.audio_render_summary()
            ))
    finally:
        world.close()

    summary = RunSummary(
        run_dir=cfg.run_dir,
        scene_label=dataset.scene_label,
        n_episodes=len(build.episodes),
        funnel=funnel,
        skipped=build.skipped,
    )
    say(summary.summary())
    return summary
