"""
anomaly_controller — the pure interrupt-resume DECISION brain of the
anomaly-response task.

The robot is given a PRIMARY find-task (an ObjectNav-style target). While
searching, a heard anomaly INTERRUPTS it: the agent diverts to the sound's
source location, CHECKs what happened, then RESUMEs and completes the primary
task, and finally REPORTs. This module owns ONLY the state-machine decision —
given the audio brain's onset/anomaly verdict and arrival geometry it returns
the next nav-mode, the active retrieval goal, an optional investigate-waypoint
to inject into the candidate pool, and the save/restore/requery directives the
runner must apply. It never touches the LTM and never imports the simulator
env, so the whole thing is pure numpy-free logic and unit-testable — mirroring
the audio_task.py "pure brain" split.

Responsibility split (same two-env boundary as audio_task):
  * the sim-env RENDERS + supplies geometry (arrival, distance);
  * audio_task DECIDES onset / anomaly-vs-benign;
  * this module DECIDES the task structure (interrupt -> investigate -> resume
    -> report);
  * episode_runner ORCHESTRATES: it applies the directives (goal swap, waypoint
    injection, candidate save/restore, forced re-query).

Every transition is gated by ``cfg.enabled`` at the call site, so when the task
is not anomaly-response the controller is simply never invoked and the existing
objectnav / audiogoal / revisit goal handling is byte-identical.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

# The active retrieval goal while investigating when the dataset did not name an
# object at the source — a sentinel category that yields no spurious LTM recall.
SOURCE_PSEUDO_GOAL = "__anomaly_source__"

Xyz = Tuple[float, float, float]


class NavMode(Enum):
    SEARCH = "search"            # pursuing the primary find-target (the default loop)
    INVESTIGATE = "investigate"  # diverted to the anomaly source waypoint
    CHECK = "check"              # arrived at source; record what was found
    RESUME = "resume"            # restore primary state + force re-query (1 tick)
    COMPLETE = "complete"        # primary target reached
    REPORTED = "reported"        # final report emitted (terminal)


@dataclass
class AnomalyControllerConfig:
    """Static config for one anomaly-response run (env-tunable from the CLI)."""
    enabled: bool = False
    investigate_max_steps: int = 40          # sub-budget for the detour; overflow -> abort
    investigate_arrive_radius_m: float = 1.5  # within this of the source => CHECK
    extend_budget: bool = False              # if True the detour does not count vs primary budget


@dataclass
class ControllerState:
    """Per-episode mutable state. ``reset(primary_goal)`` at every episode start."""
    mode: NavMode = NavMode.SEARCH
    primary_goal: Optional[str] = None
    active_goal: Optional[str] = None        # what propose/retrieve query NOW
    investigate_target_xyz: Optional[Xyz] = None
    investigate_steps: int = 0
    investigated: bool = False               # reached CHECK (arrived at the source)
    investigate_aborted: bool = False        # gave up the detour on the step budget
    resumed: bool = False                    # re-entered SEARCH after an interrupt
    benign_onset_ignored: bool = False
    n_benign_ignored: int = 0
    investigation_event: Optional[Dict[str, Any]] = None
    report: Optional[Dict[str, Any]] = None

    def reset(self, primary_goal: Optional[str]) -> None:
        self.mode = NavMode.SEARCH
        self.primary_goal = primary_goal
        self.active_goal = primary_goal
        self.investigate_target_xyz = None
        self.investigate_steps = 0
        self.investigated = False
        self.investigate_aborted = False
        self.resumed = False
        self.benign_onset_ignored = False
        self.n_benign_ignored = 0
        self.investigation_event = None
        self.report = None


@dataclass
class ControllerDecision:
    """What the runner should do this tick. ``mode`` always equals the post-step
    ``state.mode``; the booleans are one-shot directives applied by the runner."""
    mode: NavMode
    active_goal: Optional[str]
    investigate_waypoint: Optional[Xyz] = None
    force_requery: bool = False
    save_primary_state: bool = False
    restore_primary_state: bool = False
    investigation_event: Optional[Dict[str, Any]] = None
    report: Optional[Dict[str, Any]] = None


def step_controller(
    state: ControllerState,
    cfg: AnomalyControllerConfig,
    *,
    onset_fired: bool,
    is_anomaly: Optional[bool],
    source_xyz: Optional[Xyz],
    arrived_at_source: bool,
    primary_goal_reached: bool,
    anomaly_class: Optional[str] = None,
    anomaly_object: Optional[str] = None,
    keyframe_caption: Optional[str] = None,
) -> ControllerDecision:
    """Advance the interrupt-resume machine one tick and return the directives.

    ``onset_fired`` / ``is_anomaly`` come from the audio brain: ``is_anomaly``
    is True (anomaly), False (benign -> ignore), or None (gate off -> any onset
    interrupts). Mutates ``state`` in place.
    """
    primary = state.primary_goal

    if state.mode == NavMode.SEARCH:
        if primary_goal_reached:
            state.mode = NavMode.COMPLETE
            return ControllerDecision(mode=NavMode.COMPLETE, active_goal=primary)

        if onset_fired and not state.investigated:
            interrupt = (is_anomaly is True) or (is_anomaly is None)
            if interrupt and source_xyz is not None:
                goal = anomaly_object or SOURCE_PSEUDO_GOAL
                state.mode = NavMode.INVESTIGATE
                state.investigate_target_xyz = source_xyz
                state.investigate_steps = 0
                state.active_goal = goal
                return ControllerDecision(
                    mode=NavMode.INVESTIGATE, active_goal=goal,
                    investigate_waypoint=source_xyz, force_requery=True,
                    save_primary_state=True)
            if is_anomaly is False:
                state.benign_onset_ignored = True
                state.n_benign_ignored += 1
            # anomaly but no source cue, or benign: keep searching
        state.active_goal = primary
        return ControllerDecision(mode=NavMode.SEARCH, active_goal=primary)

    if state.mode == NavMode.INVESTIGATE:
        state.investigate_steps += 1
        if arrived_at_source:
            state.mode = NavMode.CHECK
            state.investigated = True
            event = {
                "anomaly_class": anomaly_class,
                "source_xyz": state.investigate_target_xyz,
                "caption": keyframe_caption,
                "investigate_steps": state.investigate_steps,
            }
            state.investigation_event = event
            return ControllerDecision(
                mode=NavMode.CHECK, active_goal=state.active_goal,
                investigation_event=event)
        if state.investigate_steps >= cfg.investigate_max_steps:
            state.mode = NavMode.RESUME
            state.investigate_aborted = True
            state.active_goal = primary
            return ControllerDecision(
                mode=NavMode.RESUME, active_goal=primary,
                restore_primary_state=True, force_requery=True)
        return ControllerDecision(
            mode=NavMode.INVESTIGATE, active_goal=state.active_goal,
            investigate_waypoint=state.investigate_target_xyz)

    if state.mode == NavMode.CHECK:
        state.mode = NavMode.RESUME
        state.active_goal = primary
        return ControllerDecision(
            mode=NavMode.RESUME, active_goal=primary,
            restore_primary_state=True, force_requery=True)

    if state.mode == NavMode.RESUME:
        state.mode = NavMode.SEARCH
        state.resumed = True
        state.active_goal = primary
        return ControllerDecision(mode=NavMode.SEARCH, active_goal=primary)

    # COMPLETE / REPORTED are terminal — idempotent.
    return ControllerDecision(mode=state.mode, active_goal=primary)


def build_report(state: ControllerState, primary_completed: bool) -> Dict[str, Any]:
    """The structured end-of-episode report; also stamps REPORTED on the state."""
    ev = state.investigation_event or {}
    report = {
        "primary_completed": bool(primary_completed),
        "investigated": bool(state.investigated),
        "investigate_aborted": bool(state.investigate_aborted),
        "resumed": bool(state.resumed),
        "anomaly_class": ev.get("anomaly_class"),
        "source_xyz": ev.get("source_xyz"),
        "n_benign_ignored": int(state.n_benign_ignored),
    }
    state.report = report
    state.mode = NavMode.REPORTED
    return report
