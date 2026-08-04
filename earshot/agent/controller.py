"""The interrupt-resume machine: SEARCH, INVESTIGATE, CHECK, RESUME, and the greedy climb.

A pure decision function. It is handed the audio brain's onset and anomaly verdict, the
agent's own recent loudness, the lateral cue and the detector's visual confirm, and it
returns the next nav mode plus the directives the runner applies — the goal swap, the
investigate waypoint, the save/restore of primary state, the forced re-query. It never
touches the simulator and never reads ground truth to *decide*, which is why ADR-0008
calls it the one module that unit-tests on a Mac and the paper's single
framing-independent positive.

**The transition table is carried unchanged.** Every guard, every ordering, every
one-shot directive is the old ``step_controller``'s. Two things around it change, both
disclosed:

- **``build_report`` is gone from this module** and lives in ``earshot/report/`` as a
  frozen ``AgentReport`` of exactly §5.1's nine fields (ADR-0013's disclosed deviation
  from ticket 10's "near-verbatim"). The old one emitted ``source_xyz`` out of the
  investigation event, returned an untyped dict, and mutated the state it was handed.
- **``ControllerState`` is frozen and every transition returns a new one.** The old
  function mutated in place. This is the discipline ``audio/onset.py`` already applies to
  ``OnsetState``, for its reason: the runner owns the episode's mutable state in one
  place, and a controller leaked across episodes cannot carry a stale ``investigated``
  into the next one — which is the bug a ``reset()`` method exists to paper over. The
  *decisions* are identical; only the plumbing moved.

**The oracle coordinate stays in the state, and that is on purpose.** "The controller
cannot see ground truth" is not available as the rule, because the oracle arm's
controller legitimately holds ``source_xyz`` as its waypoint while the task spec requires
an identical report schema in both arms. The boundary is drawn at the report type instead
(ADR-0013), so nothing privileged can appear in the testimony whatever the controller
holds. What this module does *not* hold is the line-of-sight probe — §3.3's analyst-only
signal, which ``tests/mac/test_analyst_only.py`` structurally forbids ``agent/`` from
naming.

**The lateral sign arrives already in the agent frame and gets no compensation term.**
The RIR grid rendered at identity listener yaw, so the cue was world-frame and the fusion
arc corrected for it with ``heard == -right(world-bearing)``. Live rendering uses the
agent's real transform, so ``audio/lateral.lateral_sign`` returns an agent-frame cue with
the same arithmetic on the same samples. Carried across with the old compensation the
agent turns the wrong way on every stall, and it looks like a mediocre climb rather than a
bug. ``tests/box/test_audio_box.py`` pins the convention.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional, Sequence, Tuple

from earshot.agent.config import ControllerConfig
from earshot.types import Pose, Xyz

__all__ = [
    "SOURCE_PSEUDO_GOAL",
    "ACT_FORWARD",
    "ACT_TURN_LEFT",
    "ACT_TURN_RIGHT",
    "ACT_STOP",
    "NavMode",
    "InvestigationEvent",
    "ControllerState",
    "ControllerDecision",
    "realizable_investigate_step",
    "is_diverting",
    "step_controller",
]

# The active goal while investigating when the dataset named no object at the source.
# **A consequence for ticket 25's builder:** in the realizable arm, arrival is
# peak-or-plateau *plus visual confirm*, and there is nothing to visually confirm about a
# sentinel — so an episode whose source has no named object can only ever leave
# INVESTIGATE through the step-budget abort. The builder should name the object;
# `investigate_aborted` in the funnel is what shows up if it does not.
SOURCE_PSEUDO_GOAL = "__anomaly_source__"

# Realizable-localization actions. Strings that match `sim.world`'s action names, so the
# runner passes them straight through, but this module does not import that module and
# does not know they are habitat's — STOP is not one of them there, because it is a task
# decision rather than a simulator action.
ACT_FORWARD = "move_forward"
ACT_TURN_LEFT = "turn_left"
ACT_TURN_RIGHT = "turn_right"
ACT_STOP = "stop"


def realizable_investigate_step(
    energy_history: Sequence[float],
    lateral_sign: int,
    visual_confirm: bool,
    *,
    eps: float = 1e-6,
) -> str:
    """One greedy step of realizable anomaly-source localization (ADR-0011). Carried verbatim.

    Pure, and from **agent-estimable signals only** — it never reads a ground-truth source
    distance or coordinate:

    - ``energy_history`` — the agent's own recent binaural loudness at successive poses.
      Climbing it walks toward "getting louder".
    - ``lateral_sign`` — the interaural level sign, ``+1`` source to the right, ``-1``
      left, ``0`` ambiguous. Agent-frame, uncompensated (see the module docstring).
    - ``visual_confirm`` — the detector confirms the anomaly object is here.

    Rising loudness means forward; peak-or-plateau plus visual confirm means STOP, so the
    agent stops *at* the source rather than at an arbitrary loud cell; a stall without a
    confirm turns toward the louder half-plane.

    ``turn_left`` and ``turn_right`` change the measured RMS without changing the
    distance, and §4.1 instruments that rather than fixing it — the per-step record
    carries the action taken so a rotation-driven rise is separable from a
    translation-driven one after the fact.
    """
    history = [float(e) for e in energy_history if e is not None]
    if not history:
        return ACT_FORWARD  # no reading yet, probe forward
    current = history[-1]
    previous = history[-2] if len(history) >= 2 else None
    rising = previous is None or current > previous + float(eps)
    if visual_confirm and not rising:
        return ACT_STOP
    if rising:
        return ACT_FORWARD
    if int(lateral_sign) > 0:
        return ACT_TURN_RIGHT
    if int(lateral_sign) < 0:
        return ACT_TURN_LEFT
    return ACT_TURN_LEFT  # ambiguous sign, default scan turn


class NavMode(Enum):
    SEARCH = "search"  # pursuing the primary find-target: the default loop
    INVESTIGATE = "investigate"  # diverted to the anomaly source
    CHECK = "check"  # arrived at the source; record what was found
    RESUME = "resume"  # restore primary state and force a re-query (one tick)
    COMPLETE = "complete"  # primary target reached
    REPORTED = "reported"  # the report has been emitted (terminal)


@dataclass(frozen=True)
class InvestigationEvent:
    """What the agent found at the end of the detour. Agent-estimable only.

    ``source_xyz`` is **not** here, and its replacement is ``stopped_at_pose`` — the
    agent's own pose, which §5.1 substitutes for the coordinate for exactly this reason.
    The old event carried the oracle position and a keyframe caption; the caption has no
    field in §5.1 and the coordinate belongs to the audit record, which gets it from the
    task rather than from the agent's testimony.
    """

    anomaly_class: Optional[str]
    visual_confirm_object: Optional[str]
    investigate_steps: int
    stopped_at_pose: Optional[Pose]
    realizable: bool


@dataclass(frozen=True)
class ControllerState:
    """The episode's task state. Frozen; ``step_controller`` returns the next one.

    Build a fresh one per episode — ``ControllerState(primary_goal=..., active_goal=...)``
    — rather than resetting one in place. There is no ``reset()`` because there is nothing
    to reset: a new episode is a new value.
    """

    primary_goal: Optional[str] = None
    active_goal: Optional[str] = None
    mode: NavMode = NavMode.SEARCH
    investigate_target_xyz: Optional[Xyz] = None
    investigate_steps: int = 0
    investigated: bool = False  # reached CHECK, i.e. arrived at the source
    investigate_aborted: bool = False  # gave up the detour on the step budget
    resumed: bool = False  # re-entered SEARCH after the interrupt
    n_benign_ignored: int = 0
    investigation_event: Optional[InvestigationEvent] = None

    @classmethod
    def for_episode(cls, primary_goal: Optional[str]) -> "ControllerState":
        """The starting state: searching for the primary goal, nothing investigated."""
        return cls(primary_goal=primary_goal, active_goal=primary_goal)


@dataclass(frozen=True)
class ControllerDecision:
    """What the runner does this tick. ``mode`` always equals the returned state's mode.

    The booleans are one-shot directives, true on the tick of the transition only.
    ``investigate_waypoint`` is the oracle arm's point goal, injected into the candidate
    pool as a ``SOURCE_INVESTIGATE`` candidate so it wins the scorer's pick;
    ``realizable_action`` is the realizable arm's low-level action, applied directly. They
    are mutually exclusive by construction — an arm produces one or the other — and that
    is what makes "which arm ran" readable off a decision.
    """

    mode: NavMode
    active_goal: Optional[str]
    investigate_waypoint: Optional[Xyz] = None
    realizable_action: Optional[str] = None
    force_requery: bool = False
    save_primary_state: bool = False
    restore_primary_state: bool = False
    investigation_event: Optional[InvestigationEvent] = None


def is_diverting(mode: NavMode) -> bool:
    """True while on the detour: the runner must not let the primary task terminate.

    Suppresses the primary STOP through INVESTIGATE, CHECK and RESUME. False in SEARCH
    (normal primary navigation) and in COMPLETE / REPORTED, where the primary goal has
    been reached and the STOP is legitimate — suppressing the terminal states would break
    primary success outright.
    """
    return mode in (NavMode.INVESTIGATE, NavMode.CHECK, NavMode.RESUME)


def step_controller(
    state: ControllerState,
    cfg: ControllerConfig,
    *,
    onset_fired: bool,
    is_anomaly: Optional[bool],
    primary_goal_reached: bool,
    realizable: bool = True,
    source_xyz: Optional[Xyz] = None,
    arrived_at_source: bool = False,
    anomaly_class: Optional[str] = None,
    anomaly_object: Optional[str] = None,
    energy_history: Optional[Sequence[float]] = None,
    lateral_sign: int = 0,
    visual_confirm: bool = False,
    pose: Optional[Pose] = None,
) -> Tuple[ControllerState, ControllerDecision]:
    """Advance the machine one tick. Returns ``(next_state, decision)``.

    ``onset_fired`` and ``is_anomaly`` come from the audio layer: ``is_anomaly`` is True
    (anomalous, interrupt), False (benign, ignore and count it), or None (nothing
    conditioned the verdict, so any onset interrupts).

    ``realizable`` selects the arm and defaults to the realizable one, which is what the
    smoke runs (§8) — the old flag defaulted to the oracle path so prior runs stayed
    byte-identical, and those runs are being deleted. The realizable branch never reads
    ``source_xyz`` or ``arrived_at_source``, so the reach numbers it produces are not
    measured on privileged information; the oracle branch reads both, and its privilege
    shows in its trajectory and its audit record rather than in its report.
    """
    primary = state.primary_goal

    if state.mode == NavMode.SEARCH:
        if primary_goal_reached:
            nxt = replace(state, mode=NavMode.COMPLETE, active_goal=primary)
            return nxt, ControllerDecision(mode=NavMode.COMPLETE, active_goal=primary)

        if onset_fired and not state.investigated:
            interrupt = is_anomaly is True or is_anomaly is None
            # The realizable arm needs only the onset; the oracle arm needs a coordinate
            # to point-goal to, so without one it cannot enter and keeps searching.
            if interrupt and (realizable or source_xyz is not None):
                goal = anomaly_object or SOURCE_PSEUDO_GOAL
                nxt = replace(
                    state,
                    mode=NavMode.INVESTIGATE,
                    active_goal=goal,
                    # The realizable arm stores no oracle coordinate: steering comes from
                    # the live energy climb, and holding one would make it available to a
                    # later edit that "just needs the distance".
                    investigate_target_xyz=None if realizable else source_xyz,
                    investigate_steps=0,
                )
                if realizable:
                    return nxt, ControllerDecision(
                        mode=NavMode.INVESTIGATE,
                        active_goal=goal,
                        realizable_action=realizable_investigate_step(
                            energy_history or [], lateral_sign, visual_confirm
                        ),
                        force_requery=True,
                        save_primary_state=True,
                    )
                return nxt, ControllerDecision(
                    mode=NavMode.INVESTIGATE,
                    active_goal=goal,
                    investigate_waypoint=source_xyz,
                    force_requery=True,
                    save_primary_state=True,
                )
            if is_anomaly is False:
                state = replace(state, n_benign_ignored=state.n_benign_ignored + 1)
            # An anomaly with no usable cue, or a benign sound: keep searching.

        nxt = replace(state, active_goal=primary)
        return nxt, ControllerDecision(mode=NavMode.SEARCH, active_goal=primary)

    if state.mode == NavMode.INVESTIGATE:
        state = replace(state, investigate_steps=state.investigate_steps + 1)
        # Compute the realizable action first, so a STOP both transitions to CHECK and is
        # not also handed to the runner to apply.
        action = None
        if realizable:
            action = realizable_investigate_step(
                energy_history or [], lateral_sign, visual_confirm
            )
        arrived = (action == ACT_STOP) if realizable else bool(arrived_at_source)

        if arrived:
            event = InvestigationEvent(
                anomaly_class=anomaly_class,
                visual_confirm_object=anomaly_object if visual_confirm else None,
                investigate_steps=state.investigate_steps,
                stopped_at_pose=pose,
                realizable=bool(realizable),
            )
            nxt = replace(
                state, mode=NavMode.CHECK, investigated=True, investigation_event=event
            )
            return nxt, ControllerDecision(
                mode=NavMode.CHECK,
                active_goal=state.active_goal,
                investigation_event=event,
            )

        if state.investigate_steps >= int(cfg.investigate_max_steps):
            nxt = replace(
                state,
                mode=NavMode.RESUME,
                investigate_aborted=True,
                active_goal=primary,
            )
            return nxt, ControllerDecision(
                mode=NavMode.RESUME,
                active_goal=primary,
                restore_primary_state=True,
                force_requery=True,
            )

        if realizable:
            return state, ControllerDecision(
                mode=NavMode.INVESTIGATE,
                active_goal=state.active_goal,
                realizable_action=action,
            )
        return state, ControllerDecision(
            mode=NavMode.INVESTIGATE,
            active_goal=state.active_goal,
            investigate_waypoint=state.investigate_target_xyz,
        )

    if state.mode == NavMode.CHECK:
        nxt = replace(state, mode=NavMode.RESUME, active_goal=primary)
        return nxt, ControllerDecision(
            mode=NavMode.RESUME,
            active_goal=primary,
            restore_primary_state=True,
            force_requery=True,
        )

    if state.mode == NavMode.RESUME:
        nxt = replace(state, mode=NavMode.SEARCH, resumed=True, active_goal=primary)
        return nxt, ControllerDecision(mode=NavMode.SEARCH, active_goal=primary)

    # COMPLETE and REPORTED are terminal, and idempotent.
    return state, ControllerDecision(mode=state.mode, active_goal=primary)
