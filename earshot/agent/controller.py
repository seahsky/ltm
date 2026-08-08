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

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional, Sequence, Tuple

from earshot.agent.config import ControllerConfig
from earshot.agent.occupancy import forward_xz
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
    "RISING_WINDOW",
    "RISING_SIGMAS",
    "MIN_DISPERSION_SAMPLES",
    "CAST_STEPS",
    "SCAN_STEPS",
    "UNMEASURED_EPS",
    "climb_eps",
    "is_rising",
    "cast_action",
    "next_plateau_steps",
    "realizable_investigate_step",
    "realizable_investigate_probe",
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


# How many readings each side of the comparison averages. One (the original behaviour)
# compares two adjacent samples, and `detour-2` measured what that costs: 325 of 336
# plateau windows were a SINGLE step with zero travel — the agent leaving FORWARD for one
# tick and turning. Five is the scale of those dropouts and still short against a 121-step
# detour, so a real climb is blunted by at most a few steps.
RISING_WINDOW = 5

# provenance: fake — how many standard errors the gap between the two windows must clear.
# One, because the quantity it multiplies is measured rather than assumed and a larger
# bar would cost forwards on a cue `eps-1` measured as real but small.
RISING_SIGMAS = 1.0

# Below this many readings the dispersion term is not estimated. A sample SD over two or
# three points is mostly its own noise, and at n=2 it exactly cancels the difference it is
# meant to bound, which would answer "not rising" to every early step. Under it the rule
# clears `eps` alone, which is the pre-`eps-1` behaviour and is why short histories still
# degrade to the original comparison.
MIN_DISPERSION_SAMPLES = 6

# provenance: derived — how many forward steps a cast leg commits to before the heading is
# re-derived. `ControllerConfig.investigate_probe_m` is 2.0 m and `sim.world`'s step is
# 0.25 m, so this is ONE PROBE'S REACH: the rule names a place 2 m out and the agent now
# walks to it instead of re-deriving a fresh 60-degree offset after every 0.25 m, which is
# what turned the plateau branch into an orbit.
CAST_STEPS = 8

# provenance: derived — how many dead-cue steps are spent turning BEFORE any casting. A
# `ControllerConfig.investigate_probe_turn_deg` of 60 makes six of them a full circle, so
# this is one complete sweep.
#
# **It is here because a mis-oriented agent is not a lost one.** Facing away from a live
# source also reads as a dead cue, and that case is fixed by consecutive turns — the
# lateral sign homes onto the bearing in a turn or three and the climb resumes. A cast
# that interrupts after the first turn walks the agent away from a source it was about to
# find, which is measurable in the fake world (`TestTheStallTurnsTowardTheSource`) before
# it is measurable on the box. Only once the cue has stayed dead through a whole sweep is
# orientation ruled out and travel the remaining move.
SCAN_STEPS = 6

# provenance: fake — the threshold the climb falls back to when the renderer's scatter was
# not measured. It is the pre-`detour-2` constant, kept only so an episode with no
# measurement still runs a defined rule, and it is deliberately far below any real noise
# floor: a run that silently ran at this value is a run whose climb was a coin flip, and
# the audit records which of the two was in force so that is visible rather than inferred.
UNMEASURED_EPS = 1e-6


def climb_eps(render_scatter: Optional[float]) -> float:
    """The threshold the climb actually ran with, given the episode's measured scatter.

    **One copy of the fallback policy, read by both the runner and the replay.** The
    runner decides an episode's `eps` here and `tools/detour_report` reconstructs plateau
    windows with the same call, so a replay cannot drift into judging `rising` at a
    different threshold than the agent used. Two copies of this conditional would be two
    controllers, one of which never ran.

    ``None`` means *not measured* and never zero — `calibrate_onset` leaves it ``None``
    when fewer than two repeats arrived, and a zero threshold would read as a renderer
    that agreed with itself exactly, which would be a finding rather than a default.
    """
    return UNMEASURED_EPS if render_scatter is None else float(render_scatter)


def _sample_sd(values: Sequence[float]) -> float:
    """Sample (``n - 1``) SD. Zero for fewer than two points, which the caller guards.

    Sample rather than population for the same reason `calibration.render_scatter_of`
    uses it: these are a handful of draws from a distribution, not the whole of it.
    """
    numbers = [float(v) for v in values]
    if len(numbers) < 2:
        return 0.0
    mean = sum(numbers) / len(numbers)
    return math.sqrt(sum((v - mean) ** 2 for v in numbers) / (len(numbers) - 1))


def is_rising(
    energy_history: Sequence[float],
    *,
    eps: float,
    window: int = RISING_WINDOW,
) -> bool:
    """Is the agent getting louder, judged as a trend between two windows?

    **Both sides average, and `eps-1` is why.** The version this replaces compared the
    single latest reading against a median of the preceding ``window`` — which halves the
    noise on the baseline and leaves the current side carrying all of it. That run then
    measured what the current side is up against: pooled over 20 episodes, one 0.25 m
    forward step buys a rise of **0.61 to 0.86 of the local scatter** in every band inside
    5 m, including the 2-3 m band where the detours die. The cue is real and it is smaller
    than the variation a single pose-to-pose comparison is read through, so no threshold on
    a single reading can recover it. Averaging both sides makes the signal grow with
    ``window`` while the noise falls as its square root.

    **The bar is the larger of two things, and that is the second `eps-1` finding.**
    ``eps`` is the *renderer's* disagreement with itself, median 3.3e-3 over that run. The
    scatter the agent actually walks through — the field's own pose-to-pose variation,
    turns included — measured 7e-3 to 1.2e-2, two to three times larger. A rule that clears
    only the render floor is under-thresholded in exactly the regime it operates in, so the
    gap must also clear ``RISING_SIGMAS`` standard errors of the observed dispersion:

        gap = mean(recent) - mean(baseline)
        rising = gap > max(eps, RISING_SIGMAS * s * sqrt(2 / window))

    ``s`` is the sample SD of the pooled readings, so the bar is measured on the agent's
    own trace rather than assumed, and it rises by itself in a room whose field is noisy.
    ``eps`` stays as a floor: a renderer that agreed with itself exactly would put ``s``
    near zero, and a comparison with no floor at all would answer FORWARD to arithmetic
    dust.

    **Means, not medians, now that both sides are windows.** A median is the right summary
    of a baseline whose failure mode is one bad render; it is the wrong one for a window
    whose readings carry a *trend*, because it discards the ordering that makes the trend
    visible and it is the less efficient estimator of the level the gap is measuring.

    Short histories degrade toward the old comparison by construction: the windows shrink
    to what is available, and below ``MIN_DISPERSION_SAMPLES`` readings the dispersion term
    is not estimated at all and the rule clears ``eps`` alone.

    **Rotations are not filtered out, deliberately.** `turn_left`/`turn_right` change the
    measured RMS without changing distance (see below), so a forward-only baseline is the
    tempting fix — but which action produced a reading is decided *after* the controller
    runs, and threading it back turns a pure predicate into a stateful one. Averaging
    absorbs a symmetric contamination, and it now enters ``s`` as well, which raises the
    bar in exactly the episodes where turning is adding variance. If the contamination
    turns out to be biased rather than symmetric, that is the next thing to measure, and
    the per-step record already carries the action to measure it with.
    """
    history = [float(e) for e in energy_history if e is not None]
    if len(history) < 2:
        return True  # nothing to compare against yet, so probe forward
    # Split what is available into two adjacent blocks of at most `window`. With three
    # readings that is two against one; with two it is the original single-step compare.
    half = min(int(window), len(history) // 2)
    recent = history[-half:]
    baseline = history[-2 * half : -half]
    gap = (sum(recent) / len(recent)) - (sum(baseline) / len(baseline))
    bar = float(eps)
    pooled = recent + baseline
    if len(pooled) >= MIN_DISPERSION_SAMPLES:
        dispersion = _sample_sd(pooled) * math.sqrt(2.0 / len(recent))
        bar = max(bar, RISING_SIGMAS * dispersion)
    return gap > bar


def realizable_investigate_step(
    energy_history: Sequence[float],
    lateral_sign: int,
    visual_confirm: bool,
    *,
    eps: float = UNMEASURED_EPS,
    window: int = RISING_WINDOW,
    plateau_steps: int = 0,
    cast_steps: int = CAST_STEPS,
) -> str:
    """One step of realizable anomaly-source localization (ADR-0011): SURGE, or CAST.

    Pure, and from **agent-estimable signals only** — it never reads a ground-truth source
    distance or coordinate:

    - ``energy_history`` — the agent's own recent binaural loudness at successive poses.
      Climbing it walks toward "getting louder".
    - ``lateral_sign`` — the interaural level sign, ``+1`` source to the right, ``-1``
      left, ``0`` ambiguous. Agent-frame, uncompensated (see the module docstring).
    - ``visual_confirm`` — the detector confirms the anomaly object is here.
    - ``plateau_steps`` — how many consecutive steps the cue has already been dead for.
      Not a reading; the count of them, carried on ``ControllerState`` because the runner
      trims ``energy_history`` and a plateau outlives the readings it started in.

    Rising loudness means forward, and peak-or-plateau plus visual confirm means STOP, so
    the agent stops *at* the source rather than at an arbitrary loud cell. Both carried
    unchanged. What is new is the third branch.

    **The un-cued branch now CASTS instead of turning, and `eps-1` is why.**

    Every earlier version answered "turn" whenever the cue was dead, so no branch advanced
    a plateaued agent. That defect was survivable only by accident: the pre-`detour-2` test
    was ``current > previous + 1e-6`` against a renderer scattering 2.8e-3, a coin flip on
    flat ground, and P(forward) of about a half is a random walk that covers distance.
    Calibrating the threshold removed the coin flip and cost **13.2 points of
    Anomaly-response SR** — 46.0% to 32.9% over 365 paired episodes, 15 of 16 scenes down,
    sign test p = 0.0005. The noise was the exploration.

    A threshold cannot give it back. `eps-1` measured one 0.25 m forward as worth 0.61 to
    0.86 of the field's own local scatter in every band inside 5 m, so a correctly
    calibrated single-step rule fires rarely BY CONSTRUCTION. The un-cued case needs a
    policy, not a smaller epsilon.

    So: **one turn, then ``cast_steps`` forwards, then a turn the other way.** The first
    leg turns toward the louder half-plane, which is the lateral cue's last useful word.
    Later legs ALTERNATE regardless of the sign, and that is deliberate — following a
    stable sign every leg traces a closed polygon, which is an orbit, and an agent
    circling a source at five metres reads exactly like the flat field and stable sign it
    would be circling in. Alternating cannot close: the heading oscillates between two
    values one turn apart, so the sweep drifts along their bisector and the agent always
    goes somewhere.

    The cast is interrupted by the cue rather than run to completion — ``rising`` is
    evaluated every step, and a real rise surges immediately and resets the count. And
    arrival still preempts everything: ``visual_confirm`` with a dead cue is a STOP even
    mid-leg, because a cast is what the agent does when it has not arrived.

    **The rule does not read the collision flag, and ticket 26 measured why it should
    not.** The first box episode walked 110 forwards for 6.57 m of path and never reached
    line-of-sight, which looks like a rule pushing a wall it cannot see. It is not.
    ``allow_sliding`` is **False** (``sim/world.py``, the ObjectNav benchmark's setting),
    so a collided forward leaves the pose *unchanged*; ``heard_signal`` convolves the whole
    clip every step and takes no pose, so the RMS is a pure function of pose; so the reading
    after a collision **equals** the one before it, ``rising`` is False, and the stall branch
    below already turns. Adding a collision branch produces the action that branch produces
    anyway — verified over four wall geometries against the carried rule, trajectories
    byte-identical in all four. What the flag is genuinely for is the *record*
    (``report/audit.StepRecord``), which is where it now goes.

    ``turn_left`` and ``turn_right`` change the measured RMS without changing the
    distance, and §4.1 instruments that rather than fixing it — the per-step record
    carries the action taken so a rotation-driven rise is separable from a
    translation-driven one after the fact.
    """
    history = [float(e) for e in energy_history if e is not None]
    if not history:
        return ACT_FORWARD  # no reading yet, probe forward
    rising = is_rising(history, eps=eps, window=window)
    if visual_confirm and not rising:
        return ACT_STOP
    if rising:
        return ACT_FORWARD  # surge
    return cast_action(plateau_steps, lateral_sign, cast_steps=cast_steps)


def _turn_toward(lateral_sign: int) -> str:
    """The carried stall turn: toward the louder half-plane, left when the sign says
    nothing. Ambiguous, zero and absent all scan left, which is the rule's own default."""
    return ACT_TURN_RIGHT if int(lateral_sign) > 0 else ACT_TURN_LEFT


def cast_action(
    plateau_steps: int,
    lateral_sign: int,
    *,
    cast_steps: int = CAST_STEPS,
    scan_steps: int = SCAN_STEPS,
) -> str:
    """Where in the scan-then-cast cycle ``plateau_steps`` puts the agent. Pure.

    Split out of the rule so the replay reconstructs the cycle by calling it rather than
    by re-deriving it, and so the cycle reads as one expression. ``plateau_steps`` is the
    count of consecutive dead-cue steps BEFORE this one, so zero is a plateau's first step.

    Three phases, in order:

    1. **Scan** — ``scan_steps`` turns toward the louder half-plane. A mis-oriented agent
       recovers here and never reaches the rest.
    2. **Cast** — one turn, then ``cast_steps`` forwards, repeatedly.
    3. and the legs alternate direction, so the sweep cannot close into an orbit.

    ``cast_steps = 0`` collapses phases 2 and 3 into a turn on every dead step, which is
    the pre-`eps-1` behaviour and the control arm the next sweep needs. That is why both
    lengths are arguments rather than constants read from module scope.
    """
    index = max(0, int(plateau_steps))
    if index < int(scan_steps):
        return _turn_toward(lateral_sign)
    index -= int(scan_steps)
    period = 1 + int(cast_steps)
    if index % period:
        return ACT_FORWARD  # running the leg
    # Starting one. The lateral sign steers the first leg and the alternation steers the
    # rest; see the rule's docstring for why a stable sign must not steer them all.
    #
    # **The alternation is conditional on there being legs at all.** It exists to stop a
    # TRAVELLING sweep closing into a polygon, and with `cast_steps = 0` nothing travels:
    # alternating there would oscillate the agent on the spot, which is worse than the
    # rule being controlled against rather than equal to it.
    turn_right = int(lateral_sign) > 0
    if int(cast_steps) and (index // period) % 2:
        turn_right = not turn_right
    return ACT_TURN_RIGHT if turn_right else ACT_TURN_LEFT


def next_plateau_steps(
    energy_history: Sequence[float],
    *,
    eps: float = UNMEASURED_EPS,
    window: int = RISING_WINDOW,
    plateau_steps: int = 0,
) -> int:
    """The plateau count after this tick: zero if the cue is alive, one more if not. Pure.

    The counter lives on ``ControllerState`` rather than being derived inside the rule
    because the runner trims ``energy_history`` to what `is_rising` reads, and a cast leg
    is longer than that window — a rule deriving its own plateau length from a truncated
    history would silently restart the cycle every ``ENERGY_HISTORY`` steps.

    A *replay* has the whole series and can derive it, which is what makes the reconstructed
    action checkable against the recorded one (`tools/detour_report`).
    """
    history = [float(e) for e in energy_history if e is not None]
    if not history or is_rising(history, eps=eps, window=window):
        return 0
    return max(0, int(plateau_steps)) + 1


def realizable_investigate_probe(action: str, pose: Pose, cfg: ControllerConfig) -> Xyz:
    """Where the realizable detour routes to next, given the carried rule's answer.

    **Ticket 26's structural fix, and the one change to how the realizable arm moves.**
    The arm used to apply ``realizable_investigate_step``'s action *directly* to the
    simulator, which meant that during the whole detour the agent had no planner and no
    map — ``move_forward`` was its only translation, and the energy gradient decided which
    way forward pointed. Blocked forward, flat reading, turn, gradient turns it back,
    collide: measured as a **livelock** against every wall geometry tried, ending pressed
    against the obstacle with zero lateral movement, and budget-independent. No sequence
    of that rule's actions can go around anything.

    So the rule now chooses a *direction* and this turns it into a *place*, which the
    navmesh follower reaches by whatever route exists — the same machinery SEARCH already
    uses for the primary task. **The arm stays realizable**: the heading comes from live
    binaural energy and the interaural level sign, the distance is a fixed constant, and
    the map the follower plans on is the agent's own. No source coordinate enters, which
    is the property ADR-0001 built this arm for and the one the oracle arm gives up.

    The carried rule is the single source of the decision — this reads its action rather
    than re-deriving the cue, so there is one copy of ADR-0011's logic. ``ACT_STOP`` never
    arrives here: arrival is terminal and handled before a probe is wanted.
    """
    if action == ACT_STOP:
        raise ValueError(
            "arrival is not a probe — `step_controller` transitions to CHECK on a STOP "
            "and must not also ask where to go next"
        )
    offset = math.radians(float(cfg.investigate_probe_turn_deg))
    if action == ACT_TURN_LEFT:
        heading = pose.yaw_rad + offset
    elif action == ACT_TURN_RIGHT:
        heading = pose.yaw_rad - offset
    elif action == ACT_FORWARD:
        heading = pose.yaw_rad
    else:
        raise ValueError("unknown realizable action {!r}".format(action))
    dx, dz = forward_xz(heading)
    reach = float(cfg.investigate_probe_m)
    return Xyz(
        pose.position.x + dx * reach, pose.position.y, pose.position.z + dz * reach
    )


def _probe_for(
    action: Optional[str], pose: Optional[Pose], cfg: ControllerConfig
) -> Optional[Xyz]:
    """``realizable_investigate_probe``, guarded for the two cases that have no probe.

    A ``None`` action or a ``STOP`` is an arrival or a tick with nothing to steer, and a
    missing pose **raises**: the realizable arm cannot name a place without knowing where
    it is, and returning ``None`` there would put the runner back on the planner-less path
    this replaced, silently and only sometimes.
    """
    if action is None or action == ACT_STOP:
        return None
    if pose is None:
        raise ValueError(
            "the realizable arm needs a pose to place its probe — without one the "
            "detour has no waypoint and falls back to stepping blind, which is the "
            "livelock ticket 26 measured"
        )
    return realizable_investigate_probe(action, pose, cfg)


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
    # Consecutive INVESTIGATE steps whose cue was dead, which is where the agent sits in
    # the cast cycle. Zero outside the detour and reset by any rise. It is state rather
    # than a derivation because `energy_history` reaches the rule trimmed to what
    # `is_rising` reads, and a cast leg outlives that window.
    plateau_steps: int = 0
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

    **Both arms name a place, and they stay mutually exclusive.** ``investigate_waypoint``
    is the oracle arm's point goal — the source coordinate. ``investigate_probe`` is the
    realizable arm's: a point a fixed distance along the heading live energy and the
    lateral sign chose (``realizable_investigate_probe``). Both are injected into the
    candidate pool as ``SOURCE_INVESTIGATE`` so the divert wins the scorer's pick by rank.

    Two fields rather than one, because they are not the same claim — one is where the
    source *is*, the other is where the agent has decided to *look*. Collapsing them
    would make "which arm ran" unreadable off a decision, and the audit's
    ``localization_arm`` should not be the only witness to it.

    ``realizable_action`` is retained and is now **diagnostic rather than steering**: the
    arm no longer applies it to the simulator, because doing so left the whole detour with
    no planner and no map. See ``realizable_investigate_probe`` for the livelock that
    produced.
    """

    mode: NavMode
    active_goal: Optional[str]
    investigate_waypoint: Optional[Xyz] = None
    investigate_probe: Optional[Xyz] = None
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
    rising_eps: float = UNMEASURED_EPS,
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

        # **The abort is terminal for the interrupt** (ticket 26). `investigate_aborted`
        # is read here as well as `investigated`, because an abort correctly leaves
        # `investigated` False — the source was never reached — and the guard without it
        # sees a still-firing onset and diverts again. The first box episode entered
        # INVESTIGATE six times and spent about 210 of its 250 steps re-aborting, which
        # makes `investigate_max_steps` a per-attempt budget that nothing bounds in
        # aggregate. One detour per episode: the sub-budget is the whole detour's.
        if onset_fired and not state.investigated and not state.investigate_aborted:
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
                    # A fresh detour starts a fresh cast cycle: `nxt` carries
                    # `plateau_steps=0` from the state above, so this first tick is either
                    # a surge or the turn that opens a leg.
                    entry = realizable_investigate_step(
                        energy_history or [], lateral_sign, visual_confirm,
                        eps=rising_eps, plateau_steps=0,
                    )
                    nxt = replace(nxt, plateau_steps=next_plateau_steps(
                        energy_history or [], eps=rising_eps, plateau_steps=0))
                    return nxt, ControllerDecision(
                        mode=NavMode.INVESTIGATE,
                        active_goal=goal,
                        investigate_probe=_probe_for(entry, pose, cfg),
                        realizable_action=entry,
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
                energy_history or [], lateral_sign, visual_confirm, eps=rising_eps,
                plateau_steps=state.plateau_steps,
            )
            # Advanced whatever the action was, including a forward mid-leg: the count is
            # of dead-cue STEPS, not of turns, and resetting it on the leg's own forwards
            # would restart the cycle every other tick.
            state = replace(state, plateau_steps=next_plateau_steps(
                energy_history or [], eps=rising_eps,
                plateau_steps=state.plateau_steps))
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
                investigate_probe=_probe_for(action, pose, cfg),
                realizable_action=action,
                # The probe moves with the cue, so the pool is re-proposed every tick of
                # the detour rather than on the planner's decision period. A stale probe
                # is a place the energy reading that chose it no longer supports, and the
                # detour is short — this is the one loop where paying a proposal per step
                # is cheaper than steering to yesterday's guess.
                force_requery=True,
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
