"""The interrupt-resume machine, and the greedy climb it drives.

ADR-0008 calls this the one module that unit-tests on a Mac and the paper's single
framing-independent positive, so the whole transition table is walked here — including the
orderings that are easy to get right by accident and wrong by refactor: the one-shot
directives fire on the transition tick only, CHECK and RESUME are each exactly one tick,
and a STOP that arrives as arrival is not also handed to the runner to apply.

Two things a fake cannot settle, and neither is faked here: whether the lateral cue's frame
is what ``audio/lateral.py`` says (``tests/box/test_audio_box.py``'s turn-around pair), and
whether an energy climb converges in a real room. What is settled here is that the rule
consumes the cue **without a compensation term** — the grid era's
``heard == -right(world-bearing)`` — which is the port error ticket 23 was told to watch
for.
"""

import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.agent.config import ControllerConfig
from earshot.agent.controller import (
    ACT_FORWARD,
    ACT_STOP,
    ACT_TURN_LEFT,
    ACT_TURN_RIGHT,
    CAST_STEPS,
    SCAN_STEPS,
    SOURCE_PSEUDO_GOAL,
    ControllerState,
    NavMode,
    is_diverting,
    is_rising,
    next_plateau_steps,
    realizable_investigate_probe,
    realizable_investigate_step,
    step_controller,
)
from earshot.types import Pose, Xyz

CFG = ControllerConfig(investigate_max_steps=4)
SOURCE = Xyz(3.0, 0.0, -2.0)
POSE = Pose(position=Xyz(0.5, 0.0, -0.5), yaw_rad=0.2)


def searching(goal="chair"):
    return ControllerState.for_episode(goal)


def tick(state, **kwargs):
    kwargs.setdefault("onset_fired", False)
    kwargs.setdefault("is_anomaly", None)
    kwargs.setdefault("primary_goal_reached", False)
    # The realizable arm places its probe relative to the agent, so it needs a pose and
    # raises without one rather than falling back to stepping blind. Defaulted here so
    # every test that is not *about* the pose reads as it did before.
    kwargs.setdefault("pose", POSE)
    return step_controller(state, CFG, **kwargs)


class TestTheGreedyRule(unittest.TestCase):
    def test_no_reading_yet_probes_forward(self):
        self.assertEqual(realizable_investigate_step([], 0, False), ACT_FORWARD)

    def test_rising_loudness_keeps_going(self):
        self.assertEqual(realizable_investigate_step([0.1, 0.2], 0, False), ACT_FORWARD)

    def test_a_peak_with_a_visual_confirm_stops(self):
        self.assertEqual(realizable_investigate_step([0.3, 0.3], 0, True), ACT_STOP)

    def test_a_plateau_without_a_confirm_does_not_stop(self):
        """Otherwise the agent stops at an arbitrary loud cell rather than at the source."""
        self.assertNotEqual(realizable_investigate_step([0.3, 0.3], 0, False), ACT_STOP)

    def test_a_confirm_ends_the_detour_even_while_the_climb_still_rises(self):
        """**Changed by `cast-1`, and this is the assertion that changed.**

        This used to read "still getting louder means the source is ahead, confirm or
        not" and expect FORWARD. That rule refused arrivals the agent had already made:
        seven of fifteen abandoned episodes in `DYehNKdT76V` had a closest approach inside
        the ring — 0.13 m in one — and since the confirm is a pure function of distance
        they had it, so `rising` must have been true at every in-ring step. The windowed
        test's baseline lags a real approach by up to `RISING_WINDOW` steps, which at
        0.25 m a step is more hysteresis than the ring is wide.

        What is given up is stopping AT the source rather than at the ring's edge. Under
        an oracle confirm the ring already carries that; under a caption confirm it would
        not, and this line is where that arm reconsiders it.
        """
        self.assertEqual(realizable_investigate_step([0.1, 0.4], 0, True), ACT_STOP)
        # and the confirm is still REQUIRED — a rising climb with no confirm walks on
        self.assertEqual(realizable_investigate_step([0.1, 0.4], 0, False), ACT_FORWARD)

    def test_a_stall_turns_toward_the_louder_half_plane_uncompensated(self):
        """``+1`` is a source to the RIGHT, so the agent turns right. No inversion.

        The grid rendered at identity listener yaw, so the cue was world-frame and the
        fusion arc compensated with ``heard == -right(world-bearing)``. Live rendering
        returns an agent-frame cue from the same arithmetic; carried across with the old
        compensation the agent turns the wrong way on every stall, and it looks like a
        mediocre climb rather than a bug.
        """
        self.assertEqual(realizable_investigate_step([0.3, 0.3], +1, False), ACT_TURN_RIGHT)
        self.assertEqual(realizable_investigate_step([0.3, 0.3], -1, False), ACT_TURN_LEFT)

    def test_an_ambiguous_sign_scans(self):
        self.assertEqual(realizable_investigate_step([0.3, 0.3], 0, False), ACT_TURN_LEFT)

    def test_a_single_reading_counts_as_rising(self):
        """No confirm, so this isolates its subject: what ONE reading does to `rising`.

        It passed `True` before and was reading the confirm branch as much as the rising
        one — which only went unnoticed while a confirm plus a rise happened to answer
        FORWARD as well.
        """
        self.assertEqual(realizable_investigate_step([0.2], 0, False), ACT_FORWARD)

    def test_none_readings_are_skipped(self):
        self.assertEqual(realizable_investigate_step([None, 0.3, 0.3], 0, True), ACT_STOP)

    def test_a_rise_inside_the_epsilon_is_a_plateau(self):
        self.assertEqual(
            realizable_investigate_step([0.3, 0.3 + 1e-9], 0, True), ACT_STOP
        )


class TestAWallIsAlreadyAStall(unittest.TestCase):
    """Why ticket 26 did NOT give the rule a collision branch, as an executable argument.

    The first box episode walked 110 forwards for 6.57 m of path and never reached
    line-of-sight, which reads as a rule pushing a wall it cannot see. It is not, and the
    chain is three facts that are each checked somewhere else in this suite:

    - ``allow_sliding`` is **False** (``sim/world.py``, the ObjectNav benchmark setting),
      so a collided forward leaves the pose unchanged;
    - ``heard_signal`` takes no pose and convolves the whole clip every step, so the
      measured RMS is a pure function of pose;
    - therefore the reading after a collision **equals** the one before it.

    Which lands on the rule as a plateau — and ADR-0011's stall branch already turns on a
    plateau. A collision branch would emit the action the stall branch emits anyway;
    ``test_task_runner``'s wall fixture measures that end to end over real geometry.

    So the flag is recorded and not consumed, and the real finding is one this rule cannot
    fix: the agent turns 30 degrees, the energy gradient points back into the wall, and it
    re-collides. Nothing in the arm remembers which yaws are blocked.
    """

    def test_a_collision_reads_as_a_plateau_and_the_stall_branch_turns(self):
        """An unchanged pose repeats the reading exactly, which is not ``rising``."""
        after_collision = [0.2, 0.2]  # the same RMS twice: the agent did not move
        self.assertEqual(
            realizable_investigate_step(after_collision, +1, False), ACT_TURN_RIGHT
        )
        self.assertEqual(
            realizable_investigate_step(after_collision, -1, False), ACT_TURN_LEFT
        )

    def test_the_rule_takes_no_collision_argument(self):
        """A parameter accepted and ignored is the inert surface this tree deletes."""
        import inspect

        params = inspect.signature(realizable_investigate_step).parameters
        self.assertNotIn(
            "collided",
            params,
            "the rule grew a collision argument — if that is deliberate, the wall "
            "fixture in test_task_runner has to show it changing a trajectory first",
        )


class TestTheProbeThePlannerRoutesTo(unittest.TestCase):
    """Ticket 26's structural fix: the climb names a place, not a step.

    Applying the rule's action straight to the simulator gave the detour no planner and
    no map — ``move_forward`` was its only translation and the gradient chose where
    forward pointed, so a blocked line to the source was a **livelock**: measured against
    every wall geometry tried, ending pressed flat against the obstacle with zero lateral
    movement, and unchanged by tripling the step budget.

    What a Mac can settle is the geometry and the provenance. Whether the follower
    actually gets around a wall is a navmesh property and lives in ``tests/box/``
    (ADR-0014: a capability is exercised, never proxied) — this suite's follower steers in
    a straight line, so a green here licenses nothing about routing.
    """

    CFG = ControllerConfig(investigate_probe_m=2.0, investigate_probe_turn_deg=60.0)

    def test_forward_probes_along_the_agents_own_heading(self):
        """``forward_xz(0)`` is ``-z``, habitat's forward — the frame ticket 23 pinned."""
        at_origin = Pose(position=Xyz(0.0, 0.0, 0.0), yaw_rad=0.0)
        probe = realizable_investigate_probe(ACT_FORWARD, at_origin, self.CFG)
        self.assertAlmostEqual(probe.x, 0.0)
        self.assertAlmostEqual(probe.z, -2.0)

    def test_a_turn_offsets_the_heading_in_the_frames_direction(self):
        """``turn_left`` adds to yaw (``occupancy.bearing_rel``), so its probe is to port."""
        at_origin = Pose(position=Xyz(0.0, 0.0, 0.0), yaw_rad=0.0)
        left = realizable_investigate_probe(ACT_TURN_LEFT, at_origin, self.CFG)
        right = realizable_investigate_probe(ACT_TURN_RIGHT, at_origin, self.CFG)
        self.assertLess(left.x, 0.0)
        self.assertGreater(right.x, 0.0)
        self.assertAlmostEqual(left.z, right.z)

    def test_the_offset_is_wider_than_one_simulator_turn(self):
        """A probe one 30-degree step off the blocked heading snaps back onto the wall."""
        self.assertGreater(self.CFG.investigate_probe_turn_deg, 30.0)

    def test_it_keeps_the_agents_own_height(self):
        """The probe is a floor-plane move; a y of its own would be a storey change."""
        pose = Pose(position=Xyz(1.0, 0.7, -3.0), yaw_rad=1.1)
        self.assertAlmostEqual(
            realizable_investigate_probe(ACT_FORWARD, pose, self.CFG).y, 0.7
        )

    def test_it_is_always_the_probe_distance_away(self):
        pose = Pose(position=Xyz(1.0, 0.0, -3.0), yaw_rad=2.3)
        for action in (ACT_FORWARD, ACT_TURN_LEFT, ACT_TURN_RIGHT):
            probe = realizable_investigate_probe(action, pose, self.CFG)
            self.assertAlmostEqual(
                pose.position.horizontal_distance_to(probe), self.CFG.investigate_probe_m
            )

    def test_arrival_is_not_a_probe(self):
        pose = Pose(position=Xyz(0.0, 0.0, 0.0), yaw_rad=0.0)
        with self.assertRaises(ValueError):
            realizable_investigate_probe(ACT_STOP, pose, self.CFG)

    def test_the_realizable_arm_emits_a_probe_rather_than_an_applied_action(self):
        state, decision = tick(searching(), onset_fired=True, is_anomaly=True)
        self.assertIsNotNone(decision.investigate_probe)
        state, decision = tick(state, energy_history=[0.1, 0.2])
        self.assertIsNotNone(decision.investigate_probe)

    def test_the_probe_is_derived_from_the_agents_pose_and_nothing_privileged(self):
        """Same cue, same pose, same probe — with the source moved 20 m away."""
        near, _ = tick(
            searching(), onset_fired=True, is_anomaly=True, source_xyz=SOURCE
        )
        far, _ = tick(
            searching(),
            onset_fired=True,
            is_anomaly=True,
            source_xyz=Xyz(20.0, 0.0, 20.0),
        )
        self.assertEqual(near.investigate_target_xyz, far.investigate_target_xyz)

    def test_the_two_arms_still_name_different_fields(self):
        """"Which arm ran" stays readable off a decision (ADR-0013's report boundary)."""
        _, realizable = tick(searching(), onset_fired=True, is_anomaly=True)
        _, oracle = tick(
            searching(),
            onset_fired=True,
            is_anomaly=True,
            realizable=False,
            source_xyz=SOURCE,
        )
        self.assertIsNone(realizable.investigate_waypoint)
        self.assertIsNotNone(realizable.investigate_probe)
        self.assertIsNotNone(oracle.investigate_waypoint)
        self.assertIsNone(oracle.investigate_probe)

    def test_the_realizable_arm_refuses_to_steer_without_a_pose(self):
        """It cannot name a place without knowing where it is, and a silent None here
        puts the detour back on the planner-less path — sometimes, and invisibly."""
        with self.assertRaises(ValueError):
            step_controller(
                searching(),
                CFG,
                onset_fired=True,
                is_anomaly=True,
                primary_goal_reached=False,
                pose=None,
            )


class TestSearch(unittest.TestCase):
    def test_no_onset_keeps_searching_on_the_primary_goal(self):
        state, decision = tick(searching())
        self.assertEqual(decision.mode, NavMode.SEARCH)
        self.assertEqual(decision.active_goal, "chair")
        self.assertIsNone(decision.realizable_action)

    def test_reaching_the_primary_goal_completes(self):
        state, decision = tick(searching(), primary_goal_reached=True)
        self.assertEqual(state.mode, NavMode.COMPLETE)
        self.assertEqual(decision.mode, NavMode.COMPLETE)

    def test_an_anomalous_onset_interrupts(self):
        state, decision = tick(
            searching(), onset_fired=True, is_anomaly=True, anomaly_object="toilet"
        )
        self.assertEqual(state.mode, NavMode.INVESTIGATE)
        self.assertEqual(decision.active_goal, "toilet")
        self.assertTrue(decision.save_primary_state)
        self.assertTrue(decision.force_requery)

    def test_an_unconditioned_verdict_also_interrupts(self):
        """``None`` means nothing conditioned the verdict, so any onset interrupts."""
        state, _ = tick(searching(), onset_fired=True, is_anomaly=None)
        self.assertEqual(state.mode, NavMode.INVESTIGATE)

    def test_a_benign_onset_is_ignored_and_counted(self):
        state, decision = tick(searching(), onset_fired=True, is_anomaly=False)
        self.assertEqual(decision.mode, NavMode.SEARCH)
        self.assertEqual(state.n_benign_ignored, 1)

    def test_benign_onsets_accumulate(self):
        state = searching()
        for _ in range(3):
            state, _ = tick(state, onset_fired=True, is_anomaly=False)
        self.assertEqual(state.n_benign_ignored, 3)

    def test_an_unnamed_source_gets_the_sentinel_goal(self):
        state, decision = tick(searching(), onset_fired=True, is_anomaly=True)
        self.assertEqual(decision.active_goal, SOURCE_PSEUDO_GOAL)

    def test_a_second_onset_after_investigating_does_not_re_interrupt(self):
        state = ControllerState(
            primary_goal="chair", active_goal="chair", investigated=True
        )
        state, decision = tick(state, onset_fired=True, is_anomaly=True)
        self.assertEqual(decision.mode, NavMode.SEARCH)

    def test_completing_takes_priority_over_an_onset(self):
        state, decision = tick(
            searching(), onset_fired=True, is_anomaly=True, primary_goal_reached=True
        )
        self.assertEqual(decision.mode, NavMode.COMPLETE)


class TestTheOracleArm(unittest.TestCase):
    def test_it_needs_a_coordinate_to_enter(self):
        state, decision = tick(
            searching(), onset_fired=True, is_anomaly=True, realizable=False
        )
        self.assertEqual(decision.mode, NavMode.SEARCH)

    def test_with_a_coordinate_it_injects_a_waypoint(self):
        state, decision = tick(
            searching(),
            onset_fired=True,
            is_anomaly=True,
            realizable=False,
            source_xyz=SOURCE,
        )
        self.assertEqual(decision.investigate_waypoint, SOURCE)
        self.assertIsNone(decision.realizable_action)
        self.assertEqual(state.investigate_target_xyz, SOURCE)

    def test_arrival_is_the_geometric_flag(self):
        state, _ = tick(
            searching(),
            onset_fired=True,
            is_anomaly=True,
            realizable=False,
            source_xyz=SOURCE,
        )
        state, decision = tick(
            state, realizable=False, source_xyz=SOURCE, arrived_at_source=True
        )
        self.assertEqual(decision.mode, NavMode.CHECK)
        self.assertFalse(decision.investigation_event.realizable)

    def test_the_realizable_arm_stores_no_coordinate(self):
        """Not merely unused: an oracle coordinate in the state is one edit from being read."""
        state, _ = tick(
            searching(), onset_fired=True, is_anomaly=True, source_xyz=SOURCE
        )
        self.assertIsNone(state.investigate_target_xyz)


class TestTheRealizableArm(unittest.TestCase):
    def _investigating(self):
        state, _ = tick(
            searching(), onset_fired=True, is_anomaly=True, anomaly_object="toilet"
        )
        return state

    def test_it_enters_on_the_onset_alone(self):
        state = self._investigating()
        self.assertEqual(state.mode, NavMode.INVESTIGATE)

    def test_every_tick_carries_an_action(self):
        state, decision = tick(self._investigating(), energy_history=[0.1, 0.2])
        self.assertEqual(decision.realizable_action, ACT_FORWARD)
        self.assertIsNone(decision.investigate_waypoint)

    def test_the_steps_count_up(self):
        state = self._investigating()
        state, _ = tick(state, energy_history=[0.1, 0.2])
        state, _ = tick(state, energy_history=[0.2, 0.3])
        self.assertEqual(state.investigate_steps, 2)

    def test_a_stop_is_arrival_and_is_not_also_applied(self):
        """The STOP transitions to CHECK; handing it to the runner would stop the episode."""
        state, decision = tick(
            self._investigating(),
            energy_history=[0.3, 0.3],
            visual_confirm=True,
            anomaly_object="toilet",
            anomaly_class="alarm",
            pose=POSE,
        )
        self.assertEqual(decision.mode, NavMode.CHECK)
        self.assertIsNone(decision.realizable_action)
        self.assertTrue(state.investigated)

    def test_the_event_is_agent_estimable_only(self):
        _, decision = tick(
            self._investigating(),
            energy_history=[0.3, 0.3],
            visual_confirm=True,
            anomaly_object="toilet",
            anomaly_class="alarm",
            pose=POSE,
        )
        event = decision.investigation_event
        self.assertEqual(event.anomaly_class, "alarm")
        self.assertEqual(event.visual_confirm_object, "toilet")
        self.assertEqual(event.stopped_at_pose, POSE)
        self.assertTrue(event.realizable)
        self.assertNotIn("source_xyz", [f for f in event.__dataclass_fields__])

    def test_an_unconfirmed_arrival_names_no_object(self):
        """The oracle arm can arrive without seeing anything, and must not claim it did."""
        state, _ = tick(
            self._investigating(), energy_history=[0.1, 0.2], visual_confirm=False
        )
        _, decision = tick(
            state,
            realizable=False,
            arrived_at_source=True,
            anomaly_object="toilet",
            visual_confirm=False,
            pose=POSE,
        )
        self.assertEqual(decision.mode, NavMode.CHECK)
        self.assertIsNone(decision.investigation_event.visual_confirm_object)

    def test_it_never_reads_the_oracle_arrival_flag(self):
        _, decision = tick(
            self._investigating(), energy_history=[0.1, 0.2], arrived_at_source=True
        )
        self.assertEqual(decision.mode, NavMode.INVESTIGATE)


class TestTheAbort(unittest.TestCase):
    def test_the_step_budget_resumes_the_primary_task(self):
        state, _ = tick(searching(), onset_fired=True, is_anomaly=True)
        for _ in range(CFG.investigate_max_steps):
            state, decision = tick(state, energy_history=[0.1, 0.1])
        self.assertEqual(decision.mode, NavMode.RESUME)
        self.assertTrue(state.investigate_aborted)
        self.assertTrue(decision.restore_primary_state)
        self.assertEqual(decision.active_goal, "chair")

    def test_an_abort_is_not_an_investigation(self):
        state, _ = tick(searching(), onset_fired=True, is_anomaly=True)
        for _ in range(CFG.investigate_max_steps):
            state, _ = tick(state, energy_history=[0.1, 0.1])
        self.assertFalse(state.investigated)
        self.assertIsNone(state.investigation_event)

    def test_an_aborted_detour_is_not_re_entered(self):
        """The sub-budget is the whole detour's, not one attempt's.

        SEARCH's guard was ``onset_fired and not state.investigated``, and an abort sets
        ``investigate_aborted`` without setting ``investigated`` — correctly, because the
        source was never reached. So the next SEARCH tick saw a still-firing onset and a
        still-false ``investigated`` and diverted again. The first box episode entered
        INVESTIGATE six times and spent about 210 of its 250 steps re-aborting, which
        makes ``investigate_max_steps`` a per-attempt budget nothing bounds in aggregate.
        """
        state, _ = tick(searching(), onset_fired=True, is_anomaly=True)
        for _ in range(CFG.investigate_max_steps):
            state, _ = tick(state, energy_history=[0.1, 0.1])
        self.assertTrue(state.investigate_aborted)

        state, decision = tick(state, onset_fired=True, is_anomaly=True)  # RESUME -> SEARCH
        state, decision = tick(state, onset_fired=True, is_anomaly=True)
        self.assertEqual(decision.mode, NavMode.SEARCH)
        self.assertEqual(decision.active_goal, "chair")

    def test_the_primary_task_still_completes_after_an_abort(self):
        """Terminal for the interrupt, not for the episode."""
        state, _ = tick(searching(), onset_fired=True, is_anomaly=True)
        for _ in range(CFG.investigate_max_steps):
            state, _ = tick(state, energy_history=[0.1, 0.1])
        state, _ = tick(state)  # RESUME -> SEARCH
        state, decision = tick(state, onset_fired=True, primary_goal_reached=True)
        self.assertEqual(decision.mode, NavMode.COMPLETE)

    def test_arrival_on_the_last_permitted_step_still_checks(self):
        """The arrival branch is evaluated before the budget, which is the carried order."""
        state, _ = tick(searching(), onset_fired=True, is_anomaly=True)
        for _ in range(CFG.investigate_max_steps - 1):
            state, _ = tick(state, energy_history=[0.1, 0.1])
        state, decision = tick(
            state, energy_history=[0.3, 0.3], visual_confirm=True, pose=POSE
        )
        self.assertEqual(decision.mode, NavMode.CHECK)


class TestCheckResumeAndTermination(unittest.TestCase):
    def _checked(self):
        state, _ = tick(searching(), onset_fired=True, is_anomaly=True)
        state, _ = tick(
            state, energy_history=[0.3, 0.3], visual_confirm=True, pose=POSE
        )
        return state

    def test_check_lasts_exactly_one_tick(self):
        state, decision = tick(self._checked())
        self.assertEqual(decision.mode, NavMode.RESUME)
        self.assertTrue(decision.restore_primary_state)
        self.assertTrue(decision.force_requery)

    def test_resume_lasts_exactly_one_tick_and_returns_to_search(self):
        state, _ = tick(self._checked())
        state, decision = tick(state)
        self.assertEqual(decision.mode, NavMode.SEARCH)
        self.assertTrue(state.resumed)

    def test_the_whole_loop_reaches_search_again_with_the_primary_goal(self):
        state = self._checked()
        for _ in range(2):
            state, decision = tick(state)
        self.assertEqual(state.mode, NavMode.SEARCH)
        self.assertEqual(state.active_goal, "chair")
        self.assertTrue(state.investigated)
        self.assertTrue(state.resumed)

    def test_after_resuming_the_primary_goal_can_still_complete(self):
        state = self._checked()
        for _ in range(2):
            state, _ = tick(state)
        state, decision = tick(state, primary_goal_reached=True)
        self.assertEqual(decision.mode, NavMode.COMPLETE)

    def test_terminal_states_are_idempotent(self):
        state, _ = tick(searching(), primary_goal_reached=True)
        for _ in range(3):
            state, decision = tick(state)
            self.assertEqual(decision.mode, NavMode.COMPLETE)


class TestDiverting(unittest.TestCase):
    def test_the_detour_suppresses_the_primary_stop(self):
        for mode in (NavMode.INVESTIGATE, NavMode.CHECK, NavMode.RESUME):
            self.assertTrue(is_diverting(mode))

    def test_search_and_the_terminal_states_do_not(self):
        """Suppressing the terminal states would break primary success outright."""
        for mode in (NavMode.SEARCH, NavMode.COMPLETE, NavMode.REPORTED):
            self.assertFalse(is_diverting(mode))


class TestTheStateIsAValue(unittest.TestCase):
    def test_the_state_is_frozen(self):
        with self.assertRaises(Exception):
            searching().mode = NavMode.CHECK  # type: ignore[misc]

    def test_the_caller_s_state_is_not_advanced(self):
        """The old function mutated in place, so a leaked controller carried an
        ``investigated`` into the next episode and ``reset()`` existed to paper over it."""
        state = searching()
        step_controller(
            state,
            CFG,
            onset_fired=True,
            is_anomaly=True,
            primary_goal_reached=False,
            pose=POSE,
        )
        self.assertEqual(state.mode, NavMode.SEARCH)
        self.assertEqual(state.n_benign_ignored, 0)

    def test_a_fresh_episode_is_a_fresh_value(self):
        first = searching()
        advanced, _ = tick(first, onset_fired=True, is_anomaly=False)
        second = ControllerState.for_episode("bed")
        self.assertEqual(second.n_benign_ignored, 0)
        self.assertEqual(second.primary_goal, "bed")
        self.assertEqual(advanced.n_benign_ignored, 1)

    def test_the_decisions_mode_always_equals_the_returned_states_mode(self):
        state = searching()
        script = [
            {"onset_fired": True, "is_anomaly": True},
            {"energy_history": [0.1, 0.2]},
            {"energy_history": [0.3, 0.3], "visual_confirm": True, "pose": POSE},
            {},
            {},
            {"primary_goal_reached": True},
        ]
        for kwargs in script:
            state, decision = tick(state, **kwargs)
            self.assertEqual(decision.mode, state.mode)

    def test_the_two_arms_directives_are_mutually_exclusive(self):
        state, realizable = tick(searching(), onset_fired=True, is_anomaly=True)
        _, oracle = tick(
            searching(),
            onset_fired=True,
            is_anomaly=True,
            realizable=False,
            source_xyz=SOURCE,
        )
        self.assertIsNotNone(realizable.realizable_action)
        self.assertIsNone(realizable.investigate_waypoint)
        self.assertIsNone(oracle.realizable_action)
        self.assertIsNotNone(oracle.investigate_waypoint)


class TestClimbRuleOff(unittest.TestCase):
    """ADR-0018's ``ClimbRule`` arm, on the pure rule directly.

    `pilot-2` measured the climb as already inert in BOTH arms on real energy traces —
    rise/eps below 1 in every distance band, 90-99% of detour steps plateaued, so the
    pre-registered expectation for ``ClimbRule.OFF`` is NO CHANGE on the box. Nothing
    here is tuned to make the arm look effective; this suite proves the switch is clean
    on a fixture built to fire the rule, which real traces mostly don't.
    """

    # A strictly climbing trace, chosen so `is_rising` reads a real gap against the
    # dispersion term (see the module's worked calculation in `is_rising`'s docstring):
    # both windows are full (`MIN_DISPERSION_SAMPLES` = 6 readings) and the gap (0.3)
    # clears the measured bar (~0.153) with room to spare.
    RISING = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    def test_on_reproduces_the_climb(self):
        action = realizable_investigate_step(self.RISING, 0, False, climb_enabled=True)
        print("ClimbRule.LIVE on a rising trace:", action)
        self.assertEqual(action, ACT_FORWARD)

    def test_off_forces_the_scan_cast_cycle_on_the_identical_trace(self):
        action = realizable_investigate_step(self.RISING, 0, False, climb_enabled=False)
        print("ClimbRule.OFF on the SAME rising trace:", action)
        # Not merely "not forward" -- the OFF arm must actually run the cycle: at
        # plateau_steps=0 (the default) that is the scan phase's turn.
        self.assertEqual(action, ACT_TURN_LEFT)

    def test_off_does_not_depend_on_a_doctored_epsilon(self):
        """The rule must SKIP calling `is_rising`, not call it with a huge threshold --
        a huge `eps` would make OFF depend on a magic number instead of a branch. Proven
        by showing an eps of exactly zero still fails to produce a rise."""
        action = realizable_investigate_step(
            self.RISING, 0, False, eps=0.0, climb_enabled=False
        )
        print("ClimbRule.OFF at eps=0.0:", action)
        self.assertNotEqual(action, ACT_FORWARD)

    def test_next_plateau_steps_on_reads_the_climb(self):
        n = next_plateau_steps(self.RISING, plateau_steps=0, climb_enabled=True)
        print("next_plateau_steps, climb ON, rising trace:", n)
        self.assertEqual(n, 0)  # a real rise resets the plateau counter

    def test_next_plateau_steps_off_advances_on_the_identical_trace(self):
        """The counter must honour the same short-circuit the action rule does, or the
        cast cycle restarts every tick of a detour whose climb is supposedly off."""
        n = next_plateau_steps(self.RISING, plateau_steps=0, climb_enabled=False)
        print("next_plateau_steps, climb OFF, SAME rising trace:", n)
        self.assertEqual(n, 1)

    def test_is_rising_sigmas_is_now_a_parameter_and_defaults_unchanged(self):
        """`sigmas` moved from module scope to a keyword; the default must still equal
        the module constant so no caller's behaviour shifted by adding it."""
        from earshot.agent.controller import RISING_SIGMAS

        default_verdict = is_rising(self.RISING, eps=1e-6, window=5)
        explicit_verdict = is_rising(self.RISING, eps=1e-6, window=5, sigmas=RISING_SIGMAS)
        self.assertEqual(default_verdict, explicit_verdict)
        self.assertTrue(default_verdict)


class TestLateralCueOff(unittest.TestCase):
    """ADR-0018's ``LateralCue`` arm: the sign is treated as absent, i.e. the same path
    an episode with a genuinely ambiguous cue already takes -- ``_turn_toward``'s own
    documented default of scanning left."""

    DEAD = [0.3, 0.3]  # flat: never rising, isolates the stall turn

    def test_on_turns_toward_the_cued_side(self):
        action = realizable_investigate_step(
            self.DEAD, +1, False, plateau_steps=0, lateral_cue_enabled=True
        )
        print("LateralCue.LIVE, sign=+1:", action)
        self.assertEqual(action, ACT_TURN_RIGHT)

    def test_off_scans_left_regardless_of_the_sign(self):
        action = realizable_investigate_step(
            self.DEAD, +1, False, plateau_steps=0, lateral_cue_enabled=False
        )
        print("LateralCue.OFF, sign=+1 (must read as absent):", action)
        self.assertEqual(action, ACT_TURN_LEFT)

    def test_off_flips_a_left_leaning_sign_too(self):
        """Not just "ignores +1" -- -1 must ALSO read as absent, not as a second vote
        for the same answer left already gives by coincidence."""
        action = realizable_investigate_step(
            self.DEAD, -1, False, plateau_steps=0, lateral_cue_enabled=False
        )
        print("LateralCue.OFF, sign=-1 (must ALSO read as absent):", action)
        self.assertEqual(action, ACT_TURN_LEFT)


class TestCastPolicyScanOnly(unittest.TestCase):
    """ADR-0018's ``CastPolicy`` arm, past the scan phase where the two arms diverge."""

    DEAD = [0.3, 0.3]  # flat: never rising, isolates the cast/scan cycle
    PAST_SCAN = SCAN_STEPS + 1  # one dead step into the cast phase (index == 1)

    def test_cast_runs_a_leg(self):
        action = realizable_investigate_step(
            self.DEAD, 0, False, plateau_steps=self.PAST_SCAN, cast_steps=CAST_STEPS
        )
        print("CastPolicy.CAST, one step past the scan:", action)
        self.assertEqual(action, ACT_FORWARD)

    def test_scan_only_turns_on_the_identical_dead_step(self):
        action = realizable_investigate_step(
            self.DEAD, 0, False, plateau_steps=self.PAST_SCAN, cast_steps=0
        )
        print("CastPolicy.SCAN_ONLY, the SAME dead step:", action)
        self.assertIn(action, (ACT_TURN_LEFT, ACT_TURN_RIGHT))

    def test_scan_only_never_produces_a_leg_across_several_dead_steps(self):
        """Not one lucky sample -- every plateau depth past the scan turns, never walks,
        under SCAN_ONLY. `cast_action`'s own docstring calls this the pre-`eps-1`
        control arm; this pins that a dead step never reads as a leg under it."""
        actions = [
            realizable_investigate_step(
                self.DEAD, 0, False, plateau_steps=self.PAST_SCAN + k, cast_steps=0
            )
            for k in range(5)
        ]
        print("CastPolicy.SCAN_ONLY, five consecutive dead steps:", actions)
        self.assertTrue(all(a in (ACT_TURN_LEFT, ACT_TURN_RIGHT) for a in actions))


class TestTheArmsReachBothCallSites(unittest.TestCase):
    """`step_controller` calls the realizable rule from two places: the SEARCH-entry
    tick that opens the detour, and every INVESTIGATE tick after (contract §5.2). A
    forward that lands on only one of the two is the bug where the first tick of a
    detour runs a different policy from the rest of it -- so each arm below is checked
    at BOTH call sites, not just one.
    """

    RISING = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    DEAD = [0.3, 0.3]

    def _investigating(self, plateau_steps=0):
        """A state already inside INVESTIGATE, to exercise the SECOND call site
        directly rather than through however many ticks it takes to reach it."""
        return ControllerState(
            primary_goal="chair", active_goal="toilet",
            mode=NavMode.INVESTIGATE, plateau_steps=plateau_steps,
        )

    # -- ClimbRule, at both call sites --------------------------------------

    def test_climb_on_at_the_search_entry_call_site(self):
        _, decision = tick(
            searching(), onset_fired=True, is_anomaly=True,
            energy_history=self.RISING, climb_enabled=True,
        )
        print("ClimbRule.LIVE, SEARCH-entry call site:", decision.realizable_action)
        self.assertEqual(decision.realizable_action, ACT_FORWARD)

    def test_climb_off_at_the_search_entry_call_site(self):
        _, decision = tick(
            searching(), onset_fired=True, is_anomaly=True,
            energy_history=self.RISING, climb_enabled=False,
        )
        print("ClimbRule.OFF, SEARCH-entry call site:", decision.realizable_action)
        self.assertEqual(decision.realizable_action, ACT_TURN_LEFT)

    def test_climb_on_at_the_investigate_call_site(self):
        _, decision = tick(
            self._investigating(), energy_history=self.RISING, climb_enabled=True
        )
        print("ClimbRule.LIVE, INVESTIGATE call site:", decision.realizable_action)
        self.assertEqual(decision.realizable_action, ACT_FORWARD)

    def test_climb_off_at_the_investigate_call_site(self):
        _, decision = tick(
            self._investigating(), energy_history=self.RISING, climb_enabled=False
        )
        print("ClimbRule.OFF, INVESTIGATE call site:", decision.realizable_action)
        self.assertEqual(decision.realizable_action, ACT_TURN_LEFT)

    # -- LateralCue, at both call sites --------------------------------------

    def test_lateral_on_at_the_search_entry_call_site(self):
        _, decision = tick(
            searching(), onset_fired=True, is_anomaly=True,
            energy_history=self.DEAD, lateral_sign=+1, lateral_cue_enabled=True,
        )
        print("LateralCue.LIVE, SEARCH-entry call site:", decision.realizable_action)
        self.assertEqual(decision.realizable_action, ACT_TURN_RIGHT)

    def test_lateral_off_at_the_search_entry_call_site(self):
        _, decision = tick(
            searching(), onset_fired=True, is_anomaly=True,
            energy_history=self.DEAD, lateral_sign=+1, lateral_cue_enabled=False,
        )
        print("LateralCue.OFF, SEARCH-entry call site:", decision.realizable_action)
        self.assertEqual(decision.realizable_action, ACT_TURN_LEFT)

    def test_lateral_on_at_the_investigate_call_site(self):
        _, decision = tick(
            self._investigating(), energy_history=self.DEAD,
            lateral_sign=+1, lateral_cue_enabled=True,
        )
        print("LateralCue.LIVE, INVESTIGATE call site:", decision.realizable_action)
        self.assertEqual(decision.realizable_action, ACT_TURN_RIGHT)

    def test_lateral_off_at_the_investigate_call_site(self):
        _, decision = tick(
            self._investigating(), energy_history=self.DEAD,
            lateral_sign=+1, lateral_cue_enabled=False,
        )
        print("LateralCue.OFF, INVESTIGATE call site:", decision.realizable_action)
        self.assertEqual(decision.realizable_action, ACT_TURN_LEFT)


class TestCastPolicyAcrossAFullDetour(unittest.TestCase):
    """`CastPolicy` diverges from `CastPolicy.CAST` only once the scan phase is
    exhausted, and the SEARCH-entry call always opens at `plateau_steps=0` -- inside the
    scan phase regardless of `cast_steps` -- so the two arms cannot be told apart on
    that FIRST tick alone. This drives `step_controller` the way the runner actually
    does: repeatedly, holding the arm constant tick to tick, which is the only way the
    divergence is observable and exercises both call sites end to end in the process.
    """

    DEAD = [0.3, 0.3]
    # The module-level `CFG`/`tick()` cap a detour at 4 steps, well short of the
    # `SCAN_STEPS + 1` INVESTIGATE ticks this drive needs to reach the cast leg -- an
    # abort there would resume the primary task and read as a `None` action, not as
    # either arm's answer. A local, wider budget is this class's own thing to own.
    CFG = ControllerConfig(investigate_max_steps=SCAN_STEPS + 5)

    def _drive_past_the_scan(self, *, cast_steps):
        state, decision = step_controller(
            searching(), self.CFG,
            onset_fired=True, is_anomaly=True, primary_goal_reached=False, pose=POSE,
            energy_history=self.DEAD, cast_steps=cast_steps,
        )
        for _ in range(SCAN_STEPS + 1):
            state, decision = step_controller(
                state, self.CFG,
                onset_fired=False, is_anomaly=None, primary_goal_reached=False, pose=POSE,
                energy_history=self.DEAD, cast_steps=cast_steps,
            )
        self.assertEqual(
            state.mode, NavMode.INVESTIGATE,
            "the drive aborted or arrived before reaching the cast leg -- widen CFG",
        )
        return decision

    def test_cast_runs_a_leg_once_the_scan_is_exhausted(self):
        decision = self._drive_past_the_scan(cast_steps=CAST_STEPS)
        print("CastPolicy.CAST, end of a full driven detour:", decision.realizable_action)
        self.assertEqual(decision.realizable_action, ACT_FORWARD)

    def test_scan_only_never_runs_a_leg_over_the_same_drive(self):
        decision = self._drive_past_the_scan(cast_steps=0)
        print(
            "CastPolicy.SCAN_ONLY, end of the SAME driven detour:",
            decision.realizable_action,
        )
        self.assertIn(decision.realizable_action, (ACT_TURN_LEFT, ACT_TURN_RIGHT))


if __name__ == "__main__":
    unittest.main(verbosity=2)
