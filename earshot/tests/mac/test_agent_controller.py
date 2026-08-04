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
    SOURCE_PSEUDO_GOAL,
    ControllerState,
    NavMode,
    is_diverting,
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

    def test_rising_with_a_confirm_keeps_going(self):
        """Still getting louder means the source is ahead, confirm or not."""
        self.assertEqual(realizable_investigate_step([0.1, 0.4], 0, True), ACT_FORWARD)

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
        self.assertEqual(realizable_investigate_step([0.2], 0, True), ACT_FORWARD)

    def test_none_readings_are_skipped(self):
        self.assertEqual(realizable_investigate_step([None, 0.3, 0.3], 0, True), ACT_STOP)

    def test_a_rise_inside_the_epsilon_is_a_plateau(self):
        self.assertEqual(
            realizable_investigate_step([0.3, 0.3 + 1e-9], 0, True), ACT_STOP
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
            state, CFG, onset_fired=True, is_anomaly=True, primary_goal_reached=False
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
