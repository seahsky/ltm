"""
TDD for embodied_memory/anomaly_controller.py — the pure interrupt-resume
DECISION brain of the anomaly-response task.

The robot has a PRIMARY find-task; a heard anomaly INTERRUPTS it (go to the
source, CHECK what happened) then it RESUMES and completes the primary task,
then REPORTS. This module owns ONLY the state-machine decision: given the audio
brain's onset/anomaly verdict + arrival geometry, it emits the next nav_mode,
the active retrieval goal, an optional investigate-waypoint to inject, and
save/restore/requery directives for the runner. It NEVER imports habitat_sim
(two-env boundary) and never touches the LTM — it is pure and unit-testable,
mirroring audio_task.py.

Run: /opt/anaconda3/envs/ltm-embodied/bin/python \
        embodied_memory/scripts/test_anomaly_controller.py
"""
from __future__ import annotations

import sys

from embodied_memory import anomaly_controller as ac


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _fresh(primary="chair", **cfg_kw):
    cfg = ac.AnomalyControllerConfig(enabled=True, **cfg_kw)
    st = ac.ControllerState()
    st.reset(primary)
    return cfg, st


def _search(st, cfg, **kw):
    """step_controller with sensible SEARCH-tick defaults."""
    defaults = dict(
        onset_fired=False, is_anomaly=None, source_xyz=None,
        arrived_at_source=False, primary_goal_reached=False,
        anomaly_class=None, anomaly_object=None, keyframe_caption=None,
    )
    defaults.update(kw)
    return ac.step_controller(st, cfg, **defaults)


# ----------------------------------------------------------------------
# reset / init
# ----------------------------------------------------------------------
def case_reset_initializes_search_with_primary_goal():
    _cfg, st = _fresh("bed")
    assert st.mode == ac.NavMode.SEARCH, st.mode
    assert st.primary_goal == "bed"
    assert st.active_goal == "bed"
    assert st.investigated is False
    assert st.investigation_event is None


# ----------------------------------------------------------------------
# SEARCH
# ----------------------------------------------------------------------
def case_search_no_onset_stays_search():
    cfg, st = _fresh("chair")
    d = _search(st, cfg)
    assert d.mode == ac.NavMode.SEARCH
    assert d.active_goal == "chair"
    assert d.investigate_waypoint is None
    assert d.save_primary_state is False
    assert st.active_goal == "chair"  # never diverges with no onset


def case_anomaly_onset_enters_investigate_and_saves():
    cfg, st = _fresh("chair")
    d = _search(st, cfg, onset_fired=True, is_anomaly=True,
                source_xyz=(1.0, 0.0, 2.0), anomaly_object="tv")
    assert d.mode == ac.NavMode.INVESTIGATE
    assert d.save_primary_state is True
    assert d.force_requery is True
    assert d.investigate_waypoint == (1.0, 0.0, 2.0)
    # active goal swaps to what's at the source so warm-audio memory can recall it
    assert d.active_goal == "tv"
    assert st.mode == ac.NavMode.INVESTIGATE
    assert st.investigate_target_xyz == (1.0, 0.0, 2.0)


def case_anomaly_onset_without_object_uses_sentinel_goal():
    cfg, st = _fresh("chair")
    d = _search(st, cfg, onset_fired=True, is_anomaly=True,
                source_xyz=(1.0, 0.0, 2.0), anomaly_object=None)
    assert d.mode == ac.NavMode.INVESTIGATE
    assert d.active_goal == ac.SOURCE_PSEUDO_GOAL


def case_anomaly_gate_off_onset_interrupts():
    # is_anomaly None == gate not evaluated (gate OFF) -> any onset interrupts
    cfg, st = _fresh("chair")
    d = _search(st, cfg, onset_fired=True, is_anomaly=None,
                source_xyz=(0.0, 0.0, 1.0), anomaly_object="oven")
    assert d.mode == ac.NavMode.INVESTIGATE


def case_benign_onset_ignored_stays_search_and_does_not_reinterrupt():
    cfg, st = _fresh("chair")
    d = _search(st, cfg, onset_fired=True, is_anomaly=False,
                source_xyz=(1.0, 0.0, 2.0))
    assert d.mode == ac.NavMode.SEARCH
    assert d.active_goal == "chair"
    assert d.save_primary_state is False
    assert d.investigate_waypoint is None
    assert st.benign_onset_ignored is True
    assert st.n_benign_ignored == 1
    # a SECOND benign onset is still ignored and counted, no interrupt
    d2 = _search(st, cfg, onset_fired=True, is_anomaly=False,
                 source_xyz=(1.0, 0.0, 2.0))
    assert d2.mode == ac.NavMode.SEARCH
    assert st.n_benign_ignored == 2


def case_anomaly_onset_without_source_cue_cannot_investigate():
    cfg, st = _fresh("chair")
    d = _search(st, cfg, onset_fired=True, is_anomaly=True, source_xyz=None)
    assert d.mode == ac.NavMode.SEARCH  # no waypoint -> stay searching


def case_no_reinterrupt_after_investigation_done():
    cfg, st = _fresh("chair")
    st.mode = ac.NavMode.SEARCH
    st.investigated = True  # already investigated once this episode
    d = _search(st, cfg, onset_fired=True, is_anomaly=True,
                source_xyz=(1.0, 0.0, 2.0))
    assert d.mode == ac.NavMode.SEARCH  # one investigation per episode


# ----------------------------------------------------------------------
# INVESTIGATE
# ----------------------------------------------------------------------
def case_investigate_steering_until_arrive():
    cfg, st = _fresh("chair", investigate_max_steps=40)
    st.mode = ac.NavMode.INVESTIGATE
    st.active_goal = "tv"
    st.investigate_target_xyz = (1.0, 0.0, 2.0)
    d = ac.step_controller(
        st, cfg, onset_fired=False, is_anomaly=None,
        source_xyz=(1.0, 0.0, 2.0), arrived_at_source=False,
        primary_goal_reached=False)
    assert d.mode == ac.NavMode.INVESTIGATE
    assert d.investigate_waypoint == (1.0, 0.0, 2.0)
    assert st.investigate_steps == 1
    assert d.investigation_event is None  # no CHECK yet, no STOP


def case_investigate_arrives_enters_check_no_stop():
    cfg, st = _fresh("chair")
    st.mode = ac.NavMode.INVESTIGATE
    st.active_goal = "tv"
    st.investigate_target_xyz = (1.0, 0.0, 2.0)
    d = ac.step_controller(
        st, cfg, onset_fired=False, is_anomaly=None,
        source_xyz=(1.0, 0.0, 2.0), arrived_at_source=True,
        primary_goal_reached=False, anomaly_class="alarm",
        keyframe_caption="there is a television")
    assert d.mode == ac.NavMode.CHECK
    assert st.investigated is True
    assert d.investigation_event is not None
    assert d.investigation_event["anomaly_class"] == "alarm"
    assert d.investigation_event["source_xyz"] == (1.0, 0.0, 2.0)
    assert d.investigation_event["caption"] == "there is a television"
    # oracle event byte-identical: no realizable key leaks in
    assert "realizable" not in d.investigation_event


def case_investigate_budget_overflow_aborts_to_resume():
    cfg, st = _fresh("chair", investigate_max_steps=3)
    st.mode = ac.NavMode.INVESTIGATE
    st.active_goal = "tv"
    st.investigate_target_xyz = (1.0, 0.0, 2.0)
    st.investigate_steps = 2  # next tick makes it 3 == max, still not arrived
    d = ac.step_controller(
        st, cfg, onset_fired=False, is_anomaly=None,
        source_xyz=(1.0, 0.0, 2.0), arrived_at_source=False,
        primary_goal_reached=False)
    assert d.mode == ac.NavMode.RESUME
    assert d.restore_primary_state is True
    assert d.force_requery is True
    assert st.investigate_aborted is True
    assert st.investigated is False  # never reached the source


# ----------------------------------------------------------------------
# CHECK -> RESUME -> SEARCH
# ----------------------------------------------------------------------
def case_check_transitions_to_resume_and_restores():
    cfg, st = _fresh("chair")
    st.mode = ac.NavMode.CHECK
    st.active_goal = "tv"
    d = ac.step_controller(
        st, cfg, onset_fired=False, is_anomaly=None, source_xyz=None,
        arrived_at_source=False, primary_goal_reached=False)
    assert d.mode == ac.NavMode.RESUME
    assert d.restore_primary_state is True
    assert d.force_requery is True
    assert d.active_goal == "chair"  # primary restored
    assert st.active_goal == "chair"


def case_resume_returns_to_search():
    cfg, st = _fresh("chair")
    st.mode = ac.NavMode.RESUME
    st.active_goal = "chair"
    d = ac.step_controller(
        st, cfg, onset_fired=False, is_anomaly=None, source_xyz=None,
        arrived_at_source=False, primary_goal_reached=False)
    assert d.mode == ac.NavMode.SEARCH
    assert d.active_goal == "chair"
    assert st.resumed is True


# ----------------------------------------------------------------------
# COMPLETE / REPORT
# ----------------------------------------------------------------------
def case_primary_reached_completes():
    cfg, st = _fresh("chair")
    d = _search(st, cfg, primary_goal_reached=True)
    assert d.mode == ac.NavMode.COMPLETE
    assert st.mode == ac.NavMode.COMPLETE


def case_build_report_after_investigation_and_completion():
    cfg, st = _fresh("chair")
    st.investigated = True
    st.investigation_event = {"anomaly_class": "alarm",
                              "source_xyz": (1.0, 0.0, 2.0),
                              "caption": "there is a television"}
    rep = ac.build_report(st, primary_completed=True)
    assert rep["primary_completed"] is True
    assert rep["investigated"] is True
    assert rep["anomaly_class"] == "alarm"
    assert rep["source_xyz"] == (1.0, 0.0, 2.0)


def case_build_report_on_timeout_primary_incomplete():
    cfg, st = _fresh("chair")
    st.investigated = False
    rep = ac.build_report(st, primary_completed=False)
    assert rep["primary_completed"] is False
    assert rep["investigated"] is False


# ----------------------------------------------------------------------
# invariants
# ----------------------------------------------------------------------
def case_full_episode_no_anomaly_active_goal_never_diverges():
    cfg, st = _fresh("chair")
    for _ in range(50):
        d = _search(st, cfg)
        assert d.active_goal == "chair"
        assert st.active_goal == "chair"
        assert d.mode == ac.NavMode.SEARCH


def case_is_diverting_true_for_detour_states():
    # while diverting, the runner must suppress the primary episode-ending STOP
    assert ac.is_diverting(ac.NavMode.INVESTIGATE) is True
    assert ac.is_diverting(ac.NavMode.CHECK) is True
    assert ac.is_diverting(ac.NavMode.RESUME) is True


def case_is_diverting_false_for_search_and_terminal():
    # SEARCH = normal primary nav; COMPLETE/REPORTED = primary reached, the STOP
    # is legitimate. Suppressing those would break primary success.
    assert ac.is_diverting(ac.NavMode.SEARCH) is False
    assert ac.is_diverting(ac.NavMode.COMPLETE) is False
    assert ac.is_diverting(ac.NavMode.REPORTED) is False


def case_no_habitat_sim_import():
    import importlib
    mod = importlib.import_module("embodied_memory.anomaly_controller")
    src = open(mod.__file__).read()
    assert "habitat_sim" not in src, "two-env boundary: must not import habitat_sim"
    assert "habitat" not in src.replace("habitat_env", ""), "must not import habitat"


# ----------------------------------------------------------------------
# ADR-0001 realizable localization — the pure energy-climb helper
# ----------------------------------------------------------------------
def case_realizable_probes_forward_with_no_history():
    assert ac.realizable_investigate_step([], 0, False) == ac.ACT_FORWARD


def case_realizable_climbs_while_getting_louder():
    # rising energy -> keep moving forward, even if the object is already visible
    assert ac.realizable_investigate_step([0.1, 0.2, 0.3], 0, False) == ac.ACT_FORWARD
    assert ac.realizable_investigate_step([0.1, 0.2, 0.3], 1, True) == ac.ACT_FORWARD


def case_realizable_stops_on_peak_plus_visual():
    # loudness stopped rising AND the anomaly object is confirmed -> STOP at source
    assert ac.realizable_investigate_step([0.3, 0.3], 0, True) == ac.ACT_STOP     # plateau
    assert ac.realizable_investigate_step([0.3, 0.2], 0, True) == ac.ACT_STOP     # dropped


def case_realizable_no_stop_without_visual_confirm():
    # peaked but NOT visually confirmed -> do not STOP; turn to refine
    act = ac.realizable_investigate_step([0.3, 0.2], -1, False)
    assert act != ac.ACT_STOP and act == ac.ACT_TURN_LEFT


def case_realizable_turns_toward_louder_half_plane_when_stalled():
    assert ac.realizable_investigate_step([0.3, 0.2], 1, False) == ac.ACT_TURN_RIGHT
    assert ac.realizable_investigate_step([0.3, 0.2], -1, False) == ac.ACT_TURN_LEFT
    assert ac.realizable_investigate_step([0.3, 0.2], 0, False) == ac.ACT_TURN_LEFT  # ambiguous


# ----------------------------------------------------------------------
# ADR-0001 realizable localization — the controller branch
# ----------------------------------------------------------------------
def case_realizable_entry_needs_only_onset_no_oracle_source():
    cfg, st = _fresh("chair", realizable_localization=True)
    d = _search(st, cfg, onset_fired=True, is_anomaly=True, source_xyz=None,
                anomaly_object="tv", energy_history=[0.1], lateral_sign=1,
                visual_confirm=False)
    assert d.mode == ac.NavMode.INVESTIGATE
    assert d.save_primary_state is True and d.force_requery is True
    assert d.investigate_waypoint is None            # no oracle waypoint in realizable mode
    assert d.realizable_action == ac.ACT_FORWARD      # single reading -> probe forward
    assert d.active_goal == "tv"
    assert st.investigate_target_xyz is None


def case_realizable_investigate_emits_action_not_waypoint():
    cfg, st = _fresh("chair", realizable_localization=True)
    st.mode = ac.NavMode.INVESTIGATE
    st.active_goal = "tv"
    d = ac.step_controller(
        st, cfg, onset_fired=False, is_anomaly=None, source_xyz=None,
        arrived_at_source=False, primary_goal_reached=False,
        energy_history=[0.1, 0.2, 0.3], lateral_sign=0, visual_confirm=False)
    assert d.mode == ac.NavMode.INVESTIGATE
    assert d.investigate_waypoint is None
    assert d.realizable_action == ac.ACT_FORWARD
    assert st.investigate_steps == 1


def case_realizable_stop_transitions_to_check():
    cfg, st = _fresh("chair", realizable_localization=True)
    st.mode = ac.NavMode.INVESTIGATE
    st.active_goal = "tv"
    d = ac.step_controller(
        st, cfg, onset_fired=False, is_anomaly=None, source_xyz=None,
        arrived_at_source=False, primary_goal_reached=False,
        energy_history=[0.3, 0.3], lateral_sign=0, visual_confirm=True,
        anomaly_class="alarm", keyframe_caption="a television")
    assert d.mode == ac.NavMode.CHECK
    assert st.investigated is True
    assert d.investigation_event is not None
    assert d.investigation_event["realizable"] is True


def case_realizable_branch_ignores_ground_truth_source():
    # No-GT-leak static check: the realizable decision must be IDENTICAL whether the
    # oracle source_xyz / arrived_at_source say "arrived" or are absent. If the
    # branch leaked the GT distance, arrived_at_source=True would flip it to CHECK.
    def _decide(source_xyz, arrived):
        cfg, st = _fresh("chair", realizable_localization=True)
        st.mode = ac.NavMode.INVESTIGATE
        st.active_goal = "tv"
        return ac.step_controller(
            st, cfg, onset_fired=False, is_anomaly=None,
            source_xyz=source_xyz, arrived_at_source=arrived,
            primary_goal_reached=False,
            energy_history=[0.1, 0.2, 0.3], lateral_sign=1, visual_confirm=False)
    truthful = _decide(None, False)
    leaky = _decide((0.0, 0.0, 0.0), True)     # oracle says "you're there"
    assert truthful.mode == leaky.mode == ac.NavMode.INVESTIGATE
    assert truthful.realizable_action == leaky.realizable_action == ac.ACT_FORWARD


def case_realizable_helper_reads_no_ground_truth_source_field():
    # Static check per spec: the realizable decision helper must read no ground-truth
    # source field — its signature is only the agent-estimable signals, and its body
    # references none of the privileged GT identifiers the runner computes.
    import inspect
    params = list(inspect.signature(ac.realizable_investigate_step).parameters)
    assert params == ["energy_history", "lateral_sign", "visual_confirm", "eps"], params
    src = inspect.getsource(ac.realizable_investigate_step)
    for banned in ("source_xyz", "arrived_at_source", "distance_to_goal",
                   "geodesic", "source_position"):
        assert banned not in src, f"realizable helper leaks a GT field: {banned!r}"


def case_oracle_path_byte_identical_when_realizable_off():
    # The new realizable params default-off => the oracle decision is unchanged.
    cfg, st = _fresh("chair")     # realizable_localization defaults False
    d = _search(st, cfg, onset_fired=True, is_anomaly=True,
                source_xyz=(1.0, 0.0, 2.0), anomaly_object="tv",
                energy_history=[0.5, 0.9], lateral_sign=1, visual_confirm=True)
    assert d.mode == ac.NavMode.INVESTIGATE
    assert d.investigate_waypoint == (1.0, 0.0, 2.0)   # oracle waypoint, not an action
    assert d.realizable_action is None


def main() -> int:
    cases = [
        case_reset_initializes_search_with_primary_goal,
        case_search_no_onset_stays_search,
        case_anomaly_onset_enters_investigate_and_saves,
        case_anomaly_onset_without_object_uses_sentinel_goal,
        case_anomaly_gate_off_onset_interrupts,
        case_benign_onset_ignored_stays_search_and_does_not_reinterrupt,
        case_anomaly_onset_without_source_cue_cannot_investigate,
        case_no_reinterrupt_after_investigation_done,
        case_investigate_steering_until_arrive,
        case_investigate_arrives_enters_check_no_stop,
        case_investigate_budget_overflow_aborts_to_resume,
        case_check_transitions_to_resume_and_restores,
        case_resume_returns_to_search,
        case_primary_reached_completes,
        case_build_report_after_investigation_and_completion,
        case_build_report_on_timeout_primary_incomplete,
        case_full_episode_no_anomaly_active_goal_never_diverges,
        case_is_diverting_true_for_detour_states,
        case_is_diverting_false_for_search_and_terminal,
        case_no_habitat_sim_import,
        case_realizable_probes_forward_with_no_history,
        case_realizable_climbs_while_getting_louder,
        case_realizable_stops_on_peak_plus_visual,
        case_realizable_no_stop_without_visual_confirm,
        case_realizable_turns_toward_louder_half_plane_when_stalled,
        case_realizable_entry_needs_only_onset_no_oracle_source,
        case_realizable_investigate_emits_action_not_waypoint,
        case_realizable_stop_transitions_to_check,
        case_realizable_branch_ignores_ground_truth_source,
        case_realizable_helper_reads_no_ground_truth_source_field,
        case_oracle_path_byte_identical_when_realizable_off,
    ]
    print(f"running {len(cases)} anomaly_controller cases…")
    for c in cases:
        c()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
