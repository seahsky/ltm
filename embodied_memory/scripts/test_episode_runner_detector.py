"""
Layer-2 sanity tests for the EpisodeRunner detector intercept.

We unit-test the small block that decides what action to emit at a
stop_signal=True candidate, with a mock detector. We do NOT spin up Habitat;
we only exercise the decision branch.

Uses the same stub-and-load bootstrap as ``test_propose_candidates.py`` to
load ``episode_runner`` without touching faiss/transformers/habitat.

Invoke with::

    python embodied_memory/scripts/test_episode_runner_detector.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# stub-and-load: bring up just enough of `embodied_memory` to evaluate
# `_decide_stop_or_approach` without touching faiss/transformers/habitat.
# ----------------------------------------------------------------------

_EMB_DIR = Path(__file__).resolve().parent.parent  # …/embodied_memory


def _ensure_stub_package() -> None:
    if "embodied_memory" in sys.modules:
        return
    pkg = types.ModuleType("embodied_memory")
    pkg.__path__ = [str(_EMB_DIR)]
    sys.modules["embodied_memory"] = pkg


def _stub_submodule(name: str, attrs):
    mod = types.ModuleType(name)
    for a in attrs:
        setattr(mod, a, type(a, (), {}))
    sys.modules[name] = mod


def _load_file_as(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _bootstrap():
    _ensure_stub_package()
    _stub_submodule("embodied_memory.episode_source",
                    ["Episode", "EpisodeSource", "Step"])
    _stub_submodule("embodied_memory.memory_bridge",
                    ["EmbodiedMemoryBridge"])
    _stub_submodule("embodied_memory.perception",
                    ["CLIPKeyframeEncoder", "Keyframe", "SemanticCaptioner"])
    _stub_submodule("embodied_memory.remembr_backbone",
                    ["ReMEmbRBuilder", "ReMEmbRPlanner",
                     "_caption_mentions", "_goal_terms"])
    hab = types.ModuleType("embodied_memory.habitat_env")
    hab._ACTION_NAMES = [
        "stop", "move_forward", "turn_left", "turn_right", "look_up", "look_down",
    ]
    sys.modules["embodied_memory.habitat_env"] = hab
    # Real frontier_planner — pure-Python, no heavy deps.
    _load_file_as("embodied_memory.frontier_planner",
                  _EMB_DIR / "frontier_planner.py")
    # Real episode_runner — imports the above by name.
    er = _load_file_as("embodied_memory.episode_runner",
                       _EMB_DIR / "episode_runner.py")
    return er


er_mod = _bootstrap()
_decide_stop_or_approach = er_mod._decide_stop_or_approach  # NEW helper (Step 3)
_approach_arrived = er_mod._approach_arrived  # NEW helper (c7)
_detector_memory_agrees = er_mod._detector_memory_agrees  # NEW helper (c9)
_oracle_stop_override = er_mod._oracle_stop_override  # NEW helper (oracle ladder)
_arrival_stop = er_mod._arrival_stop  # NEW helper (waypoint-arrival STOP)
ACTION_STOP = er_mod.ACTION_STOP
ACTION_FORWARD = er_mod.ACTION_FORWARD


class _Cand:
    """Minimal stand-in for a FrontierCandidate (source + raw_score)."""
    def __init__(self, source, raw_score):
        self.source = source
        self.raw_score = raw_score


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


class _MockDetector:
    def __init__(self, returns):
        self.returns = returns
        self.calls = 0

    def locate(self, **kwargs):
        self.calls += 1
        return self.returns


def _intrinsics():
    return {"fx": 1.0, "fy": 1.0, "cx": 1.0, "cy": 1.0, "image_hw": (256, 256)}


def case_detector_off_emits_stop_unchanged():
    counters = {"n_detector_called": 0, "n_detector_localized": 0,
                "n_detector_locate_failed": 0}
    action, approach_wp = _decide_stop_or_approach(
        detector_enabled=False, detector=None,
        rgb=np.zeros((256, 256, 3), dtype=np.uint8),
        depth=np.full((256, 256), 2.0, dtype=np.float32),
        goal_category="chair",
        agent_pose=np.eye(4, dtype=np.float32),
        intrinsics=_intrinsics(),
        counters=counters,
    )
    assert action == ACTION_STOP
    assert approach_wp is None
    assert counters == {"n_detector_called": 0, "n_detector_localized": 0,
                        "n_detector_locate_failed": 0}
    print("  case_detector_off_emits_stop_unchanged: OK")


def case_detector_on_locate_none_falls_back_to_stop():
    counters = {"n_detector_called": 0, "n_detector_localized": 0,
                "n_detector_locate_failed": 0}
    det = _MockDetector(returns=None)
    action, approach_wp = _decide_stop_or_approach(
        detector_enabled=True, detector=det,
        rgb=np.zeros((256, 256, 3), dtype=np.uint8),
        depth=np.full((256, 256), 2.0, dtype=np.float32),
        goal_category="chair",
        agent_pose=np.eye(4, dtype=np.float32),
        intrinsics=_intrinsics(),
        counters=counters,
    )
    assert action == ACTION_STOP
    assert approach_wp is None
    assert det.calls == 1
    assert counters["n_detector_called"] == 1
    assert counters["n_detector_localized"] == 0
    assert counters["n_detector_locate_failed"] == 1  # regression guard for Issue 4 rename
    print("  case_detector_on_locate_none_falls_back_to_stop: OK")


def case_detector_on_locate_returns_waypoint_installs_approach():
    counters = {"n_detector_called": 0, "n_detector_localized": 0,
                "n_detector_locate_failed": 0}
    wp = np.array([1.5, 0.0, 2.3], dtype=np.float32)
    det = _MockDetector(returns=wp)
    action, approach_wp = _decide_stop_or_approach(
        detector_enabled=True, detector=det,
        rgb=np.zeros((256, 256, 3), dtype=np.uint8),
        depth=np.full((256, 256), 2.0, dtype=np.float32),
        goal_category="chair",
        agent_pose=np.eye(4, dtype=np.float32),
        intrinsics=_intrinsics(),
        counters=counters,
    )
    assert action is None   # caller must drive toward approach_wp
    assert np.allclose(approach_wp, wp, atol=1e-6)
    assert counters["n_detector_called"] == 1
    assert counters["n_detector_localized"] == 1
    assert counters["n_detector_locate_failed"] == 0
    print("  case_detector_on_locate_returns_waypoint_installs_approach: OK")


def case_pathfinder_wired_before_decide_in_run_episode():
    """Regression for c5 crash: ``GoalDetector`` is constructed with
    ``pathfinder=None`` in ``run_hm3d_pol`` (the Habitat sim doesn't exist
    yet at that point) and is wired lazily inside ``_run_episode``. If the
    wiring runs AFTER ``_decide_stop_or_approach`` (which calls
    ``detector.locate(...)`` -> ``pathfinder.snap_point(...)``), the first
    detector tick crashes with 'NoneType has no attribute snap_point'.

    Pin the contract via source inspection: the wiring assignment must
    appear in ``_run_episode`` BEFORE any ``_decide_stop_or_approach``
    call site within the same method body.
    """
    src = (_EMB_DIR / "episode_runner.py").read_text()
    run_ep = "def _run_episode("
    assert run_ep in src
    body_start = src.index(run_ep)
    # Heuristic: scan until the next top-level def or class (same indent).
    # All _run_episode body lines are indented >= 8 spaces; the next
    # method/class returns to 4 spaces. We slice to that boundary.
    nl = src.index("\n", body_start)
    lines = src[nl + 1:].splitlines()
    body_lines = []
    for ln in lines:
        if ln and not ln.startswith(" ") and not ln.startswith("\t"):
            break
        # detect end of method body — next 4-space def/class at method level
        if ln.startswith("    def ") or ln.startswith("    class "):
            break
        body_lines.append(ln)
    body = "\n".join(body_lines)

    wire = "self.goal_detector.pathfinder = "
    call = "_decide_stop_or_approach("
    assert wire in body, "wiring assignment missing from _run_episode"
    assert call in body, "_decide_stop_or_approach call missing from _run_episode"
    assert body.index(wire) < body.index(call), (
        "pathfinder wiring must come BEFORE _decide_stop_or_approach in _run_episode "
        "(c5 crashed because the only existing wiring lived inside the post-locate "
        "branch and never ran on the first detector call)"
    )
    print("  case_pathfinder_wired_before_decide_in_run_episode: OK")


def case_detector_counters_match_renamed_key():
    """Regression: counter name is n_detector_locate_failed, not n_detector_offnavmesh."""
    counters = {"n_detector_called": 0, "n_detector_localized": 0,
                "n_detector_locate_failed": 0}
    det = _MockDetector(returns=None)
    action, approach_wp = _decide_stop_or_approach(
        detector_enabled=True, detector=det,
        rgb=np.zeros((256, 256, 3), dtype=np.uint8),
        depth=np.full((256, 256), 2.0, dtype=np.float32),
        goal_category="chair",
        agent_pose=np.eye(4, dtype=np.float32),
        intrinsics=_intrinsics(),
        counters=counters,
    )
    assert action == ACTION_STOP
    assert approach_wp is None
    assert counters["n_detector_called"] == 1
    assert counters["n_detector_localized"] == 0
    assert counters["n_detector_locate_failed"] == 1, counters
    print("  case_detector_counters_match_renamed_key: OK")


def case_approach_arrived_via_force_repropose():
    """The follower-STOP/None path (force_repropose=True) counts as arrival
    even when the agent is still far from the raw waypoint."""
    arrived, stop_distance = _approach_arrived(
        True, [1.0, 0.0, 1.0], [5.0, 0.0, 5.0], 0.25
    )
    assert arrived is True
    assert np.isclose(stop_distance, float(np.sqrt(32.0)), atol=1e-6), stop_distance
    print("  case_approach_arrived_via_force_repropose: OK")


def case_approach_arrived_via_soft_backstop():
    """Inside the tight 0.25 m ring, arrival fires without a follower signal."""
    arrived, stop_distance = _approach_arrived(
        False, [5.1, 0.0, 5.0], [5.0, 0.0, 5.0], 0.25
    )
    assert arrived is True
    assert np.isclose(stop_distance, 0.1, atol=1e-6), stop_distance
    print("  case_approach_arrived_via_soft_backstop: OK")


def case_approach_not_arrived_when_far_and_no_signal():
    """No follower signal + outside the ring → keep approaching (no premature STOP)."""
    arrived, stop_distance = _approach_arrived(
        False, [0.0, 0.0, 0.0], [5.0, 0.0, 5.0], 0.25
    )
    assert arrived is False
    assert np.isclose(stop_distance, float(np.sqrt(50.0)), atol=1e-6), stop_distance
    print("  case_approach_not_arrived_when_far_and_no_signal: OK")


def case_memory_agrees_when_near_sighting():
    """Detector point within agree_radius of a same-category LTM sighting -> agree."""
    wp = np.array([5.0, 0.0, 5.0], dtype=np.float32)
    sightings = [np.array([4.0, 6.0], dtype=np.float32)]  # xz, dist=sqrt(2)~1.41 < 2.0
    assert _detector_memory_agrees(wp, sightings, 2.0) is True
    print("  case_memory_agrees_when_near_sighting: OK")


def case_memory_disagrees_when_far_from_all_sightings():
    """Detector point far from every sighting -> disagree (wrong-instance guard)."""
    wp = np.array([5.0, 0.0, 5.0], dtype=np.float32)
    sightings = [np.array([0.0, 0.0], dtype=np.float32),   # dist ~7.07
                 np.array([10.0, 10.0], dtype=np.float32)]  # dist ~7.07
    assert _detector_memory_agrees(wp, sightings, 2.0) is False
    print("  case_memory_disagrees_when_far_from_all_sightings: OK")


def case_memory_gate_cold_no_sightings():
    """Empty sighting list (cold visit, no recall) -> disagree (cold suppression)."""
    wp = np.array([5.0, 0.0, 5.0], dtype=np.float32)
    assert _detector_memory_agrees(wp, [], 2.0) is False
    print("  case_memory_gate_cold_no_sightings: OK")


def case_memory_gate_disabled_when_none():
    """mem_world_xys=None -> gate disabled (legacy c7 always-commit behavior)."""
    wp = np.array([5.0, 0.0, 5.0], dtype=np.float32)
    assert _detector_memory_agrees(wp, None, 2.0) is True
    print("  case_memory_gate_disabled_when_none: OK")


def case_decide_gates_to_stop_on_disagreement():
    """_decide_stop_or_approach: locate succeeds but memory disagrees -> STOP,
    n_detector_gated incremented; n_detector_localized still counts the locate."""
    counters = {"n_detector_called": 0, "n_detector_localized": 0,
                "n_detector_locate_failed": 0, "n_detector_gated": 0}
    wp = np.array([5.0, 0.0, 5.0], dtype=np.float32)
    det = _MockDetector(returns=wp)
    action, approach_wp = _decide_stop_or_approach(
        detector_enabled=True, detector=det,
        rgb=np.zeros((256, 256, 3), dtype=np.uint8),
        depth=np.full((256, 256), 2.0, dtype=np.float32),
        goal_category="chair",
        agent_pose=np.eye(4, dtype=np.float32),
        intrinsics=_intrinsics(),
        counters=counters,
        mem_world_xys=[np.array([0.0, 0.0], dtype=np.float32)],  # far -> disagree
        agree_radius=2.0,
    )
    assert action == ACTION_STOP
    assert approach_wp is None
    assert counters["n_detector_localized"] == 1
    assert counters["n_detector_gated"] == 1
    print("  case_decide_gates_to_stop_on_disagreement: OK")


def case_decide_commits_on_agreement():
    """_decide_stop_or_approach: locate succeeds and memory agrees -> approach;
    n_detector_gated stays 0."""
    counters = {"n_detector_called": 0, "n_detector_localized": 0,
                "n_detector_locate_failed": 0, "n_detector_gated": 0}
    wp = np.array([5.0, 0.0, 5.0], dtype=np.float32)
    det = _MockDetector(returns=wp)
    action, approach_wp = _decide_stop_or_approach(
        detector_enabled=True, detector=det,
        rgb=np.zeros((256, 256, 3), dtype=np.uint8),
        depth=np.full((256, 256), 2.0, dtype=np.float32),
        goal_category="chair",
        agent_pose=np.eye(4, dtype=np.float32),
        intrinsics=_intrinsics(),
        counters=counters,
        mem_world_xys=[np.array([5.5, 5.0], dtype=np.float32)],  # dist 0.5 -> agree
        agree_radius=2.0,
    )
    assert action is None
    assert np.allclose(approach_wp, wp, atol=1e-6)
    assert counters["n_detector_localized"] == 1
    assert counters["n_detector_gated"] == 0
    print("  case_decide_commits_on_agreement: OK")


def case_arrival_stop_fires_on_confident_memory_arrival():
    """Arrived at a high-cosine memory waypoint with caption confirm -> STOP."""
    cand = _Cand("memory", 0.55)
    assert _arrival_stop(True, cand, True, 0.4) is True
    print("  case_arrival_stop_fires_on_confident_memory_arrival: OK")


def case_arrival_stop_requires_arrival():
    """Not yet arrived -> never STOP (keep navigating)."""
    cand = _Cand("memory", 0.55)
    assert _arrival_stop(False, cand, True, 0.4) is False
    print("  case_arrival_stop_requires_arrival: OK")


def case_arrival_stop_requires_memory_source():
    """Frontier/remembr waypoint arrival -> no STOP (only memory recalls)."""
    assert _arrival_stop(True, _Cand("frontier", 0.9), True, 0.4) is False
    assert _arrival_stop(True, None, True, 0.4) is False
    print("  case_arrival_stop_requires_memory_source: OK")


def case_arrival_stop_requires_high_cosine():
    """Low-confidence memory recall -> no STOP (avoid wrong-instance stops)."""
    assert _arrival_stop(True, _Cand("memory", 0.30), True, 0.4) is False
    print("  case_arrival_stop_requires_high_cosine: OK")


def case_arrival_stop_requires_caption_confirm():
    """Arrived at a confident memory waypoint but caption doesn't mention the
    goal -> no STOP."""
    assert _arrival_stop(True, _Cand("memory", 0.55), False, 0.4) is False
    print("  case_arrival_stop_requires_caption_confirm: OK")


def case_arrival_stop_wired():
    """Source-scan: waypoint-arrival STOP wired in runner + caption matcher reused."""
    src = (_EMB_DIR / "episode_runner.py").read_text()
    assert "def _arrival_stop" in src
    assert "n_arrival_stop" in src
    assert "_arrival_stop_cos" in src
    assert "ARRIVAL_STOP_COS" in src
    assert '"0.4"' in src  # default cosine gate
    # reuses remembr_backbone's whole-word caption matcher
    assert "_caption_mentions" in src and "_goal_terms" in src
    print("  case_arrival_stop_wired: OK")


def case_oracle_stop_forces_stop_within_radius():
    """Oracle-STOP: inside the GT success ring -> force STOP regardless of action."""
    assert _oracle_stop_override(ACTION_FORWARD, 0.05, 0.1) == ACTION_STOP
    print("  case_oracle_stop_forces_stop_within_radius: OK")


def case_oracle_stop_passthrough_outside_radius():
    """Outside the ring -> action unchanged (agent keeps navigating)."""
    assert _oracle_stop_override(ACTION_FORWARD, 0.5, 0.1) == ACTION_FORWARD
    print("  case_oracle_stop_passthrough_outside_radius: OK")


def case_oracle_stop_passthrough_unknown_d2g():
    """No GT distance available -> never force STOP (safe passthrough)."""
    assert _oracle_stop_override(ACTION_FORWARD, None, 0.1) == ACTION_FORWARD
    print("  case_oracle_stop_passthrough_unknown_d2g: OK")


def case_oracle_ladder_wired():
    """Source-scan: oracle-STOP/location toggles wired in runner + CLI."""
    src = (_EMB_DIR / "episode_runner.py").read_text()
    assert "def _oracle_stop_override" in src
    assert "self.oracle_stop" in src
    assert "self.oracle_location" in src
    assert "oracle_stop_radius" in src
    # location override steers to the GT goal inside the loop
    assert "target_position" in src
    cli = (_EMB_DIR / "run_hm3d_pol.py").read_text()
    assert "--oracle-stop" in cli
    assert "--oracle-location" in cli
    print("  case_oracle_ladder_wired: OK")


def case_memory_gate_wired():
    """Source-scan: the detector-memory agreement gate is wired end-to-end."""
    src = (_EMB_DIR / "episode_runner.py").read_text()
    assert "def _detector_memory_agrees" in src
    assert "n_detector_gated" in src
    assert "_detector_mem_agree_m" in src
    assert '"2.0"' in src  # default DETECTOR_MEM_AGREE_M
    assert "mem_world_xys" in src
    print("  case_memory_gate_wired: OK")


def case_approach_uses_tight_follower():
    """Source-scan: the dedicated 0.25 m approach follower is wired and the
    approach branches use it; _waypoint_action treats follower-STOP as arrival."""
    src = (_EMB_DIR / "episode_runner.py").read_text()
    assert "def _init_approach_follower" in src
    assert "self.approach_follower" in src
    assert "_approach_goal_radius" in src
    assert '"0.25"' in src  # default DETECTOR_APPROACH_RADIUS
    assert "use_approach_follower=True" in src
    assert "ACTION_STOP" in src  # referenced in _waypoint_action arrival wiring
    print("  case_approach_uses_tight_follower: OK")


def main() -> int:
    print("episode_runner detector intercept tests")
    case_detector_off_emits_stop_unchanged()
    case_detector_on_locate_none_falls_back_to_stop()
    case_detector_on_locate_returns_waypoint_installs_approach()
    case_detector_counters_match_renamed_key()
    case_pathfinder_wired_before_decide_in_run_episode()
    case_approach_arrived_via_force_repropose()
    case_approach_arrived_via_soft_backstop()
    case_approach_not_arrived_when_far_and_no_signal()
    case_approach_uses_tight_follower()
    case_memory_agrees_when_near_sighting()
    case_memory_disagrees_when_far_from_all_sightings()
    case_memory_gate_cold_no_sightings()
    case_memory_gate_disabled_when_none()
    case_decide_gates_to_stop_on_disagreement()
    case_decide_commits_on_agreement()
    case_memory_gate_wired()
    case_oracle_stop_forces_stop_within_radius()
    case_oracle_stop_passthrough_outside_radius()
    case_oracle_stop_passthrough_unknown_d2g()
    case_oracle_ladder_wired()
    case_arrival_stop_fires_on_confident_memory_arrival()
    case_arrival_stop_requires_arrival()
    case_arrival_stop_requires_memory_source()
    case_arrival_stop_requires_high_cosine()
    case_arrival_stop_requires_caption_confirm()
    case_arrival_stop_wired()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
