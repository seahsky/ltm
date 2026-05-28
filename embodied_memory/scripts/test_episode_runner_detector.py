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
                    ["ReMEmbRBuilder", "ReMEmbRPlanner"])
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
ACTION_STOP = er_mod.ACTION_STOP


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


def main() -> int:
    print("episode_runner detector intercept tests")
    case_detector_off_emits_stop_unchanged()
    case_detector_on_locate_none_falls_back_to_stop()
    case_detector_on_locate_returns_waypoint_installs_approach()
    case_detector_counters_match_renamed_key()
    case_pathfinder_wired_before_decide_in_run_episode()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
