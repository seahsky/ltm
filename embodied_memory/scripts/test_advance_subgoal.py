"""
Sanity tests for the MultiON sub-goal cursor in ``episode_runner``.

Two layers, both habitat/torch-free (stub-and-load bootstrap of
``test_episode_runner_detector.py``):

1. ``_advance_subgoal`` — the pure advance decision (found / finished) from
   (distance-to-active-subgoal, caption confirm, cursor position, radius).
   This is the only load-bearing core-loop change; everything else is gated
   behind ``K > 1``.
2. Runner-level with a stub source:
   * a single-goal episode produces NO multion keys in the episode log and
     terminates exactly as before (the K==1 path is untouched), and
   * a K=3 episode advances the cursor on (distance < found_radius AND
     caption confirm), logs ordered ``subgoals_found`` events, and STOPs
     after the final sub-goal (success_multion / progress / ppl populated).

Invoke with::

    python embodied_memory/scripts/test_advance_subgoal.py
"""
from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# ----------------------------------------------------------------------
# stub-and-load bootstrap (pattern of test_episode_runner_detector.py)
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
    return mod


def _load_file_as(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _goal_terms_stub(goal):
    return [str(goal).strip().lower().replace("_", " ")]


def _caption_mentions_stub(caption, terms):
    cap = str(caption).lower()
    for t in terms:
        if re.search(r"\b" + re.escape(t) + r"\b", cap):
            return t
    return None


def _bootstrap():
    _ensure_stub_package()
    _stub_submodule("embodied_memory.episode_source",
                    ["Episode", "EpisodeSource", "Step"])
    _stub_submodule("embodied_memory.memory_bridge", ["EmbodiedMemoryBridge"])
    percep = _stub_submodule(
        "embodied_memory.perception",
        ["CLIPKeyframeEncoder", "SemanticCaptioner"])

    class _Keyframe:  # kwargs-tolerant stand-in for perception.Keyframe
        def __init__(self, **kw):
            self.__dict__.update(kw)

    percep.Keyframe = _Keyframe
    remembr = _stub_submodule("embodied_memory.remembr_backbone",
                              ["ReMEmbRBuilder", "ReMEmbRPlanner"])
    # The advance check + arrival-STOP call these as functions — stub with
    # behaviour-equivalent implementations (word-boundary keyword match).
    remembr._goal_terms = _goal_terms_stub
    remembr._caption_mentions = _caption_mentions_stub
    hab = types.ModuleType("embodied_memory.habitat_env")
    hab._ACTION_NAMES = [
        "stop", "move_forward", "turn_left", "turn_right", "look_up", "look_down",
    ]
    sys.modules["embodied_memory.habitat_env"] = hab
    _load_file_as("embodied_memory.frontier_planner",
                  _EMB_DIR / "frontier_planner.py")
    return _load_file_as("embodied_memory.episode_runner",
                         _EMB_DIR / "episode_runner.py")


er = _bootstrap()
_advance_subgoal = er._advance_subgoal
ACTION_STOP = er.ACTION_STOP
ACTION_FORWARD = er.ACTION_FORWARD


# ----------------------------------------------------------------------
# 1. pure helper
# ----------------------------------------------------------------------


def case_far_not_found():
    assert _advance_subgoal(5.0, True, 0, 3, 1.0) == (False, False)
    print("  case_far_not_found: OK")


def case_near_without_caption_not_found():
    # Lenient strictness: proximity alone never advances — the caption must
    # confirm the active category (wrong-category confirms never abort either,
    # they simply don't advance).
    assert _advance_subgoal(0.4, False, 0, 3, 1.0) == (False, False)
    print("  case_near_without_caption_not_found: OK")


def case_found_mid_advances_not_finished():
    assert _advance_subgoal(0.4, True, 1, 3, 1.0) == (True, False)
    print("  case_found_mid_advances_not_finished: OK")


def case_found_last_finishes():
    assert _advance_subgoal(0.4, True, 2, 3, 1.0) == (True, True)
    print("  case_found_last_finishes: OK")


def case_none_distance_not_found():
    # Base EpisodeSource.distance_to_category returns None (no sim / mock).
    assert _advance_subgoal(None, True, 0, 3, 1.0) == (False, False)
    print("  case_none_distance_not_found: OK")


def case_boundary_distance_not_found():
    assert _advance_subgoal(1.0, True, 0, 3, 1.0) == (False, False)
    assert _advance_subgoal(0.999, True, 0, 3, 1.0) == (True, False)
    print("  case_boundary_distance_not_found: OK")


# ----------------------------------------------------------------------
# 2. runner-level with a stub source
# ----------------------------------------------------------------------


class _RR:
    selected = None
    debug_info = {"top_scores": []}


class _StubBridge:
    text_encode_fn = staticmethod(lambda s: np.zeros(8, dtype=np.float32))
    _pending: list = []

    def begin_episode(self, *a, **k):
        pass

    def observe_keyframe(self, *a, **k):
        pass

    def propose_memory_candidates(self, *a, **k):
        return []

    def rerank(self, **k):
        return _RR(), {}

    def consolidate(self, *a, **k):
        pass

    def stats(self):
        return {}


class _StubPlanner:
    decision_period = 10

    def reset(self, agent_pos=None):
        pass

    def update(self, *a, **k):
        pass

    def is_decision_step(self):
        return False

    def propose(self, pos, yaw):
        return [er.FrontierCandidate(
            candidate_id=1, world_xy=np.array([5.0, 5.0], dtype=np.float32),
            grid_rc=(0, 0), distance_m=5.0, bearing_rad=0.0, cluster_size=1,
            raw_score=0.5, source="frontier", metadata={})]

    def step_controller(self, cand, pos, yaw):
        return ACTION_FORWARD

    def grid_stats(self):
        return {"cells_free": 0, "cells_occupied": 0,
                "cells_unknown": 0, "frontier_cells": 0}

    def controller_stats(self):
        return {"astar_path": 0, "astar_reachable_fallback": 0,
                "astar_fallback": 0, "collision_escape": 0,
                "replan_scheduled": 0, "replan_forced": 0, "replan_stuck": 0}


class _StubSource:
    """Scripted env: categories become 'visible+near' at fixed step counts
    (chair@2, bed@5, toilet@8). Captions mention every currently-visible
    category; distance_to_category drops below the radius at the same step."""

    _SCHED = {"chair": 2, "bed": 5, "toilet": 8}

    def __init__(self, seq=None):
        self.t = 0
        self.seq = seq

    # -- helpers -------------------------------------------------------
    def _caption(self):
        seen = [c for c, t0 in self._SCHED.items() if self.t >= t0]
        return "a room with " + " and a ".join(seen) if seen else "an empty hallway"

    def _mk_step(self, action, done):
        return SimpleNamespace(
            step_idx=self.t,
            rgb=np.zeros((4, 4, 3), dtype=np.uint8),
            depth=np.ones((4, 4), dtype=np.float32),
            semantic=None,
            agent_state=SimpleNamespace(
                position=np.zeros(3, dtype=np.float32), rotation_yaw=0.0),
            action=action,
            reward=0.0,
            done=done,
            info={"caption": self._caption(),
                  "text_embedding": np.zeros(8, dtype=np.float32),
                  "visual_embedding": np.zeros(8, dtype=np.float32),
                  "distance_to_goal": 5.0},
        )

    # -- EpisodeSource surface used by _run_episode --------------------
    def reset(self, episode_idx):
        self.t = 0
        meta = {"max_steps": 20}
        if self.seq:
            meta["object_categories"] = list(self.seq)
        ep = SimpleNamespace(
            episode_id="m0", scene_id="S",
            target_category=(self.seq[0] if self.seq else "chair"),
            target_position=None, metadata=meta, success=False, spl=0.0)
        return self._mk_step(action=None, done=False), ep

    def step(self, action):
        self.t += 1
        return self._mk_step(action=action, done=(action == ACTION_STOP))

    def get_sim(self):
        return None

    def distance_to_category(self, agent_pos, category):
        return 0.5 if self.t >= self._SCHED.get(category, 10**9) else 5.0

    def nearest_category_viewpoint(self, agent_pos, category):
        return 3.0, np.array([1.0, 0.0, 1.0], dtype=np.float32)


def _mk_runner(source, **kw):
    tmp = tempfile.mkdtemp(prefix="multion-runner-test-")
    return er.EpisodeRunner(
        source=source, planner=_StubPlanner(), bridge=_StubBridge(),
        clip_encoder=None, captioner=None, out_dir=tmp,
        target_category="chair", keyframe_every_m=1,
        max_steps_per_episode=20, backbone="frontier", **kw)


def case_single_goal_run_has_no_multion_keys():
    runner = _mk_runner(_StubSource(seq=None))
    ep_log, metrics = runner._run_episode(0)
    multion_keys = {"is_multion", "target_categories", "subgoals_found",
                    "progress", "ppl", "spl_multion", "success_multion",
                    "path_len_taken", "geodesic_optimal",
                    "geodesic_optimal_partial", "recall_assisted_advances"}
    leaked = multion_keys & set(ep_log)
    assert not leaked, f"single-goal log must carry no multion keys: {leaked}"
    # K==1 never STOPs via the multion cursor: runs to the step budget.
    assert ep_log["action_stop"] == 0, ep_log["action_stop"]
    assert ep_log["n_steps"] == 19, ep_log["n_steps"]
    print("  case_single_goal_run_has_no_multion_keys: OK")


def case_k3_run_advances_cursor_and_stops():
    seq = ["chair", "bed", "toilet"]
    runner = _mk_runner(_StubSource(seq=seq))
    ep_log, metrics = runner._run_episode(0)

    assert ep_log["is_multion"] is True
    assert ep_log["target_categories"] == seq
    found = ep_log["subgoals_found"]
    assert [f["category"] for f in found] == seq, found
    assert [f["subgoal_idx"] for f in found] == [0, 1, 2], found
    # ordered in time
    steps = [f["step_idx"] for f in found]
    assert steps == sorted(steps) and len(set(steps)) == 3, steps
    for f in found:
        assert f["distance"] < 1.0
        assert f["memory_assisted"] is False  # stub bridge never fires
        assert "path_len_at_found" in f
    assert ep_log["progress"] == 1.0
    assert ep_log["success_multion"] is True
    # STOP only after the FINAL sub-goal
    assert ep_log["action_stop"] == 1, ep_log["action_stop"]
    assert ep_log["n_steps"] < 19, "must terminate before the step budget"
    # L_opt = 3 ordered legs of 3.0 from the stub seam
    assert abs(ep_log["geodesic_optimal"] - 9.0) < 1e-6
    assert ep_log["geodesic_optimal_partial"] is False
    assert ep_log["ppl"] is not None and ep_log["ppl"] > 0.0
    assert ep_log["recall_assisted_advances"] == 0
    print("  case_k3_run_advances_cursor_and_stops: OK")


def case_k3_partial_progress_no_stop_until_timeout():
    # toilet never becomes reachable -> 2 of 3 found, no STOP, progress 2/3.
    class _Partial(_StubSource):
        _SCHED = {"chair": 2, "bed": 5, "toilet": 10**9}

    runner = _mk_runner(_Partial(seq=["chair", "bed", "toilet"]))
    ep_log, _ = runner._run_episode(0)
    assert [f["category"] for f in ep_log["subgoals_found"]] == ["chair", "bed"]
    assert ep_log["success_multion"] is False
    assert abs(ep_log["progress"] - 2.0 / 3.0) < 1e-9
    assert ep_log["action_stop"] == 0, "must not STOP before the final sub-goal"
    assert ep_log["n_steps"] == 19
    print("  case_k3_partial_progress_no_stop_until_timeout: OK")


# ----------------------------------------------------------------------
# 3. absorbing-loop diagnostics (multion-micro2: turn-forever ep with 741
#    re-proposes; wall-pushing eps with 700+ forwards and zero displacement
#    while collision_escape stayed 0 — that counter belongs to the dead
#    grid-controller path, so the follower era needs its own counters)
# ----------------------------------------------------------------------


def case_waypoint_outcome_pure():
    # not a follower-done tick -> no classification
    assert er._waypoint_outcome(False, 0.4) is None
    # done within goal_radius+slack -> reached
    assert er._waypoint_outcome(True, 0.4) == "reached"
    assert er._waypoint_outcome(True, 0.79) == "reached"
    # done while still far -> the follower gave up: unreachable
    assert er._waypoint_outcome(True, 2.0) == "unreachable"
    # no candidate distance available -> conservative: unreachable
    assert er._waypoint_outcome(True, None) == "unreachable"
    print("  case_waypoint_outcome_pure: OK")


def case_forward_no_progress_pure():
    assert er._forward_no_progress(ACTION_FORWARD, 0.01) is True
    assert er._forward_no_progress(ACTION_FORWARD, 0.2) is False
    # non-forward actions never count (turning in place is not wall-pushing)
    assert er._forward_no_progress(er.ACTION_TURN_LEFT, 0.0) is False
    print("  case_forward_no_progress_pure: OK")


def case_runner_logs_loop_diagnostics():
    # The stub source never moves the agent (position pinned at zeros) and the
    # stub planner always drives FORWARD -> every forward must be flagged
    # no-progress; the no-sim grid fallback never sets force_repropose ->
    # waypoint counters stay 0 but the keys must exist in the episode log.
    runner = _mk_runner(_StubSource(seq=None))
    ep_log, _ = runner._run_episode(0)
    for key in ("n_waypoint_reached", "n_waypoint_unreachable",
                "n_forward_no_progress"):
        assert key in ep_log, f"missing diagnostic key: {key}"
    assert ep_log["action_forward"] > 0
    assert ep_log["n_forward_no_progress"] == ep_log["action_forward"], \
        (ep_log["n_forward_no_progress"], ep_log["action_forward"])
    assert ep_log["n_waypoint_reached"] == 0
    assert ep_log["n_waypoint_unreachable"] == 0
    print("  case_runner_logs_loop_diagnostics: OK")


def main() -> int:
    print("multion sub-goal cursor sanity tests")
    case_far_not_found()
    case_near_without_caption_not_found()
    case_found_mid_advances_not_finished()
    case_found_last_finishes()
    case_none_distance_not_found()
    case_boundary_distance_not_found()
    case_single_goal_run_has_no_multion_keys()
    case_k3_run_advances_cursor_and_stops()
    case_k3_partial_progress_no_stop_until_timeout()
    case_waypoint_outcome_pure()
    case_forward_no_progress_pure()
    case_runner_logs_loop_diagnostics()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
