"""
Sanity tests for the multion reached-thrash escape (multion-micro3 ep0:
``rerank_calls≈700`` of 749 steps, ~95% turns, because the candidate pool kept
yielding waypoints already inside the 0.5 m re-propose radius — re-propose →
turn toward the near pick → "reached" again → spin forever).

Two pure helpers + runner-level behavior (habitat/torch-free stub-and-load
bootstrap, pattern of ``test_advance_subgoal.py``):

1. ``_filter_near_candidates`` — drop pool candidates already within
   ``min_target_m`` of the agent (strict ``<``, matching the
   ``candidate_reached`` trigger and ``_advance_subgoal`` convention, so a
   boundary-distance candidate that can never re-trigger "reached" survives);
   never drop ``stop_signal`` candidates; fall back to the original list when
   every non-stop candidate would be dropped.
2. ``_cooldown_elapsed`` — the reached-triggered re-propose cooldown backstop.
3. Runner-level: multion runs filter the pool and bound reached re-proposes;
   K=1 keeps the legacy behavior byte-identical (counters log-only).

Invoke with::

    python embodied_memory/scripts/test_filter_near_candidates.py
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
# stub-and-load bootstrap (pattern of test_advance_subgoal.py)
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

    class _Keyframe:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    percep.Keyframe = _Keyframe
    remembr = _stub_submodule("embodied_memory.remembr_backbone",
                              ["ReMEmbRBuilder", "ReMEmbRPlanner"])
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
_filter_near_candidates = er._filter_near_candidates
_cooldown_elapsed = er._cooldown_elapsed
ACTION_STOP = er.ACTION_STOP
ACTION_FORWARD = er.ACTION_FORWARD


def _cand(cid, xy, stop=False, source="frontier"):
    return er.FrontierCandidate(
        candidate_id=cid, world_xy=np.array(xy, dtype=np.float32),
        grid_rc=(0, 0),
        distance_m=float(np.hypot(xy[0], xy[1])),
        bearing_rad=0.0, cluster_size=1, raw_score=0.5, source=source,
        metadata=({"stop_signal": True} if stop else {}))


# ----------------------------------------------------------------------
# 1. pure: _filter_near_candidates
# ----------------------------------------------------------------------


def case_near_dropped_far_survives():
    cands = [_cand(1, (0.1, 0.0)), _cand(2, (5.0, 5.0))]
    out, n = _filter_near_candidates(cands, (0.0, 0.0), 0.5)
    assert [c.candidate_id for c in out] == [2], out
    assert n == 1, n
    print("  case_near_dropped_far_survives: OK")


def case_near_stop_signal_preserved():
    # A stop candidate is near BY DESIGN — it must never be filtered.
    cands = [_cand(1, (0.0, 0.0), stop=True), _cand(2, (5.0, 5.0))]
    out, n = _filter_near_candidates(cands, (0.0, 0.0), 0.5)
    assert [c.candidate_id for c in out] == [1, 2], out
    assert n == 0, n
    print("  case_near_stop_signal_preserved: OK")


def case_all_near_returns_original_list():
    # Never leave the agent waypoint-less: if every non-stop candidate is
    # near, fall back to the unfiltered pool (and report 0 dropped).
    cands = [_cand(1, (0.1, 0.0)), _cand(2, (0.0, 0.2))]
    out, n = _filter_near_candidates(cands, (0.0, 0.0), 0.5)
    assert [c.candidate_id for c in out] == [1, 2], out
    assert n == 0, n
    print("  case_all_near_returns_original_list: OK")


def case_all_near_plus_stop_returns_original_list():
    # Stop candidates don't count as "kept navigation targets" — with only a
    # stop survivor the non-stop pool would be empty, so fall back.
    cands = [_cand(1, (0.0, 0.0), stop=True), _cand(2, (0.1, 0.0))]
    out, n = _filter_near_candidates(cands, (0.0, 0.0), 0.5)
    assert [c.candidate_id for c in out] == [1, 2], out
    assert n == 0, n
    print("  case_all_near_plus_stop_returns_original_list: OK")


def case_boundary_distance_survives():
    # Strict <, matching _advance_subgoal AND the candidate_reached trigger:
    # a candidate at exactly min_target_m can never re-trigger "reached"
    # (that trigger is also strict <), so it is NOT part of the thrash loop.
    cands = [_cand(1, (0.5, 0.0)), _cand(2, (0.499, 0.0)), _cand(3, (5.0, 5.0))]
    out, n = _filter_near_candidates(cands, (0.0, 0.0), 0.5)
    assert [c.candidate_id for c in out] == [1, 3], out
    assert n == 1, n
    print("  case_boundary_distance_survives: OK")


def case_agent_offset_uses_world_frame():
    # Distance is agent→candidate in world (x, z), not candidate norm.
    cands = [_cand(1, (10.0, 10.0)), _cand(2, (14.0, 14.0))]
    out, n = _filter_near_candidates(cands, (10.1, 10.0), 0.5)
    assert [c.candidate_id for c in out] == [2], out
    assert n == 1, n
    print("  case_agent_offset_uses_world_frame: OK")


# ----------------------------------------------------------------------
# 2. pure: _cooldown_elapsed
# ----------------------------------------------------------------------


def case_cooldown_elapsed_pure():
    assert _cooldown_elapsed(10, 7, 3) is True
    assert _cooldown_elapsed(9, 7, 3) is False
    assert _cooldown_elapsed(8, 7, 3) is False
    # episode-start sentinel: always elapsed
    assert _cooldown_elapsed(1, -10**9, 3) is True
    # cooldown 0 disables the backstop
    assert _cooldown_elapsed(7, 7, 0) is True
    print("  case_cooldown_elapsed_pure: OK")


# ----------------------------------------------------------------------
# 2b. pure: _filter_candidates_near_points (the unreachable-waypoint
#     blacklist — multion-full1 third absorbing mode: the follower reports
#     the chosen frontier waypoint unreachable, the candidate is dropped,
#     and the planner re-proposes the SAME cluster next tick, forever:
#     S1 6/8 eps, S3 3/8, top pick re-chosen 593-732x)
# ----------------------------------------------------------------------


def case_blacklist_point_dropped_far_survives():
    cands = [_cand(1, (5.0, 5.0)), _cand(2, (8.0, 0.0))]
    out, n = er._filter_candidates_near_points(cands, [(5.2, 5.0)], 0.5)
    assert [c.candidate_id for c in out] == [2], out
    assert n == 1, n
    print("  case_blacklist_point_dropped_far_survives: OK")


def case_blacklist_multiple_points():
    cands = [_cand(1, (5.0, 5.0)), _cand(2, (8.0, 0.0)), _cand(3, (1.0, 1.0))]
    out, n = er._filter_candidates_near_points(
        cands, [(5.0, 5.0), (1.2, 1.0)], 0.5)
    assert [c.candidate_id for c in out] == [2], out
    assert n == 2, n
    print("  case_blacklist_multiple_points: OK")


def case_blacklist_preserves_stop_and_falls_back():
    # stop candidates survive; all non-stop blacklisted -> original list.
    cands = [_cand(1, (0.0, 0.0), stop=True), _cand(2, (5.0, 5.0))]
    out, n = er._filter_candidates_near_points(cands, [(5.0, 5.0)], 0.5)
    assert [c.candidate_id for c in out] == [1, 2], out
    assert n == 0, n
    print("  case_blacklist_preserves_stop_and_falls_back: OK")


def case_blacklist_empty_points_noop():
    cands = [_cand(1, (5.0, 5.0))]
    out, n = er._filter_candidates_near_points(cands, [], 0.5)
    assert out == cands and n == 0
    print("  case_blacklist_empty_points_noop: OK")


# ----------------------------------------------------------------------
# 3. runner-level (stubs from test_advance_subgoal.py)
# ----------------------------------------------------------------------


class _RR:
    selected = None
    debug_info = {"top_scores": []}


class _StubBridge:
    text_encode_fn = staticmethod(lambda s: np.zeros(8, dtype=np.float32))

    def __init__(self):
        self._pending: list = []

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

    def consolidate_subgoal_boundary(self, *a, **k):
        return {"fine": [], "mid": [], "coarse": []}

    def stats(self):
        return {}


class _BasePlanner:
    decision_period = 10

    def reset(self, agent_pos=None):
        pass

    def update(self, *a, **k):
        pass

    def is_decision_step(self):
        return False

    def step_controller(self, cand, pos, yaw):
        return ACTION_FORWARD

    def grid_stats(self):
        return {"cells_free": 0, "cells_occupied": 0,
                "cells_unknown": 0, "frontier_cells": 0}

    def controller_stats(self):
        return {"astar_path": 0, "astar_reachable_fallback": 0,
                "astar_fallback": 0, "collision_escape": 0,
                "replan_scheduled": 0, "replan_forced": 0, "replan_stuck": 0}


class _NearFarPlanner(_BasePlanner):
    """Pool always offers a near (already-reached) pick first plus a far one —
    the micro3 absorbing-loop geometry."""

    def propose(self, pos, yaw):
        return [
            er.FrontierCandidate(
                candidate_id=1, world_xy=np.array([0.1, 0.0], dtype=np.float32),
                grid_rc=(0, 0), distance_m=0.1, bearing_rad=0.0, cluster_size=1,
                raw_score=0.9, source="frontier", metadata={}),
            er.FrontierCandidate(
                candidate_id=2, world_xy=np.array([5.0, 5.0], dtype=np.float32),
                grid_rc=(0, 0), distance_m=7.1, bearing_rad=0.0, cluster_size=1,
                raw_score=0.5, source="frontier", metadata={}),
        ]


class _AllNearPlanner(_BasePlanner):
    """Pathological geometry: EVERY proposal is already reached — the filter
    must fall back (never waypoint-less) and the cooldown must bound the
    re-propose rate."""

    def propose(self, pos, yaw):
        return [er.FrontierCandidate(
            candidate_id=1, world_xy=np.array([0.1, 0.0], dtype=np.float32),
            grid_rc=(0, 0), distance_m=0.1, bearing_rad=0.0, cluster_size=1,
            raw_score=0.9, source="frontier", metadata={})]


class _StubSource:
    """Agent pinned at the origin; sub-goals never advance (distance stays
    far), so a multion episode runs to the step budget."""

    def __init__(self, seq=None):
        self.t = 0
        self.seq = seq

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
            info={"caption": "an empty hallway",
                  "text_embedding": np.zeros(8, dtype=np.float32),
                  "visual_embedding": np.zeros(8, dtype=np.float32),
                  "distance_to_goal": 5.0},
        )

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
        return 5.0

    def nearest_category_viewpoint(self, agent_pos, category):
        return 3.0, np.array([1.0, 0.0, 1.0], dtype=np.float32)


def _mk_runner(source, planner, cls=None):
    tmp = tempfile.mkdtemp(prefix="filter-near-test-")
    return (cls or er.EpisodeRunner)(
        source=source, planner=planner, bridge=_StubBridge(),
        clip_encoder=None, captioner=None, out_dir=tmp,
        target_category="chair", keyframe_every_m=1,
        max_steps_per_episode=20, backbone="frontier")


def case_multion_filters_near_candidates():
    runner = _mk_runner(_StubSource(seq=["chair", "bed", "toilet"]),
                        _NearFarPlanner())
    ep_log, metrics = runner._run_episode(0)
    # Every propose dropped the near pick, so the chosen waypoint is the far
    # one and the reached-thrash trigger never fires.
    assert ep_log["n_candidates_filtered_near"] > 0, ep_log
    assert ep_log["n_propose_reached"] == 0, ep_log
    for d in ep_log["decisions"]:
        assert all(c["id"] != 1 for c in d["candidates"]), d
    print("  case_multion_filters_near_candidates: OK")


def case_single_goal_unfiltered_byte_identity():
    # K=1: no filtering, no cooldown — the legacy reached-triggered re-propose
    # fires every tick exactly as before; the new counters are log-only.
    runner = _mk_runner(_StubSource(seq=None), _NearFarPlanner())
    ep_log, metrics = runner._run_episode(0)
    assert ep_log["n_candidates_filtered_near"] == 0, ep_log
    # near pick (raw top-1) keeps winning -> reached every tick -> ~n_steps
    # re-proposes (the thrash, deliberately preserved on the K=1 path).
    assert ep_log["n_propose_reached"] >= 15, ep_log["n_propose_reached"]
    assert ep_log["rerank_calls"] >= 16, ep_log["rerank_calls"]
    assert any(c["id"] == 1 for d in ep_log["decisions"]
               for c in d["candidates"])
    # Every decision records WHY it proposed (multion-full1 post-mortem: a
    # per-tick re-propose mode existed that no counter attributed).
    triggers = [d["trigger"] for d in ep_log["decisions"]]
    assert triggers[0] == "no_candidate", triggers[:3]
    assert triggers.count("reached") >= 15, triggers
    assert all(t in ("no_candidate", "scheduled", "reached")
               for t in triggers), triggers
    print("  case_single_goal_unfiltered_byte_identity: OK")


def case_multion_cooldown_bounds_repropose():
    # All-near pool: the filter falls back to the original list (waypoint-less
    # is never allowed), so "reached" fires every tick — the cooldown backstop
    # must bound the re-propose rate to ~n_steps/cooldown instead of ~n_steps.
    runner = _mk_runner(_StubSource(seq=["chair", "bed", "toilet"]),
                        _AllNearPlanner())
    ep_log, metrics = runner._run_episode(0)
    assert ep_log["n_steps"] == 19, ep_log["n_steps"]
    assert ep_log["n_candidates_filtered_near"] == 0, ep_log
    # 19 ticks / cooldown 3 -> at most ~7 reached-triggered re-proposes.
    assert 1 <= ep_log["n_propose_reached"] <= 7, ep_log["n_propose_reached"]
    assert ep_log["rerank_calls"] <= 10, ep_log["rerank_calls"]
    print("  case_multion_cooldown_bounds_repropose: OK")


# ----------------------------------------------------------------------
# 3b. runner-level: unreachable-waypoint blacklist (full1 third mode)
# ----------------------------------------------------------------------


class _UnreachableTopPlanner(_BasePlanner):
    """Pool always led by the same 'best' waypoint — which the follower will
    report unreachable — plus a reachable alternative. The full1 absorbing
    geometry: unreachable -> drop -> SAME cluster re-proposed -> forever."""

    def propose(self, pos, yaw):
        return [
            er.FrontierCandidate(
                candidate_id=1, world_xy=np.array([5.0, 5.0], dtype=np.float32),
                grid_rc=(0, 0), distance_m=7.1, bearing_rad=0.0, cluster_size=3,
                raw_score=0.9, source="frontier", metadata={}),
            er.FrontierCandidate(
                candidate_id=2, world_xy=np.array([8.0, 0.0], dtype=np.float32),
                grid_rc=(0, 0), distance_m=8.0, bearing_rad=0.0, cluster_size=1,
                raw_score=0.5, source="frontier", metadata={}),
        ]


class _UnreachableFollowerRunner(er.EpisodeRunner):
    """Waypoint (5,5) is navmesh-unreachable: the follower signals 'done'
    while the agent is still far (force_repropose + TURN, exactly what
    _waypoint_action does on a None/STOP from the real follower)."""

    def _waypoint_action(self, candidate, agent_pos, agent_yaw,
                         use_approach_follower=False):
        if (abs(float(candidate.world_xy[0]) - 5.0) < 1e-6
                and abs(float(candidate.world_xy[1]) - 5.0) < 1e-6):
            self._waypoint_force_repropose = True
            return er.ACTION_TURN_LEFT
        self._waypoint_force_repropose = False
        return ACTION_FORWARD


def case_multion_blacklists_unreachable_waypoint():
    runner = _mk_runner(_StubSource(seq=["chair", "bed", "toilet"]),
                        _UnreachableTopPlanner(),
                        cls=_UnreachableFollowerRunner)
    ep_log, metrics = runner._run_episode(0)
    # The bad waypoint is tried ONCE, blacklisted, and never re-chosen.
    assert ep_log["n_waypoint_unreachable"] == 1, ep_log["n_waypoint_unreachable"]
    assert ep_log["n_candidates_filtered_unreachable"] >= 1, ep_log
    assert ep_log["rerank_calls"] <= 6, ep_log["rerank_calls"]
    chosen_ids = [d["chosen_id"] for d in ep_log["decisions"]]
    assert chosen_ids[0] == 1 and all(c == 2 for c in chosen_ids[1:]), chosen_ids
    # Bookkeeping fix: the counters reach ep_metrics (summary rows showed 0
    # in multion-full1 while the episode JSONs carried 662 — misled the
    # post-mortem).
    assert metrics["n_waypoint_unreachable"] == ep_log["n_waypoint_unreachable"]
    assert metrics["n_waypoint_reached"] == ep_log["n_waypoint_reached"]
    assert metrics["n_forward_no_progress"] == ep_log["n_forward_no_progress"]
    print("  case_multion_blacklists_unreachable_waypoint: OK")


def case_single_goal_unreachable_loop_preserved():
    # K=1 byte-identity: no blacklist — the unreachable top pick is re-chosen
    # every tick exactly as before (the legacy loop, deliberately preserved).
    runner = _mk_runner(_StubSource(seq=None), _UnreachableTopPlanner(),
                        cls=_UnreachableFollowerRunner)
    ep_log, metrics = runner._run_episode(0)
    assert ep_log["n_waypoint_unreachable"] >= 15, ep_log["n_waypoint_unreachable"]
    assert ep_log["rerank_calls"] >= 16, ep_log["rerank_calls"]
    assert ep_log["n_candidates_filtered_unreachable"] == 0, ep_log
    print("  case_single_goal_unreachable_loop_preserved: OK")


def main() -> int:
    print("reached-thrash escape sanity tests")
    case_near_dropped_far_survives()
    case_near_stop_signal_preserved()
    case_all_near_returns_original_list()
    case_all_near_plus_stop_returns_original_list()
    case_boundary_distance_survives()
    case_agent_offset_uses_world_frame()
    case_cooldown_elapsed_pure()
    case_blacklist_point_dropped_far_survives()
    case_blacklist_multiple_points()
    case_blacklist_preserves_stop_and_falls_back()
    case_blacklist_empty_points_noop()
    case_multion_filters_near_candidates()
    case_single_goal_unfiltered_byte_identity()
    case_multion_cooldown_bounds_repropose()
    case_multion_blacklists_unreachable_waypoint()
    case_single_goal_unreachable_loop_preserved()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
