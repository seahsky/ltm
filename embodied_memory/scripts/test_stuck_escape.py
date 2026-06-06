"""
Sanity tests for the multion stuck-loop escapes (multion-full2 post-mortem:
two of eight S3 episodes burned the entire 750-step budget in diagnosed
absorbing loops the existing counters saw but nothing acted on).

Pathology A (ep5, turn-in-place): the chosen frontier waypoint is
navmesh-unreachable (wp_unreachable=724, 742/749 turns). The 4fa6679
blacklist fires but (a) the never-empty fallback in
``_filter_candidates_near_points`` returned the RAW unfiltered pool when all
non-stop candidates would drop, re-admitting the bad candidate, and (b) the
agent never moves, so nothing ever breaks the cycle.

Pathology B (ep4, forward-into-wall): n_forward_no_progress=656 with
collision_escape=0 — the grid-era escape needs two CONSECUTIVE forwards but
the stuck loop alternates FORWARD/TURN, and ``_forward_no_progress`` was
computed every tick yet only counted, never acted on.

Pure helpers + runner-level behavior (habitat/torch-free stub-and-load
bootstrap, pattern of ``test_filter_near_candidates.py``):

1. ``_farthest_from_points`` — the non-stop candidate FARTHEST from the
   filter points (empty/all-stop -> None).
2. ``_filter_candidates_near_points(..., prefer_farthest=True)`` — when all
   non-stop candidates would drop, return the single farthest non-stop
   candidate (+ preserved stop candidates) instead of the raw pool; the
   default ``prefer_farthest=False`` raw-pool fallback is unchanged
   (regression guard — the reached-thrash near-filter call site keeps it).
3. ``_should_snap_unreachable`` — ``>=`` threshold, 0 disables.
4. ``_no_progress_escape`` — full window with >= M no-progress forwards.
5. Runner-level: the snap escape breaks the unreachable turn-forever loop
   (forwards resume); the no-progress escape blacklists the wall waypoint
   and re-proposes (far candidate chosen next); the two escapes compose;
   single-goal (non-multion) behavior is byte-identical (both counters 0);
   counter bookkeeping (ep_metrics == ep_log).

Invoke with::

    python embodied_memory/scripts/test_stuck_escape.py
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# ----------------------------------------------------------------------
# stub-and-load bootstrap (pattern of test_filter_near_candidates.py)
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
ACTION_STOP = er.ACTION_STOP
ACTION_FORWARD = er.ACTION_FORWARD


def _cand(cid, xy, stop=False, source="frontier"):
    return er.FrontierCandidate(
        candidate_id=cid, world_xy=np.array(xy, dtype=np.float32),
        grid_rc=(0, 0),
        distance_m=float(np.hypot(xy[0], xy[1])),
        bearing_rad=0.0, cluster_size=1, raw_score=0.5, source=source,
        metadata=({"stop_signal": True} if stop else {}))


def _with_env(env, fn):
    """Run ``fn`` with env overrides applied; restore afterwards (the runner
    reads the escape knobs from os.environ at construction)."""
    old = {k: os.environ.get(k) for k in env}
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        return fn()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ----------------------------------------------------------------------
# 1. pure: _farthest_from_points
# ----------------------------------------------------------------------


def case_farthest_picks_farthest():
    cands = [_cand(1, (1.0, 0.0)), _cand(2, (3.0, 0.0))]
    out = er._farthest_from_points(cands, [(0.0, 0.0)])
    assert out is not None and out.candidate_id == 2, out
    print("  case_farthest_picks_farthest: OK")


def case_farthest_uses_min_distance_to_any_point():
    # Distance to the SET of points is the min over points: c1 sits right on
    # the second point, c2 is 1 m from both -> c2 is farther from the set.
    cands = [_cand(1, (4.0, 0.0)), _cand(2, (1.0, 0.0))]
    out = er._farthest_from_points(cands, [(0.0, 0.0), (4.0, 0.0)])
    assert out is not None and out.candidate_id == 2, out
    print("  case_farthest_uses_min_distance_to_any_point: OK")


def case_farthest_skips_stop_and_empty():
    # stop candidates are never navigation targets; empty/all-stop -> None.
    assert er._farthest_from_points([], [(0.0, 0.0)]) is None
    only_stop = [_cand(1, (9.0, 9.0), stop=True)]
    assert er._farthest_from_points(only_stop, [(0.0, 0.0)]) is None
    mixed = [_cand(1, (9.0, 9.0), stop=True), _cand(2, (1.0, 0.0))]
    out = er._farthest_from_points(mixed, [(0.0, 0.0)])
    assert out is not None and out.candidate_id == 2, out
    print("  case_farthest_skips_stop_and_empty: OK")


# ----------------------------------------------------------------------
# 2. pure: _filter_candidates_near_points prefer_farthest fallback
# ----------------------------------------------------------------------


def case_prefer_farthest_fallback_returns_farthest_not_raw():
    # All non-stop candidates within the blacklist radius: the old fallback
    # returned the RAW pool (re-admitting the bad candidate -> full2 ep5
    # turn-in-place); prefer_farthest returns the least-bad single candidate.
    cands = [_cand(1, (0.0, 0.0), stop=True),
             _cand(2, (5.0, 5.0)), _cand(3, (5.3, 5.0))]
    out, n = er._filter_candidates_near_points(
        cands, [(5.1, 5.0)], 0.5, prefer_farthest=True)
    assert [c.candidate_id for c in out] == [1, 3], out
    assert n == 1, n
    print("  case_prefer_farthest_fallback_returns_farthest_not_raw: OK")


def case_prefer_farthest_partial_filter_unchanged():
    # When at least one non-stop candidate survives, prefer_farthest is
    # irrelevant — normal filtering applies.
    cands = [_cand(1, (5.0, 5.0)), _cand(2, (8.0, 0.0))]
    out, n = er._filter_candidates_near_points(
        cands, [(5.2, 5.0)], 0.5, prefer_farthest=True)
    assert [c.candidate_id for c in out] == [2], out
    assert n == 1, n
    print("  case_prefer_farthest_partial_filter_unchanged: OK")


def case_default_raw_pool_fallback_unchanged():
    # Regression guard: the near-filter call site keeps the raw-pool
    # fallback — the default (prefer_farthest omitted) must not change.
    cands = [_cand(1, (5.0, 5.0)), _cand(2, (5.3, 5.0))]
    out, n = er._filter_candidates_near_points(cands, [(5.1, 5.0)], 0.5)
    assert [c.candidate_id for c in out] == [1, 2], out
    assert n == 0, n
    print("  case_default_raw_pool_fallback_unchanged: OK")


# ----------------------------------------------------------------------
# 3. pure: _should_snap_unreachable
# ----------------------------------------------------------------------


def case_should_snap_unreachable_boundary():
    assert er._should_snap_unreachable(8, 8) is True   # >= boundary
    assert er._should_snap_unreachable(9, 8) is True
    assert er._should_snap_unreachable(7, 8) is False
    assert er._should_snap_unreachable(100, 0) is False  # 0 disables
    print("  case_should_snap_unreachable_boundary: OK")


# ----------------------------------------------------------------------
# 4. pure: _no_progress_escape
# ----------------------------------------------------------------------


def case_no_progress_escape_pure():
    # fires at min events with a full window
    assert er._no_progress_escape([True] * 12 + [False] * 8, 12, 20) is True
    # below min: no fire
    assert er._no_progress_escape([True] * 11 + [False] * 9, 12, 20) is False
    # partial window: no fire even if every entry is a no-progress forward
    assert er._no_progress_escape([True] * 12, 12, 20) is False
    # 0 disables
    assert er._no_progress_escape([True] * 20, 0, 20) is False
    print("  case_no_progress_escape_pure: OK")


# ----------------------------------------------------------------------
# 5. runner-level (stubs from test_filter_near_candidates.py)
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


class _SnapSimSource(_StubSource):
    """Exposes a stub sim whose pathfinder snaps any query to (4, 0, 4) —
    the navmesh point the snap escape should re-commit toward."""

    def get_sim(self):
        return SimpleNamespace(pathfinder=SimpleNamespace(
            snap_point=lambda g: np.array([4.0, 0.0, 4.0], dtype=np.float32)))


def _mk_runner(source, planner, cls=None):
    tmp = tempfile.mkdtemp(prefix="stuck-escape-test-")
    return (cls or er.EpisodeRunner)(
        source=source, planner=planner, bridge=_StubBridge(),
        clip_encoder=None, captioner=None, out_dir=tmp,
        target_category="chair", keyframe_every_m=1,
        max_steps_per_episode=20, backbone="frontier")


class _OnlyUnreachablePlanner(_BasePlanner):
    """The full2 ep5 geometry: the pool ONLY ever offers the unreachable
    cluster — the blacklist alone can never escape (the never-empty fallback
    re-admits it every propose)."""

    def propose(self, pos, yaw):
        return [er.FrontierCandidate(
            candidate_id=1, world_xy=np.array([5.0, 5.0], dtype=np.float32),
            grid_rc=(0, 0), distance_m=7.1, bearing_rad=0.0, cluster_size=3,
            raw_score=0.9, source="frontier", metadata={})]


class _UnreachableAt55Runner(er.EpisodeRunner):
    """Waypoint (5,5) is navmesh-unreachable (follower 'done' while far ->
    force_repropose + TURN, what _waypoint_action does on None/STOP); any
    OTHER waypoint — e.g. the snapped (4,4) — steers fine (FORWARD)."""

    def _waypoint_action(self, candidate, agent_pos, agent_yaw,
                         use_approach_follower=False):
        if (abs(float(candidate.world_xy[0]) - 5.0) < 1e-6
                and abs(float(candidate.world_xy[1]) - 5.0) < 1e-6):
            self._waypoint_force_repropose = True
            return er.ACTION_TURN_LEFT
        self._waypoint_force_repropose = False
        return ACTION_FORWARD


def case_multion_snap_escape_breaks_unreachable_loop():
    def run():
        runner = _mk_runner(_SnapSimSource(seq=["chair", "bed", "toilet"]),
                            _OnlyUnreachablePlanner(),
                            cls=_UnreachableAt55Runner)
        return runner._run_episode(0)

    ep_log, metrics = _with_env({"REMEMBR_UNREACHABLE_SNAP_N": 4}, run)
    # The loop runs exactly SNAP_N unreachables, then the snap escape
    # re-commits the waypoint at the navmesh-snapped point and FORWARDs
    # resume — full2 ep5 had 724 unreachables and 742/749 turns.
    assert ep_log["n_unreachable_escape"] >= 1, ep_log["n_unreachable_escape"]
    assert ep_log["n_waypoint_unreachable"] < 15, ep_log["n_waypoint_unreachable"]
    assert ep_log["action_forward"] >= 5, ep_log["action_forward"]
    # bookkeeping: counters reach ep_metrics (summary rows)
    assert metrics["n_unreachable_escape"] == ep_log["n_unreachable_escape"]
    assert metrics["n_no_progress_escape"] == ep_log["n_no_progress_escape"]
    print("  case_multion_snap_escape_breaks_unreachable_loop: OK")


def case_multion_snap_disabled_keeps_blacklist_drop():
    # REMEMBR_UNREACHABLE_SNAP_N=0 disables the snap: the unreachable loop is
    # back to blacklist+drop (the pre-fix behavior) — counter stays 0.
    def run():
        runner = _mk_runner(_SnapSimSource(seq=["chair", "bed", "toilet"]),
                            _OnlyUnreachablePlanner(),
                            cls=_UnreachableAt55Runner)
        return runner._run_episode(0)

    ep_log, metrics = _with_env({"REMEMBR_UNREACHABLE_SNAP_N": 0}, run)
    assert ep_log["n_unreachable_escape"] == 0, ep_log["n_unreachable_escape"]
    assert ep_log["n_waypoint_unreachable"] >= 15, ep_log["n_waypoint_unreachable"]
    print("  case_multion_snap_disabled_keeps_blacklist_drop: OK")


class _WallFarPlanner(_BasePlanner):
    """The full2 ep4 geometry: the top-scored pick drives the agent into a
    wall (forward forever, zero displacement); a far alternative exists."""

    def propose(self, pos, yaw):
        return [
            er.FrontierCandidate(
                candidate_id=1, world_xy=np.array([2.0, 0.0], dtype=np.float32),
                grid_rc=(0, 0), distance_m=2.0, bearing_rad=0.0, cluster_size=2,
                raw_score=0.9, source="frontier", metadata={}),
            er.FrontierCandidate(
                candidate_id=2, world_xy=np.array([8.0, 0.0], dtype=np.float32),
                grid_rc=(0, 0), distance_m=8.0, bearing_rad=0.0, cluster_size=1,
                raw_score=0.5, source="frontier", metadata={}),
        ]


class _AlwaysForwardRunner(er.EpisodeRunner):
    """The follower never reports done — it just keeps emitting FORWARD while
    the (pinned) agent slides against the wall: collision_escape never fires
    (needs consecutive forwards through the grid controller, which is not in
    this path) and _forward_no_progress only counts."""

    def _waypoint_action(self, candidate, agent_pos, agent_yaw,
                         use_approach_follower=False):
        self._waypoint_force_repropose = False
        return ACTION_FORWARD


def case_multion_no_progress_escape_blacklists_and_reproposes():
    def run():
        runner = _mk_runner(_StubSource(seq=["chair", "bed", "toilet"]),
                            _WallFarPlanner(), cls=_AlwaysForwardRunner)
        return runner._run_episode(0)

    ep_log, metrics = _with_env(
        {"REMEMBR_NO_PROGRESS_WINDOW": 5, "REMEMBR_NO_PROGRESS_MIN": 5}, run)
    assert ep_log["n_no_progress_escape"] >= 1, ep_log["n_no_progress_escape"]
    # The wall waypoint lands on the unreachable blacklist, so the next
    # proposal pool drops it and the FAR candidate is chosen.
    assert ep_log["n_candidates_filtered_unreachable"] >= 1, ep_log
    chosen_ids = [d["chosen_id"] for d in ep_log["decisions"]]
    assert chosen_ids[0] == 1, chosen_ids
    assert 2 in chosen_ids, chosen_ids
    # bookkeeping
    assert metrics["n_no_progress_escape"] == ep_log["n_no_progress_escape"]
    assert metrics["n_unreachable_escape"] == ep_log["n_unreachable_escape"]
    print("  case_multion_no_progress_escape_blacklists_and_reproposes: OK")


def case_multion_no_progress_disabled_keeps_counting_only():
    # REMEMBR_NO_PROGRESS_MIN=0 disables the escape: forward-into-wall is
    # back to count-only (the pre-fix behavior).
    def run():
        runner = _mk_runner(_StubSource(seq=["chair", "bed", "toilet"]),
                            _WallFarPlanner(), cls=_AlwaysForwardRunner)
        return runner._run_episode(0)

    ep_log, metrics = _with_env(
        {"REMEMBR_NO_PROGRESS_WINDOW": 5, "REMEMBR_NO_PROGRESS_MIN": 0}, run)
    assert ep_log["n_no_progress_escape"] == 0, ep_log["n_no_progress_escape"]
    assert ep_log["n_forward_no_progress"] >= 15, ep_log["n_forward_no_progress"]
    print("  case_multion_no_progress_disabled_keeps_counting_only: OK")


class _UnreachableAndWallPlanner(_BasePlanner):
    """Both pathologies in one pool: the top pick (5,5) is unreachable, the
    alternative (2,0) is a wall the agent forwards into without progress."""

    def propose(self, pos, yaw):
        return [
            er.FrontierCandidate(
                candidate_id=1, world_xy=np.array([5.0, 5.0], dtype=np.float32),
                grid_rc=(0, 0), distance_m=7.1, bearing_rad=0.0, cluster_size=3,
                raw_score=0.9, source="frontier", metadata={}),
            er.FrontierCandidate(
                candidate_id=2, world_xy=np.array([2.0, 0.0], dtype=np.float32),
                grid_rc=(0, 0), distance_m=2.0, bearing_rad=0.0, cluster_size=1,
                raw_score=0.5, source="frontier", metadata={}),
        ]


def case_multion_escapes_compose():
    # unreachable top pick -> blacklist -> wall alternative -> no-progress
    # escape -> blacklist -> prefer_farthest re-admits the unreachable pick
    # -> consecutive unreachables -> snap escape. Both counters fire.
    def run():
        runner = _mk_runner(_SnapSimSource(seq=["chair", "bed", "toilet"]),
                            _UnreachableAndWallPlanner(),
                            cls=_UnreachableAt55Runner)
        return runner._run_episode(0)

    ep_log, metrics = _with_env(
        {"REMEMBR_UNREACHABLE_SNAP_N": 3,
         "REMEMBR_NO_PROGRESS_WINDOW": 5,
         "REMEMBR_NO_PROGRESS_MIN": 5}, run)
    assert ep_log["n_no_progress_escape"] >= 1, ep_log["n_no_progress_escape"]
    assert ep_log["n_unreachable_escape"] >= 1, ep_log["n_unreachable_escape"]
    print("  case_multion_escapes_compose: OK")


def case_single_goal_byte_identity():
    # K=1: neither escape may act — the unreachable loop and the wall-forward
    # loop are the (deliberately preserved) legacy behaviors, and the new
    # counters are log-only zeros.
    env = {"REMEMBR_UNREACHABLE_SNAP_N": 4,
           "REMEMBR_NO_PROGRESS_WINDOW": 5,
           "REMEMBR_NO_PROGRESS_MIN": 5}

    def run_unreachable():
        runner = _mk_runner(_SnapSimSource(seq=None),
                            _OnlyUnreachablePlanner(),
                            cls=_UnreachableAt55Runner)
        return runner._run_episode(0)

    ep_log, metrics = _with_env(env, run_unreachable)
    assert ep_log["n_unreachable_escape"] == 0, ep_log["n_unreachable_escape"]
    assert ep_log["n_no_progress_escape"] == 0, ep_log["n_no_progress_escape"]
    assert ep_log["n_waypoint_unreachable"] >= 15, ep_log["n_waypoint_unreachable"]
    assert metrics["n_unreachable_escape"] == 0
    assert metrics["n_no_progress_escape"] == 0

    def run_wall():
        runner = _mk_runner(_StubSource(seq=None), _WallFarPlanner(),
                            cls=_AlwaysForwardRunner)
        return runner._run_episode(0)

    ep_log, metrics = _with_env(env, run_wall)
    assert ep_log["n_no_progress_escape"] == 0, ep_log["n_no_progress_escape"]
    assert ep_log["n_forward_no_progress"] >= 15, ep_log["n_forward_no_progress"]
    assert ep_log["n_candidates_filtered_unreachable"] == 0, ep_log
    print("  case_single_goal_byte_identity: OK")


def main() -> int:
    print("stuck-loop escape sanity tests")
    case_farthest_picks_farthest()
    case_farthest_uses_min_distance_to_any_point()
    case_farthest_skips_stop_and_empty()
    case_prefer_farthest_fallback_returns_farthest_not_raw()
    case_prefer_farthest_partial_filter_unchanged()
    case_default_raw_pool_fallback_unchanged()
    case_should_snap_unreachable_boundary()
    case_no_progress_escape_pure()
    case_multion_snap_escape_breaks_unreachable_loop()
    case_multion_snap_disabled_keeps_blacklist_drop()
    case_multion_no_progress_escape_blacklists_and_reproposes()
    case_multion_no_progress_disabled_keeps_counting_only()
    case_multion_escapes_compose()
    case_single_goal_byte_identity()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
