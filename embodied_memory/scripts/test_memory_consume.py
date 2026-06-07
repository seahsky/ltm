"""
Sanity tests for the multion memory-consumption fix (multion-full3
post-mortem: the LTM-specific S3 regression was driven by a NEW absorbing
mode the full3 escapes don't address — the wrong-instance recall attractor).

multion-full3 evidence:

* ep12 (catastrophic): a remembered "toilet" sighting ~0.5-0.8 m away; the
  follower instantly reports reached -> candidate dropped -> re-propose fires
  through the ``no_candidate`` trigger, WHICH THE REACHED-COOLDOWN DOES NOT
  GATE (n_propose_reached=0, rerank=949/1049) -> the same memory candidate
  wins again (mem_chosen=945) -> soft_spl 0, adv 0.
* ep10/11/13 (chronic): the distance-trigger variant — candidate_reached
  re-proposes ~53x/episode, the same memory waypoint re-chosen all episode,
  adv 0. The near-filter can't drop it (raw distance > 0.5 m) and the
  unreachable blacklist can't (outcome is "reached", not "unreachable").
* ep0/ep6: the snap escape fired 100-125x but looped — the snapped point is
  ALSO follower-unreachable, so the cycle restarts every SNAP_N ticks.

Three fixes (all multion-gated; K=1 byte-identical):

F1 — CONSUME a memory-source candidate once REACHED without an advance
     (both the follower-done outcome and the candidate_reached trigger):
     its position joins a per-sub-goal consumed list filtered from later
     pools; cleared on sub-goal advance (a new sub-goal may legitimately
     revisit). ``REMEMBR_CONSUME_REACHED_MEM=0`` disables. Counters
     ``n_memory_consumed`` / ``n_candidates_filtered_consumed``.
F2 — follower-done drops set a cooldown on the ``no_candidate`` re-propose
     path (the ep12 hole): at most one propose per REMEMBR_PROPOSE_COOLDOWN
     window after a follower-done drop. The sub-goal-advance recall moment
     still re-queries immediately.
F3 — snap-once: a waypoint is snapped at most once (``snap_retried``
     metadata mark); if the snapped point is also unreachable -> blacklist +
     drop instead of re-snapping forever. REMEMBR_UNREACHABLE_SNAP_N default
     8 -> 1 (snap on first failure; the full3 ep5 interleaved-reset pattern
     defeated the consecutive-count threshold).

Invoke with::

    python embodied_memory/scripts/test_memory_consume.py
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
# stub-and-load bootstrap (pattern of test_stuck_escape.py)
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


def _with_env(env, fn):
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
# stubs
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


class _MemRR:
    """Rerank result that selects the candidate at ``mem_xy`` by the stable
    'go to (x.x, y.y)' prefix _chosen_candidate_index matches on — so the
    MEMORY candidate wins whenever it is in the pool, and the index-0
    fallback (the frontier pick) wins when it has been filtered out."""

    def __init__(self, mem_xy):
        self.selected = SimpleNamespace(
            response=f"go to ({mem_xy[0]:.1f}, {mem_xy[1]:.1f})",
            final_score=0.9)
        self.debug_info = {"top_scores": []}


class _MemBridge(_StubBridge):
    """Always re-proposes the SAME remembered sighting (a fresh candidate
    object each call, like the real bridge re-querying the fine LTM) and
    reranks it to the top — the full3 wrong-instance recall attractor."""

    def __init__(self, mem_xy=(0.6, 0.0)):
        super().__init__()
        self.mem_xy = tuple(float(v) for v in mem_xy)

    def propose_memory_candidates(self, *a, **k):
        return [er.FrontierCandidate(
            candidate_id=999,
            world_xy=np.array(self.mem_xy, dtype=np.float32),
            grid_rc=(0, 0),
            distance_m=float(np.hypot(*self.mem_xy)),
            bearing_rad=0.0, cluster_size=1, raw_score=0.9,
            source="memory", metadata={})]

    def rerank(self, **k):
        return _MemRR(self.mem_xy), {}


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


class _FarFrontierPlanner(_BasePlanner):
    """One far frontier alternative — what the agent SHOULD fall back to
    once the bad recall is consumed."""

    def propose(self, pos, yaw):
        return [er.FrontierCandidate(
            candidate_id=1, world_xy=np.array([8.0, 0.0], dtype=np.float32),
            grid_rc=(0, 0), distance_m=8.0, bearing_rad=0.0, cluster_size=1,
            raw_score=0.5, source="frontier", metadata={})]


class _NearFrontierPlanner(_BasePlanner):
    """A frontier waypoint the follower instantly reports reached (0.6 m,
    inside goal_radius+slack) — isolates the F2 no_candidate cooldown hole
    WITHOUT memory in the pool (frontier picks are never consumed)."""

    def propose(self, pos, yaw):
        return [er.FrontierCandidate(
            candidate_id=1, world_xy=np.array([0.6, 0.0], dtype=np.float32),
            grid_rc=(0, 0), distance_m=0.6, bearing_rad=0.0, cluster_size=1,
            raw_score=0.9, source="frontier", metadata={})]


class _StubSource:
    """Agent pinned at the origin; sub-goals never advance."""

    def __init__(self, seq=None, caption="an empty hallway"):
        self.t = 0
        self.seq = seq
        self.caption = caption

    def _agent_pos(self):
        return np.zeros(3, dtype=np.float32)

    def _mk_step(self, action, done):
        return SimpleNamespace(
            step_idx=self.t,
            rgb=np.zeros((4, 4, 3), dtype=np.uint8),
            depth=np.ones((4, 4), dtype=np.float32),
            semantic=None,
            agent_state=SimpleNamespace(
                position=self._agent_pos(), rotation_yaw=0.0),
            action=action,
            reward=0.0,
            done=done,
            info={"caption": self.caption,
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


class _MovingSource(_StubSource):
    """FORWARD advances the agent +0.25 m along x — lets the committed
    candidate's distance fall below propose_reached_m WITHOUT the follower
    reporting done (the ep10/11/13 chronic variant)."""

    def __init__(self, seq=None, caption="an empty hallway", step_m=0.25):
        super().__init__(seq, caption)
        self.x = 0.0
        self.step_m = float(step_m)

    def _agent_pos(self):
        return np.array([self.x, 0.0, 0.0], dtype=np.float32)

    def reset(self, episode_idx):
        self.x = 0.0
        return super().reset(episode_idx)

    def step(self, action):
        if action == ACTION_FORWARD:
            self.x += self.step_m
        return super().step(action)


class _AdvanceOnChairSource(_StubSource):
    """The FIRST sub-goal (chair) is findable immediately (distance below
    found_radius + caption confirm); later categories never advance —
    exercises the consumed-list clear at the sub-goal boundary."""

    def __init__(self, seq=None):
        super().__init__(seq, caption="a chair in a hallway")

    def distance_to_category(self, agent_pos, category):
        return 0.5 if category == "chair" else 5.0


class _SameSnapSimSource(_StubSource):
    """Stub sim whose pathfinder snap is a NO-OP (returns the same
    unreachable point) — the full3 ep0/ep6 snap-loop geometry: snapping
    cannot help, so the escape must give up after one retry per waypoint."""

    def get_sim(self):
        return SimpleNamespace(pathfinder=SimpleNamespace(
            snap_point=lambda g: np.array([5.0, 0.0, 5.0], dtype=np.float32)))


class _OnlyUnreachablePlanner(_BasePlanner):
    def propose(self, pos, yaw):
        return [er.FrontierCandidate(
            candidate_id=1, world_xy=np.array([5.0, 5.0], dtype=np.float32),
            grid_rc=(0, 0), distance_m=7.1, bearing_rad=0.0, cluster_size=3,
            raw_score=0.9, source="frontier", metadata={})]


class _MemReachedRunner(er.EpisodeRunner):
    """Follower instantly reports done when steering toward the remembered
    waypoint (it is ~0.5-0.8 m away -> outcome 'reached'); any other
    waypoint steers fine (FORWARD)."""

    MEM_XY = (0.6, 0.0)

    def _waypoint_action(self, candidate, agent_pos, agent_yaw,
                         use_approach_follower=False):
        if (abs(float(candidate.world_xy[0]) - self.MEM_XY[0]) < 1e-6
                and abs(float(candidate.world_xy[1]) - self.MEM_XY[1]) < 1e-6):
            self._waypoint_force_repropose = True
            return er.ACTION_TURN_LEFT
        self._waypoint_force_repropose = False
        return ACTION_FORWARD


class _AlwaysForwardRunner(er.EpisodeRunner):
    def _waypoint_action(self, candidate, agent_pos, agent_yaw,
                         use_approach_follower=False):
        self._waypoint_force_repropose = False
        return ACTION_FORWARD


class _UnreachableAt55Runner(er.EpisodeRunner):
    def _waypoint_action(self, candidate, agent_pos, agent_yaw,
                         use_approach_follower=False):
        if (abs(float(candidate.world_xy[0]) - 5.0) < 1e-6
                and abs(float(candidate.world_xy[1]) - 5.0) < 1e-6):
            self._waypoint_force_repropose = True
            return er.ACTION_TURN_LEFT
        self._waypoint_force_repropose = False
        return ACTION_FORWARD


def _mk_runner(source, planner, bridge=None, cls=None):
    tmp = tempfile.mkdtemp(prefix="mem-consume-test-")
    return (cls or er.EpisodeRunner)(
        source=source, planner=planner, bridge=(bridge or _StubBridge()),
        clip_encoder=None, captioner=None, out_dir=tmp,
        target_category="chair", keyframe_every_m=1,
        max_steps_per_episode=20, backbone="frontier")


def _chosen_sources(ep_log):
    return [d["chosen_source"] for d in ep_log["decisions"]]


# ----------------------------------------------------------------------
# F1 — consume on reached-without-advance
# ----------------------------------------------------------------------


def case_follower_reached_memory_consumed():
    # full3 ep12: follower-done memory thrash. The recall must be consumed
    # after ONE visit; the next pool filters it and the frontier pick wins.
    runner = _mk_runner(_StubSource(seq=["chair", "bed", "toilet"]),
                        _FarFrontierPlanner(), bridge=_MemBridge((0.6, 0.0)),
                        cls=_MemReachedRunner)
    ep_log, metrics = runner._run_episode(0)
    assert ep_log["n_memory_consumed"] >= 1, ep_log["n_memory_consumed"]
    assert ep_log["n_candidates_filtered_consumed"] >= 1, ep_log
    sources = _chosen_sources(ep_log)
    assert sources.count("memory") == 1, sources
    assert "frontier" in sources, sources
    # ep12 burned 949/1049 reranks; consumed + cooldown-gated must collapse it.
    assert ep_log["rerank_calls"] <= 6, ep_log["rerank_calls"]
    # bookkeeping: counters reach ep_metrics (summary rows)
    assert metrics["n_memory_consumed"] == ep_log["n_memory_consumed"]
    assert (metrics["n_candidates_filtered_consumed"]
            == ep_log["n_candidates_filtered_consumed"])
    print("  case_follower_reached_memory_consumed: OK")


def case_distance_trigger_memory_consumed():
    # full3 ep10/11/13: the chronic variant — the agent WALKS within the
    # 0.5 m re-propose radius (no follower-done), candidate_reached fires,
    # and the same recall is re-chosen all episode. Consume on that trigger
    # too: the memory candidate is chosen exactly once.
    runner = _mk_runner(_MovingSource(seq=["chair", "bed", "toilet"]),
                        _FarFrontierPlanner(), bridge=_MemBridge((0.7, 0.0)),
                        cls=_AlwaysForwardRunner)
    ep_log, metrics = runner._run_episode(0)
    assert ep_log["n_memory_consumed"] >= 1, ep_log["n_memory_consumed"]
    sources = _chosen_sources(ep_log)
    assert sources.count("memory") == 1, sources
    # Later proposes (agent far past the recall) must filter it from the
    # pool via the consumed list, not the near-filter.
    assert ep_log["n_candidates_filtered_consumed"] >= 1, ep_log
    print("  case_distance_trigger_memory_consumed: OK")


def case_consume_disabled_keeps_legacy_thrash():
    # REMEMBR_CONSUME_REACHED_MEM=0: the recall attractor is back (memory
    # re-chosen repeatedly, bounded only by the F2 cooldown) and the new
    # counters stay 0.
    def run():
        runner = _mk_runner(_StubSource(seq=["chair", "bed", "toilet"]),
                            _FarFrontierPlanner(),
                            bridge=_MemBridge((0.6, 0.0)),
                            cls=_MemReachedRunner)
        return runner._run_episode(0)

    ep_log, metrics = _with_env({"REMEMBR_CONSUME_REACHED_MEM": 0}, run)
    assert ep_log["n_memory_consumed"] == 0, ep_log["n_memory_consumed"]
    assert ep_log["n_candidates_filtered_consumed"] == 0, ep_log
    sources = _chosen_sources(ep_log)
    assert sources.count("memory") >= 3, sources
    print("  case_consume_disabled_keeps_legacy_thrash: OK")


def case_consumed_cleared_on_subgoal_advance():
    # The consumed list is PER SUB-GOAL: after the chair advance the same
    # remembered position is offered again (a new sub-goal may legitimately
    # want that area) — memory is chosen again post-advance.
    runner = _mk_runner(_AdvanceOnChairSource(seq=["chair", "bed", "toilet"]),
                        _FarFrontierPlanner(), bridge=_MemBridge((0.6, 0.0)),
                        cls=_MemReachedRunner)
    ep_log, metrics = runner._run_episode(0)
    assert len(ep_log["subgoals_found"]) >= 1, ep_log["subgoals_found"]
    sources = _chosen_sources(ep_log)
    # chosen for the chair hunt AND again after the advance (list cleared)
    assert sources.count("memory") >= 2, sources
    assert ep_log["n_memory_consumed"] >= 2, ep_log["n_memory_consumed"]
    print("  case_consumed_cleared_on_subgoal_advance: OK")


# ----------------------------------------------------------------------
# F2 — follower-done drop cooldown (the ep12 no_candidate hole)
# ----------------------------------------------------------------------


def case_follower_drop_repropose_cooldown():
    # A frontier waypoint the follower instantly reports reached: drop ->
    # no_candidate -> re-propose fired EVERY TICK in full3 (rerank=949/1049,
    # n_propose_reached=0 — the cooldown never saw it). The drop must now
    # set the same cooldown: ~n/cooldown proposes, not ~n.
    # _MemReachedRunner's MEM_XY (0.6, 0.0) matches the near frontier pick,
    # so the follower instantly reports done on it — no memory in the pool.
    runner = _mk_runner(_StubSource(seq=["chair", "bed", "toilet"]),
                        _NearFrontierPlanner(), cls=_MemReachedRunner)
    ep_log, metrics = runner._run_episode(0)
    assert ep_log["n_steps"] == 19, ep_log["n_steps"]
    # 19 ticks / cooldown 3 -> at most ~8 proposes (legacy: ~19).
    assert 3 <= ep_log["rerank_calls"] <= 8, ep_log["rerank_calls"]
    assert ep_log["n_memory_consumed"] == 0, ep_log  # frontier never consumed
    print("  case_follower_drop_repropose_cooldown: OK")


def case_single_goal_byte_identity():
    # K=1: no consumption, no drop-cooldown — the legacy per-tick re-propose
    # thrash is deliberately preserved and the new counters are log-only 0s.
    runner = _mk_runner(_StubSource(seq=None), _FarFrontierPlanner(),
                        bridge=_MemBridge((0.6, 0.0)), cls=_MemReachedRunner)
    ep_log, metrics = runner._run_episode(0)
    assert ep_log["n_memory_consumed"] == 0, ep_log["n_memory_consumed"]
    assert ep_log["n_candidates_filtered_consumed"] == 0, ep_log
    assert ep_log["rerank_calls"] >= 16, ep_log["rerank_calls"]
    sources = _chosen_sources(ep_log)
    assert sources.count("memory") >= 15, sources
    assert metrics["n_memory_consumed"] == 0
    assert metrics["n_candidates_filtered_consumed"] == 0
    print("  case_single_goal_byte_identity: OK")


# ----------------------------------------------------------------------
# F3 — snap-once (the ep0/ep6 snap-loop)
# ----------------------------------------------------------------------


def case_snap_once_then_blacklist():
    # Snap is a no-op (same unreachable point back): the escape must mark the
    # waypoint snap_retried and give up on the SECOND failure (blacklist +
    # drop) instead of re-snapping every SNAP_N ticks forever (full3 ep0:
    # escape=125 with wp_unreach=1023 — each snap bought 8 more turns).
    runner = _mk_runner(_SameSnapSimSource(seq=["chair", "bed", "toilet"]),
                        _OnlyUnreachablePlanner(), cls=_UnreachableAt55Runner)
    ep_log, metrics = runner._run_episode(0)
    # Default SNAP_N is now 1: snap fires on the FIRST unreachable of each
    # committed waypoint, and at most once per waypoint.
    assert ep_log["n_unreachable_escape"] >= 2, ep_log["n_unreachable_escape"]
    # Per cycle: 1 unreachable -> snap, 1 unreachable -> drop, cooldown gap.
    # Without snap-once the same 19-tick run shows ~19 unreachables.
    assert ep_log["n_waypoint_unreachable"] <= 2 * ep_log["n_unreachable_escape"] + 2, \
        (ep_log["n_waypoint_unreachable"], ep_log["n_unreachable_escape"])
    print("  case_snap_once_then_blacklist: OK")


def main() -> int:
    print("memory-consumption sanity tests")
    case_follower_reached_memory_consumed()
    case_distance_trigger_memory_consumed()
    case_consume_disabled_keeps_legacy_thrash()
    case_consumed_cleared_on_subgoal_advance()
    case_follower_drop_repropose_cooldown()
    case_single_goal_byte_identity()
    case_snap_once_then_blacklist()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
