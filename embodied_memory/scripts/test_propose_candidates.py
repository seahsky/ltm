"""
Module-level sanity test for ``EpisodeRunner._propose_candidates`` (Run 4).

Run-4 added frontier-planner injection on top of the LLM-only ReMEmbR
proposal pool. This test exercises the three cases described in the
Phase-2 Run-4 plan without standing up the full Habitat / model stack:

  (a) STOP short-circuit preserved when the LLM returns a stop_signal
      candidate as element 0 — frontier candidates must NOT be appended.
  (b) Frontier candidates appear in the output when the LLM returns
      non-STOP candidates.
  (c) De-dup drops frontier candidates within REMEMBR_MIN_WAYPOINT_DIST
      of any LLM candidate.

The test sidesteps the ``embodied_memory`` package ``__init__`` (which
imports ``faiss`` via ``dialogue_memory``) by stubbing the heavy
submodules in ``sys.modules`` first, then loading
``frontier_planner.py`` and ``episode_runner.py`` directly through
``importlib.util.spec_from_file_location``. Same pattern as the prior
collision-escape sanity test (commit ``117028d``).

Invoke with::

    python embodied_memory/scripts/test_propose_candidates.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np


# ----------------------------------------------------------------------
# stub-and-load: bring up just enough of `embodied_memory` to evaluate
# `EpisodeRunner._propose_candidates` without touching faiss/transformers.
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
        # Lightweight placeholder classes; episode_runner only references
        # these names as type hints, never instantiates them.
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
    # Heavy submodules — placeholders only.
    _stub_submodule("embodied_memory.episode_source",
                    ["Episode", "EpisodeSource", "Step"])
    _stub_submodule("embodied_memory.memory_bridge",
                    ["EmbodiedMemoryBridge"])
    _stub_submodule("embodied_memory.perception",
                    ["CLIPKeyframeEncoder", "Keyframe", "SemanticCaptioner"])
    _stub_submodule("embodied_memory.remembr_backbone",
                    ["ReMEmbRBuilder", "ReMEmbRPlanner"])
    # habitat_env stub: only its _ACTION_NAMES list is needed (by the oracle
    # action map); the real module imports AgentState which the episode_source
    # stub doesn't provide, so we can't load it here.
    hab = types.ModuleType("embodied_memory.habitat_env")
    hab._ACTION_NAMES = [
        "stop", "move_forward", "turn_left", "turn_right", "look_up", "look_down",
    ]
    sys.modules["embodied_memory.habitat_env"] = hab
    # Real frontier_planner — pure-Python, no heavy deps.
    fp = _load_file_as("embodied_memory.frontier_planner",
                       _EMB_DIR / "frontier_planner.py")
    # Real episode_runner — imports the above by name.
    er = _load_file_as("embodied_memory.episode_runner",
                       _EMB_DIR / "episode_runner.py")
    return er.EpisodeRunner, fp.FrontierCandidate


EpisodeRunner, FrontierCandidate = _bootstrap()


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _make_cand(
    cid: int,
    x: float,
    z: float,
    source: str,
    *,
    stop_signal: bool = False,
) -> "FrontierCandidate":
    meta = {"stop_signal": True} if stop_signal else {}
    return FrontierCandidate(
        candidate_id=cid,
        world_xy=np.array([x, z], dtype=np.float32),
        grid_rc=(-1, -1),
        distance_m=0.0,
        bearing_rad=0.0,
        cluster_size=0,
        raw_score=0.5,
        source=source,
        metadata=meta,
    )


def _make_runner(n_frontier_inject: int):
    r = EpisodeRunner.__new__(EpisodeRunner)
    r.backbone = "remembr"
    r.target_category = "chair"
    r.n_frontier_inject = n_frontier_inject
    return r


def _make_step():
    agent_state = SimpleNamespace(
        position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        rotation_yaw=0.0,
    )
    return SimpleNamespace(agent_state=agent_state, step_idx=42)


def _make_ep():
    return SimpleNamespace(target_category="chair")


# ----------------------------------------------------------------------
# cases
# ----------------------------------------------------------------------


def case_a_stop_short_circuit():
    """(a) STOP short-circuit: LLM returns stop_signal first → frontier
    injection must be skipped, output equals LLM output."""
    r = _make_runner(n_frontier_inject=3)
    stop = _make_cand(99000, 0.0, 0.0, source="stop", stop_signal=True)
    r.remembr_planner = SimpleNamespace(propose=lambda **kw: [stop])
    sentinel = []
    def _no_call(*a, **kw):
        sentinel.append(("called", a, kw))
        return []
    r.planner = SimpleNamespace(propose_diverse=_no_call)
    out = r._propose_candidates(_make_step(), _make_ep())
    assert out == [stop], f"expected [stop], got {out}"
    assert not sentinel, f"frontier planner must not be called on STOP, got {sentinel}"
    print("  case (a) STOP short-circuit: OK")


def case_b_frontier_injected():
    """(b) Frontier candidates appear when LLM returns non-STOP."""
    r = _make_runner(n_frontier_inject=3)
    llm = [
        _make_cand(1, 5.0, 0.0, source="remembr"),
        _make_cand(2, -5.0, 0.0, source="remembr"),
    ]
    frontier = [
        _make_cand(10, 0.0, 3.0, source="planner"),
        _make_cand(11, 0.0, -3.0, source="planner"),
        _make_cand(12, 3.0, 3.0, source="planner"),
    ]
    r.remembr_planner = SimpleNamespace(propose=lambda **kw: llm)
    r.planner = SimpleNamespace(propose_diverse=lambda *a, **kw: frontier)
    out = r._propose_candidates(_make_step(), _make_ep())
    assert len(out) == len(llm) + len(frontier), \
        f"expected {len(llm) + len(frontier)} cands, got {len(out)}"
    n_frontier = sum(1 for c in out if c.source == "frontier")
    n_remembr = sum(1 for c in out if c.source == "remembr")
    assert n_frontier == 3, f"expected 3 frontier-tagged cands, got {n_frontier}"
    assert n_remembr == 2, f"expected 2 remembr cands, got {n_remembr}"
    print("  case (b) frontier injected (no overlap): OK")


def case_c_dedup_close_frontier():
    """(c) De-dup: frontier candidates within 0.5 m of LLM picks dropped."""
    os.environ["REMEMBR_MIN_WAYPOINT_DIST"] = "0.5"
    r = _make_runner(n_frontier_inject=3)
    llm = [
        _make_cand(1, 1.0, 1.0, source="remembr"),
    ]
    frontier = [
        _make_cand(10, 1.1, 1.1, source="planner"),   # ~0.14 m from LLM — drop
        _make_cand(11, 5.0, 5.0, source="planner"),   # far — keep
        _make_cand(12, 1.0, 1.4, source="planner"),   # 0.4 m from LLM — drop
    ]
    r.remembr_planner = SimpleNamespace(propose=lambda **kw: llm)
    r.planner = SimpleNamespace(propose_diverse=lambda *a, **kw: frontier)
    out = r._propose_candidates(_make_step(), _make_ep())
    assert len(out) == 2, \
        f"expected 2 cands, got {len(out)}: {[(c.candidate_id, c.source) for c in out]}"
    kept = [c for c in out if c.source == "frontier"]
    assert len(kept) == 1 and kept[0].candidate_id == 11, \
        f"expected only cand_id=11 kept, got {[(c.candidate_id, c.world_xy.tolist()) for c in kept]}"
    print("  case (c) de-dup within 0.5 m: OK")


def case_d_zero_inject():
    """Defensive: REMEMBR_FRONTIER_INJECT=0 disables injection entirely."""
    r = _make_runner(n_frontier_inject=0)
    llm = [_make_cand(1, 5.0, 0.0, source="remembr")]
    sentinel = []
    r.remembr_planner = SimpleNamespace(propose=lambda **kw: llm)
    r.planner = SimpleNamespace(propose_diverse=lambda *a, **kw: sentinel.append(("called",)) or [])
    out = r._propose_candidates(_make_step(), _make_ep())
    assert out == llm, f"expected just LLM cands, got {out}"
    assert not sentinel, \
        f"frontier planner must not be called when n_frontier_inject=0, got {sentinel}"
    print("  case (d) n_frontier_inject=0 disables injection: OK")


def case_e_frontier_backbone_unchanged():
    """Defensive: frontier backbone path unchanged — no LLM call, no merge."""
    r = EpisodeRunner.__new__(EpisodeRunner)
    r.backbone = "frontier"
    r.target_category = "chair"
    r.n_frontier_inject = 3
    frontier = [_make_cand(10, 0.0, 3.0, source="planner")]
    r.planner = SimpleNamespace(propose=lambda *a, **kw: frontier)
    r.remembr_planner = None  # must not be touched
    out = r._propose_candidates(_make_step(), _make_ep())
    assert out == frontier, f"frontier backbone must return planner output, got {out}"
    print("  case (e) frontier backbone unchanged: OK")


def case_f_propose_diverse_compass_fallback():
    """FrontierPlanner.propose_diverse must emit k compass candidates when
    the occupancy grid has nothing real to offer (random_walk fallback)."""
    from embodied_memory.frontier_planner import FrontierPlanner
    fp = FrontierPlanner()
    fp.reset()
    out = fp.propose_diverse(
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        agent_yaw=0.0,
        k=3,
    )
    assert len(out) == 3, f"expected 3 compass candidates, got {len(out)}"
    for c in out:
        assert c.metadata.get("fallback") == "compass", \
            f"expected compass fallback, got {c.metadata}"
        # Empty grid → no FREE/OCCUPIED data → raw_score should be the
        # 0.7 unknown-baseline (Run-4 smoke-3 behavior preserved).
        assert abs(c.raw_score - 0.7) < 1e-6, \
            f"empty-grid compass should score 0.7, got {c.raw_score}"
    # Pairwise xy distances must be > 0.5 m so de-dup against a forward LLM
    # pick cannot wipe out all three.
    import itertools
    for a, b in itertools.combinations(out, 2):
        d = float(np.linalg.norm(a.world_xy - b.world_xy))
        assert d > 0.5, f"compass cands too close: {a.world_xy} vs {b.world_xy} ({d:.3f} m)"
    print("  case (f) propose_diverse compass fallback (k=3, baseline 0.7): OK")


def case_g_compass_occupancy_aware():
    """When the grid has FREE cells in one direction and OCCUPIED in
    another, the compass picks must score the FREE direction strictly
    higher than the OCCUPIED direction."""
    from embodied_memory.frontier_planner import (
        CELL_FREE, CELL_OCCUPIED, FrontierPlanner,
    )
    fp = FrontierPlanner()
    fp.reset()

    # Agent at world (0, 0); yaw=0 means forward is +z (per the
    # (sin θ, cos θ) convention).
    # Paint FREE cells along the +z ray (forward, offset=0).
    # Paint OCCUPIED cells along the +y plane = -z direction (offset=π).
    import math as _math
    ax, az = 0.0, 0.0
    for i in range(1, 22):  # cover ~2 m at 0.1 m resolution
        rr = i * fp.grid.resolution_m
        # forward (yaw=0, +z): FREE
        r, c = fp.grid.world_to_grid(ax, az + rr)
        fp.grid.mark(r, c, CELL_FREE)
        # backward (yaw=π, -z): OCCUPIED
        r, c = fp.grid.world_to_grid(ax, az - rr)
        fp.grid.mark(r, c, CELL_OCCUPIED)
        # sides (yaw=±π/2): UNKNOWN (leave default)

    # k=2 isolates the {forward, backward} axis without polluting from
    # the 2π/3 fan; both picks lie at exact offsets 0 and π.
    out = fp._compass_fallback(
        np.array([ax, 0.0, az], dtype=np.float32),
        agent_yaw=0.0,
        k=2,
    )
    assert len(out) == 2, f"expected 2 cands, got {len(out)}"
    fwd, bwd = out[0], out[1]
    assert fwd.metadata["offset_rad"] == 0.0, \
        f"first cand should be offset=0, got {fwd.metadata}"
    assert abs(fwd.metadata["offset_rad"] - 0.0) < 1e-6
    assert abs(bwd.metadata["offset_rad"] - _math.pi) < 1e-6
    # Forward (all FREE): raw_score should saturate near 1.0.
    assert fwd.raw_score > 0.95, \
        f"FREE-direction compass should score near 1.0, got {fwd.raw_score:.3f}"
    # Backward (all OCCUPIED): raw_score should be near 0.2.
    assert bwd.raw_score < 0.3, \
        f"OCCUPIED-direction compass should score near 0.2, got {bwd.raw_score:.3f}"
    # And the spread must be huge so the rerank actually prefers FREE.
    assert (fwd.raw_score - bwd.raw_score) > 0.5, \
        f"spread {fwd.raw_score - bwd.raw_score:.3f} too small"
    print(f"  case (g) compass occupancy-aware "
          f"(FREE={fwd.raw_score:.3f}, OCC={bwd.raw_score:.3f}): OK")


def case_h_grid_recenters_on_reset():
    """planner.reset(agent_pos=...) must shift the grid origin so the
    agent's start xz lands at the grid's center cell. Without this,
    HM3D agent positions like z=-17.77 fall outside the default
    20 m-square grid centered at world origin."""
    from embodied_memory.frontier_planner import FrontierPlanner
    fp = FrontierPlanner()
    # Default (no agent_pos): origin at (-10, -10) per __init__.
    assert fp.grid.origin_xy == (-10.0, -10.0), \
        f"unexpected default origin: {fp.grid.origin_xy}"
    # HM3D-style position 7.77 m outside the default grid.
    agent_pos = np.array([-0.23, 0.0, -17.77], dtype=np.float32)
    fp.reset(agent_pos=agent_pos)
    ox, oz = fp.grid.origin_xy
    # New origin must place agent at grid center (size/2 from each edge).
    assert abs(ox - (-0.23 - 10.0)) < 1e-6, f"x origin off: {ox}"
    assert abs(oz - (-17.77 - 10.0)) < 1e-6, f"z origin off: {oz}"
    # Agent's xz must now be in-bounds.
    r, c = fp.grid.world_to_grid(-0.23, -17.77)
    assert fp.grid.in_bounds(r, c), \
        f"agent xz still out-of-bounds after recenter: ({r}, {c})"
    # And the ray-occupancy helper must report in-bounds samples for a
    # forward ray from the agent (no FREE/OCC marks yet, just bounds).
    frac_f, frac_o = fp._ray_occupancy_fractions(
        ax=-0.23, az=-17.77, theta=0.0, max_dist=2.0,
    )
    # All UNKNOWN at this point — both fractions zero, but the helper
    # would have returned (0, 0) for out-of-bounds too. Disambiguate by
    # confirming a marked cell shows up: paint one FREE cell along the
    # forward ray and re-evaluate.
    from embodied_memory.frontier_planner import CELL_FREE
    fr, fc = fp.grid.world_to_grid(-0.23, -17.77 + 1.0)  # 1 m forward in +z
    fp.grid.mark(fr, fc, CELL_FREE)
    frac_f, _ = fp._ray_occupancy_fractions(
        ax=-0.23, az=-17.77, theta=0.0, max_dist=2.0,
    )
    assert frac_f > 0.0, \
        f"FREE cell at agent+1m must register after recenter (frac_free={frac_f})"
    print(f"  case (h) grid recenters on reset (origin={fp.grid.origin_xy}): OK")


# ----------------------------------------------------------------------
# Run-5 cases: densified depth splat + grid stats + oracle backbone
# ----------------------------------------------------------------------


# Intrinsics the densified splat derives from hfov=79° on a square 256 sensor.
# f_px = (w/2)/tan(hfov/2); cx = cy = 128. Tests below pick rows/depths so the
# height gate lands cleanly on either side of obstacle_min_h (=0.3 m), given the
# default camera_height_m (=0.88 m) → floor_y = agent_y - 0.88.


def case_densify_grid():
    """Run-5 Change 2: the multi-row per-pixel splat must carve far more FREE
    cells than the old single eye-level scanline, and expose frontier cells.

    Baseline mimics the old behavior: only the eye-level band is valid, and it
    sees a near wall (short rays). Dense adds far floor rows (long rays)."""
    from embodied_memory.frontier_planner import FrontierPlanner

    agent_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Baseline: only an eye-level band (rows 110-145) sees a near wall at 0.6 m.
    depth_base = np.zeros((256, 256), dtype=np.float32)
    depth_base[110:146, :] = 0.6
    fp_base = FrontierPlanner()
    fp_base.reset(agent_pos=agent_pos)
    fp_base.update(depth_base, agent_pos, agent_yaw=0.0)
    base_free = fp_base.grid_stats()["cells_free"]

    # Dense: same near wall PLUS far floor rows (146-255) at 4.0 m.
    depth_dense = depth_base.copy()
    depth_dense[146:256, :] = 4.0
    fp = FrontierPlanner()
    fp.reset(agent_pos=agent_pos)
    fp.update(depth_dense, agent_pos, agent_yaw=0.0)
    stats = fp.grid_stats()
    dense_free = stats["cells_free"]

    assert dense_free > 200, f"dense splat carved too few FREE cells: {dense_free}"
    assert dense_free > base_free * 3, \
        f"dense free ({dense_free}) not >> single-row baseline ({base_free})"
    assert stats["frontier_cells"] > 0, \
        f"densified grid exposed no frontier cells: {stats}"
    print(f"  case densify_grid (base_free={base_free}, dense_free={dense_free}, "
          f"frontier={stats['frontier_cells']}): OK")


def case_height_gate():
    """Run-5 Change 2: the endpoint height gate marks floor (low world_h) FREE
    and obstacles (high world_h) OCCUPIED.

    A far floor band sits well below obstacle_min_h → FREE-only endpoints.
    An eye-level band sits at camera height (0.88 m > 0.3 m) → OCCUPIED."""
    from embodied_memory.frontier_planner import FrontierPlanner

    agent_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Far floor band (rows 175-185 look downward) → low world_h → FREE.
    depth_floor = np.zeros((256, 256), dtype=np.float32)
    depth_floor[175:186, :] = 2.0
    fp_floor = FrontierPlanner()
    fp_floor.reset(agent_pos=agent_pos)
    fp_floor.update(depth_floor, agent_pos, agent_yaw=0.0)
    s_floor = fp_floor.grid_stats()
    assert s_floor["cells_occupied"] == 0, \
        f"floor band must produce no OCCUPIED cells, got {s_floor['cells_occupied']}"
    assert s_floor["cells_free"] > 0, "floor band produced no FREE cells"

    # Eye-level band (rows 150-160) at camera height → OCCUPIED.
    depth_wall = np.zeros((256, 256), dtype=np.float32)
    depth_wall[150:161, :] = 2.0
    fp_wall = FrontierPlanner()
    fp_wall.reset(agent_pos=agent_pos)
    fp_wall.update(depth_wall, agent_pos, agent_yaw=0.0)
    s_wall = fp_wall.grid_stats()
    assert s_wall["cells_occupied"] > 0, \
        f"eye-level band must produce OCCUPIED cells, got {s_wall}"
    print(f"  case height_gate (floor_occ={s_floor['cells_occupied']}, "
          f"wall_occ={s_wall['cells_occupied']}): OK")


def case_grid_stats_schema():
    """Run-5 Change 3: grid_stats() exposes four int keys; the free/occupied/
    unknown census sums to the full grid (n*n = 40000 at 20 m / 0.1 m)."""
    from embodied_memory.frontier_planner import FrontierPlanner

    fp = FrontierPlanner()
    fp.reset(agent_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    fp.update(
        np.full((256, 256), 3.0, dtype=np.float32),
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        agent_yaw=0.0,
    )
    stats = fp.grid_stats()
    assert set(stats.keys()) == {
        "cells_free", "cells_occupied", "cells_unknown", "frontier_cells",
    }, f"unexpected grid_stats keys: {sorted(stats.keys())}"
    for k, v in stats.items():
        assert isinstance(v, int), f"{k} is not int: {type(v)}"
    n = fp.grid.n
    total = stats["cells_free"] + stats["cells_occupied"] + stats["cells_unknown"]
    assert total == n * n, f"census {total} != n*n ({n * n})"
    print(f"  case grid_stats_schema (n*n={n * n}, free={stats['cells_free']}): OK")


def case_oracle_action_map():
    """Run-5 Change 1: _oracle_action maps the follower's return to the
    discrete action id. Names go through _ACTION_NAMES.index; None → STOP."""
    r = EpisodeRunner.__new__(EpisodeRunner)
    r.backbone = "oracle"
    r._oracle_goal_radius = 1.0
    holder = {"next": None}
    # follower already set → _oracle_action must not try to build one (which
    # would dereference r.source); mirrors the _make_runner sentinel pattern.
    r.follower = SimpleNamespace(get_next_action=lambda goal: holder["next"])
    ep = SimpleNamespace(target_position=np.array([1.0, 0.0, 1.0], dtype=np.float32))
    expected = {"move_forward": 1, "turn_left": 2, "stop": 0, None: 0}
    for name, want in expected.items():
        holder["next"] = name
        got = r._oracle_action(ep)
        assert got == want, f"follower→{name!r} should map to {want}, got {got}"
    # No goal → STOP without touching the follower.
    r.follower = SimpleNamespace(
        get_next_action=lambda goal: (_ for _ in ()).throw(AssertionError("called")))
    assert r._oracle_action(SimpleNamespace(target_position=None)) == 0
    print("  case oracle_action_map (move_forward/turn_left/stop/None → 1/2/0/0): OK")


def case_oracle_short_circuit():
    """Run-5 Change 1: with backbone='oracle' and bridge=None, _run_episode's
    oracle branch never calls _propose_candidates or dereferences the bridge,
    and still logs grid stats."""
    r = EpisodeRunner.__new__(EpisodeRunner)
    r.backbone = "oracle"
    r.bridge = None
    r.clip_encoder = None
    r.captioner = None
    r.max_steps_per_episode = 3
    r.keyframe_every_m = 5
    r.follower = None
    r._oracle_goal_radius = 1.0

    propose_calls: list = []
    r._propose_candidates = lambda *a, **kw: propose_calls.append(1) or []
    oracle_calls: list = []
    r._oracle_action = lambda ep: oracle_calls.append(1) or 0  # ACTION_STOP

    agent_state = SimpleNamespace(
        position=np.zeros(3, dtype=np.float32), rotation_yaw=0.0)
    depth = np.zeros((4, 4), dtype=np.float32)
    step0 = SimpleNamespace(step_idx=0, depth=depth, rgb=None, semantic=None,
                            agent_state=agent_state, action=None, reward=0.0,
                            done=False, info={})
    step1 = SimpleNamespace(step_idx=1, depth=depth, rgb=None, semantic=None,
                            agent_state=agent_state, action=0, reward=0.0,
                            done=True,
                            info={"success": False, "distance_to_goal": 5.0,
                                  "spl": 0.0})
    ep = SimpleNamespace(episode_id="e0", scene_id="s0",
                         target_category="chair",
                         target_position=np.array([1.0, 0.0, 1.0], dtype=np.float32),
                         success=False, spl=0.0)
    r.source = SimpleNamespace(reset=lambda idx: (step0, ep),
                               step=lambda a: step1)
    grid = {"cells_free": 0, "cells_occupied": 0,
            "cells_unknown": 40000, "frontier_cells": 0}
    r.planner = SimpleNamespace(
        reset=lambda agent_pos=None: None,
        update=lambda *a, **kw: None,
        grid_stats=lambda: grid,
        is_decision_step=lambda: True,
    )

    ep_log, ep_metrics = r._run_episode(0)
    assert not propose_calls, "oracle must NOT call _propose_candidates"
    assert oracle_calls, "oracle must call _oracle_action"
    assert ep_metrics["success"] is False
    assert ep_log["grid_cells_unknown"] == 40000, \
        f"grid stats not logged: {ep_log.get('grid_cells_unknown')}"
    assert ep_log["bridge_stats_after"] == {}, "None bridge must log empty stats"
    print("  case oracle_short_circuit (no bridge/propose deref, grid logged): OK")


# ----------------------------------------------------------------------
# Run-6: collision-aware step controller (grid A*)
# ----------------------------------------------------------------------


def case_astar_routes_through_gap():
    """A* must route through a wall's single gap, never stepping on OCCUPIED."""
    from embodied_memory.frontier_planner import astar, CELL_FREE, CELL_OCCUPIED

    g = np.full((8, 8), CELL_FREE, dtype=np.uint8)
    g[4, 0:8] = CELL_OCCUPIED
    g[4, 6] = CELL_FREE  # single gap
    path = astar(g, (1, 1), (6, 1), inflate_radius_cells=0)
    assert path is not None, "no path found through the gap"
    assert (4, 6) in path, f"path did not route through the gap: {path}"
    assert all(g[r, c] != CELL_OCCUPIED for (r, c) in path), \
        f"path stepped on an OCCUPIED cell: {path}"
    assert path[0] == (1, 1) and path[-1] == (6, 1)
    print("  case astar_routes_through_gap: OK")


def case_astar_none_when_walled_off():
    """A* returns None when the goal is fully enclosed by OCCUPIED cells."""
    from embodied_memory.frontier_planner import astar, CELL_FREE, CELL_OCCUPIED

    g = np.full((8, 8), CELL_FREE, dtype=np.uint8)
    g[2, 2:6] = CELL_OCCUPIED
    g[6, 2:6] = CELL_OCCUPIED
    g[2:7, 2] = CELL_OCCUPIED
    g[2:7, 5] = CELL_OCCUPIED
    assert astar(g, (0, 0), (4, 3), inflate_radius_cells=0) is None, \
        "walled-off goal must be unreachable"
    print("  case astar_none_when_walled_off: OK")


def case_astar_inflation_seals_one_cell_gap():
    """1-cell obstacle inflation seals a 1-cell gap (agent-radius clearance)."""
    from embodied_memory.frontier_planner import astar, CELL_FREE, CELL_OCCUPIED

    g = np.full((8, 8), CELL_FREE, dtype=np.uint8)
    g[4, 0:8] = CELL_OCCUPIED
    g[4, 4] = CELL_FREE  # 1-cell gap
    assert astar(g, (1, 4), (6, 4), inflate_radius_cells=0) is not None, \
        "gap should be passable without inflation"
    assert astar(g, (1, 4), (6, 4), inflate_radius_cells=1) is None, \
        "1-cell inflation should seal the 1-cell gap"
    print("  case astar_inflation_seals_one_cell_gap: OK")


def case_astar_goal_occupied_snaps():
    """_snap_to_free redirects an OCCUPIED goal to the nearest FREE cell."""
    from embodied_memory.frontier_planner import (
        _inflate_occupied, _snap_to_free, CELL_FREE, CELL_OCCUPIED,
    )

    g = np.full((8, 8), CELL_FREE, dtype=np.uint8)
    g[3, 3] = CELL_OCCUPIED
    blocked = _inflate_occupied(g, 0)
    snapped = _snap_to_free(blocked, (3, 3), max_radius=5)
    assert snapped is not None, "snap found no free cell"
    assert g[snapped[0], snapped[1]] == CELL_FREE, "snapped cell is not FREE"
    assert max(abs(snapped[0] - 3), abs(snapped[1] - 3)) <= 5, "snap exceeded radius"
    print("  case astar_goal_occupied_snaps: OK")


def case_astar_start_equals_goal():
    """Degenerate start==goal returns a single-cell path; no crash."""
    from embodied_memory.frontier_planner import astar, CELL_FREE

    g = np.full((8, 8), CELL_FREE, dtype=np.uint8)
    path = astar(g, (3, 3), (3, 3))
    assert path == [(3, 3)], f"expected [(3,3)], got {path}"
    print("  case astar_start_equals_goal: OK")


def case_astar_first_action_not_into_wall():
    """step_controller must TURN toward an offset gap, not FORWARD into a wall
    that sits straight ahead of the agent."""
    from embodied_memory.frontier_planner import (
        FrontierPlanner, FrontierCandidate, CELL_FREE, CELL_OCCUPIED,
        ACTION_TURN_LEFT, ACTION_TURN_RIGHT,
    )

    fp = FrontierPlanner()
    agent_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    fp.reset(agent_pos=agent_pos)
    ar, ac = fp.grid.world_to_grid(0.0, 0.0)  # forward (yaw=0) is +z → +row
    # Free room ahead, a full-width wall 5 cells ahead, a 3-cell gap offset to
    # +col (survives the default 1-cell inflation as a 1-cell effective gap).
    fp.grid.grid[ar - 2:ar + 12, ac - 8:ac + 10] = CELL_FREE
    wall_r = ar + 5
    fp.grid.grid[wall_r, :] = CELL_OCCUPIED
    fp.grid.grid[wall_r, ac + 5:ac + 8] = CELL_FREE
    goal_rc = (ar + 9, ac)  # straight ahead, beyond the wall
    cand = FrontierCandidate(
        candidate_id=1,
        world_xy=np.array(fp.grid.grid_to_world(*goal_rc), dtype=np.float32),
        grid_rc=goal_rc,
        distance_m=0.9,
        bearing_rad=0.0,  # straight-line controller would FORWARD into the wall
        cluster_size=1,
        raw_score=1.0,
    )
    action = fp.step_controller(cand, agent_pos, agent_yaw=0.0)
    assert action in (ACTION_TURN_LEFT, ACTION_TURN_RIGHT), \
        f"A* should turn toward the gap, not drive into the wall; got {action}"
    print("  case astar_first_action_not_into_wall: OK")


def case_astar_lookahead_waypoint():
    """The ~0.4 m (4-cell) lookahead must not cut a far corner: on a straight
    leg with the bend 6 cells away, the controller goes FORWARD."""
    from embodied_memory.frontier_planner import (
        FrontierPlanner, FrontierCandidate, CELL_FREE, CELL_OCCUPIED,
        ACTION_FORWARD,
    )

    # inflate=0 so the 1-cell corridor stays passable and the path is exact.
    fp = FrontierPlanner(inflate_radius_cells=0)
    agent_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    fp.reset(agent_pos=agent_pos)
    ar, ac = fp.grid.world_to_grid(0.0, 0.0)
    # Box, then carve an L: straight forward 6 cells (col ac), bend right.
    fp.grid.grid[ar - 1:ar + 8, ac - 3:ac + 8] = CELL_OCCUPIED
    fp.grid.grid[ar:ar + 7, ac] = CELL_FREE        # vertical leg
    fp.grid.grid[ar + 6, ac:ac + 6] = CELL_FREE    # horizontal leg (the bend)
    goal_rc = (ar + 6, ac + 5)
    cand = FrontierCandidate(
        candidate_id=1,
        world_xy=np.array(fp.grid.grid_to_world(*goal_rc), dtype=np.float32),
        grid_rc=goal_rc,
        distance_m=1.0,
        bearing_rad=0.0,
        cluster_size=1,
        raw_score=1.0,
    )
    action = fp.step_controller(cand, agent_pos, agent_yaw=0.0)
    assert action == ACTION_FORWARD, \
        f"lookahead cut the far corner instead of going straight; got {action}"
    print("  case astar_lookahead_waypoint: OK")


def case_controller_fallback_on_none():
    """When A* finds no path, step_controller falls back to the straight-line
    bearing and forces a replan."""
    from embodied_memory.frontier_planner import (
        FrontierPlanner, FrontierCandidate, CELL_FREE, CELL_OCCUPIED,
        ACTION_TURN_LEFT,
    )

    fp = FrontierPlanner()
    agent_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    fp.reset(agent_pos=agent_pos)
    ar, ac = fp.grid.world_to_grid(0.0, 0.0)
    # Seal the agent in a 3x3 FREE pocket → no route to anywhere outside.
    fp.grid.grid[ar - 2:ar + 3, ac - 2:ac + 3] = CELL_OCCUPIED
    fp.grid.grid[ar - 1:ar + 2, ac - 1:ac + 2] = CELL_FREE
    goal_rc = (ar + 10, ac + 10)  # unreachable (UNKNOWN, but agent is boxed in)
    cand = FrontierCandidate(
        candidate_id=1,
        world_xy=np.array([5.0, 5.0], dtype=np.float32),
        grid_rc=goal_rc,
        distance_m=1.0,
        bearing_rad=float(np.deg2rad(40.0)),  # >15° → straight-line TURN_LEFT
        cluster_size=1,
        raw_score=1.0,
    )
    fp._force_replan = False
    action = fp.step_controller(cand, agent_pos, agent_yaw=0.0)
    assert action == ACTION_TURN_LEFT, \
        f"fallback should use straight-line bearing (40°→TURN_LEFT); got {action}"
    assert fp._force_replan is True, "A*-None fallback must force a replan"
    print("  case controller_fallback_on_none: OK")


def main() -> int:
    print("Run-4/Run-5 sanity tests")
    case_a_stop_short_circuit()
    case_b_frontier_injected()
    case_c_dedup_close_frontier()
    case_d_zero_inject()
    case_e_frontier_backbone_unchanged()
    case_f_propose_diverse_compass_fallback()
    case_g_compass_occupancy_aware()
    case_h_grid_recenters_on_reset()
    case_densify_grid()
    case_height_gate()
    case_grid_stats_schema()
    case_oracle_action_map()
    case_oracle_short_circuit()
    # Run-6: collision-aware step controller (grid A*)
    case_astar_routes_through_gap()
    case_astar_none_when_walled_off()
    case_astar_inflation_seals_one_cell_gap()
    case_astar_goal_occupied_snaps()
    case_astar_start_equals_goal()
    case_astar_first_action_not_into_wall()
    case_astar_lookahead_waypoint()
    case_controller_fallback_on_none()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
