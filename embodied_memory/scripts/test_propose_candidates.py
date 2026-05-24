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


def main() -> int:
    print("Run-4 _propose_candidates sanity tests")
    case_a_stop_short_circuit()
    case_b_frontier_injected()
    case_c_dedup_close_frontier()
    case_d_zero_inject()
    case_e_frontier_backbone_unchanged()
    case_f_propose_diverse_compass_fallback()
    case_g_compass_occupancy_aware()
    case_h_grid_recenters_on_reset()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
