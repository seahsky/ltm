"""
render_rir_grid — batch RIR grid renderer (M0a of the AudioGoal/FSD50K plan).

Generalizes ``soundspaces_rir_smoke.py`` from one impulse response to a grid:
sample navigable listener cells across an HM3D scene, render the binaural RIR
from a SINGLE anomaly source to each cell, and serialize them to a plain
``.npz`` (``audio.save_rir_grid``) for the live ``ltm-embodied`` runner to do
O(1) nearest-cell lookup + fftconvolve at run time.

Run inside the dedicated ``soundspaces-spike`` conda env (built by
``scripts/race-soundspaces-spike.sh`` from the habitat-sim
``RLRAudioPropagationUpdate`` branch with --audio). NEVER run in the working
``ltm-embodied`` env — it imports a different habitat-sim build. (habitat-lab
fails to import in the audio env; that is fine — rendering uses habitat_sim
directly, offline only.)

GREEN = ``--n-cells`` (or at least ``--min-cells``) listener cells render with
finite, non-zero-energy binaural IRs and the grid is saved. Anything else exits
non-zero with the exact failure.

    python embodied_memory/scripts/render_rir_grid.py \
        --scene data/hm3d/.../wcojb4TFT35.basis.glb \
        --out runs/audiogoal/wcojb4TFT35_rir_grid.npz --n-cells 20
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import List, Sequence

import numpy as np


# ----------------------------------------------------------------------
# Pure cell-selection logic (no habitat_sim) — unit-tested in
# test_render_rir_grid.py.
# ----------------------------------------------------------------------


def select_cells(
    points,
    geo_dists,
    *,
    max_cells: int,
    min_dist_m: float,
    max_dist_m: float,
    min_spacing_m: float,
) -> List[int]:
    """Pick a well-spread, in-range, source-reachable subset of candidate cells.

    ``points`` : (M, 3) candidate navigable positions.
    ``geo_dists`` : (M,) geodesic distance of each candidate to the source
        (``inf`` for disconnected navmesh islands).

    Keeps a candidate only if its geodesic-to-source is finite and within
    ``[min_dist_m, max_dist_m]``; greedily drops any candidate within
    ``min_spacing_m`` (2-D xz) of an already-kept one (so the grid covers the
    scene rather than clustering); caps at ``max_cells``. Candidates are
    considered in ascending geodesic order for determinism. Returns the kept
    indices into ``points``.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    geo = np.asarray(geo_dists, dtype=np.float64).reshape(-1)
    xz = pts[:, [0, 2]]

    order = sorted(
        range(len(pts)),
        key=lambda i: geo[i] if np.isfinite(geo[i]) else math.inf,
    )
    kept: List[int] = []
    kept_xz: List[np.ndarray] = []
    for i in order:
        d = float(geo[i])
        if not math.isfinite(d) or d < min_dist_m or d > max_dist_m:
            continue
        p = xz[i]
        if any(float(np.linalg.norm(p - q)) < min_spacing_m for q in kept_xz):
            continue
        kept.append(i)
        kept_xz.append(p)
        if len(kept) >= max_cells:
            break
    return kept


# ----------------------------------------------------------------------
# habitat-sim render path (soundspaces-spike env only)
# ----------------------------------------------------------------------


def _load_audio():
    """Load ``embodied_memory/audio.py`` WITHOUT importing the embodied_memory
    package ``__init__`` (which pulls memory_bridge → dialogue_memory → faiss).

    This renderer runs in the ``soundspaces-spike`` env, which has numpy + a
    habitat_sim audio build but NOT faiss; ``audio.py`` itself imports only
    numpy at module load (scipy/perception are lazy), so a direct file load is
    self-sufficient there. Mirrors the standalone-load pattern the tests use.
    """
    import importlib.util

    audio_path = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "audio.py"))
    spec = importlib.util.spec_from_file_location("_audiogoal_audio", audio_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _geodesic(pathfinder, a, b, habitat_sim) -> float:
    sp = habitat_sim.ShortestPath()
    sp.requested_start = a
    sp.requested_end = b
    found = pathfinder.find_path(sp)
    return float(sp.geodesic_distance) if found else float("inf")


def _parse_xyz(s: str):
    parts = [float(v) for v in s.replace(",", " ").split()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"--source must be 'x,y,z', got {s!r}")
    return np.asarray(parts, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True, help="HM3D scene .glb / .basis.glb")
    ap.add_argument("--out", default="runs/audiogoal/rir_grid.npz")
    ap.add_argument("--n-cells", type=int, default=20, help="target listener cells")
    ap.add_argument("--min-cells", type=int, default=8,
                    help="GREEN floor: fewer valid cells than this is RED")
    ap.add_argument("--candidates", type=int, default=400,
                    help="random navigable points sampled before selection")
    ap.add_argument("--source", type=_parse_xyz, default=None,
                    help="fixed source world xyz 'x,y,z' (default: random navigable)")
    ap.add_argument("--min-dist", type=float, default=1.0)
    ap.add_argument("--max-dist", type=float, default=12.0)
    ap.add_argument("--min-spacing", type=float, default=0.5)
    ap.add_argument("--ear-height", type=float, default=1.5)
    ap.add_argument("--sample-rate", type=int, default=48000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if not os.path.isfile(args.scene):
        print(f"RED: scene file not found: {args.scene}")
        return 2

    try:
        import quaternion  # noqa: F401  (must precede habitat_sim — issue #1813)
    except ImportError as e:
        print(f"RED: numpy-quaternion missing (pip install numpy-quaternion): {e}")
        return 2
    try:
        import habitat_sim
        import habitat_sim.sensor
    except ImportError as e:
        print(f"RED: habitat_sim import failed (audio build broken?): {e}")
        return 2
    clt = getattr(habitat_sim.sensor, "RLRAudioPropagationChannelLayoutType", None)
    if clt is None or not hasattr(clt, "Binaural"):
        print("RED: habitat_sim built WITHOUT --audio "
              "(RLRAudioPropagationChannelLayoutType missing/stub) — rebuild "
              "from the RLRAudioPropagationUpdate branch with --audio "
              f"(habitat_sim {getattr(habitat_sim, '__version__', '?')} at "
              f"{habitat_sim.__file__})")
        return 2

    save_rir_grid = _load_audio().save_rir_grid

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = args.scene
    backend_cfg.load_semantic_mesh = False     # HM3D semantics absent → materials OFF
    backend_cfg.enable_physics = False
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
    sim.seed(args.seed)
    scene_id = os.path.basename(args.scene).split(".")[0]
    print(f"  sim up: scene={scene_id} navmesh_loaded={sim.pathfinder.is_loaded}")
    if not sim.pathfinder.is_loaded:
        print("RED: navmesh not loaded — expected <scene>.basis.navmesh next to the .glb")
        sim.close()
        return 2

    pf = sim.pathfinder

    # Source: fixed, or a random navigable point that has enough reachable cells.
    rng = np.random.default_rng(args.seed)
    ear = np.array([0.0, args.ear_height, 0.0], dtype=np.float32)

    def _sample_candidates(k: int):
        return [np.asarray(pf.get_random_navigable_point(), dtype=np.float32)
                for _ in range(k)]

    source_pt = None
    chosen_idx: List[int] = []
    candidates: List[np.ndarray] = []
    for attempt in range(1 if args.source is not None else 8):
        src = args.source if args.source is not None else \
            np.asarray(pf.get_random_navigable_point(), dtype=np.float32)
        candidates = _sample_candidates(args.candidates)
        geo = np.array([_geodesic(pf, src, c, habitat_sim) for c in candidates],
                       dtype=np.float64)
        idx = select_cells(np.stack(candidates), geo, max_cells=args.n_cells,
                           min_dist_m=args.min_dist, max_dist_m=args.max_dist,
                           min_spacing_m=args.min_spacing)
        if len(idx) >= args.min_cells:
            source_pt, chosen_idx = src, idx
            break
        print(f"  [attempt {attempt}] source={np.round(src, 2).tolist()} gave only "
              f"{len(idx)} reachable cells (< {args.min_cells}); resampling source")
    if source_pt is None:
        print(f"RED: could not find a source with >= {args.min_cells} reachable "
              f"cells in 8 draws (disconnected navmesh / bad bounds?)")
        sim.close()
        return 1

    cell_positions = [candidates[i] for i in chosen_idx]
    print(f"  source={np.round(source_pt, 2).tolist()} "
          f"cells={len(cell_positions)} (target {args.n_cells})")

    # Audio sensor at ear height; constant source for the whole grid.
    spec = habitat_sim.AudioSensorSpec()
    spec.uuid = "audio_sensor"
    spec.enableMaterials = False
    spec.channelLayout.type = clt.Binaural
    spec.channelLayout.channelCount = 2
    spec.position = ear.tolist()
    spec.acousticsConfig.sampleRate = args.sample_rate
    spec.acousticsConfig.indirect = True
    sim.add_sensor(spec)
    agent = sim.get_agent(0)
    audio_sensor = agent._sensors["audio_sensor"]
    audio_sensor.setAudioSourceTransform(
        np.asarray(source_pt, dtype=np.float32) + ear)

    valid_cells: List[np.ndarray] = []
    valid_irs: List[np.ndarray] = []
    n_zero = 0
    for cell in cell_positions:
        st = agent.get_state()
        st.position = np.asarray(cell, dtype=np.float32)
        agent.set_state(st)
        ir = np.asarray(sim.get_sensor_observations()["audio_sensor"], dtype=np.float32)
        if ir.ndim != 2 or ir.shape[0] != 2 or ir.size == 0:
            n_zero += 1
            continue
        if not np.all(np.isfinite(ir)):
            n_zero += 1
            continue
        if float(np.sum(np.square(ir, dtype=np.float64))) <= 0.0:
            n_zero += 1
            continue
        valid_cells.append(np.asarray(cell, dtype=np.float32))
        valid_irs.append(ir)
    sim.close()

    if len(valid_cells) < args.min_cells:
        print(f"RED: only {len(valid_cells)} cells produced finite non-zero IRs "
              f"(< {args.min_cells}); {n_zero} were empty/zero/non-finite")
        return 1

    save_rir_grid(
        args.out,
        cell_positions=np.stack(valid_cells),
        source_position=np.asarray(source_pt, dtype=np.float32),
        irs=valid_irs,
        sample_rate=args.sample_rate,
        scene_id=scene_id,
    )
    energies = [float(np.sum(np.square(ir, dtype=np.float64))) for ir in valid_irs]
    print(f"GREEN: rendered {len(valid_cells)} cells (dropped {n_zero}) "
          f"sr={args.sample_rate} scene={scene_id} "
          f"energy[min={min(energies):.4g} max={max(energies):.4g}] -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
