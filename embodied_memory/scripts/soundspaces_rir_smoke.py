"""
soundspaces_rir_smoke — GO/NO-GO spike: render ONE room impulse response in
an HM3D scene via habitat-sim's RLRAudioPropagation (SoundSpaces 2.0).

Run inside the dedicated `soundspaces-spike` conda env (built by
scripts/race-soundspaces-spike.sh from the habitat-sim
`RLRAudioPropagationUpdate` branch with --audio). NEVER run this in the
working `ltm-embodied` env — it imports a different habitat-sim build.

GREEN = an IR waveform with finite, non-zero energy is rendered and saved.
Anything else exits non-zero with the exact failure, which IS the spike's
deliverable (the version-delta / blocker list for the Friday reassessment).

    python embodied_memory/scripts/soundspaces_rir_smoke.py \
        --scene data/hm3d/.../wcojb4TFT35.basis.glb \
        --out runs/soundspaces-spike/rir.npy
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def _geodesic(pathfinder, a, b, habitat_sim) -> float:
    sp = habitat_sim.ShortestPath()
    sp.requested_start = a
    sp.requested_end = b
    found = pathfinder.find_path(sp)
    return float(sp.geodesic_distance) if found else float("inf")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True, help="HM3D scene .glb / .basis.glb")
    ap.add_argument("--out", default="runs/soundspaces-spike/rir.npy")
    ap.add_argument(
        "--materials", action="store_true",
        help="EXPERIMENTAL: enable acoustic materials. Needs semantic "
             "annotations + --scene-dataset-config + --materials-json; HM3D "
             "semantics are absent/broken so the documented SoundSpaces mode "
             "for HM3D is materials OFF (the default here).",
    )
    ap.add_argument(
        "--scene-dataset-config", default="",
        help="SceneDatasetConfig json (only used with --materials; e.g. "
             "hm3d_annotated_basis.scene_dataset_config.json)",
    )
    ap.add_argument(
        "--materials-json", default="",
        help="category->acoustic-material json (only used with --materials; "
             "ships in the rlr-audio-propagation submodule at "
             "RLRAudioPropagationPkg/data/mp3d_material_config.json)",
    )
    ap.add_argument("--sample-rate", type=int, default=48000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if not os.path.isfile(args.scene):
        print(f"RED: scene file not found: {args.scene}")
        return 2

    try:
        # Official SoundSpaces workaround: quaternion MUST be imported before
        # habitat_sim or audio builds can crash with free(): invalid pointer
        # at import (habitat-sim issue #1813 / INSTALLATION.md known issues).
        import quaternion  # noqa: F401
    except ImportError as e:
        print(f"RED: numpy-quaternion missing (pip install numpy-quaternion): {e}")
        return 2
    try:
        import habitat_sim
        import habitat_sim.sensor
    except ImportError as e:
        print(f"RED: habitat_sim import failed (audio build broken?): {e}")
        return 2
    # Probe an audio-GATED symbol: AudioSensorSpec is bound even in non-audio
    # builds, but RLRAudioPropagationChannelLayoutType is py::none() there
    # (issue #2340's signature) — so this catches a without---audio build that
    # a hasattr(habitat_sim, "AudioSensorSpec") check would false-GREEN.
    clt = getattr(habitat_sim.sensor, "RLRAudioPropagationChannelLayoutType", None)
    if clt is None or not hasattr(clt, "Binaural"):
        print("RED: habitat_sim built WITHOUT --audio "
              "(RLRAudioPropagationChannelLayoutType missing/stub) — rebuild "
              "from the RLRAudioPropagationUpdate branch with --audio "
              f"(habitat_sim {getattr(habitat_sim, '__version__', '?')} at "
              f"{habitat_sim.__file__})")
        return 2

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = args.scene
    backend_cfg.load_semantic_mesh = bool(args.materials)
    if args.materials and args.scene_dataset_config:
        backend_cfg.scene_dataset_config_file = args.scene_dataset_config
    backend_cfg.enable_physics = False
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
    sim.seed(args.seed)
    print(f"  sim up: scene={os.path.basename(args.scene)} "
          f"navmesh_loaded={sim.pathfinder.is_loaded}")
    if not sim.pathfinder.is_loaded:
        print("RED: navmesh not loaded — expected <scene>.basis.navmesh next "
              "to the .glb (random navigable points would be NaN)")
        sim.close()
        return 2

    spec = habitat_sim.AudioSensorSpec()
    spec.uuid = "audio_sensor"
    spec.enableMaterials = bool(args.materials)
    spec.channelLayout.type = clt.Binaural
    spec.channelLayout.channelCount = 2
    spec.position = [0.0, 1.5, 0.0]          # ear height
    spec.acousticsConfig.sampleRate = args.sample_rate
    spec.acousticsConfig.indirect = True      # reverberant tail, not just direct
    sim.add_sensor(spec)

    agent = sim.get_agent(0)
    audio_sensor = agent._sensors["audio_sensor"]
    if args.materials and args.materials_json:
        audio_sensor.setAudioMaterialsJSON(args.materials_json)

    # Multi-floor HM3D scenes have disconnected navmesh islands; two blind
    # random draws can produce an all-zero IR (false RED on a working build).
    # Keep the first listener/source pair with a finite, bounded geodesic.
    pair = None
    for _ in range(20):
        listener_pt = sim.pathfinder.get_random_navigable_point()
        source_pt = sim.pathfinder.get_random_navigable_point()
        d = _geodesic(sim.pathfinder, listener_pt, source_pt, habitat_sim)
        if np.isfinite(d) and 1.0 <= d <= 15.0:
            pair = (listener_pt, source_pt, d)
            break
    if pair is None:
        print("RED: no connected listener/source pair within [1,15] m in 20 "
              "draws (disconnected navmesh islands?)")
        sim.close()
        return 1
    listener_pt, source_pt, geo_d = pair

    state = agent.get_state()
    state.position = listener_pt
    agent.set_state(state)
    audio_sensor.setAudioSourceTransform(
        np.asarray(source_pt, dtype=np.float32) + np.array([0.0, 1.5, 0.0],
                                                           dtype=np.float32)
    )
    print(f"  listener={np.round(listener_pt, 2).tolist()} "
          f"source={np.round(source_pt, 2).tolist()} geodesic={geo_d:.2f}m")

    ir = np.asarray(sim.get_sensor_observations()["audio_sensor"])
    sim.close()

    if ir.size == 0:
        print("RED: rendered IR is empty (size 0)")
        return 1
    if not np.all(np.isfinite(ir)):
        print("RED: rendered IR contains non-finite samples")
        return 1
    energy = float(np.sum(np.square(ir, dtype=np.float64)))
    peak = float(np.max(np.abs(ir)))
    if energy <= 0.0:
        print("RED: rendered IR is all-zero despite a connected pair — "
              "engine/material issue, paste the full log")
        return 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, ir)
    print(f"GREEN: RIR rendered — shape={ir.shape} sr={args.sample_rate} "
          f"geodesic={geo_d:.2f}m peak={peak:.6f} energy={energy:.6f} "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
