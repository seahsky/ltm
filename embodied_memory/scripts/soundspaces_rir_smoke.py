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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True, help="HM3D scene .glb / .basis.glb")
    ap.add_argument("--out", default="runs/soundspaces-spike/rir.npy")
    ap.add_argument(
        "--materials", action="store_true",
        help="enable acoustic materials (needs semantic annotations; default "
             "OFF — HM3D's semantics are absent/broken, the documented "
             "SoundSpaces fallback is enableMaterials=False)",
    )
    ap.add_argument("--sample-rate", type=int, default=48000)
    args = ap.parse_args()

    if not os.path.isfile(args.scene):
        print(f"RED: scene file not found: {args.scene}")
        return 2

    try:
        import habitat_sim
    except ImportError as e:
        print(f"RED: habitat_sim import failed (audio build broken?): {e}")
        return 2
    if not hasattr(habitat_sim, "AudioSensorSpec"):
        print("RED: habitat_sim has no AudioSensorSpec — built without --audio "
              "or not from the RLRAudioPropagationUpdate branch "
              f"(habitat_sim {getattr(habitat_sim, '__version__', '?')} at "
              f"{habitat_sim.__file__})")
        return 2

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = args.scene
    backend_cfg.load_semantic_mesh = bool(args.materials)
    backend_cfg.enable_physics = False
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
    print(f"  sim up: scene={os.path.basename(args.scene)} "
          f"navmesh_loaded={sim.pathfinder.is_loaded}")

    spec = habitat_sim.AudioSensorSpec()
    spec.uuid = "audio_sensor"
    spec.enableMaterials = bool(args.materials)
    spec.channelLayout.type = (
        habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
    )
    spec.channelLayout.channelCount = 2
    spec.position = [0.0, 1.5, 0.0]          # ear height
    spec.acousticsConfig.sampleRate = args.sample_rate
    spec.acousticsConfig.indirect = True      # reverberant tail, not just direct
    sim.add_sensor(spec)

    agent = sim.get_agent(0)
    listener_pt = sim.pathfinder.get_random_navigable_point()
    state = agent.get_state()
    state.position = listener_pt
    agent.set_state(state)

    source_pt = sim.pathfinder.get_random_navigable_point()
    audio_sensor = agent._sensors["audio_sensor"]
    audio_sensor.setAudioSourceTransform(
        np.asarray(source_pt, dtype=np.float32) + np.array([0.0, 1.5, 0.0],
                                                           dtype=np.float32)
    )
    print(f"  listener={np.round(listener_pt, 2).tolist()} "
          f"source={np.round(source_pt, 2).tolist()}")

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
        print("RED: rendered IR is all-zero (no acoustic path? source/listener "
              "in disconnected components?) — retry may pick better points")
        return 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, ir)
    print(f"GREEN: RIR rendered — shape={ir.shape} sr={args.sample_rate} "
          f"peak={peak:.6f} energy={energy:.6f} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
