"""
CLI entry point for the HM3D proof-of-life run.

Usage (live):
    python -m embodied_memory.run_hm3d_pol \
        --scene <hm3d-val-scene-id> \
        --n-episodes 5 \
        --target chair \
        --out-dir runs/pol-001

Usage (cached escape hatch):
    python -m embodied_memory.run_hm3d_pol \
        --mode cached \
        --cached-bundle path/to/bundle.npz \
        --n-episodes 5 \
        --out-dir runs/pol-cached

Exit code:
    0  if all 5 pass conditions are met
    1  if any pass condition fails (the runner still writes summary.json)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np

from .episode_runner import EpisodeRunner
from .frontier_planner import FrontierPlanner
from .memory_bridge import EmbodiedMemoryBridge
from .perception import CLIPKeyframeEncoder, SemanticCaptioner


# ----------------------------------------------------------------------
# encoder factory (text side)
# ----------------------------------------------------------------------


def _build_text_encoder(name: str):
    """Return ``(encode_fn, embed_dim)`` where encode_fn: str -> np.ndarray."""
    if name == "mock":
        from dialogue_memory.encoder import MockEncoder
        enc = MockEncoder(embed_dim=384)
        return (lambda s: enc.encode(s)), 384
    # Default: SBERT all-MiniLM-L6-v2 (384-d) — small, fast, MPS-friendly.
    from dialogue_memory.encoder import SentenceTransformerEncoder
    enc = SentenceTransformerEncoder(model_name="all-MiniLM-L6-v2")
    dim = enc.embed_dim
    return (lambda s: np.asarray(enc.encode(s), dtype=np.float32)), int(dim)


# ----------------------------------------------------------------------
# source factory
# ----------------------------------------------------------------------


def _build_source(args):
    if args.mode == "cached":
        from .cached_source import CachedEpisodeSource, write_synthetic_bundle
        bundle = args.cached_bundle
        if bundle is None:
            # Convenience: synth a tiny bundle in out_dir so the user can run
            # the pipeline end-to-end with zero downloads.
            bundle = os.path.join(args.out_dir, "_synthetic_bundle.npz")
            os.makedirs(args.out_dir, exist_ok=True)
            write_synthetic_bundle(bundle)
            print(f"[run_hm3d_pol] no --cached-bundle given; wrote synthetic to {bundle}")
        return CachedEpisodeSource(bundle_path=bundle, n_episodes=args.n_episodes)

    # live mode
    from .habitat_env import HabitatObjectNavSource
    return HabitatObjectNavSource(
        scene_id=args.scene,
        scene_dataset_path=args.scene_dataset_path,
        episodes_path=args.episodes_path,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        target_category=args.target,
        image_hw=(args.image_hw, args.image_hw),
    )


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="HM3D proof-of-life runner")
    parser.add_argument("--mode", choices=["live", "cached"], default="live")
    parser.add_argument("--scene", type=str, default=None,
                        help="HM3D scene id (live mode)")
    parser.add_argument("--scene-dataset-path", type=str, default=None)
    parser.add_argument("--episodes-path", type=str, default=None)
    parser.add_argument("--cached-bundle", type=str, default=None)
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--target", type=str, default="chair")
    parser.add_argument("--out-dir", type=str, default="runs/pol-001")
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--image-hw", type=int, default=256)
    parser.add_argument("--keyframe-every", type=int, default=5)
    parser.add_argument("--decision-period", type=int, default=10)
    parser.add_argument("--n-candidates", type=int, default=4)
    parser.add_argument("--text-encoder", type=str, default="sentence_transformer",
                        choices=["sentence_transformer", "mock"])
    parser.add_argument("--clip-device", type=str, default=None,
                        help="Override CLIP device (mps / cpu / cuda)")
    parser.add_argument("--no-strict-pass", action="store_true",
                        help="Always exit 0 (don't fail on pass-condition misses)")

    args = parser.parse_args(argv)

    if args.mode == "live" and not args.scene:
        parser.error("--scene is required in live mode")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[run_hm3d_pol] mode={args.mode} out_dir={args.out_dir}")

    # 1. text encoder.
    text_encode_fn, text_dim = _build_text_encoder(args.text_encoder)

    # 2. perception.
    clip_encoder = CLIPKeyframeEncoder(device=args.clip_device)
    captioner = SemanticCaptioner()

    # 3. planner.
    planner = FrontierPlanner(
        decision_period=args.decision_period,
        n_candidates=args.n_candidates,
    )

    # 4. memory bridge — seed coarse layer with a small HM3D-Semantics
    # category set so coarse retrieval is non-empty from step 0.
    seed_cats = [
        "chair", "sofa", "couch", "bed", "table", "tv_monitor", "toilet",
        "plant", "sink", "refrigerator",
    ]
    bridge = EmbodiedMemoryBridge(
        text_embed_dim=text_dim,
        visual_embed_dim=clip_encoder.embed_dim,
        text_encode_fn=text_encode_fn,
        cluster_every_n_episodes=3,
        consolidation_top_k=5,
        coarse_seed_categories=seed_cats,
    )

    # 5. source + runner.
    source = _build_source(args)
    runner = EpisodeRunner(
        source=source,
        planner=planner,
        bridge=bridge,
        clip_encoder=clip_encoder,
        captioner=captioner,
        out_dir=args.out_dir,
        target_category=args.target,
        keyframe_every_m=args.keyframe_every,
        max_steps_per_episode=args.max_steps,
    )

    summary = runner.run(args.n_episodes)
    source.close()

    print("\n=== Pass conditions ===")
    for k, v in summary.pass_conditions.items():
        marker = "PASS" if v else "FAIL"
        print(f"  [{marker}] {k}")
    print(f"\nSummary: {json.dumps(summary.to_dict(), indent=2, default=str)}")

    all_pass = all(summary.pass_conditions.values())
    if all_pass or args.no_strict_pass:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
