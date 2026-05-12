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
from typing import List, Optional

import numpy as np

from .episode_runner import EpisodeRunner
from .frontier_planner import FrontierPlanner
from .memory_bridge import EmbodiedMemoryBridge
from .perception import CLIPKeyframeEncoder, SemanticCaptioner
from .remembr_backbone import ReMEmbRBuilder, ReMEmbRConfig, ReMEmbRPlanner


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


_VALID_SPLITS = ("val_mini", "val", "train")


def _resolve_episodes_path_for_split(split: str) -> Optional[str]:
    """Canonical HM3D-ObjectNav layout for a given split. Returns the first
    existing candidate or None if no dataset is on disk."""
    candidates = [
        f"data/hm3d/datasets/objectnav/hm3d/v1/{split}/{split}.json.gz",
        f"data/datasets/objectnav/hm3d/v1/{split}/{split}.json.gz",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


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
    # "any" / "" disables the target-category filter so we see every episode in
    # the dataset (needed when the ablation wants to fill minival with all
    # available episodes, not just chair-targets).
    target = args.target
    if target is None or str(target).strip().lower() in {"", "any", "all"}:
        target = None

    # Resolve effective episodes path. Precedence:
    #   1. explicit --episodes-path  (legacy override)
    #   2. canonical layout for --split  (Phase 2)
    #   3. habitat_env's _default_episodes_path  (prefers val_mini, then val)
    effective_ep_path = args.episodes_path
    if not effective_ep_path and args.split:
        effective_ep_path = _resolve_episodes_path_for_split(args.split)
        if not effective_ep_path:
            raise RuntimeError(
                f"--split {args.split} requested but no dataset found at "
                f"data/hm3d/datasets/objectnav/hm3d/v1/{args.split}/{args.split}.json.gz"
            )

    # `--scene` accepts:
    #   - a single id (legacy):   "00800-TEEsavR23oF"
    #   - comma-separated list:   "00800-TEEsavR23oF,00802-wcojb4TFT35"
    #   - "all" / "minival":      auto-discover from episodes content dir
    scenes: List[str] = _resolve_scene_list(args.scene, effective_ep_path)
    return HabitatObjectNavSource(
        scene_id=scenes if len(scenes) > 1 else scenes[0],
        scene_dataset_path=args.scene_dataset_path,
        episodes_path=effective_ep_path,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        target_category=target,
        image_hw=(args.image_hw, args.image_hw),
    )


def _resolve_scene_list(scene_arg: str, episodes_path: Optional[str]) -> list:
    """Expand the --scene argument into a list of scene ids habitat can load."""
    if "," in scene_arg:
        return [s.strip() for s in scene_arg.split(",") if s.strip()]
    if scene_arg.lower() in {"all", "minival", "auto"}:
        # Auto-discover from the episodes content/ directory adjacent to the
        # dataset json.gz (matches habitat-lab's ObjectNav layout).
        ep_path = episodes_path
        if not ep_path:
            from .habitat_env import HabitatObjectNavSource
            ep_path = HabitatObjectNavSource._default_episodes_path()
        if not ep_path:
            raise RuntimeError("--scene all requested but no episodes dataset on disk")
        content_dir = os.path.join(os.path.dirname(ep_path), "content")
        if not os.path.isdir(content_dir):
            raise RuntimeError(f"--scene all requires {content_dir} to exist")
        scenes = sorted(
            os.path.splitext(os.path.splitext(f)[0])[0]
            for f in os.listdir(content_dir) if f.endswith(".json.gz")
        )
        if not scenes:
            raise RuntimeError(f"--scene all found no .json.gz files in {content_dir}")
        return scenes
    return [scene_arg]


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="HM3D proof-of-life runner")
    parser.add_argument("--mode", choices=["live", "cached"], default="live")
    parser.add_argument("--scene", type=str, default=None,
                        help="HM3D scene id (live mode)")
    parser.add_argument("--scene-dataset-path", type=str, default=None)
    parser.add_argument("--episodes-path", type=str, default=None,
                        help="Explicit override; if set, --split is ignored.")
    parser.add_argument("--split", type=str, default="val_mini",
                        choices=list(_VALID_SPLITS),
                        help="HM3D-ObjectNav split to load (default: val_mini). "
                             "Resolves to data/hm3d/datasets/objectnav/hm3d/v1/<split>/<split>.json.gz.")
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
    parser.add_argument("--setting", type=int, choices=[1, 2, 3], default=None,
                        help="Ablation preset (semantics depend on --backbone):\n"
                             "  --backbone frontier:\n"
                             "    1 = memory-off baseline (STM/LTM/rerank all disabled);\n"
                             "    2 = STM only (LTM + rerank disabled);\n"
                             "    3 = full hierarchical LTM + memory-injected candidates + rerank.\n"
                             "  --backbone remembr:\n"
                             "    1 = vanilla ReMEmbR (flat memory only; hierarchical LTM + rerank off);\n"
                             "    2 = + STM consolidation into hierarchical LTM (rerank still off);\n"
                             "    3 = + memory-injected candidates + reranking on top of ReMEmbR.")
    parser.add_argument("--disable-stm", action="store_true",
                        help="Skip per-episode keyframe buffering (overrides --setting).")
    parser.add_argument("--disable-ltm", action="store_true",
                        help="Skip consolidation, coarse seeding, and LTM retrieval (overrides --setting).")
    parser.add_argument("--disable-rerank", action="store_true",
                        help="Pass through the planner's raw top-1 instead of running the reranker (overrides --setting).")
    parser.add_argument("--backbone", type=str, default="frontier",
                        choices=["frontier", "remembr"],
                        help="Primary candidate generator. 'frontier' uses the "
                             "Phase-1 CLIP+occupancy-grid stand-in (default). "
                             "'remembr' uses a local VLM captioner + LLM agent "
                             "planner; falls back to deterministic stubs when "
                             "model weights are unavailable.")

    args = parser.parse_args(argv)

    if args.mode == "live" and not args.scene:
        parser.error("--scene is required in live mode")

    # Resolve ablation toggles. --setting picks a preset; explicit per-toggle
    # flags can additionally disable a module on top of the preset (so e.g.
    # --setting 2 --disable-rerank is the same as --setting 2; and
    # --disable-rerank alone leaves STM + consolidation on).
    setting_presets = {
        1: (True, True, True),
        2: (False, True, True),
        3: (False, False, False),
    }
    if args.setting is not None:
        s_stm, s_ltm, s_rerank = setting_presets[args.setting]
    else:
        s_stm = s_ltm = s_rerank = False
    disable_stm = bool(s_stm or args.disable_stm)
    disable_ltm = bool(s_ltm or args.disable_ltm)
    disable_rerank = bool(s_rerank or args.disable_rerank)

    # Canonical setting label for the summary if the resolved combo matches a
    # preset; otherwise None (custom mix).
    resolved_setting: Optional[int] = None
    for k, v in setting_presets.items():
        if v == (disable_stm, disable_ltm, disable_rerank):
            resolved_setting = k
            break

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
        disable_stm=disable_stm,
        disable_ltm=disable_ltm,
        disable_rerank=disable_rerank,
        clip_encoder=clip_encoder,
    )
    print(
        f"[run_hm3d_pol] ablation: backbone={args.backbone} "
        f"setting={resolved_setting} disable_stm={disable_stm} "
        f"disable_ltm={disable_ltm} disable_rerank={disable_rerank}"
    )

    # 5. ReMEmbR backbone (only constructed when requested). Uses the CLIP
    # text encoder for caption-embedding indexing so retrievals live in the
    # same joint space as the LTM's coarse-layer category priors.
    remembr_builder: Optional[ReMEmbRBuilder] = None
    remembr_planner: Optional[ReMEmbRPlanner] = None
    if args.backbone == "remembr":
        rmb_cfg = ReMEmbRConfig()
        remembr_builder = ReMEmbRBuilder(
            config=rmb_cfg,
            text_embed_fn=clip_encoder.encode_text,
        )
        remembr_planner = ReMEmbRPlanner(builder=remembr_builder, config=rmb_cfg)

    # 6. source + runner.
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
        run_config={
            "setting": resolved_setting,
            "disable_stm": disable_stm,
            "disable_ltm": disable_ltm,
            "disable_rerank": disable_rerank,
            "backbone": args.backbone,
        },
        backbone=args.backbone,
        remembr_builder=remembr_builder,
        remembr_planner=remembr_planner,
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
