"""
Download local copies of the captioner + planner used by the ReMEmbR backbone.

Defaults pull LLaVA-1.6 Mistral-7B (captioner) and Mistral-7B-Instruct-v0.3
(planner). Override with environment variables to swap models without
touching code; see ``models/README.md``.

Usage:
    python models/download_remembr_models.py             # default cache
    python models/download_remembr_models.py --cache-dir models/hf_cache
    python models/download_remembr_models.py --captioner-only

Notes:
- Llama-3 is gated; run ``huggingface-cli login`` first if you swap to it.
- Mistral and LLaVA are ungated.
- Total download is ~30 GB; ``--captioner-only`` / ``--planner-only`` let
  you stage one half at a time.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List


def _resolve_models(args) -> List[tuple]:
    captioner = args.captioner or os.environ.get(
        "REMEMBR_CAPTIONER_MODEL", "llava-hf/llava-v1.6-mistral-7b-hf"
    )
    planner = args.planner or os.environ.get(
        "REMEMBR_PLANNER_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"
    )
    out: List[tuple] = []
    if not args.planner_only:
        out.append(("captioner", captioner))
    if not args.captioner_only:
        out.append(("planner", planner))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Download ReMEmbR captioner + planner")
    parser.add_argument("--captioner", type=str, default=None)
    parser.add_argument("--planner", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="HF cache dir (default: ~/.cache/huggingface/hub)")
    parser.add_argument("--captioner-only", action="store_true")
    parser.add_argument("--planner-only", action="store_true")
    parser.add_argument("--revision", type=str, default=None,
                        help="Pin to a specific revision for both models.")
    args = parser.parse_args(argv)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is required. Install via:")
        print("  pip install huggingface_hub")
        return 1

    for role, model_id in _resolve_models(args):
        print(f"\n=== downloading {role}: {model_id} ===")
        try:
            path = snapshot_download(
                repo_id=model_id,
                cache_dir=args.cache_dir,
                revision=args.revision,
                allow_patterns=None,
            )
            print(f"    -> cached at {path}")
        except Exception as e:
            print(f"    !! failed: {type(e).__name__}: {e}")
            if "gated" in str(e).lower() or "access" in str(e).lower():
                print("    (run `huggingface-cli login` and request access on the HF page)")
            return 2

    print("\nAll requested models downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
