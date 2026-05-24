#!/usr/bin/env python3
"""Probe whether the ReMEmbR backbone loads real weights or falls back to stub.

Fast and Habitat-free: instantiates the ReMEmbR captioner + planner from the
env-configured model ids (``REMEMBR_CAPTIONER_MODEL`` / ``REMEMBR_PLANNER_MODEL``),
runs ONE caption and ONE propose, and reports REAL vs STUB for each component —
printing the actual load error (missing weights, OOM, gated repo, template
mismatch, ...) when it stubs. Run this before a multi-hour ``--backbone remembr``
ablation to confirm ``stub_mode: false`` in ~1 minute instead of discovering a
stub run after the fact.

The per-episode JSON does *not* record the builder/planner stub flag, so this
probe is the canonical "is ReMEmbR actually alive?" check (runbook G2/L2).

Usage::

    conda activate ltm-embodied
    python embodied_memory/scripts/check_remembr.py

Honors the same env vars as the runner (set by ``scripts/race-setup.sh``):
``REMEMBR_CAPTIONER_MODEL``, ``REMEMBR_PLANNER_MODEL``, ``REMEMBR_*_DTYPE``,
``REMEMBR_DEVICE``. Exits 0 only if BOTH components are real.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Repo root on path so `from embodied_memory...` resolves when run as
# `python embodied_memory/scripts/check_remembr.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402

from embodied_memory.remembr_backbone import (  # noqa: E402
    ReMEmbRBuilder,
    ReMEmbRConfig,
    ReMEmbRPlanner,
)


def _banner(s: str) -> None:
    print(f"\n========== {s} ==========")


def main() -> int:
    cfg = ReMEmbRConfig()
    # Surface the real load error instead of silently falling back to stub.
    cfg.strict = True

    _banner("ReMEmbR backbone probe")
    print(f"  captioner_model: {cfg.captioner_model}  (dtype {cfg.captioner_dtype})")
    print(f"  planner_model:   {cfg.planner_model}  (dtype {cfg.planner_dtype})")
    print(f"  device override: {cfg.device or '(auto)'}")
    try:
        import torch

        print(f"  torch:           {torch.__version__}  "
              f"cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"  gpu:             {torch.cuda.get_device_name(0)}  "
                  f"({free / 1e9:.1f} GB free / {total / 1e9:.1f} GB)")
    except Exception as e:  # noqa: BLE001
        print(f"  torch:           NOT importable ({e})")

    # A zero/garbage caption embedding is fine — the probe only checks that the
    # captioner and planner *weights load*, not retrieval quality.
    builder = ReMEmbRBuilder(cfg, text_embed_fn=lambda s: np.zeros(512, dtype=np.float32))
    planner = ReMEmbRPlanner(builder, cfg)

    captioner_real = False
    _banner("[1/2] captioner (Qwen-VL / LLaVA)")
    builder.reset()
    rgb = (np.random.RandomState(0).rand(480, 640, 3) * 255).astype(np.uint8)
    try:
        rec = builder.caption_and_index(
            rgb, agent_position=np.zeros(3, dtype=np.float32), timestep=0
        )
        captioner_real = not builder.stub_mode
        print(f"  REAL — caption: {rec.caption!r}")
    except Exception:  # noqa: BLE001
        print("  STUB — captioner failed to load/caption:")
        traceback.print_exc()

    planner_real = False
    _banner("[2/2] planner (LLM agent)")
    planner.reset()
    try:
        cands = planner.propose(
            goal="chair",
            agent_pose=np.zeros(3, dtype=np.float32),
            agent_yaw=0.0,
            current_step=1,
        )
        # planner.stub_mode ORs in the builder; check the planner's OWN flag.
        planner_real = not planner._stub_mode  # noqa: SLF001 — diagnostic probe
        tr = planner.last_trace
        n_tools = len(tr.tool_calls) if tr else 0
        print(f"  REAL — {len(cands)} candidate(s); {n_tools} tool-call(s)")
        if tr:
            for tc in tr.tool_calls[:4]:
                print(f"    tool_call: {tc}")
    except Exception:  # noqa: BLE001
        print("  STUB — planner failed to load/propose:")
        traceback.print_exc()

    _banner("VERDICT")
    print(f"  captioner: {'REAL' if captioner_real else 'STUB'}")
    print(f"  planner:   {'REAL' if planner_real else 'STUB'}")
    real = captioner_real and planner_real
    if real:
        print("  => ReMEmbR backbone is REAL (stub_mode=false). Safe to run the "
              "full --backbone remembr ablation.")
    else:
        print("  => ReMEmbR backbone is STUB / partial. Fix the errors above "
              "(pull weights via models/download_remembr_models.py, free VRAM, "
              "or hf login) before the ablation — a stub run is not paper-faithful.")
    return 0 if real else 1


if __name__ == "__main__":
    sys.exit(main())
