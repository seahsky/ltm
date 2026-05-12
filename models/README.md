# Model weights for the ReMEmbR backbone

This directory holds local copies of the captioner and planner used by
`embodied_memory/remembr_backbone.py`. Weights are gitignored — only this
README, the download script, and `embodied/` (where embodied-trained
predictor/scorer weights will land) are tracked.

## Required models

| Role | Default model | Approx VRAM (fp16) | Disk |
|---|---|---|---|
| Captioner | `llava-hf/llava-v1.6-mistral-7b-hf` | ~15 GB | ~14 GB |
| Planner   | `mistralai/Mistral-7B-Instruct-v0.3` | ~14 GB | ~14 GB |

Total: ~30 GB VRAM in fp16. Both can be quantized (bitsandbytes 4-bit) for
a ~12 GB total at the cost of some captioning quality. The runner pulls
the model ids from environment variables, so you can swap to a smaller
captioner (e.g. `llava-hf/llava-1.5-7b-hf`) or planner
(`meta-llama/Meta-Llama-3-8B-Instruct`) without code changes.

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `REMEMBR_CAPTIONER_MODEL` | `llava-hf/llava-v1.6-mistral-7b-hf` | HF repo id for the VLM |
| `REMEMBR_PLANNER_MODEL`   | `mistralai/Mistral-7B-Instruct-v0.3` | HF repo id for the LLM |
| `REMEMBR_CAPTIONER_DTYPE` | `float16` | One of `float16` / `bfloat16` / `float32` |
| `REMEMBR_PLANNER_DTYPE`   | `float16` | Same options |
| `REMEMBR_DEVICE`          | (auto) | Override device — `cuda`, `cuda:1`, `mps`, `cpu` |
| `REMEMBR_MAX_TOOL_CALLS`  | `4` | Per-decision-step LLM tool-call budget |
| `REMEMBR_MAX_CAPTION_TOKENS` | `64` | Max new tokens per caption |
| `REMEMBR_MAX_PLANNER_TOKENS` | `256` | Max new tokens per planner turn |
| `REMEMBR_STRICT`          | `0` | Set `1` to crash instead of falling back to stub mode |

## Download

```bash
conda activate ltm-embodied
# Make sure transformers + huggingface_hub are installed (already in the env).
python models/download_remembr_models.py
```

The script uses `huggingface_hub.snapshot_download` so the cached weights
live under `~/.cache/huggingface/hub/`, not inside this repo. Pass
`--cache-dir models/hf_cache` to keep them repo-local (still gitignored).

You will need a Hugging Face access token for Llama-3 (Mistral is
ungated). `huggingface-cli login` once before running.

## Stub mode

`remembr_backbone.py` falls back to deterministic stub outputs when:

- `transformers` / `torch` are not importable, OR
- `from_pretrained` raises (missing weights, no GPU memory, etc.)

This lets `--backbone remembr` run end-to-end in CI / on a laptop without
GPUs. Set `REMEMBR_STRICT=1` to disable the fallback and surface the real
error instead. The per-episode JSON log records `stub_mode: true` on
every decision so you can never confuse a stub run with a real one.

## `embodied/`

Reserved for **embodied-trained** predictor / scorer checkpoints
produced by Step 3b of the Phase-2 plan (see CLAUDE.md). Empty until
`dialogue_memory/train_predictor.py --embodied` or
`train_scorer.py --embodied` has run on `runs/<dir>` data.
