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
(`meta-llama/Meta-Llama-3-8B-Instruct`) without code changes. See
**Lightweight backbone pair** below for the validated sub-24 GB combo.

## Lightweight backbone pair (single 24 GB GPU)

For budget-constrained runs (e.g. a single L4) you can swap both models
to a smaller Qwen pair that fits fp16 on one 24 GB GPU:

| Role | Model | Approx VRAM (fp16) | Disk |
|---|---|---|---|
| Captioner | `Qwen/Qwen2-VL-2B-Instruct` | ~5 GB | ~4.5 GB |
| Planner   | `Qwen/Qwen2.5-3B-Instruct` | ~6 GB | ~6 GB |

Total: ~11 GB VRAM fp16, ~10 GB disk. Both share the Qwen tokenizer
family (`<|im_start|>` chat template), so there's no per-model
template-handling code — the planner's `_format_chat` uses
`tokenizer.apply_chat_template()` and works for any HF chat-tuned LLM.

```bash
export REMEMBR_CAPTIONER_MODEL=Qwen/Qwen2-VL-2B-Instruct
export REMEMBR_PLANNER_MODEL=Qwen/Qwen2.5-3B-Instruct
# dtypes default to float16 — no change needed
```

**Caveat.** This is a different experiment from the default 7B+7B fp16
backbone. The Phase-2 ablation report in `PHASE2_ABLATION_REPORT.md`
uses the default pair; the Qwen pair produces a separate baseline. Both
are scientifically valid — the small-pair run additionally shows
whether the LTM gain holds with a 5× smaller backbone.

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

Holds **embodied-trained** predictor / scorer checkpoints produced by the
G3 trainers (see `Phase-2 readiness check.md`). The CLIs:

```bash
conda activate ltm-embodied

# U_i predictor (surprise / forward-modeling head)
python -m dialogue_memory.train_predictor \
    --embodied runs/abl-s1 runs/abl-s2 runs/abl-s3 \
    --encoder clip \
    --out models/embodied/predictor.pt \
    --epochs 5

# R_i importance scorer (BCE on soft_spl since Phase-1 has 0 binary successes)
python -m dialogue_memory.train_scorer \
    --embodied runs/abl-s1 runs/abl-s2 runs/abl-s3 \
    --encoder clip \
    --label-mode soft_spl \
    --out models/embodied/scorer.pt \
    --epochs 5
```

`--encoder clip` is the operationally-correct choice — it embeds captions
in the **same CLIP-512 joint space** that the embodied LTM is indexed in,
so the trained heads compose with the rest of the pipeline.
`--encoder sbert` (384-d) is supported for fast laptop smoke tests but
its checkpoints don't compose with the CLIP-indexed bridge.

Each CLI saves the best-val checkpoint to `--out` and can be reloaded via
`PredictionTrainer.load()` / `ScorerTrainer.load()`.

## Phase-2 operator runbook (CUDA host)

These steps need a GPU box with ≥24 GB VRAM and the weights in §Download.

```bash
# G1: pull weights (~30 GB)
python models/download_remembr_models.py

# G2: single-episode smoke — confirms ReMEmbR is alive (stub_mode: false)
python -m embodied_memory.run_hm3d_pol --mode live --backbone remembr \
    --setting 3 --n-episodes 1 --target any --out-dir runs/remembr-smoke

# G3: train embodied predictor/scorer on Phase-1 episode JSONs (CPU-OK)
python -m dialogue_memory.train_predictor --embodied runs/abl-s3 \
    --encoder clip --out models/embodied/predictor.pt --epochs 5
python -m dialogue_memory.train_scorer --embodied runs/abl-s3 \
    --encoder clip --label-mode soft_spl \
    --out models/embodied/scorer.pt --epochs 5

# G4: full 3-setting ablation with real backbone
for s in 1 2 3; do
  python -m embodied_memory.run_hm3d_pol --mode live --backbone remembr \
    --setting $s --scene all --n-episodes 30 --target any \
    --out-dir runs/abl-s${s}-remembr
done
python embodied_memory/scripts/analyze_ablation.py \
    runs/abl-s1-remembr runs/abl-s2-remembr runs/abl-s3-remembr
# Look for: "=== phase 2 gate === ... gate: PASS"

# G5 (only if G4 produced ≥1 success): refresh affordance table on real runs
python -m embodied_memory.run_hm3d_pol --mode live --backbone remembr \
    --setting 3 --scene all --n-episodes 30 --target any \
    --affordance-from-runs runs/abl-s1-remembr runs/abl-s2-remembr \
                           runs/abl-s3-remembr \
    --out-dir runs/abl-s3-remembr-affordance
```

## Phase-2 operator runbook — lightweight pair (single 24 GB GPU)

Same flow as above but with the Qwen pair from §Lightweight backbone
pair. Fits comfortably on a single L4. Disk pull is ~10 GB instead of
~30 GB.

```bash
conda activate ltm-embodied
export REMEMBR_CAPTIONER_MODEL=Qwen/Qwen2-VL-2B-Instruct
export REMEMBR_PLANNER_MODEL=Qwen/Qwen2.5-3B-Instruct

# L1: pull weights (~10 GB)
python models/download_remembr_models.py

# L2: single-episode smoke — validates template + tokenizer compat on
# the new pair before burning a full ablation pass.
python -m embodied_memory.run_hm3d_pol --mode live --backbone remembr \
    --setting 3 --n-episodes 1 --target any --out-dir runs/remembr-smoke-qwen
# Confirm runs/remembr-smoke-qwen/episode_000.json shows stub_mode: false
# and at least one parsed TOOL: or ANSWER: line in trace.tool_calls.

# L3: full 3-setting ablation, ~5-8 hr per pass on a single L4.
for s in 1 2 3; do
  python -m embodied_memory.run_hm3d_pol --mode live --backbone remembr \
    --setting $s --scene all --n-episodes 30 --target any \
    --out-dir runs/abl-s${s}-qwen
done
python embodied_memory/scripts/analyze_ablation.py \
    runs/abl-s1-qwen runs/abl-s2-qwen runs/abl-s3-qwen
```

If the smoke at L2 emits `kind: unparseable` for every turn, the small
LLM is drifting from the `TOOL: ...` / `ANSWER: ...` line format —
either add a one-shot exemplar to `sys_prompt` in
`remembr_backbone._llm_propose`, or bump `REMEMBR_MAX_TOOL_CALLS` so
the planner has more retries before the stub fallback engages.
