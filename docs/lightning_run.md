# Lightning AI run — query-construction fix validation

Runbook for validating the **query-side instance fix** on a fresh Lightning AI Studio (or any single-GPU Linux box).

## What this validates

The powered 97-cell val matrix made the memory soft-SPL a **null at scale** (warm S3−S1 +0.020, n=48, p=0.29), diagnosed to **wrong-instance over-fire**: the bare category query `"there is a {cat}"` scores every same-category instance about equally (rank gap ~0.047), so retrieval can't prefer the goal instance.

The `$0` encoder-swap gate returned **GO-QUERY**: a stronger text embedder does *not* lift the live category-query gap, but querying with the recalled **prior-sighting captions** does (caption-query gap +0.080 on the live encoder, +0.102 on bge).
So the lever is **query construction, not the embedder** — no re-embed, no backbone swap.

This run A/B's that fix live: per cell it runs `S1` (memory-off), `S3` (baseline, bare category query), and `S3qx` (`LTM_QUERY_EXPANSION` on), then a paired soft-SPL compare of `S3qx` (B) vs `S3` (A).

**Honest caveat.**
The `$0` gate measured the *ceiling* — it queried with the goal instance's own captions (leave-one-out).
The realizable fix is pseudo-relevance feedback: it refines the query toward whatever the *first-pass category query* surfaced, which can itself be the wrong instance.
So the realizable gain is genuinely uncertain — it can sharpen toward the true instance (help) or amplify a wrong first pass (hurt).
That is exactly why it needs a live A/B; the gate cannot settle it.
`prf` (keeps the category anchor, conservative) is the default arm; `caption` (pure centroid) is more aggressive.

## Two-environment split (why a free box works)

- **Main loop** (`ltm-embodied`): headless `habitat-sim` + CLIP/CLAP/SBERT + faiss + transformers (ReMEmbR: Qwen2-VL-2B captioner, Qwen2.5-7B planner).
  Runtime audio is pre-rendered RIR convolved in O(1) — **SoundSpaces is NOT imported at runtime.**
- **Grid render** (`soundspaces-spike`): SoundSpaces 2.0, used **once** per cell to render the RIR grid at the anomaly source.

A free GPU box only needs the main-loop env for the bulk of the compute; the render env is a one-time per-cell cost.

## One-time setup

```bash
git clone <repo> ~/ltm && cd ~/ltm && git checkout lifelong-revisit-eval

# 1. main-loop env (name MUST be ltm-embodied — the drivers source race-setup.sh)
conda env create -f embodied_memory/environment.yml      # → env "ltm-embodied", python 3.9, habitat-sim 0.3.3

# 2. RIR-render env (one-time build, ~1h)
bash scripts/race-soundspaces-spike.sh                    # → env "soundspaces-spike"

# 3. Matterport token for HM3D mesh download (val_mini meshes may already ship; val needs this)
printf 'MATTERPORT_TOKEN_ID=...\nMATTERPORT_TOKEN_SECRET=...\n' > .env
```

`scripts/race-setup.sh` assumes `~/miniconda3` and an env named `ltm-embodied`; if Lightning's conda lives elsewhere, either symlink it or set `LTM_ENV_NAME` and adjust `MINICONDA` in that script.

## Run the validation

Scoped smoke on the two `val_mini` scenes — baseline **and** query-fix in one pass:

```bash
nrun bash scripts/race-anomaly-response-matrix.sh \
    --split val_mini --query-expansion prf --tag-prefix qfix
```

Each cell runs `S1`, `S3`, `S3qx`; the driver then prints the pooled paired compare.
`--query-expansion caption` runs the aggressive arm instead.
The child self-guards: if `n_query_expanded == 0` (expansion never fired, e.g. a cell that never recalls) the arm aborts loudly rather than reporting a vacuous tie.

Full `val` scale-up (adds mesh download; ~40–50 h serial, resumable):

```bash
nrun bash scripts/race-anomaly-response-matrix.sh --split val --download --max-cells 0 --tag-prefix qfixv   # fetch meshes, then exit
nrun bash scripts/race-anomaly-response-matrix.sh --split val --query-expansion prf --tag-prefix qfixv       # the matrix
```

### Resumability and session length

- **Resumable.** Re-run the *same* command; a cell is skipped only once all its arms (`S1`, `S3`, `S3qx`) have a `summary.json`. A cell missing the expansion arm is re-run.
- **Lightning free tier fits.** The 10-minute idle auto-sleep does not fire while a job is actively computing, and background execution keeps the job alive after the browser closes. The real limit is monthly GPU credits (~25 L4-hours), not session length. Because the run is resumable, it chains across sessions/restarts if credits or a per-session cap interrupt it. Verify the exact free-GPU continuous window on the dashboard at signup.

## Reading the result

Two distinct verdicts land in `runs/qfix*-...`:

1. **`[7/8] pooled soft-SPL`** (`runs/<prefix>-matrix-analysis.log`) — baseline warm `S3−S1`, the LTM effect. Reproduces the powered number for these cells.
2. **`[8/8] query-fix A/B`** (`runs/<prefix>-queryexp-compare.log`) — paired `S3qx − S3`. **The headline for this run:** does the query change beat the wrong-instance over-fire baseline? Positive = the realizable fix works; a tie/negative = the null is deeper than query construction (next lever: query with the *recorded cold-pass goal caption* rather than PRF over ambiguous first-pass hits).
3. **`[7/7] controller census** — unchanged systems headline (interrupt→investigate→resume→report); the query fix does not touch the controller.

`n_query_expanded` in each `S3qx` `summary.json` confirms the arm actually fired.
