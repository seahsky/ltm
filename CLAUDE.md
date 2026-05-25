# LTM-Embodied Agent

## Mission

Build a **lifelong hierarchical long-term memory** system for embodied agents,
extending the **ReMEmbR** backbone (Anwar et al., ICRA 2025) and evaluated in
the **Habitat simulator on the HM3D dataset** (Ramakrishnan et al., 2021).

The four core modules — STM, bio-inspired consolidation, hierarchical LTM
(fine / mid / coarse), and memory-guided plan re-ranking — are specified in
`Research Proposal_Embodied Agent.md`.

**Current state.** Two parallel paths:

- `dialogue_memory/` — text-only LTM prototype validated on MSC multi-session
  dialogue (see `MSC_BENCHMARK_REPORT.md`). Don't touch this when working on
  the embodied side; the bridge subclasses/swaps where needed.
- `embodied_memory/` — Habitat ObjectNav port of the same LTM stack. The
  Phase-1 ablation has been run end-to-end on HM3D `val_mini` (2 scenes ×
  ~30 paired episodes per setting). The LTM fine layer is indexed on the
  **caption TEXT embedding (SBERT)**. (It briefly used CLIP image embeddings
  after the original SBERT-text index went inert — the HM3D semantic sensor
  returns all-zeros so every caption defaulted to "room interior" — but the
  CLIP image-text cosine proved flat/non-discriminative (~0.25 sighting vs
  ~0.228 baseline) and made memory pick wrong instances. With the real Qwen-VL
  captioner the captions are rich again, so SBERT goal-vs-caption similarity is
  the discriminative signal.) Memory **injects waypoint candidates** into the
  frontier planner's pool (option 2 in the design notes), not just reranks.

**Phase-1 outcome.** Memory is in the action loop (~1,400 memory candidates
chosen across 30 S3 episodes vs zero in the rerank-only versions), and the
previously persistent S3 < S1 soft-SPL gap (−0.027) closed to −0.002 (95% CI
straddles zero). Three episodes show clear positive gains (max +0.80 soft-SPL
from a single memory pick); two regress slightly. Binary SPL is still 0
everywhere because the stand-in CLIP+frontier backbone times out at 250 steps
before any episode succeeds, so the Phase-1→Phase-2 gate as written (mean SPL
> 0, p < 0.1) technically still FAILs — but on hard-SPL only, not for "memory
is inert/harmful". See `runs/abl-s{1,2,3}/summary.json` for the latest numbers
and `runs/abl-s{1,2,3}-v{1,2,3}/` for the development history.

**What's still missing**: operationally running the real **ReMEmbR backbone**
on a CUDA host (the code path exists at `embodied_memory/remembr_backbone.py`
and `--backbone remembr` is wired; weights aren't pulled yet — this is why
cosines cap at ~0.27 vs the 0.32 saturation point and binary success stays
at 0) and multi-scene lifelong eval beyond 2-scene minival. The remaining
code seams (consolidator R-weighting of failed episodes, embodied-data
training of `train_predictor` / `train_scorer`, coarse-layer affordance
learning) are all wired in this branch — see `models/README.md`
"Phase-2 operator runbook" for the exact commands.

## Next milestone

**Phase 2: integrate real ReMEmbR.** Phase 1 confirmed the LTM machinery
works end-to-end and produces a small but real soft-SPL signal once memory
gets to *propose* candidates (not just rerank). The remaining headroom — and
the path to non-zero hard SPL — is in the perception+planning backbone.

When ReMEmbR lands, re-run the same 3-setting ablation (`--setting 1|2|3` in
`embodied_memory.run_hm3d_pol`); the harness and paired-bootstrap analyzer
(`embodied_memory/scripts/analyze_ablation.py`) are already wired. Setting 1
will then be paper-faithful (vanilla ReMEmbR with native flat keyframe
memory) rather than the current memory-off stand-in.

Headline metrics: ObjectNav SPL, success rate, steps-to-success, and — for
settings 2–3 — `n_memory_chosen`, `retrieval_hits`, and the paired soft-SPL
delta from `analyze_ablation.py`.

## Running the ablation

```bash
conda activate ltm-embodied
# Setting 1: memory-off baseline (STM/LTM/rerank all disabled)
python -m embodied_memory.run_hm3d_pol --mode live --scene all \
    --setting 1 --n-episodes 30 --target any --out-dir runs/abl-s1
# Setting 2: STM only
python -m embodied_memory.run_hm3d_pol --mode live --scene all \
    --setting 2 --n-episodes 30 --target any --out-dir runs/abl-s2
# Setting 3: full system + memory-injected candidates
python -m embodied_memory.run_hm3d_pol --mode live --scene all \
    --setting 3 --n-episodes 30 --target any --out-dir runs/abl-s3
# Paired-bootstrap delta report
python embodied_memory/scripts/analyze_ablation.py \
    runs/abl-s1 runs/abl-s2 runs/abl-s3
```

`--scene all` auto-discovers minival scenes from
`data/hm3d/datasets/objectnav/hm3d/v1/val_mini/content/`. `--target any`
disables the per-episode category filter so all minival episodes run.

## Repo orientation

- `dialogue_memory/` — LTM modules: `ltm.py` (3-layer FAISS memory),
  `consolidation.py` (importance scoring `I = αR + βU + γN`),
  `pattern_cluster.py` (mid-layer), `reranking.py`, `encoder.py`
  (pluggable embeddings), `train_predictor.py`, `train_scorer.py`,
  `msc_benchmark.py` (eval harness).
- `embodied_memory/` — Habitat ObjectNav port. `memory_bridge.py` glues
  STM → consolidation → CLIP-indexed fine/mid/coarse LTM → memory-injected
  reranking; `frontier_planner.py` is the backbone stand-in;
  `perception.py` exposes CLIP image + text encoders; `episode_runner.py`
  orchestrates; `run_hm3d_pol.py` is the CLI;
  `scripts/analyze_ablation.py` is the paired-bootstrap analyzer.
- `data/` — datasets. MSC for dialogue; HM3D under `data/hm3d/`.
- `run_msc_*.sh` — baseline / quick / full MSC evaluation scripts.
- `Research Proposal_Embodied Agent.md` — method spec (source of truth
  for the research goal).
- `README_LTM_MSC_EVAL.md`, `README_MSC_EVAL.md`, `MSC_BENCHMARK_REPORT.md`
  — dialogue-side architecture notes and early results.

## Conventions

- Method spec lives in the research proposal; treat it as authoritative
  for the embodied design. The dialogue code is reference, not constraint.
- Don't break the dialogue/MSC path while building the embodied path —
  keep them as independent entry points.
