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

**Phase-2 outcome (2026-05-25, real ReMEmbR — see `PHASE2_ABLATION_REPORT.md`
Run 7 for the full arc).** The real ReMEmbR backbone (Qwen2-VL-2B captioner +
Qwen2.5-7B planner) now runs in the loop on a CUDA host (the "weights aren't
pulled" note above is stale). Two fixes made the ablation meaningful:
a **navmesh point-goal controller** (`episode_runner._waypoint_action` steers
to the agent's self-chosen waypoint via Habitat's `ShortestPathFollower`,
replacing the grid-A\* that couldn't route — this fixed navigation), and a
**re-index of the LTM onto discriminative SBERT caption-text** (the CLIP
image-text cosine was flat ~0.25 and made memory pick wrong instances). The
gate was reframed to soft-SPL (binary SPL@0.1 m is perception-bound: caption
detection can't localize to the 0.1 m success radius). **Final 3×30 G4
(`runs/abl-s{1,2,3}-qwen`): C1 PASS (navigation works, soft-SPL S1 ≈ 0.089),
C2 FAIL — the hierarchical LTM is net-neutral (soft-SPL S3−S1 = −0.009, n.s.;
S3 ~18 steps slower).** This is a structural property of the eval, not a bug:
ObjectNav is single-goal-per-episode, so the LTM's recall-past-sighting value
rarely applies. The memory mechanism is verified correct and discriminative.

## Next milestone

**Lifelong / revisit eval.** The Phase-2 result shows the LTM is net-neutral on
the single-goal-per-episode `val_mini`; a positive memory effect needs an eval
where past observations are actually relevant — the same scene traversed
repeatedly with the LTM carrying over, and recurring goals so a past sighting is
retrievable and useful. This is eval-infrastructure work, not a fix to the
memory stack. A separate lever for non-zero **binary** SPL is a real object
detector / precise goal-approach (the 0.1 m localization the captioner can't
provide). The remaining code seams (consolidator R-weighting, embodied-data
training of `train_predictor` / `train_scorer`, coarse-layer affordance
learning) are wired — see `models/README.md` "Phase-2 operator runbook".

The 3-setting ablation + paired-bootstrap analyzer
(`embodied_memory/scripts/analyze_ablation.py`, soft-SPL-primary gate) are the
measurement harness. Headline metrics: soft-SPL S3−S1 (primary), `success@1m` /
`min_d2g` (reach diagnostics), `n_memory_chosen` / `n_remembr_chosen`, and the
honest binary SPL@0.1 m.

## Running the ablation

```bash
conda activate ltm-embodied   # on RACE: source scripts/race-setup.sh
# IMPORTANT: pass --backbone remembr for the real ReMEmbR ablation. Omitting it
# silently uses the 'frontier' stand-in (a wrong-backbone G4 cost a re-run).
for s in 1 2 3; do   # 1=memory-off, 2=STM-only, 3=full system
  python -m embodied_memory.run_hm3d_pol --mode live --scene all --backbone remembr \
      --setting $s --n-episodes 30 --target any --out-dir runs/abl-s$s-qwen
done
# Paired-bootstrap delta report + soft-SPL-primary Phase-2 gate
python embodied_memory/scripts/analyze_ablation.py \
    runs/abl-s1-qwen runs/abl-s2-qwen runs/abl-s3-qwen
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
