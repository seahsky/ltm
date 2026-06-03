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

**Phase-3 outcome (2026-05-27, lifelong/revisit eval — see
`PHASE2_ABLATION_REPORT.md` Run 8).** The Run-7 "net-neutral" verdict was
**confounded by a captioning bug**, not a structural property. The LTM fine layer
was indexed via `episode_runner._build_keyframe` → `SemanticCaptioner`, which —
because HM3D's semantic sensor returns all-zeros — emitted a degenerate
`"… room interior"` caption for *every* keyframe. So memory had no discriminative
content (goal-query cosine pinned ~0.17, below the 0.23 bar → memory never fired).
The rich Qwen-VL caption went only to ReMEmbR's separate flat memory. **Fix: when
`backbone==remembr`, index the LTM on the VLM caption** (plus a fix chain: SBERT
L2-normalization, proper cosine in `propose_memory_candidates`, `spl_guard` for
cold-start-on-goal, same-category reachable warm starts). On the controlled-start
**revisit** smoke (`wcojb4TFT35`, chair+bed): **Gate A GREEN — warm soft-SPL
S1 0.079 → S3 0.375, paired Δ +0.296, 90% CI [+0.100,+0.517], p=0.002; first
non-zero binary SPL (warm S3 0.378); memory fire-rate 0.833.** The LTM helps when
content is discriminative AND past observations are relevant. This recontextualizes
all prior embodied results where the semantic sensor was zero.

**Phase-C outcome (2026-05-27, multi-scene generalization — see
`PHASE2_ABLATION_REPORT.md` Run 9).** Scaled the revisit eval to the full
**3-setting ablation (S1/S2/S3) across 2 scenes (`wcojb4TFT35`, `TEEsavR23oF`) ×
{chair,bed}**, 16 episodes/setting. **Gate A = (a) GREEN, generalizes:** warm
paired soft-SPL **S3−S1 = +0.240, 90% CI [+0.073,+0.417], p=0.008** (n=12 pairs),
binary SPL 0→0.196 on both scenes, success@1m 33%→67%. The added **S2 (STM-only)
decomposition cleanly attributes the gain to the LTM**: S2−S1 = exactly 0.000
(STM alone does nothing) so the entire effect is **S3−S2 = +0.240** (consolidation
+ hierarchical LTM + rerank, the proposal's modules 2–4); cold control S3−S1 ≈ 0
(p=0.315, memory inert without a prior sighting). The harness added
`episode_order.pin_episode_order` (pins shuffle=False + group_by_scene=True for
multi-scene cold-first ordering) and the S2 delta reporting in `analyze_revisit.py`;
the memory stack itself was unchanged from Run 8.

## Next milestone

**Scorer-head training CLOSED — heuristic R is at/near the ceiling (2026-06-03, Run 13;
see `PHASE2_ABLATION_REPORT.md` Run 13).** On-thesis attempt to *train* the LTM's own
importance head (R in `I = αR + βU + γN`, which gates the top-k fine-layer writes that
retrieval queries against) rather than re-measure the frozen stack. Wired the dormant
`train_scorer` checkpoints into inference for the first time (`load_scorer` →
`DialogueConsolidation(relevance_scorer=…)` → bridge `scorer_ckpt` → `--scorer-ckpt`,
with a loud raise on encoder-dim mismatch). **Two labels: (d1) episode soft-SPL is
unlearnable for a caption head (Val Acc flat 0.32) → R goes inert → memory over-fires
(210→625) and thrashes → REGRESSES (warm soft-SPL S3−S1 +0.236→+0.125). (d2) a
per-keyframe `goal_object` label (caption names an HM3D goal object) IS learnable (Val Acc
0.32→0.76), kills the over-fire, and RECOVERS to heuristic-competitive on warm (+0.194 vs
+0.236 — statistical tie) — but does NOT beat it and significantly HURTS cold-start
(−0.152).** Verdict: the hand-tuned heuristic R is at/near the ceiling; training doesn't
beat it here. The exercise proved the consolidation-importance path is load-bearing (a bad
R breaks the LTM, a good R restores it) and reproduced the thesis an **8th time** (heuristic
warm +0.2357, p=0.001). Driver `scripts/race-train-scorer.sh`. Remaining on-thesis pushes
(orthogonal): widen the revisit matrix (higher-n estimate) or apply the now-proven
train+wire pattern to the predictor (U) / coarse-affordance heads.

**Embodied binary-SPL work CLOSED — localization-bound (2026-06-03, Run 12; see
`PHASE2_ABLATION_REPORT.md` Run 12).** A component diagnosis (`diagnose_pipeline.py`)
showed observe+retrieve work on warm visits; the **oracle ladder**
(`scripts/race-oracle-ladder.sh`, `--oracle-stop`/`--oracle-location`) proved
**termination is the entire recoverable gap** — a perfect STOP lifts warm succ@0.1m
0.167 → **0.750**. But the realizable proxy (`_arrival_stop`: STOP at a confident memory
waypoint + caption-confirm; `ARRIVAL_STOP_COS`/`ARRIVAL_STOP_RADIUS`) was **net-zero
across 3 iterations** because **the memory waypoint is a VIEWING POSE (~0.5–1.5m from the
object), not the goal point** — stopping there is outside the 0.1m ring by construction.
**So binary SPL@0.1m is genuinely localization-bound** (confirmed twice: caption-grounding
can't localize to 0.1m, memory recall is a viewing pose). The LTM thesis reproduced a
**6th time** (warm soft-SPL S3−S1 +0.21–0.24, p≈0.001). Headline config: **detector OFF,
arrival-STOP off-by-default**; `success@1m ≈ 0.67` warm is the comparable-metric headline.
Remaining binary-SPL levers (real <0.1m detector; relaxed success ring) are out of thesis
scope. On-thesis ways to push the *positive* result: widen the revisit matrix or train the
LTM heads (below).

**Detector arc CLOSED — headline config is detector OFF (2026-06-02, Run 11; see
`PHASE2_ABLATION_REPORT.md` Run 11).** The goal-detector binary-SPL push (c1–c9) is
finished. c7 (precise 0.25 m approach + a counter re-diagnosis: Habitat's follower
signals arrival via the STOP action, so `n_detector_approach_success` was a mis-wired
metric — confirmed 0→8) and c9 (detector–memory agreement gate, `DETECTOR_MEM_AGREE_M`,
`n_detector_gated`) both ran the full 6-cell RACE matrix. **The caption-grounding detector
is net-neutral-to-negative under every variant — detector OFF strictly dominates** (WARM
soft-SPL S3 0.344 vs ON 0.231; binary SPL 0.051 vs 0.000). The gate fires correctly
(`n_detector_gated` 0→6) but over-suppresses (S3 gated 6/7 → 1 commit): a **detector-quality
ceiling, not a radius knob**. The detector code stays env-gated and OFF by default; the
standard non-detector path is unaffected. **The LTM thesis reproduced a 3rd time** (OFF
S3−S1 soft-SPL +0.2343, p=0.001 — Phase C +0.240, c7 +0.217, c9 +0.234). Higher **binary**
SPL@0.1 m now needs a strong object detector (GroundingDINO/OWLv2/Detic) — a separate
project on the orthogonal axis — or is accepted as perception-bound. On-thesis ways to push
the *positive* result further: widen the revisit matrix (below) or train the LTM's own heads
(`train_predictor` / `train_scorer` / coarse-affordance) on embodied data.

**Fold the revisit eval into the standard harness — DONE (2026-05-27).** Phase C
confirmed the LTM effect generalizes; the visit-order revisit analysis is now a
**first-class mode of the standard analyzer**: `python
embodied_memory/scripts/analyze_ablation.py --revisit <run_dirs>` (one front
door). It lazily delegates to `analyze_revisit.py`, which **remains runnable
standalone** as a back-compat alias (identical output). Opt-in only — the
standard `analyze_ablation` Phase-2 gate is unchanged unless `--revisit` is
passed. The controlled-start dataset build stays revisit-specific
(`make_revisit_smoke.py` / `scripts/race-revisit.sh`); only the driver's final
analysis call moved to `analyze_ablation --revisit`. A separate lever for higher
**binary** SPL is still a real
object detector / precise goal-approach (Phase C's binary SPL 0.196 is perception-
bound at the 0.1 m success radius; memory gets the agent to the goal region, not
always within 0.1 m). Optional: widen the revisit matrix (tv_monitor / plant /
toilet; more scenes — the driver supports it via `--scenes` / `--categories`) to
tighten the estimate. The remaining code seams (consolidator R-weighting,
embodied-data training of `train_predictor` / `train_scorer`, coarse-layer
affordance learning) are wired — see `models/README.md` "Phase-2 operator runbook".

**Revisit harness:** `scripts/race-revisit.sh` drives
`make_revisit_smoke.py` → `run_hm3d_pol.py --episodes-path --scene all` →
`analyze_ablation.py --revisit` (warm-only paired soft-SPL bootstrap + S2-STM-only
decomposition + Gate-A a/b/c verdict; `analyze_revisit.py` is the standalone
alias for the same output). A bare `bash scripts/race-revisit.sh
--tag <t>` runs the Phase-C default: both val_mini scenes × {chair,bed} ×
{S1,S2,S3}, n_warm 3 (48 episodes). The single-goal
3-setting ablation + `analyze_ablation.py` (soft-SPL-primary gate) remain the
val_mini harness. Headline metrics: soft-SPL S3−S1 (primary), `success@1m` /
`min_d2g` (reach diagnostics), `n_memory_chosen` / `n_remembr_chosen`, binary
SPL@0.1 m.

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
