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

## ICRA-2027 paper push — AudioGoal anomaly-response arc

The project pivoted (2026-06-10) to an ICRA-2027 paper: **LTM-grounded temporal-context
anomaly response**. The agent maps a home (silent pass → persistent SBERT-indexed LTM),
then an FSD50K anomaly sound (baby cry / alarm / glass break) fires from a source
co-located with a captioned goal object; the warm agent recalls the region. CLAP is a
3-way onset trigger/classifier only (class→category decorative); retrieval reuses
`propose_memory_candidates` verbatim. Audio is rendered offline (`render_rir_grid.py`,
soundspaces env) and convolved O(1) in the live runner — the two-env split holds. Full
arc in the `paper-push-icra2027` memory; M0–M2 (audio path / wiring / dataset builder)
landed; **M3 = the powered milestone (see `PHASE2_ABLATION_REPORT.md` → "AudioGoal M3").**

**AudioGoal M3 outcome (2026-06-18, real ReMEmbR, 2 scenes × 3 cells × S1/S2/S3,
`runs/m3-*`).** Pooled warm soft-SPL **S3−S1 = +0.171 (n=18, 90% CI [+0.070,+0.277],
p=0.002)** — a legitimate independent reproduction of the warm-relevant-LTM thesis in the
new audio task (~12th repro; between priors +0.115/+0.24). Clean decomposition (S2−S1 n.s.
⇒ 100 % LTM-specific S3−S2 +0.172, p=0.004) and cold control ≈ 0. **First significant
binary SPL@0.1 m: +0.139, p=0.003** (code-verified real STOPs; *refines* not retracts the
localization-bound finding — 0.1 m reachable because the cold seed starts at a goal
viewpoint; concentrated in wcojb glass:chair). Heterogeneity HIGH (4/6 cells win, 2
regress): alarm:toilet/TEEsav −0.113 is the documented wrong-instance over-fire (fired 93×,
min_d2g *worsened*) = the SBERT instance ceiling surfacing, not a bug. Caveats: cell-fragile
(leave-best-out → +0.095, p≈0.07); within-scene **not** cross-env; quote both 0.1 m and
1.0 m rings. The analyzer pairing was re-keyed to the renumbering-invariant
`(scene_id, target_category, visit_order)` (stage-1 silently dropped pairs on Habitat
`episode_id` renumbering). Next = **M4 temporal-context head** (default-OFF, A/B'd).

**AudioGoal M4 outcome (2026-06-18, real ReMEmbR, `--temporal` A/B on the M3 warm matrix,
`runs/m3t-*` — see `PHASE2_ABLATION_REPORT.md` → "AudioGoal M4").** The one net-new mechanism
the ICRA plan promised: a **temporal-context head** (recency≈reliability — additive recency
bonus, max `LTM_TEMPORAL_WEIGHT`=0.05, on the SBERT-cos `raw_score` of already-recalled
same-category sightings; env-gated `LTM_TEMPORAL_CONTEXT`, default-OFF). A/B'd (A=baseline S3
`m3-*`, B=temporal-on S3 `m3t-*`, paired on the renumbering-invariant key). **Clean honest
negative: it does not change warm outcomes — warm soft-SPL B−A = −0.0005 (n=18, CI
[−0.0015,+0.0000] — a tie at the floor, NOT a regression), binary SPL +0.0000, succ@1m/steps/
fire-rate bit-identical.** The predicted over-fire DID appear at the *selection* layer (warm
mem_chosen 271→339, +25%, concentrated in the documented wrong-instance `alarm:toilet`/TEEsav
cell = 161) but was harmless: +25% extra picks are credit **re-attribution**, not re-routing
(same goal category, same already-relevant region → same destination). Code-verified mechanism
(3-lens adversarial review, all agree): the bonus is **read-side** (mutates `c.raw_score` after
candidates are emitted, `memory_bridge.py:983-988`) and **never touches write-gating** — the
genuine difference from the R/U importance heads (which changed write-gating → stored more
wrong-instance frames). So this is a *cleaner* negative than coarse (never-chosen): the head was
exercised (fired more) yet inert on outcomes. Cold is head-independent (mem_chosen=0 both arms ⇒
the bonus block can't fire; the +0.068/step-swing is backbone variance; cold n=6 underpowered).
**M4 joins coarse-affordance + R/U as built/correct/conservative-but-not-beating-the-heuristic;
the bottleneck stays SBERT instance discrimination, default stays OFF.** Caveats for the paper:
the eval lacks the head's *design regime* (a changed world — static map between cold-map and
warm-visit, so no stale-vs-fresh signal); B is a *re-run* of M3's A arm (NOT a new repro — count
stays on M3 +0.171); M4-inert proves "recency adds nothing on top," NOT "gain=recall" (that rests
on the M3 decomposition S2−S1 n.s. / S3−S2 +0.172). Also fixed the analyzer's floor-artifact
verdict line (`analyze_revisit._compare_verdict` + `_VERDICT_TIE_BAND`=0.005 → sub-band |Δ| now
reports "tie at the floor"; 4 TDD cases, 33 total). Paper value: pre-empts the "did you try
recency weighting?" reviewer question and sharpens the M3 story.

## Audit caveats (2026-06-08)

A read-only fact-check (the "diagnose-first" program; full version in
`PHASE2_ABLATION_REPORT.md` → "AUDIT CAVEATS") found the positive thesis solid but
five claims overstated/mislabeled — state them precisely:

1. **Provenance.** The committed `runs/abl-s{1,2,3}-qwen` dirs held **Run-2** data
   (verified before removal: `abl-s1-qwen` soft_spl **0.0279**, steps **9.6**;
   `abl-s3-qwen` mem **21**; degenerate `"room interior"` captions), **not** the
   Run-7 numbers (S1 ≈ 0.089) cited against them. Because that data was misleading
   and provenance-mismatched, it was **git-removed on 2026-06-12** (recoverable via
   git history). The real headline data (`revisit-*`, wide matrix, `scorer-*`,
   `predictor-*`) lives only on RACE. `runs/abl-s*-tier1` is a 3-ep pre-Run-1 smoke,
   not Run 19.
2. **Headline magnitude.** The advertised **+0.24** is the **n=12** chair+bed
   subset; the better-powered **n=26** wide-matrix estimate is **+0.115** (p=0.005,
   ~half). Quote both. Distinct datasets ≈ 2–3 (Phase-C n=12, wide n=26), not "8–10
   reproductions" (most are reruns/re-analyses of the same arm).
3. **"Hierarchical 3-layer LTM" → fine + rerank.** Every local S3 run has
   `ltm_mid=false` (mid empty); coarse is seeded but `propose_memory_candidates`
   queries the **fine layer only**. The measured action-path effect is **fine layer +
   memory-injected rerank**, not a working 3-layer hierarchy.
4. **Cold control = two experiments.** Phase-C cold S3−S1 **+0.020, p=0.315**
   (same-category, inert) vs Run-17 cold **+0.157, p<0.001** (cross-category lifelong
   transfer) are NOT contradictory — different controls.
5. **Success ring.** Binary SPL is at **0.1 m** (localization-bound); the benchmark
   uses **1.0 m**, where warm SR = S3 0.667 vs S1 0.333 (Phase-C). Quote both.

**Scope — cross-env transfer is structurally ABSENT (crossenv-2, 2026-06-08, VERIFIED).**
The positive result is within-scene, same-category recall, NOT the proposal's
cross-environment reuse — the injector hard-filters to the current scene
(`memory_bridge.py:829`). The redesigned cross-env eval (`crossenv-2`: home cold
sighting in scene A, one query/category in scene B, `analyze_cross_env.py` role-based,
recall counted by scene_id) ran on the real backbone: **cross-scene recall counter =
1208 (the home sighting IS recalled in scene B) but counted-not-injected → no
waypoint.** A 12-agent adversarial code audit found **no home→away injection path**
across every memory→waypoint emitter (fine seam scene-gated; ReMEmbR flat memory reset
per-episode; coarse layer has no position). The **away S3−S1 = +0.1695 (p=0.004, n=4)
is same-(away-)scene CROSS-EPISODE memory** (the 4 away episodes share one LTM;
within-episode consolidation is MultiON-gated off for single-goal — `episode_runner.py:872`),
**NOT cross-env transfer** — and it is FRAGILE (rides on one episode, an upper bound).
The lone cross-scene READ (rerank S_sim via un-scene-filtered `retrieve()`, weight 0.30)
is a goal-irrelevant non-navigable score perturbation, so home sightings aren't literally
zero-influence but cannot manufacture transfer. **Net: cross-env reuse via the fine layer
is structurally impossible for a waypoint; positive transfer needs step 4 (coarse-affordance)
or a better instance-discriminating embedding** — confirming audit overstatement #1 with a
controlled experiment. (`crossenv-1` first measured this WRONG — n_warm=3 made the analyzer
read within-away revisit, and the recall counter read 0 only because Habitat renumbers
episode_id; both fixed in the redesign.) **`crossenv-3` (`--isolate`, 2026-06-08) makes the
conclusion OVER-DETERMINED:** freezing away-scene LTM writes (`LTM_FREEZE_SCENE`, so each away
episode queries only the home sightings) collapsed away S3−S1 **+0.1695 → +0.0218** (p=0.066,
CI [0,+0.054]) with **mem_chosen=0 on every away episode** (no injectable memory), while the
recall counter rose to **4055**. Per-episode: 3 of 4 away episodes were byte-identical between
runs; **only bed changed (0.639 mem=3 → 0.048 mem=0)** — proving the entire +0.1695 was that one
episode's cross-episode same-scene recall, not transfer. Three independent lines now agree
(12-agent code audit: no injection path; mem_chosen=0 under isolation; delta collapse): **the
LTM recalls the cross-scene sighting but it yields zero navigation benefit → cross-env transfer
is structurally absent → step 4 (coarse-affordance) is the required mechanism.**

**Step 4 coarse-affordance — BUILT + CONSERVATIVE but DOMINATED → arc CLOSED (2026-06-09,
Run 20; see `PHASE2_ABLATION_REPORT.md` Run 20).** The coarse-affordance head (the only
untouched LTM head) was built end-to-end: a position-free `category→preferred_room` prior
grounded to the current scene via a **CLIP zero-shot room classifier** (keyframe image vs
CLIP-text "a photo of a {room}"; calibrated `min_cos=0.292` on real ViT-B/32 cosines ~0.30) +
frontier-grounding, fully instrumented (`_last_coarse_diag`). Two bugs were found AND fixed
by the new instrumentation: (1) thresholds calibrated for a synthetic cosine world (13-agent
adversarial review; defaults 0.20/0.005→0.25/0.02 + a per-run calibration diagnostic), and
(2) a **frontier-grounded self-dedup** (a coarse target's xy IS a frontier's xy, but
`planner_world_xys` included the frontiers → every target deduped at distance 0; symptom
`n_coarse_room_matched`≈12 yet `n_coarse_candidates=0`; fixed by skipping the planner dedup for
frontier-grounded targets). **Final (clip2, real backbone): the head CLIP-tags real cosines and
PROPOSES (1–4/episode) but is NEVER CHOSEN (`n_coarse_chosen=0`) at the rerank weight 0.7 —
it always loses to concrete frontier (≈0.8–1.0) and memory candidates. So `revon-s3` is
byte-identical to `revoff-s3` → ZERO over-fire, warm S3−S1 reproduced at +0.2127 (p=0.002,
~11th repro).** Verdict: the coarse head is correct, CLIP-grounded, and provably conservative
(harmless in warm) but **inert** — the reranker correctly prefers concrete sightings over a
position-free room prior, so cross-env transfer is NOT demonstrated (an honest, well-
instrumented negative, not a bug). Demonstrating coarse value needs a competitive rerank
weight (env-tunable `_COARSE_PRIOR_WEIGHT`) AND a properly-powered brand-new-scene first-visit
eval (the cross-env arm is n=4, same-scene-confounded) — deferred. **All LTM heads are now
exhausted; the genuinely different remaining lever is a better instance-discriminating
embedding/detector (a separate, larger project), not another LTM head.** Drivers
`scripts/archive/race-room-clip.sh` (calibrate→cross-env A/B→revisit over-fire A/B).

**Instance bottleneck — now MEASURED ($0), was asserted.**
`diagnose_sbert_cosines.py` (instance section; `runs/diagnose-instance-sep.txt`):
within-instance caption cosine **0.628** vs between-instance-same-category **0.535**
→ sep **+0.093** (signal EXISTS), but the live query `"there is a {cat}"` collapses
instances to a **0.047** rank gap. **Verdict MIXED: the embedding carries instance
signal; the category query throws it away → the first lever is query/retrieval
construction, NOT a detector.** "Train a detector" is not yet justified.

## Next milestone

**MultiON arc CLOSED — clean null on cold K=3 chains (2026-06-07, Run 15; see
`PHASE2_ABLATION_REPORT.md` Run 15).** The on-thesis answer to Run 7's "single-goal
doesn't reward recall": chain K=3 categories per episode so within-episode recall can
compound (Progress/PPL metrics, gap-by-sub-goal-index analyzer, `scripts/archive/race-multion.sh`).
Four RACE matrices (full1→full4) each diagnosed and killed an absorbing mode — the
decisive on-thesis fix being **memory consumption** (a memory waypoint reached without
advancing the sub-goal is consumed per-sub-goal; full3's wrong-instance recall attractor
re-chose one bad recall 945× and reversed the result), plus a follower-drop re-propose
cooldown and snap-once escape. **full4 (n=14/setting, clean mechanics): Progress
S1 = S2 = S3 = 0.190 exactly, paired S3−S1 = +0.0000, gap-by-index identical at every
index — zero compounding.** Why null when revisit is +0.24: the backbone's exploration
ceiling starves the premise (idx0 found-rate 0.5, idx1 0.07 — the agent rarely survives
to the recall moment), and cold *incidental* sightings are not warm *relevant* priors
(SBERT captions can't tell instances apart). Net thesis statement: **the LTM helps when
past observations are relevant (warm revisit +0.24, 8 reproductions) and is cleanly
neutral when they are incidental (cold MultiON)** — a sharper claim than either result
alone. full2's +0.167 (n=8) was config-sensitivity, not signal. Durable artifacts:
consumption semantics (`REMEMBR_CONSUME_REACHED_MEM`), the escape/counter suite,
`test_{stuck_escape,memory_consume}.py`.

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
warm +0.2357, p=0.001). Driver `scripts/archive/race-train-scorer.sh`.

**Run 14 (2026-06-04) confirms + refines this at scale.** Widened the revisit matrix to
6 categories × 2 scenes (n=26 warm / 10 cold) and added a **direct paired
trained-vs-heuristic test** (`analyze_revisit.py --compare`). Head-to-head: trained − heuristic
warm soft-SPL = **+0.060, 90% CI [−0.019, +0.150], p=0.116 — NOT significant (parity)**, so
"heuristic at/near ceiling" holds on soft-SPL. The one *significant* head-to-head difference
favors the **heuristic** (binary SPL@0.1 m −0.043, CI excludes 0; succ@1m 0.577 vs 0.538). But
two Run-13 artifacts were corrected: the n=4 cold regression was **noise** (n=10 cold trained ≥
heuristic), and the trained head is **~20% more step-efficient** (94.5 vs 119.7). Net: a
precision-vs-efficiency tradeoff at parity, not a winner — **heuristic stays the default when
exact arrival matters; scorer-head lever stays CLOSED**.
**Predictor (U) head training CLOSED too — it REGRESSES warm (2026-06-08, Run 18;
`PHASE2_ABLATION_REPORT.md` Run 18).** The β-weight `U` head was trained as a *self-supervised*
next-caption forward model (`U=(1−cos(pred,actual))/2`), deliberately avoiding the scorer-d1
weak-label trap. Full fresh wide-matrix run (build eval set → 30 training eps → S1/S3-heur/S3-
trained-u). **Trained U REGRESSES the thesis-relevant warm condition: warm soft-SPL S3−S1 +0.112
(heuristic, 10th repro, p=0.011) → +0.061 (trained, p=0.069, CI straddles 0)** — the *same
over-fire signature as scorer-d1*: warm `mem_chosen` 856→1165 (+36%, the forward model assigns
high surprise to most captions, inflating `I` broadly → over-retrieval → thrash), warm `succ@1m`
collapses 0.577→0.385 (exactly back to S1), min_d2g worse; ~8% fewer steps (same efficiency-for-
precision trade as the scorer). Cold cross-category transfer survives (both p=0.001).
**Importance-head training lever EXHAUSTED — three U fixes all REGRESS via one mechanism (2026-06-08,
Run 19; `PHASE2_ABLATION_REPORT.md` Run 19).** After Run 18, three research agents diagnosed the U
head (grounded in code): (1) a confirmed train/serve normalization SKEW (dataset fed UN-normalized
SBERT targets but inference is L2-normalized → corrupt cosine readout); (2) no discriminative spread
(U≈0.30±0.05, a flat offset); (3) the deepest — forward-model surprise is novelty-like, redundant
with γN, orthogonal to goal-relevance (retrieval is **pure caption-goal cosine + position→waypoint**).
**Tier-1 fix (e2, commit 05e0cef): L2-normalize training pairs + cosine loss + per-episode U
calibration (`_calibrate_uniqueness_pool`) → NEGLIGIBLE (warm S3−S1 +0.0613→+0.0609).** **Tier-3
(p1, commit c491a36): a goal-PROXIMITY U** (log per-step geodesic `distance_to_goal`; binary ≤1.0m
label; scorer in the U slot via `--utility-scorer-ckpt`) **also REGRESSES — warm soft-SPL +0.067,
binary SPL −0.016 (NEGATIVE, only head to hurt binary), succ@1m 0.346 < memory-off S1 0.385.** All
three U formulations land at warm soft-SPL ≈ +0.06 (half the heuristic +0.112) with the same
**over-fire** (mem_chosen ~1130 vs heuristic 856). Unifying cause: **the SBERT caption embedding
can't distinguish object instances, so any head storing *more* goal-ish frames surfaces more
*wrong-instance* candidates at retrieval → over-fire → worse arrival.** The heuristic U wins because
it is conservative *and* R-derived; conservatism is the right bias in an instance-ambiguous space.
Verdict: **the importance-head training lever is CLOSED (5 angles: R scorer-d1/d3; U surprise /
calibrated / proximity — all ≤ heuristic). The bottleneck is the embedding's instance discrimination,
not the importance signal.** Heuristic importance stays default. Drivers
`scripts/archive/race-train-{predictor,utility-scorer}.sh`. The only untouched LTM head is the **coarse-layer
affordance** head; the genuinely different remaining lever for further gains is a better
embedding/detector (instance discrimination — a separate, bigger project).
**Run 17 (2026-06-07) completed the wide matrix's module attribution** (`race-wide-s2.sh`,
S2 on the exact scorer-d3 dataset): warm S2−S1 = +0.012 n.s. / binary exactly 0.000, warm
S3−S2 = **+0.103, p=0.017** (binary +0.074, p=0.039) — **~90% of the soft-SPL effect and
100% of the binary-precision effect is LTM-specific**, reproducing Phase C's decomposition
at n=26 (9th warm-thesis reproduction). Cold S3−S1 +0.157 is cross-category lifelong
transfer (scene mapped while hunting other categories), not a leak.

**Embodied binary-SPL work CLOSED — localization-bound (2026-06-03, Run 12; see
`PHASE2_ABLATION_REPORT.md` Run 12).** A component diagnosis (`diagnose_pipeline.py`)
showed observe+retrieve work on warm visits; the **oracle ladder**
(`scripts/archive/race-oracle-ladder.sh`, `--oracle-stop`/`--oracle-location`) proved
**termination is the entire recoverable gap** — a perfect STOP lifts warm succ@0.1m
0.167 → **0.750**. But the realizable proxy (`_arrival_stop`: STOP at a confident memory
waypoint + caption-confirm; `ARRIVAL_STOP_COS`/`ARRIVAL_STOP_RADIUS`) was **net-zero
across 3 iterations** because **the memory waypoint is a VIEWING POSE (~0.5–1.5m from the
object), not the goal point** — stopping there is outside the 0.1m ring by construction.
**So binary SPL@0.1m is genuinely localization-bound** (confirmed twice: caption-grounding
can't localize to 0.1m, memory recall is a viewing pose). The LTM thesis reproduced a
**6th time** (warm soft-SPL S3−S1 +0.21–0.24, p≈0.001). Headline config: **detector OFF,
arrival-STOP off-by-default**; the comparable-metric headline is the **true benchmark SR
(STOP within 1.0 m), verified by log recompute (Run 16, 2026-06-07): warm S3 0.667 vs S1
0.333 on the Phase-C harness (exact — stop_rate 1.0 there, so reach = at-STOP) and 0.500
vs 0.308 on the wide 6-category matrix** (the older "~58%" was STOP-independent reach).
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
