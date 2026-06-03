# LTM-Embodied Agent — Status & Gap Report

**Date:** 2026-06-02
**Scope:** Where the embodied (HM3D ObjectNav) path stands, and how far it is
from "optimal," benchmarked against the relevant literature.
**Author:** generated from `runs/`, `PHASE2_ABLATION_REPORT.md`, `CLAUDE.md`,
and the `detector-c7` RACE run.

---

## 0. TL;DR

- **The research hypothesis is validated.** The lifelong hierarchical LTM
  produces a **positive, statistically significant** improvement on the
  revisit eval: paired warm **soft-SPL S3−S1 = +0.240, 90% CI [+0.073, +0.417],
  p = 0.008** (Phase C, 2 scenes × {chair, bed}), and the S2 (STM-only)
  decomposition attributes essentially the **entire** gain to the LTM modules
  (S2−S1 = 0.000). On this axis — *"does memory help?"* — we are **at the
  intended result**, not far from it.
- **Absolute navigation performance is far from benchmark SOTA**, but that gap
  is driven by the **zero-shot modular backbone and perception**, not by the
  memory. Our agent is not a trained ObjectNav policy; it is a frozen
  Qwen2-VL captioner + Qwen2.5-7B planner + our LTM.
- **A metric caveat dominates every cross-paper comparison:** our success ring
  is **0.1 m**; the standard HM3D ObjectNav benchmark uses **1.0 m**. Our
  `binary SPL@0.1 m` (~0.04–0.20) is therefore **not comparable** to published
  `SPL@1.0 m` (~0.27–0.34). The comparable metric is **success@1 m**, where our
  system reaches ~0.50–0.67 on the (curated) revisit subset.
- **Current frontier of work:** detector arc c7 (done, verified) → c9 (done,
  awaiting RACE). c9 gates the precise-approach detector on detector–memory
  agreement to fix the wrong-instance / cold-fire regressions c7 exposed.

---

## 1. What we are actually measuring (read this first)

| Axis | Our setup | Standard HM3D ObjectNav |
|---|---|---|
| **Backbone** | Frozen ReMEmbR: Qwen2-VL-2B captioner + Qwen2.5-7B planner (zero-shot, **no ObjectNav training**) | Usually a policy trained on ObjectNav (PIRLNav, OVRL-v2) or a zero-shot modular method |
| **Eval split** | `val_mini`: 2 scenes (`wcojb4TFT35`, `TEEsavR23oF`); revisit subset = 2 cats × cold/warm | Full HM3D val (~2000 episodes, 20 scenes) |
| **Success radius** | **0.1 m** (probed) for "binary SPL"; **1.0 m** for the `success@1m` diagnostic | **1.0 m** geodesic to a goal viewpoint |
| **Primary metric** | **soft-SPL** (continuous, distance-discounted) + paired S3−S1 delta | binary **success** + **SPL@1.0 m** |
| **What the eval isolates** | The *marginal* value of the LTM (S3 full vs S1 memory-off), under a controlled revisit | Absolute task competence of one agent |

**Why soft-SPL is our primary metric.** At a 0.1 m ring, binary SPL is
*perception-bound*: caption-based detection localizes an object at ~1.5 m
visibility but cannot place the agent within 0.1 m of the goal viewpoint, so
binary SPL@0.1 m is near-zero **by construction** regardless of memory quality.
soft-SPL (and `success@1m` / `min_d2g`) measure the navigation signal the eval
can actually resolve. This was a deliberate, documented reframe (Phase-2 Run 7).

> **Implication for "distance to optimal":** there are two different optima.
> (A) the **research** optimum — a robust positive memory effect — which we have
> reached; (B) the **absolute benchmark** optimum (SOTA SPL@1.0 m), which is an
> orthogonal axis bottlenecked by the backbone and perception, not the LTM.

---

## 2. Current results (measured)

### 2.1 Standard `val_mini` single-goal ablation (Phase 2, Run 7, real ReMEmbR)

| Setting | soft-SPL (S1 ref) | binary SPL@0.1 m | note |
|---|---|---|---|
| S1 (memory-off) | **0.089** | ~0 | navigation works (C1 PASS) |
| S3 − S1 (full − off) | **−0.009** (n.s., p≈0.70) | ~0 | memory net-neutral here |

**Verdict:** on single-goal-per-episode ObjectNav, the LTM's recall-a-past-sighting
value rarely applies, so the effect is null. This is a **structural property of
the eval**, not a defect — and it motivated the revisit eval.

### 2.2 Revisit eval (Phase C, Run 9 — the headline positive result)

| Metric | S1 (off) | S3 (full) | Paired Δ (S3−S1, warm) |
|---|---|---|---|
| **soft-SPL (warm)** | — | — | **+0.240**, 90% CI [+0.073, +0.417], **p=0.008** |
| binary SPL@0.1 m | 0.000 | **0.196** | (first non-zero binary SPL) |
| success@1 m | ~0.33 | ~0.67 | +0.34 |
| S2−S1 (STM-only) | — | — | **0.000** (gain is *not* from STM) |
| S3−S2 (LTM modules) | — | — | **+0.240** (the entire effect) |
| Cold S3−S1 (control) | — | — | ~0 (p=0.315 — memory inert with no prior sighting, as expected) |

**Verdict:** the LTM helps when content is discriminative **and** a past
observation is relevant; the gain is cleanly attributed to consolidation +
hierarchical LTM + rerank (proposal modules 2–4), generalizes across both scenes,
and the cold control behaves correctly. **This is the project's core claim, and
it holds.**

### 2.3 Detector arc — `detector-c7` RACE run (2026-06-02, this session)

6-cell matrix (S1/S2/S3 × detector off/on), 16 episodes/setting, 12 warm.

| Cell (warm, n=12) | soft-SPL | binary SPL@0.1 m | success@1 m | min_d2g |
|---|---|---|---|---|
| S1 off | 0.127 | 0.032 | 0.583 | 1.80 m |
| **S3 off** | **0.344** | 0.051 | 0.667 | 1.22 m |
| S1 on (detector) | 0.125 | 0.000 | 0.500 | 1.81 m |
| S3 on (detector) | 0.223 | 0.035 | 0.500 | 1.62 m |

Paired deltas:
- **soft-SPL S3−S1, detector OFF = +0.217**, 90% CI [+0.094, +0.346], p=0.001 → reproduces Phase C.
- soft-SPL S3−S1, detector ON = +0.098 (p=0.039) → **the detector halved the memory gain.**
- `n_detector_approach_success` **0 → 8** → confirmed the c7 counter re-diagnosis (the agent *was* stopping at the waypoint; the metric was mis-wired).

**Verdict:** the c7 precision mechanism works, but binary SPL did **not** recover
(0.035 ≪ 0.30 target) because the detector grounds the **wrong object instance**
~half the time and fired on **cold** episodes (cold S3−S1 = −0.094), truncating
the memory-guided exploration that produced the gain. → motivated **c9**.

### 2.4 Detector arc — `detector-c9` RACE run (gate → **detector OFF wins**)

c9 added a detector–memory agreement gate: commit the precise approach only if the
localized point is within `DETECTOR_MEM_AGREE_M` (2.0 m) of a retrieved same-category
LTM sighting; else fall back to plain STOP. **The gate fires correctly
(`n_detector_gated` 0 → 6) but the detector still net-hurts** — detector OFF is
strictly the best configuration:

| WARM (n=12) | soft-SPL S3 | binary SPL S3 | soft-SPL S3−S1 | cold S3−S1 |
|---|---|---|---|---|
| **detector OFF** | **0.344** | **0.051** | **+0.2343** (p=0.001) | +0.022 |
| detector ON (gated) | 0.231 | **0.000** | +0.1209 (p=0.011) | −0.152 |

**Verdict — detector arc CLOSED.** Across c1–c6, c7 (precise), c9 (gated) the
caption-grounding detector is net-neutral-to-negative under *every* variant. The gate
over-suppresses (S3-det gated 6 of 7 localizations → 1 commit) because caption-grounded
points rarely co-locate with a memory sighting — a **detector-quality ceiling, not a
radius knob**. Headline config is **detector OFF**. The thesis reproduced a **3rd time**
(detector-OFF S3−S1 soft-SPL +0.2343, p=0.001 — Phase C +0.240, c7 +0.217, c9 +0.234).

### 2.5 Bottleneck isolation — oracle ladder + waypoint-arrival STOP (Run 12)

A component-level diagnosis (`diagnose_pipeline.py`, no GPU) showed observe + retrieve
both *work* on warm visits (observation rate 1.0, retrieval on-target 0.66); several warm
episodes reach `min_d2g = 0.00` yet fail — the agent reaches the goal but STOPs elsewhere.
The **oracle ladder** quantified it:

| Cell (warm succ@0.1m) | nomem | ours | oracle-loc | **oracle-stop** | oracle-both |
|---|---|---|---|---|---|
| succ@0.1m | 0.250 | 0.167 | 0.500 | **0.750** | 0.667 |

**`ours → oracle-stop`: 0.167 → 0.750 from a perfect STOP alone** (agent's own nav
untouched) → termination is the entire recoverable gap. But the *realizable* proxy
(`_arrival_stop`: STOP at a confident memory waypoint + caption-confirm) was **net-zero**
across 3 iterations (arrival-1/2/3) — because **the memory waypoint is a *viewing pose*
(~0.5–1.5 m from the object), not the goal point.** Stopping there lands outside the 0.1 m
ring by construction; oracle-STOP reached 0.75 only by using **GT distance** to catch the
transient closest approach.

**Conclusion:** binary SPL@0.1 m is **genuinely localization-bound**, confirmed from two
angles — caption-grounding can't localize the object to 0.1 m (detector arc), and memory
recall gives a viewing pose not the goal (arrival arc). Only GT closes it. The thesis
reproduced a **6th time** (soft-SPL S3−S1 +0.21–0.24, p≈0.001–0.002). Headline config
stays **detector OFF + arrival-STOP off-by-default**; binary SPL@0.1 m needs a real
detector (GroundingDINO/OWLv2/Detic) or a relaxed success ring — both out of thesis scope.

---

## 3. Literature comparison

> ⚠️ **Published numbers below are from the cited papers (HM3D ObjectNav `val`,
> success@1.0 m / SPL@1.0 m) and should be verified against each source. They are
> on the FULL val split with a 1.0 m success ring — NOT directly comparable to our
> 2-scene / 0.1 m binary SPL. They are included to locate our approach class,
> not to claim a head-to-head SPL.**

### 3.1 HM3D ObjectNav `val` — published baselines

| Method | Class | Success@1m | SPL@1m |
|---|---|---|---|
| PIRLNav (Ramrakhya+, CVPR'23) | Trained (IL+RL) | ~64% | ~27% |
| OVRL-v2 (Yadav+, '23) | Trained (RL) | ~65% | ~28% |
| Habitat'22 Challenge winner | Trained | ~60% | ~0.30 |
| **VLFM (Yokoyama+, ICRA'24)** | **Zero-shot (frozen VLM)** | **~53%** | **~30%** |
| L3MVN (Yu+, IROS'23) | Zero-shot (LLM) | ~50% | ~23% |
| ESC (Zhou+, ICML'23) | Zero-shot (LLM) | ~39% | ~22% |
| PixNav (Cai+, '24) | Zero-shot | ~38% | ~21% |
| ZSON (Majumdar+, NeurIPS'22) | Zero-shot | ~26% | ~13% |
| **Ours (S3, revisit warm)** | **Zero-shot LLM + LTM** | **~0.50–0.67** (subset, 1 m) | **n/a** (we report 0.1 m / soft-SPL) |

**Reading this honestly:** our approach is in the **zero-shot modular LLM** class
(no ObjectNav training), where SOTA is ~50–53% success / ~23–30% SPL on full val.
Our `success@1m` on the *curated revisit subset* (~0.50–0.67) is in that band, but
**on an easier, tiny, warm-started subset** — so it is suggestive of
"competitive-class navigation," not a benchmarked claim. We have **not** run the
full val split, and our SPL@1.0 m is not computed.

### 3.2 Relationship to ReMEmbR (the backbone we extend)

ReMEmbR (Anwar et al., ICRA 2025) is **not an HM3D ObjectNav SPL method.** It is a
long-horizon spatio-temporal **video-QA + goal-pose** system evaluated on its own
**NaVQA** benchmark (answer correctness, position error in metres, latency) on a
real Nova Carter robot. We reuse its *architecture* (VLM captioner → retrieval →
LLM planner) as the backbone and **extend its flat, task-scoped memory into a
lifelong hierarchical LTM.** So the right comparison to ReMEmbR is **architectural,
not metric**: our ablation's **S1 (= ReMEmbR-style flat memory / memory-off) vs S3
(= +lifelong hierarchical LTM)** *is* the "what does our extension add over
ReMEmbR" experiment. Answer: **+0.240 soft-SPL on revisit (p=0.008).**

---

## 4. Gap analysis — "how far from optimal?"

### Axis A — the research contribution (LTM value). **~At target.**
- Target: demonstrate a robust, attributable positive effect of the lifelong LTM.
- Status: **achieved** — +0.240 soft-SPL (p=0.008), generalizes across 2 scenes,
  cleanly attributed to LTM (S2−S1=0), correct cold control.
- Remaining: **widen** the estimate (more categories: tv/plant/toilet; more scenes
  — harness already supports `--scenes`/`--categories`) to tighten the CI and
  strengthen the publishable claim. This is *consolidation*, not a missing result.

### Axis B — absolute navigation (benchmark SPL). **Large gap, mostly not the LTM.**
Decomposition of the gap to ~30% SPL@1m zero-shot SOTA:

| Gap source | Evidence | Owner |
|---|---|---|
| **Success radius** (0.1 m vs 1.0 m) | binary SPL@0.1 m ~0 while success@1m ~0.5–0.67 | metric definition — reframed to soft-SPL |
| **Perception / detector** | detector grounds wrong instance ~50%; fires on cold; can't localize to 0.1 m | **c9 (in flight)** + a real object detector |
| **Zero-shot backbone** | no ObjectNav training; planner is obstacle-blind, replan-thrash | swap in a trained policy or VLFM-style value map (out of scope for the memory thesis) |
| **Eval size** | 2 scenes, ≤16 ep/cell → wide CIs | scale the matrix |

**Key point:** none of the Axis-B gap is attributable to the memory being
inert or harmful — every mechanical failure in the chain has been eliminated and
the memory mechanism is verified discriminative and in-the-loop (Phase C:
~1,400+ memory-chosen decisions; fire-rate 0.75–0.83 on warm visits).

---

## 5. Distance-to-optimal scorecard

| Dimension | Optimal | Current | Distance |
|---|---|---|---|
| LTM produces positive, significant effect | p<0.05 positive | **+0.240, p=0.008** | **Reached** |
| Effect attributed to LTM (not STM/backbone) | clean decomposition | **S2−S1=0, S3−S2=+0.240** | **Reached** |
| Generalizes across scenes | ≥2 scenes consistent | 2/2 scenes | **Reached** (widen to strengthen) |
| binary SPL@0.1 m (precise stop) | ≥0.30 | 0.035 (c7) | **Far** — perception-bound; c9 + real detector |
| success@1 m on full val | ~0.50 (zero-shot SOTA) | ~0.50–0.67 on subset; **full val unrun** | **Unknown** — needs full-val run |
| SPL@1.0 m vs zero-shot SOTA (~30%) | ~0.30 | **not computed** | **Unmeasured** |

---

## 6. What would move each remaining lever

1. **c9 (done, awaiting RACE)** — detector–memory agreement gate. Expected to fix
   the c7 regressions (cold harm, wrong-instance). Success = `n_detector_gated>0`,
   cold S3−S1 → 0, detector-ON soft-SPL → back toward +0.217.
2. **Real object detector** (GroundingDINO / OWLv2 / Detic) — the single biggest
   lever for **binary SPL@0.1 m**; caption-grounding is the precision ceiling.
3. **Full-val + SPL@1.0 m run** — to make a *defensible* literature comparison on
   the standard metric (currently we only have a 2-scene subset).
4. **Widen revisit matrix** — more categories/scenes to tighten the +0.240 CI.
5. **(Optional, out of memory scope)** stronger backbone (VLFM-style value map or a
   trained policy) to lift absolute success — but this is orthogonal to the thesis.

---

## 7. Honest limitations

- All positive results are on a **small, curated revisit subset** (2 scenes), with
  **warm starts deliberately seeded near a prior sighting**. The effect is real and
  attributed, but the absolute numbers are not a full-benchmark result.
- **No full HM3D val run** and **no SPL@1.0 m** — so all cross-paper rows in §3 are
  *class-locating*, not head-to-head.
- Published baseline numbers in §3.1 are recalled from the literature and should be
  **verified against the cited papers** before any external use.
- Binary success at 0.1 m is **perception-bound**; treat soft-SPL / success@1m as
  the live signals until a real detector lands.
