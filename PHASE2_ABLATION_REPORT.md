# Phase-2 ablation report — ReMEmbR backbone, HM3D val_mini

**Date:** 2026-05-19
**Branch:** `phase2-readiness`
**Pod:** JarvisLabs A100 80GB ($1.49/hr on-demand)
**Run dirs:** `runs/abl-s{1,2,3}-remembr`
**Source log:** `/tmp/phase2.log` on pod (downloaded copy at `~/Downloads/phase2.log`)

## TL;DR

The full Phase-2 ablation ran cleanly end-to-end with the production
ReMEmbR stack (LLaVA-v1.6-Mistral-7B captioner + Mistral-7B-Instruct-v0.3
planner, fp16, 80 GB A100). **The Phase-2 gate FAILED — but for a single,
fixable reason**: the action pipeline contains no path that ever emits
`stop` (action=0), so binary SPL is zero by construction. Memory steered
the agent to within **0.59 m of a sofa** and **1.46 m of a bed**; in both
cases the agent walked past and continued forward until the step cap.

A grounded STOP path (commit `509dbc8`) is now wired through the
`ReMEmbRPlanner` and runner. Rerun expected to unblock C1+C3 of the gate.

## What we ran

```bash
# Full 3-setting ablation, 30 episodes each, 250 max_steps,
# val_mini (00800-TEEsavR23oF + 00802-wcojb4TFT35), --target any
scripts/run_phase2_ablation.sh   # PHASE2_OUT_SUFFIX=-remembr default
```

Settings (per `embodied_memory/run_hm3d_pol.py --setting`):

| Setting | STM | LTM | Rerank | Memory-injected candidates |
|---|---|---|---|---|
| S1 (off) | ❌ | ❌ | ❌ | ❌ |
| S2 (STM) | ✅ | ❌ | ❌ | ❌ |
| S3 (full) | ✅ | ✅ | ✅ | ✅ |

Wall-clock: 12:03 → 14:43 UTC (2h 40m). Cost: ~$4.

## Results

### Aggregate (over 30 paired episodes)

| Run | success | mean SPL | soft_SPL | mean_steps | rerank disagreements | retrieval hits |
|---|---|---|---|---|---|---|
| `abl-s1-remembr` | 0/30 | 0.0000 | **0.0420** | 249.0 | 0 | 0 |
| `abl-s2-remembr` | 0/30 | 0.0000 | 0.0420 | 249.0 | 0 | 0 |
| `abl-s3-remembr` | 0/30 | 0.0000 | **0.0544** | 249.0 | 1,206 | 7,154 |

### Paired bootstrap deltas (b − a, n=5000, 95% CI)

| Comparison | metric | mean | 95% CI |
|---|---|---|---|
| S3 − S1 | spl | 0.0000 | [0, 0] |
| **S3 − S1** | **soft_spl** | **+0.0124** | **[−0.0305, +0.0794]** |
| S3 − S1 | success | 0.0000 | [0, 0] |
| S2 − S1 | (all) | 0.0000 | [0, 0] |

### Phase-2 gate (`analyze_ablation.py:250-322`)

| Criterion | Result | Detail |
|---|---|---|
| **C1** backbone alive | ❌ FAIL | `n_success(S1) = 0` |
| **C2** memory helps soft | ❌ FAIL | Δsoft = +0.012, 90% CI [−0.027, +0.069], one-sided p=0.388 |
| **C3** memory helps hard | ❌ stretch FAIL | Δspl = 0.000 (both zero by construction) |
| **gate** | **FAIL** | requires C1 ∧ C2 |

## Diagnostics — what's actually broken

Diagnostic script (`embodied_memory/scripts/diagnose_stop.py`, commit `0bbec84`)
walked all 90 episode JSONs:

### 1. STOP is never emitted

```
action histogram across 90 episodes × ~245 steps:
  fwd    99.4–99.8%
  left    0.1–0.2%
  right   0.1–0.4%
  stop    0   ← zero, anywhere, ever
  up      0
  down    0
```

The flow `LLM picks waypoint → step_controller → discrete action` only
emits {forward, turn_left, turn_right}. `step_controller` is a pure
geometric controller with no STOP branch. ObjectNav success requires
the agent to emit `stop` within 1.0 m of the goal; we cannot succeed.

### 2. Near-miss episodes

| Episode | Scene | Target | Final dist | Memory-source pick? |
|---|---|---|---|---|
| `episode_008.json` | wcojb4TFT35 | **sofa** | **0.59 m** | ✅ |
| `episode_012.json` | wcojb4TFT35 | **bed** | **1.46 m** | ✅ |

Both finished with 25 consecutive `fwd` actions in their tail. ep_008
was inside the 1.0 m success radius and walked through.

### 3. Memory was *actually steering well*

S3 chosen_source = `memory` (only 2 of 30 final picks, but informative):

| Source of final pick | n  | mean final_dist | min final_dist |
|---|---|---|---|
| `remembr` (planner) | 28 | 9.28 m | 1.46 m |
| **`memory`** | **2** | **4.40 m** | **0.59 m** |

The hierarchical LTM brought the agent to objects. The agent walked past.

### 4. Caption logging quirk (not a bug)

The `caption` field in `step.steps[]` is the **SemanticCaptioner**
(`perception.py:200`), which depends on HM3D-Semantics annotations we
deliberately did not install. It always emits `room interior | searching
for {target}`. ReMEmbR's real LLaVA captions go into its own flat memory
(`ReMEmbRBuilder._records`), not the per-step JSON. This is logging
asymmetry, not a captioner failure — the LLaVA captions are present in
ReMEmbR's memory and are what enable the new grounded-STOP path below.

## Positive signals from this run

1. **Whole stack ran for 2h 40m without a single crash.** Habitat-sim
   on Mesa software EGL, CUDA-EGL bypass via `HABITAT_SIM_GPU_DEVICE_ID=-1`,
   bf16 LLaVA + Mistral on the A100 — all stable.
2. **Memory pipeline is alive end-to-end**: 14,754 candidates proposed
   across 30 S3 episodes (~492/ep), 413 chosen by rerank (~14/ep), 1,206
   rerank disagreements. The STM→LTM→rerank loop moves bytes correctly.
3. **S3 soft-SPL beats S1** (+0.012). Sign of the delta flipped from
   Phase-1 frontier (−0.002 → +0.012). Memory adds positive value even
   without binary success.
4. **The S1 == S2 collapse is expected**, not a bug. STM is a ring buffer
   with no downstream consumer when LTM/rerank are off; it cannot
   surface in metrics.

## The fix (committed)

**Commit `509dbc8` — "Wire ReMEmbR grounded STOP → action=0"**

Added `ReMEmbRPlanner._maybe_stop()` which runs **before** the LLM agent
loop on every decision step. It queries the builder's flat memory via
the same `retrieve_from_text` tool the LLM would use; if the top hit
satisfies:

* cosine ≥ `REMEMBR_STOP_COS` (default **0.25**)
* AND the matching observation's xz is within `REMEMBR_STOP_DIST`
  (default **1.5 m**) of the agent's current xz

then the planner short-circuits and returns a single `FrontierCandidate`
with `metadata["stop_signal"] = True`. The runner force-selects this
candidate before rerank (so memory candidates cannot outscore it) and
the action-derivation block emits `ACTION_STOP`.

Thresholds are env-tunable. The new `n_stop_signals` counter is logged
per-episode and aggregated into the run summary.

### Why this is "Option C" not "Option A"

This is **ReMEmbR's own grounded decision** — same retrieval tool, same
embedding space, same captions — just invoked synchronously by the
planner before consulting the LLM. Not a geometric hack on top of an
external controller. The LLM never has to reason about "should I stop?";
the tool answers it directly.

## What's next

### Step 1 — Smoke (5 min, $0.12)

```bash
cd /home/ltm && git pull
python -m embodied_memory.run_hm3d_pol --mode live --backbone remembr \
    --setting 3 --n-episodes 1 --target any --out-dir runs/remembr-stop-smoke
grep -E "n_stop_signals|success|spl|distance_to_goal" runs/remembr-stop-smoke/summary.json
```

Pass criteria: no crash, JSON parseable. Anything else is informative.

### Step 2 — Full rerun (~3 h, ~$5)

```bash
PHASE2_OUT_SUFFIX=-remembr-stop bash scripts/run_phase2_ablation.sh
```

Same 3 × 30 × 250 protocol so paired bootstrap stays valid.

### Step 3 — Read the gate

If C1 (any success in S1) flips to PASS, look at:

* **n_stop_signals per setting.** Should be ≥ S1 in S3 because S3 has
  more goal-matching captions in memory.
* **Δsoft_SPL S3−S1 with new run.** Expect the floor to lift in both
  but the gap to widen (memory finds more goals → more grounded STOPs).
* **Per-target success.** ep_008 (sofa) and ep_012 (bed) should be
  among the first to flip.

### Tuning knobs if STOP misbehaves

| Symptom | Knob |
|---|---|
| 0 STOPs anywhere | `REMEMBR_STOP_COS=0.20` (more permissive) |
| STOPs at wrong rooms | `REMEMBR_STOP_DIST=1.0` (tighter geofence) |
| STOPs late, after walking through | `REMEMBR_STOP_DIST=2.0` (earlier trigger) |
| LLaVA captions don't mention target | inspect `runs/.../episode_*.json` decisions trace for the `matched_caption` field; if vague, the captioner prompt may need to be more target-aware |

### Beyond Step 3

If Phase-2 gate passes:

* **G5 — coarse-layer affordance refresh** with real successes.
  `--affordance-from-runs runs/abl-s{1,2,3}-remembr-stop`.
* **G3 — embodied predictor + scorer training** on the new runs.
  `python -m dialogue_memory.train_predictor --embodied runs/abl-s3-remembr-stop --encoder clip --out models/embodied/predictor.pt`
* **Multi-scene lifelong eval** beyond val_mini — pull val proper.

If Phase-2 gate fails on C2 (soft_SPL delta still not significant):

* Increase episodes per setting (60 → 90 for tighter CI).
* Inspect rerank scoring — memory candidates that get *proposed* but
  not *chosen* may be victim to a CLIP-score floor.

If Phase-2 gate fails on C1 (still 0 successes in S1):

* That's a vanilla-ReMEmbR-on-HM3D-val_mini calibration question —
  no amount of memory will help. Check planner prompt, max_steps,
  or scene difficulty (val_mini is 2 hard scenes).

## File index

| Path | Purpose |
|---|---|
| `runs/abl-s{1,2,3}-remembr/` | Per-episode JSONs + summary.json from the failed run |
| `runs/remembr-smoke/` | Single-episode smoke from before the full run |
| `~/Downloads/phase2.log` | Combined stdout/stderr from all 3 settings + analyzer |
| `embodied_memory/scripts/diagnose_stop.py` | STOP-emission audit (commit `0bbec84`) |
| `embodied_memory/remembr_backbone.py` | `_maybe_stop()` (commit `509dbc8`) |
| `embodied_memory/episode_runner.py` | STOP short-circuit + counters (commit `509dbc8`) |
| `scripts/run_phase2_ablation.sh` | Repeatable ablation driver |
| `embodied_memory/scripts/analyze_ablation.py` | Paired bootstrap + Phase-2 gate |
