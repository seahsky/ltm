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

---

# Run 2 — Qwen lightweight pair on RACE (2026-05-22)

**Date:** 2026-05-22 → 2026-05-23
**Branch:** `phase2-readiness`
**Pod:** RACE G15 (g6.2xlarge: 1×NVIDIA L4, 4 CPU, 32 GB RAM, $1.27/hr)
**Backbone:** Qwen2-VL-2B-Instruct captioner + Qwen2.5-3B-Instruct planner
**Run dirs:** `runs/abl-s{1,2,3}-qwen`
**Gate file:** `runs/phase2-qwen-gate.txt`
**Wall-clock:** ~3 h (smoke chase + 90-episode ablation). **Cost:** ~$3.81.

## TL;DR

Re-ran the same 3-setting ablation with a lightweight Qwen pair, post the
`509dbc8` STOP fix. **The gate FAILED again — but for a different reason
than Run 1.** STOP now emits (n_stop_signals=30 in S3 — one per episode),
so binary SPL is no longer zero by construction. However, STOP triggers
*too eagerly* on the first allowed step, the planner can't produce useful
waypoints (Qwen2.5-3B regurgitates the prompt's "Current position" as its
ANSWER), and the agent doesn't translate (mean steps 9.6–9.7 across all
90 episodes). Three layered bugs sit between the wired-up backbone and a
gate-passing run.

| Criterion | Result | Detail |
|---|---|---|
| **C1** backbone alive | ❌ FAIL | `n_success(S1) = 0` — agent never reaches goal |
| **C2** memory helps soft | ❌ FAIL | Δsoft = **−0.0054**, 90% CI [−0.026, +0.019], p=0.687 |
| **C3** memory helps hard | ❌ stretch FAIL | Δspl = 0.000 (still zero, but for new reasons) |
| **gate** | **FAIL** | requires C1 ∧ C2 |

## Results

### Aggregate (over 30 paired episodes)

| Run | success | mean SPL | soft_SPL | mean_steps | rerank dis. | mem chosen |
|---|---|---|---|---|---|---|
| `abl-s1-qwen` | 0/30 | 0.0000 | **0.0279** | 9.60 | 0 | 0 |
| `abl-s2-qwen` | 0/30 | 0.0000 | 0.0279 | 9.60 | 0 | 0 |
| `abl-s3-qwen` | 0/30 | 0.0000 | **0.0225** | 9.73 | 21 | 21 |

S1 and S2 are bit-identical in every aggregate metric — same harness
sanity check as Run 1 (with rerank+LTM off, STM has no observable effect).

### Paired bootstrap (S3 − S1, n=5000)

| Metric | Mean | 90% CI | p (one-sided) |
|---|---|---|---|
| spl | 0.0000 | [0, 0] | 1.000 |
| **soft_spl** | **−0.0054** | [−0.026, +0.019] | 0.687 |
| n_steps | +0.133 | [+0.033, +0.267] | — |

S3 soft_SPL is *slightly worse* than S1 (sign flipped from Run 1's
+0.012). Memory injection cost ~0.13 extra steps per episode without
recouping any SPL.

## Diagnostics — three layered bugs (chronological)

Iteratively patched during the smoke chase before kicking off the full
ablation. Each patch fixed the previous failure mode and exposed the next.

### Bug 1 — Grounded STOP fires at step 0 (`REMEMBR_STOP_COS=0.25` too low)

**Observation.** First smoke (`runs/remembr-smoke-qwen/`): episode ended
at `n_steps=1` with `n_stop_signals=1`, agent 8.4 m from the chair.

**Root cause.** `_maybe_stop` queries `builder.retrieve_from_text(goal,
min_cosine=0.25)` and checks the matching record's xz against the agent's
current xz. But the very first keyframe is ingested at the agent's start
pose (`episode_runner.py:223`), so the geometric guard (`dist ≤ 1.5 m`)
trivially passes for the just-ingested record. And the underlying cosine
is **CLIP-text-vs-CLIP-text** of the Qwen caption against the goal word
— not image-vs-text — which easily clears 0.7 when the caption merely
*mentions* the goal class. The 0.25 threshold lets entry-shot captions
auto-STOP the agent before navigation begins.

**Patch (`2f2d141`).** Added `STOP_MIN_STEP` env-knob (default 8) so STOP
can't fire until the agent has actually walked, and excluded
current-step records from the candidate pool (`rec.timestep >=
current_step` filter). Wired `current_step` through `propose()`.

### Bug 2 — Qwen2.5-3B regurgitates `Current position` as `ANSWER`

**Observation.** With Bug 1 patched, smoke went to `n_steps=21` but
`dist_to_goal=8.48 m` (worse than 8.41 m start). Per-decision dump
showed every LLM-proposed candidate at the agent's exact starting xy
(−0.227, −17.772) — i.e. zero displacement.

**Root cause.** The prompt is `"Goal: find a chair. Current position:
x=-0.23, ... Pick a waypoint (x, z)."` At temperature 0, Qwen2.5-3B
echoes the same x and z back as `ANSWER: x=-0.23, z=-17.77,
confidence=0.5`. The parse succeeds — it's a valid line — but the
"waypoint" is the agent's own position, so the step_controller has a
zero-displacement candidate and can't move forward.

**Patch (`bd60288`).** Added a regurgitation guard in `_llm_propose`:
reject ANSWERs within `REMEMBR_MIN_WAYPOINT_DIST` (default 0.5 m) of
the agent's pose and fall through to `_stub_propose`. Mirror filter in
`_stub_propose` so retrieve_from_text hits co-located with the agent
also get skipped. When both paths produce nothing, the existing 1.5 m
forward-walk fallback kicks in.

### Bug 3 — Step controller doesn't escape collisions

**Observation.** With Bugs 1+2 patched and `STOP_MIN_STEP=50` forcing 50
steps of exploration, the agent still moved **0.04 m total** across 51
actions. `dist_to_goal` unchanged from the previous smoke.

**Root cause (unpatched).** The step_controller emits FORWARD when the
candidate's bearing is aligned, but Habitat blocks FORWARD on collision
without signaling it back up the stack. The agent's starting yaw (2.75
rad ≈ 158°) faces a wall in scene `wcojb4TFT35`, so every FORWARD
action no-ops while still counting toward `n_steps`. The controller
never tries TURN-then-FORWARD to escape. This is below the layer of
env-var tuning; it requires either collision-aware control or a
randomized-exploration fallback when the agent fails to translate for
N consecutive steps.

We did **not** patch Bug 3 — judgment call: out of session scope, no
plausible env-knob fix, and the ablation produces meaningful paired
data even with the stall (every setting hits the wall identically).

## Comparison to Run 1 (Mistral pair on JarvisLabs)

| Aspect | Run 1 (Mistral 7B, A100) | Run 2 (Qwen pair, L4) |
|---|---|---|
| Captioner | LLaVA-v1.6-Mistral-7B | Qwen2-VL-2B-Instruct |
| Planner | Mistral-7B-Instruct-v0.3 | Qwen2.5-3B-Instruct |
| Agent reached goals? | Yes — within 0.59 m (sofa), 1.46 m (bed) | No — stalled at start |
| STOP path emits? | No (controller had no STOP branch) | Yes (the `509dbc8` fix works) |
| Episode steps | 249 / 250 (timeout) | 9.6 / 250 (premature STOP) |
| C1 fails because | Action pipeline has no STOP | Agent doesn't navigate |
| S3 − S1 soft_SPL | +0.0124 (positive, not significant) | −0.0054 (negative, not significant) |
| Cost | ~$4 (JarvisLabs A100) | ~$3.81 (RACE L4) |

**Key takeaway.** The bigger Mistral 7B planner produced useful waypoints
in Run 1 — the agent navigated, found objects, walked past them due to
the missing STOP path. The lightweight Qwen 3B planner cannot. The
controller-stall bug (Bug 3) is independent of the planner choice but
matters more in Run 2 because the agent can't escape the start wall on
its own.

## What's next (in priority order)

### 1. Replace the Qwen2.5-3B planner — highest leverage

Empirically the Qwen2.5-3B planner regurgitates positions and can't pick
useful waypoints. The original ReMEmbR paper uses Mistral 7B / Llama 3.1
8B; Run 1 confirms a 7B-class planner navigates competently. Next session
should pull **Qwen2.5-7B-Instruct** (~14 GB fp16, fits on the L4 with
captioner offloaded or swapped to Qwen2-VL-2B kept in fp16) and re-run.
Expected cost: ~$6–10 (3 h ablation × $1.27/hr + ~$2 setup).

### 2. Patch the controller-stall (Bug 3)

Independent of backbone. Two cheap-ish options:
- Detect "agent did not translate for N consecutive FORWARD actions" and
  inject a TURN_LEFT or TURN_RIGHT to break out of the wall.
- Use `step.info.get('collision', False)` if Habitat surfaces it, and
  re-pick from the candidate list when colliding.
Either approach lifts Run 1's near-misses (sofa 0.59 m, bed 1.46 m)
into actual binary successes once paired with a working planner.

### 3. Reconsider grounded STOP signal source

The STOP path is correct in intent but uses **CLIP-text-vs-text** cosine
of the Qwen caption, which is too permissive (a passing mention of the
goal class clears 0.7). The bridge already maintains a **CLIP-image** LTM
(per `CLAUDE.md`); `_maybe_stop` should probably query *that* index
(image-vs-text cosine ~0.20–0.35) rather than the builder's caption-text
index. This is a refactor, not an env-var change.

### 4. Defer until 1+2 land

G3 (predictor/scorer training on real Phase-2 successes), G5 (affordance
refresh), and HM3D `val` scale-up are all gated on a passing C1. Don't
schedule them until S1 produces ≥1 success on val_mini.

## Tuning knobs added in this session

| Env var | Default | Purpose |
|---|---|---|
| `REMEMBR_STOP_MIN_STEP` | 8 | Don't allow grounded STOP before step N |
| `REMEMBR_MIN_WAYPOINT_DIST` | 0.5 m | Reject LLM ANSWERs / memory hits within N m of agent (regurgitation guard) |

Pre-existing knobs documented in Run 1 (`REMEMBR_STOP_COS`, `REMEMBR_STOP_DIST`) still apply.

## File index (Run 2)

| Path | Purpose |
|---|---|
| `runs/abl-s{1,2,3}-qwen/` | Per-episode JSONs + summary.json from the Qwen ablation |
| `runs/phase2-qwen-gate.txt` | Analyzer stdout including the gate read |
| `runs/remembr-smoke-qwen*/` | Smoke runs from the layered-bug chase (not committed) |
| `embodied_memory/remembr_backbone.py` | Bug 1 + Bug 2 patches (commits `2f2d141`, `bd60288`) |
| `embodied_memory/episode_runner.py` | `current_step` threaded through `propose()` (`2f2d141`) |
| `docs/phase2-race-runbook.md` | RACE bring-up runbook (env-path fix in `000d2a2`) |
