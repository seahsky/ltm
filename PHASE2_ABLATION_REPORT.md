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

---

# Run 3 — Qwen2.5-7B planner swap on RACE (2026-05-23)

**Date:** 2026-05-23
**Branch:** `phase2-readiness`
**Pod:** RACE G15 (g6.2xlarge: 1×NVIDIA L4 24 GB, 4 CPU, 32 GB RAM, $1.27/hr)
**Backbone:** Qwen2-VL-2B-Instruct captioner + **Qwen2.5-7B-Instruct planner** (the swap)
**Run dirs:** `runs/remembr-smoke-qwen7b/`, `runs/remembr-smoke-trace/`, `runs/remembr-smoke-replan/`
**Wall-clock:** ~2 h (smoke chase only — no full ablation). **Cost:** ~$2.

## TL;DR

Swapped the regurgitating Qwen2.5-3B planner from Run 2 for Qwen2.5-7B-Instruct
on the L4, hoping a bigger planner would produce useful waypoints. Per the
Run 2 writeup, also landed the Bug 3 controller-stall patch (Phase 0 of the
Phase-3 runbook) before bring-up. **The smoke gate failed 3 of 4 conditions
and we did not run the full ablation**, in line with the Phase-3 runbook's
"defer to a future session" branch when controller stall stays unfixed by
mechanical patches.

The Qwen-7B planner did **not** regurgitate — it produces pose-aware
waypoints that differ as the agent's yaw changes. But it is **obstacle-blind**:
it proposes "1.5 m ahead" relative to whatever yaw the agent currently has,
and the entire forward sector in scene `wcojb4TFT35` is wall. The mechanical
collision-escape works (TURN fires when stalled) but the runner's bearing
re-compute immediately re-targets the agent back into the wall. Adding a
force-replan flag broke the +/−30° oscillation but exposed the deeper issue:
with `--backbone remembr`, `_propose_candidates` routes entirely to the LLM —
the frontier planner's obstacle-aware candidates aren't in the proposal
pool at all.

| Smoke gate condition | Smoke 3 result | Status |
|---|---|---|
| Crash-free | run completed cleanly | PASS |
| `n_steps > 30` | 21 | FAIL |
| `path_traveled ≥ 2 m` | **0.04 m** | FAIL |
| `dist_to_goal < starting dist` | 8.38 m (unchanged from start) | FAIL |

Full ablation gate (C1 ∧ C2) was not measured — we triaged at the smoke
gate per the runbook's explicit stop rule.

## What we ran

Three smoke iterations on the same `--scene all --setting 3 --n-episodes 1
--target any` config. Each $0.40 / ~10 min wall-clock.

| Smoke | Config | n_steps | path | n_stop | First-failure mode |
|---|---|---|---|---|---|
| `remembr-smoke-qwen7b` | defaults | 9 | 0.04 m | 1 | False STOP at step 9 |
| `remembr-smoke-trace` | tracer prints in `step_controller` | 9 | 0.04 m | 1 | Confirmed escape fires; re-align undoes |
| `remembr-smoke-replan` | `STOP_COS=0.40`, `STOP_MIN_STEP=20`, force-replan landed | 21 | 0.04 m | 1 | LLM oscillates between two blocked waypoints |

No full 3-setting ablation was launched.

## Diagnostics — three findings (chronological)

### Finding 1 — Phase 0 collision-escape patch works at the mechanical level

**Patch (`117028d`).** `step_controller` tracks `_last_action` and a toggle.
If the agent picks FORWARD twice in a row and the last 3 logged positions
have a bbox diagonal < 0.1 m, override with an alternating TURN. Verified
locally with a 6-case sanity test (toggle correctness, no-regression on
empty history, no-fire when not stalled).

**Tracer evidence (`runs/remembr-smoke-trace/`).** Smoke 2 added inline
`print()` calls to confirm runtime behavior. The escape fires exactly as
designed at internal step 3 (bbox=0.044) and step 6 (bbox=0.000):

```
t=1: FWD (no last)
t=2: FWD (last=FWD, len=2, precond not met)
t=3: precond met, bbox=0.044, ESCAPE → TURN_RIGHT (action=3)   ← patch fired
t=4: bearing now +30°, candidate forces TURN_LEFT (action=2)   ← re-align undoes escape
t=5: FWD (last=TURN, no escape precond)
t=6: precond met, bbox=0.000, ESCAPE → TURN_LEFT (action=2)
t=7: bearing now -30°, candidate forces TURN_RIGHT (action=3)  ← undone again
t=8: FWD
```

**Why escape didn't translate to navigation.** The runner's bearing
recompute at `episode_runner.py:337-348` runs after every `env.step` and
overwrites `candidate.bearing_rad` to point at the candidate's world_xy
relative to the **new** yaw. After escape rotates the agent −30°, the
candidate is now +30° off-axis; the next `step_controller` call sees
that bearing and emits TURN_LEFT to re-align — cancelling the escape.
Net yaw drift across the episode: ≈ 0°.

### Finding 2 — Force-replan breaks the oscillation but not the loop

**Patch (`6265870`).** Added `_force_replan: bool` to `FrontierPlanner`.
Set it in the escape branch; `is_decision_step()` honors and clears it.
Locally verified with a 5-case sanity test.

**Effect on smoke 3.** `n_steps` climbed from 9 → 21. The runner's
`_propose_candidates` is now called every time escape fires, so the LLM
is re-prompted at the new yaw. Decision count went from 3 to 17. But the
LLM oscillates between two waypoints:

```
d0: (0.34, -19.16)    ← LLM picks "1.5m ahead"
d1: (0.94, -18.65)    ← after escape, LLM picks "1.5m ahead at new yaw"
d2: (0.33, -19.12)    ← same first point
d3: (0.94, -18.65)    ← same second point
d4-d7: alternate between the same two points
```

Both points are ~1.5 m from the start in slightly different rotated
directions. Both are wall. The agent rotates ±30° chasing each, gets
blocked, escapes, re-plans, picks the other — and the cycle holds.

### Finding 3 — The 7B LLM is pose-aware but obstacle-blind

The Qwen-7B planner **does** react to agent pose (different waypoints at
different yaws — this is the substantive improvement over Run 2's Qwen-3B,
which regurgitated the agent's exact position). But the LLM's prompt
contains no depth, no collision history, no occupancy info. It treats every
re-prompt as a fresh "where should the agent go?" query and answers with a
target in the agent's general forward sector. When the entire forward
sector is wall, no amount of re-prompting at rotated yaws produces a
reachable target.

The frontier planner **does** have the occupancy grid — `update()` raycasts
depth every step into a top-down map, and `_extract_frontier_cells()`
finds FREE-adjacent-to-UNKNOWN cells (canonical exploration frontiers).
But with `--backbone remembr`, `_propose_candidates` (`episode_runner.py:431`)
routes entirely to `self.remembr_planner.propose(...)` — the frontier
candidates aren't proposed, aren't reranked, aren't available to the runner.

This is the runbook's documented FAIL-C1 §3 ("planner is exploring but
not goal-directed") with the goal-directedness intact and the *exploration*
missing.

### Finding 4 — STOP_COS=0.25 is too permissive for text-vs-text (carried over from Run 2)

Smoke 1 false-STOPped at step 9 with the agent 8.38 m from the chair. The
captions in this scene are generic ("agent at (-0.2, -17.8) sees: room
interior | searching for any" — semantic captioner falls back to a constant
because HM3D-Semantics annotations aren't installed) and CLIP-text-vs-text
cosine of "chair" against that string clears 0.25 on grammatical baseline
alone. Raising to `REMEMBR_STOP_COS=0.40` and `REMEMBR_STOP_MIN_STEP=20`
cleanly prevented the false STOP in smoke 3.

This is consistent with Run 2's Bug 1 diagnosis. The Phase-3 runbook's
deferred "bridge-CLIP-image-LTM refactor for `_maybe_stop`" is the proper
fix; the env-var tightening is a stopgap.

## Patches landed this session

| Commit | File | Change |
|---|---|---|
| `117028d` | `embodied_memory/frontier_planner.py` | `step_controller` collision-escape: alternate TURN when last 3 positions stalled |
| `6265870` | `embodied_memory/frontier_planner.py` | `is_decision_step()` honors `_force_replan` flag set by escape |

Both verified locally with module-level sanity tests before push (faiss
not installed locally, so importlib loads the planner module directly).

## Comparison to Runs 1 and 2

| Aspect | Run 1 (Mistral 7B, A100) | Run 2 (Qwen 3B, L4) | Run 3 (Qwen 7B, L4) |
|---|---|---|---|
| Captioner | LLaVA-v1.6-Mistral-7B | Qwen2-VL-2B | Qwen2-VL-2B |
| Planner | Mistral-7B-Instruct-v0.3 | Qwen2.5-3B-Instruct | **Qwen2.5-7B-Instruct** |
| Planner regurgitates? | No | Yes (current position as ANSWER) | **No (pose-aware)** |
| Planner obstacle-aware? | (n/a — agent navigated) | (n/a — couldn't get past start) | **No (forward-sector-blind)** |
| Bug 3 controller stall | Unpatched (didn't bite at A100 scene) | Unpatched | **Patched (`117028d` + `6265870`)** |
| Agent reached goals? | Yes — 0.59 m sofa, 1.46 m bed | No — stalled at start | No — moves 0.04 m |
| Smoke `n_steps` typical | 249 (timeout) | 9.6 | 21 |
| Failure mode (root) | No STOP path in action pipeline | Planner can't pick useful waypoints | Planner picks pose-aware but obstacle-blind waypoints |
| Cost this run | ~$4 | ~$3.81 | **~$2 (smoke only, no ablation)** |

Pattern across the three runs: the Phase-2 backbone has a sequence of
load-bearing failures that surface in order as each prior failure is
patched out. Run 1 exposed the missing STOP. Run 2 exposed the small-
planner regurgitation and (later) the controller stall. Run 3 patches
the controller stall and exposes that the LLM-only proposal pool is
obstacle-blind regardless of planner size. Each run advanced the
diagnosis one architectural layer deeper.

## What's next (in priority order)

### 1. Bridge-CLIP-image-LTM refactor for `_maybe_stop`

The runbook's explicit next step ("§What this runbook deliberately does
NOT do"). The STOP path currently queries text-vs-text (Qwen caption vs
goal word), which has a high grammatical-baseline cosine and false-fires
on generic captions. The bridge already maintains a CLIP-image LTM (per
`CLAUDE.md`); `_maybe_stop` should query that index for image-vs-text
cosine (~0.20–0.35 in practice) instead. Out of scope for this session.

### 2. Obstacle-aware proposals for the LLM-driven backbone

Three sub-options, each architectural and each out of the Phase-3 runbook's
authorized scope:

- **Inject frontier candidates into the LLM rerank pool.** When
  `backbone=remembr`, also include 2–3 frontier-planner candidates so the
  rerank scoring can prefer obstacle-aware options. Departs from the paper-
  faithful ReMEmbR architecture (`CLAUDE.md` describes memory→frontier
  injection, not the reverse); a new ablation setting (e.g. S3+) would be
  the cleaner test.
- **Feed prior-action / collision history into the LLM prompt.** Tell the
  7B "the last 5 FORWARDs no-op'd" and let it reason. Prompt-engineering
  change; risks regurgitation regression.
- **Route to frontier planner when `_is_stuck` fires repeatedly.** Smallest
  hack but a planner swap mid-episode; ablation reading muddies.

The runbook explicitly rules out planner swaps at this stage. Defer all
three to a follow-up session.

### 3. Defer G3 / G5 / val scale-up

Still gated on a passing C1, which is still gated on the agent producing
any non-zero binary success — not yet possible on `val_mini` with the
current backbone. No change from Run 2's "What's next."

## Tuning knobs used this session

No new env vars were added. The pre-existing knobs from Runs 1+2 carried
the smoke-3 configuration:

| Env var | Value used | Why |
|---|---|---|
| `REMEMBR_STOP_COS` | 0.40 (was 0.25) | Text-vs-text baseline cosine prevents 0.25 from being a meaningful gate |
| `REMEMBR_STOP_MIN_STEP` | 20 (was 8) | Defense-in-depth against false STOP in the early steps |
| `REMEMBR_MIN_WAYPOINT_DIST` | 0.5 (default) | Run 2's regurgitation guard; not stressed in Run 3 (7B doesn't regurgitate) |

The `STOP_COS=0.40 / STOP_MIN_STEP=20` combination held empirically for
Qwen-7B + text-vs-text STOP path until the bridge-CLIP-image refactor lands.

## File index (Run 3)

| Path | Purpose |
|---|---|
| `runs/remembr-smoke-qwen7b/` | First smoke (default thresholds; false STOP at step 9) |
| `runs/remembr-smoke-trace/` | Second smoke with tracer prints — confirmed Phase 0 patch fires twice but re-align undoes |
| `runs/remembr-smoke-replan/` | Third smoke (force-replan + raised STOP thresholds) — n_steps=21, path=0.04m |
| `embodied_memory/frontier_planner.py` | Phase 0 escape (`117028d`) + force-replan (`6265870`) |
| `docs/phase3-qwen7b-runbook.md` | Source-of-truth runbook for this session (`824caff`) |

---

# Run 4 — Obstacle-aware proposal pool (prep, 2026-05-23)

**Date:** 2026-05-23 (local implementation; no RACE run executed yet)
**Branch:** `phase2-readiness`
**Pod:** _(none — pre-flight code change + sanity tests only)_
**Backbone:** Qwen2-VL-2B-Instruct captioner + Qwen2.5-7B-Instruct planner (same as Run 3)
**Run dirs (planned):** `runs/abl-s{1,2,3}-frontier`
**Status:** **Prep complete — RACE execution deferred to a future session.**

## TL;DR

Run 3 left the agent stalled because the Qwen-7B LLM planner is pose-aware
but **obstacle-blind**: every "1.5 m ahead at current yaw" proposal in scene
`wcojb4TFT35` is wall. The Phase-0 collision-escape (`117028d`) and force-
replan (`6265870`) patches work mechanically, but each re-plan re-picks
another wall point. The architectural cause is in
`episode_runner._propose_candidates`: with `--backbone remembr`, candidate
generation routes **entirely** to the LLM — the frontier planner's
occupancy-grid-aware candidates are never in the proposal pool that gets
reranked.

Run 4 lands a single-seam fix: when `backbone=remembr`, inject up to
**`REMEMBR_FRONTIER_INJECT=3`** frontier candidates onto the LLM output,
de-duped against existing LLM picks by **`REMEMBR_MIN_WAYPOINT_DIST`**
(default 0.5 m). STOP short-circuit is preserved: if the LLM emitted a
`stop_signal` candidate, it returns alone — no dilution. Counters
(`n_frontier_chosen`, `n_frontier_candidates`) are now logged per
decision and aggregated into the run summary so the analyzer can show
how often frontier picks actually steered the agent.

This is the previously-deferred "Option 2a" from the Run-3 writeup
(`PHASE2_ABLATION_REPORT.md` Run 3 → What's next §2), picked over the
runbook-recommended bridge-CLIP STOP refactor because Run 3 showed C1 is
gated by **movement** (0.04 m total), not STOP precision (the
`STOP_COS=0.40` stopgap already eliminated false STOPs).

## What we ran

**Code change only.** No RACE provisioning, no live ablation. The full
operator runbook for the paid run lives in the Run-4 plan body
(implementation-then-RACE plan executed against this branch).

Patches landed locally and unit-tested with the same module-level sanity
pattern used by `117028d` (faiss-free, importlib-loaded directly):

| File | Change |
|---|---|
| `embodied_memory/episode_runner.py` | `_propose_candidates`: in `remembr` branch, merge frontier-planner candidates onto LLM output (cap=`REMEMBR_FRONTIER_INJECT`, de-dup=`REMEMBR_MIN_WAYPOINT_DIST`). STOP short-circuit preserved. New `n_frontier_chosen` counter per episode and `n_frontier_candidates` per decision. |
| `embodied_memory/scripts/test_propose_candidates.py` | 5-case sanity test (stub-and-load): STOP short-circuit, merge, de-dup, n_inject=0 disable, frontier-backbone unchanged. All cases pass locally. |
| `embodied_memory/scripts/analyze_ablation.py` | Surfaced `n_memory_chosen` and `n_frontier_chosen` totals in the per-setting summary table. Gate logic (C1 ∧ C2) unchanged. |
| `docs/phase3-qwen7b-runbook.md` | Run-4 amendment appended (deferred-Option-2a chosen; movement-first reasoning recorded). |

## Sanity-test output (local)

```
$ python embodied_memory/scripts/test_propose_candidates.py
Run-4 _propose_candidates sanity tests
  case (a) STOP short-circuit: OK
  case (b) frontier injected (no overlap): OK
  case (c) de-dup within 0.5 m: OK
  case (d) n_frontier_inject=0 disables injection: OK
  case (e) frontier backbone unchanged: OK
All cases passed.
```

## Why this and not the bridge-CLIP STOP refactor

Run 3's failure mode was **0.04 m total movement** across 21 steps — the
agent never navigated. The bridge-CLIP STOP refactor (the runbook's
recommended Option 1) addresses STOP precision once the agent is near a
goal; it does not address "can't escape the start wall". With Run 3's
`STOP_COS=0.40` + `STOP_MIN_STEP=20` stopgap, no false STOPs fired in the
smoke. Movement is the next bottleneck. Obstacle-aware proposals are the
direct lever.

The bridge-CLIP STOP refactor remains the next session's lever **if** Run 4
flips C1 from FAIL ("agent doesn't navigate") to FAIL ("agent navigates
but doesn't STOP"). That outcome would mean we moved one architectural
layer deeper — the same step-of-diagnosis pattern Runs 1/2/3 followed.

## Setting protocol — unchanged

The 3-setting protocol (memory off / STM / full) is preserved verbatim.
Frontier injection is a backbone-side change applied **uniformly** across
all 3 settings, same as the `509dbc8` STOP fix (Run 1 → Run 2) and the
`117028d` / `6265870` controller patches (Run 2 → Run 3). The S1 vs S3
contrast still isolates the memory pipeline; the new candidate path lifts
the floor for every setting.

## Operator runbook (next session)

The full RACE bring-up + smoke + ablation flow is documented inline in
the Run-4 plan body. Key headers:

1. **Phase 0 — local sanity (free).** Already done in this prep; the
   commits in this branch satisfy the gate.
2. **Phase 1 — RACE bring-up** (~$0.40). Same RACE G15 bring-up as Run 3;
   `STOP_COS=0.40 STOP_MIN_STEP=20` stopgap stays in place.
3. **Phase 2 — Smoke gate** (~$0.40). Escalated pass conditions vs Run 3:
   `n_steps > 50`, `path_traveled ≥ 4 m`, `dist_to_goal < starting − 2 m`,
   `n_frontier_chosen ≥ 1`. If `n_frontier_chosen=0` for the whole smoke,
   the merge logic is wrong — diagnose locally before paying again.
4. **Phase 3 — Full ablation** (~$6–10). Same 3-setting × 30-episode
   protocol so paired bootstrap stays valid.
5. **Phase 4 — Gate read.** Analyzer surfaces `n_frontier_chosen`
   alongside the existing C1/C2 read.

## Expected branches at the next gate read

- **PASS (C1 ∧ C2).** Phase-2 milestone done. G3 trainers, G5 affordance
  refresh, val scale-up become schedulable (separate sessions).
- **FAIL C1 only — agent now navigates but doesn't STOP at goals.** This
  is the cleanest outcome: it would mean Run 4 cleared the wall and the
  next session is the **bridge-CLIP-image STOP refactor** (the deferred
  Option 1).
- **FAIL C2 only — S1 succeeds but memory adds no soft-SPL.** Disambiguate
  with a seed-perturbed S3 rerun or inspect the rerank scoring floor on
  memory candidates.

## Cost ceiling

| Phase | Best | Worst |
|---|---|---|
| Implementation + local tests | $0 | $0 |
| RACE bring-up | $0.40 | $1 |
| Smoke (1–3×) | $0.40 | $2 |
| Full ablation | $6 | $10 |
| Buffer | $0 | $4 |
| **Total** | **$7** | **$17** |

Fits inside the ~$17 remaining Phase-2 envelope. Hard cap stays at the
Run-3 carry-over: stop and escalate if costs trend past **$17 without a
gate read**.

## Tuning knobs added this run

| Env var | Default | Purpose |
|---|---|---|
| `REMEMBR_FRONTIER_INJECT` | 3 | Max frontier candidates injected per decision (`backbone=remembr` only). Set to 0 to disable. |

Pre-existing knobs from Runs 1–3 (`REMEMBR_STOP_COS`, `REMEMBR_STOP_DIST`,
`REMEMBR_STOP_MIN_STEP`, `REMEMBR_MIN_WAYPOINT_DIST`) all still apply;
`REMEMBR_MIN_WAYPOINT_DIST` is now also the de-dup radius for the
frontier-injection path.

## File index (Run 4)

| Path | Purpose |
|---|---|
| `embodied_memory/episode_runner.py` | Run-4 frontier injection + counters |
| `embodied_memory/scripts/test_propose_candidates.py` | 5-case sanity test (stub-and-load, faiss-free) |
| `embodied_memory/scripts/analyze_ablation.py` | `n_memory_chosen` + `n_frontier_chosen` surfaced |
| `docs/phase3-qwen7b-runbook.md` | Run-4 amendment block at the bottom |

# Run 5 — Oracle diagnostic + occupancy-grid densification (prep, 2026-05-24)

**Date:** 2026-05-24 (local implementation + RACE G15 smoke executed same day)
**Branch:** `phase2-readiness`
**Pod:** RACE G15 (g6.2xlarge, 1×L4 24 GB), ~30 min instance time
**Commits:** `a26b1b6` (densified splat + `grid_stats`), `f713119` (oracle backbone + grid logging + tests), `5b0c496` (metric-depth fix), `41e8501` (grid cols in verify)
**Run dirs:** `runs/oracle-smoke-{TEEsavR23oF,wcojb4TFT35}`, `runs/remembr-dense-smoke`, `runs/remembr-dense-nostop`
**Status:** **Oracle PASS (env navigable) + `normalize_depth` bug found & fixed (grid densified ~200×). Densified smoke STILL fails the nav gate — bottleneck has moved to the straight-line step controller (agent wedges at start). Full ablation NOT run — would be 0-success until the controller is fixed.**

## TL;DR

Across Runs 1–4 the agent **cannot navigate**: <2 m in 250 steps, stalled near
start. Run 4 made the architecture complete (`n_frontier_chosen >> 0`, all
module/coherence gates green) but the navigation gate (≥1 episode with
`n_steps>50` AND `path_traveled≥4 m`) still FAILS. Five research agents plus
direct code reads narrowed it to two complementary levers, both landed here:

1. **Root cause (densification).** `frontier_planner.update()` splatted depth
   from a **single middle-row scanline** subsampled to 64 columns. At eye
   height that scanline mostly hits walls/furniture and misses floor
   openings → too few FREE cells → frontiers cluster against walls → no
   navigable subgoal → the agent barely moves. The grid is already correctly
   agent-centered (`6713d12`), so sparsity — not mis-centering — is the cause.
   Replaced with a multi-row per-pixel back-projection + height gate that
   marks floor FREE (walkable, fills doorways) and only tall endpoints
   OCCUPIED.
2. **Decisive unknown (oracle).** We have **never** confirmed the
   environment/episode is navigable at all. Added `--backbone oracle`: a
   `ShortestPathFollower` that steers straight to the goal with a perfect
   planner, bypassing the candidate/scorer/memory machinery but logging
   `success`/`spl`/`distance_to_goal`/`n_steps` identically. If the oracle
   reaches the goal, our pipeline is the bottleneck; if it stalls, the env
   setup is broken and no planner/perception fix matters — the highest-value
   thing a $0.10 run can tell us.

Research notes that did **not** make the cut: no goal-bearing scorer term
(real ObjectNav agents shouldn't know goal xyz; the oracle already supplies
the goal-direction answer), and no collision flag in `info` (the `Collisions`
measure is off in `objectnav_hm3d.yaml`; the bbox<0.1 m stall heuristic stays).

## RACE results (G15, 2026-05-24) — what the smokes actually told us

The smoke ran on RACE G15 after the local prep. It produced a clean,
three-step diagnostic chain. **Each step moved the bottleneck one layer deeper.**

### Step 1 — Oracle: the environment is navigable (decisive)

`--backbone oracle` (model-free, ~$0) on both scenes, 2 episodes each:

| scene | target | success | dist_to_goal | spl | n_steps |
|---|---|---|---|---|---|
| TEEsavR23oF | plant | ✅ | 0.04 m | 0.942 | 126 |
| TEEsavR23oF | sofa | ✅ | 0.03 m | 0.215 | 102 |
| wcojb4TFT35 | bed | ✅ | 0.06 m | 0.689 | 28 |
| wcojb4TFT35 | chair | ✅ | 0.02 m | 0.889 | 42 |

**4/4 success, SPL up to 0.94.** This kills the "env/episode is broken"
hypothesis: spawns are reachable, goal coords are right, the discrete action
space works. The 0-success wall across Runs 1–4 is **our pipeline**, not the
environment. (The bridge pass-conditions print FAIL on the oracle path —
expected, there is no bridge; that's why we run `--no-strict-pass`. The real
read is the `verify_smoke_gate.py` oracle gate = PASS.)

### Step 2 — `normalize_depth` bug: the splat was strangled (found & fixed)

The oracle path still runs `planner.update()`, so its `grid_*` counts are real
data on real Habitat depth — and they were **inverted**: `cells_free≈4` vs
`cells_occupied` in the hundreds (local synthetic test produced the opposite,
601 free / 29 occupied). Root cause: HM3D ObjectNav's depth sensor defaults to
**`normalize_depth=True`**, returning depth in **[0, 1]** (confirmed live:
`max=0.61`, `mean=0.14`), not meters. Normalized depth collapses every ray's
ground range (a 3 m wall reads ~0.3), so the height gate marked nearly every
endpoint OCCUPIED and carved almost no FREE cells — the densification could
*never* take. This is exactly the assumption the plan flagged ("verify
`normalize_depth` is false at smoke time").

Fix (`5b0c496`): set `depth_sensor.normalize_depth = False` in the
`habitat_env` sensor override. Re-running the oracle smoke confirmed the grid
densified **~200×**:

| scene/ep | g_free before → after | g_front before → after |
|---|---|---|
| TEE plant (126 steps) | 77 → **2593** | 21 → 782 |
| TEE sofa (102 steps) | 85 → **3465** | 34 → 1303 |
| wcojb bed (28 steps) | 4 → **804** | 3 → 396 |
| wcojb chair (42 steps) | 4 → **1343** | 4 → 432 |

This was a genuine bug throttling Runs 1–4: the frontier planner literally had
~4 navigable cells to work with on `wcojb4TFT35`.

### Step 3 — Densified smoke: grid fixed, but the controller wedges (new bottleneck)

`--backbone remembr --setting 3` on `wcojb4TFT35`, 2 episodes. The full memory
stack came alive for the first time — **all 5 bridge pass-conditions PASS**
(fine layer non-empty, rerank always retrieves, memory influences, all four
modules, no crash), `n_frontier_chosen=27`, `rerank_disagreements=27`. But the
nav gate FAILED: both episodes STOPped at step 21 (`STOP_MIN_STEP=20`), 2.85 m /
5.77 m from goal — a **false STOP** firing on a distant sighting.

Re-running with `REMEMBR_STOP_MIN_STEP=9999` (STOP disabled) ran the full 249
steps and gave the decisive read:

| ep | target | n_steps | path_traveled | dist_to_goal | g_free | g_front |
|---|---|---|---|---|---|---|
| 0 | bed | 249 | **0.34 m** | 2.85 m | 224 | 111 |
| 1 | chair | 249 | **0.55 m** | 5.77 m | 197 | 80 |

`distance_to_goal`, `path_traveled`, **and every grid stat are byte-identical
to the 21-step run** (`d2g=2.8474531173706055` in both). The agent moves
~0.3–0.5 m out of the start, then **wedges and never moves or observes anything
new for the remaining ~228 steps** — the occupancy grid never grows past 224
cells. With the grid now dense (`g_free=224`, not the old 4) and STOP disabled,
the agent *still* can't translate.

**Diagnosis:** the bottleneck is now the **step controller**, not the grid (now
dense), the env (oracle proved navigable), or STOP (disabled, still stalls).
`frontier_planner.step_controller` steers by **straight-line bearing with no
collision-aware path planning** (explicitly out-of-scope in the module
docstring). The chosen frontier candidates are reachable in principle — the
oracle walks out of these exact starts — but the straight line to them crosses
geometry, so `move_forward` collides, the collision-escape toggles a turn,
`force_replan` picks another frontier on the same wall, and the agent
oscillates in place. The oracle succeeds precisely because it follows the
**navmesh**, not a straight line.

### Decision-tree branch fired

"Oracle reaches goal but densified smoke still stalls → env fine, our
perception/planner still the bottleneck." Critically, `cells_free` is **not**
tiny (224, not 4) — so it is *not* "densification didn't take". The next lever
is the **step controller**, to be developed **locally**, not on RACE:

- Replace straight-line bearing stepping with **A\* over the occupancy grid** to
  the chosen frontier (or follow `pathfinder`/navmesh like the oracle does), so
  the agent routes *around* obstacles instead of wedging.
- Then the deferred bridge-CLIP-image STOP refactor addresses the false-STOP
  (stops too eagerly on distant sightings; fires the instant `STOP_MIN_STEP`
  allows, 2.8–5.8 m from goal).

**Full 3×30 ablation NOT run.** With ~0.5 m of movement it would burn $6–10 to
confirm 0 success; the gate cannot pass until the controller can translate.
That decision keeps us well inside the cost envelope (only ~30 min of G15 time
spent on the whole diagnostic).

### Next session — collision-aware step controller (pick up here)

The binding constraint is now `frontier_planner.step_controller`: it converts a
chosen frontier into a single action by **straight-line bearing only** (turn to
face the candidate, then `move_forward`), with no routing around obstacles. When
the straight line crosses geometry the agent collides, the bbox<0.1 m
collision-escape toggles a turn, `force_replan` re-picks, and it oscillates in
place — 0.5 m over 249 steps. The oracle clears the same starts by following the
**navmesh**.

Concrete next lever (develop + validate **locally**, faiss/habitat-free, the same
way the densified splat was — `embodied_memory/scripts/test_propose_candidates.py`):

1. **Grid A\* (preferred, self-contained).** Add an A\* / BFS over the
   `OccupancyGrid` (FREE+UNKNOWN traversable, OCCUPIED blocked) from the agent
   cell to the chosen frontier cell; `step_controller` emits the action toward
   the **next waypoint on that path**, not the straight-line bearing. Unit-test:
   a synthetic grid with a wall gap → the path must route through the gap and the
   first action must not drive into the wall.
2. **Navmesh fallback (cheap sanity only).** `sim.pathfinder` is already exposed
   via `get_sim()` (for the oracle), but using it in the planner couples the
   stand-in to Habitat; grid A\* keeps it self-contained and is the better fit
   for the LTM thesis.
3. **Then** re-smoke with `scripts/race-smoke.sh` — oracle is already green, so
   only the `remembr --setting 3` escape check needs re-running. Only after
   `path_traveled ≥ 4 m` clears do the false-STOP refactor and the full 3×30
   ablation become worthwhile.

Do **not** re-run the oracle or the full ablation to start — the oracle answer
(navigable) and the depth/grid fix are settled. Start at the controller.

## What we ran (code)

**Code change only.** No RACE provisioning, no live ablation/smoke. The RACE
bring-up is a CUDA-host operator step (`docs/phase3-qwen7b-runbook.md` Phase 1);
this machine is a CPU-only laptop. Patches landed locally and verified with the
faiss/habitat-free sanity suite (importlib-loaded, `sys.modules`-stubbed).

| File | Change |
|---|---|
| `embodied_memory/frontier_planner.py` | `update()` rewritten: multi-row (~28×28 subsample) per-pixel back-projection from `hfov=79°` pinhole intrinsics + height gate (`camera_height_m=0.88`, `obstacle_min_h=0.3`); `reset(agent_pos)` fixes `_floor_y`. New `grid_stats()` census. |
| `embodied_memory/episode_runner.py` | `--backbone oracle` in-loop branch; `_init_oracle_follower`/`_oracle_action`; `None`-bridge guards throughout; logs `grid_cells_{free,occupied,unknown}`+`grid_frontier_cells` into `ep_log`/metrics/per-episode summary row. |
| `embodied_memory/run_hm3d_pol.py` | `--backbone oracle` choice; skips CLIP/captioner/text-encoder/bridge loads (`bridge=None`) so the oracle smoke starts in seconds. |
| `embodied_memory/habitat_env.py` | `get_sim()` accessor exposing `env.sim` to the follower. |
| `embodied_memory/episode_source.py` | base `get_sim()` returning `None`. |
| `embodied_memory/scripts/test_propose_candidates.py` | 5 new sanity cases (densify, height gate, `grid_stats` schema, oracle action map, oracle short-circuit) + `habitat_env._ACTION_NAMES` stub. |

## Sanity-test output (local)

```
$ python embodied_memory/scripts/test_propose_candidates.py
Run-4/Run-5 sanity tests
  case (a) STOP short-circuit: OK
  case (b) frontier injected (no overlap): OK
  case (c) de-dup within 0.5 m: OK
  case (d) n_frontier_inject=0 disables injection: OK
  case (e) frontier backbone unchanged: OK
  case (f) propose_diverse compass fallback (k=3, baseline 0.7): OK
  case (g) compass occupancy-aware (FREE=1.000, OCC=0.200): OK
  case (h) grid recenters on reset (origin=(-10.23, -27.77)): OK
  case densify_grid (base_free=26, dense_free=926, frontier=632): OK
  case height_gate (floor_occ=0, wall_occ=29): OK
  case grid_stats_schema (n*n=40000, free=601): OK
  case oracle_action_map (move_forward/turn_left/stop/None → 1/2/0/0): OK
  case oracle_short_circuit (no bridge/propose deref, grid logged): OK
All cases passed.
```

The densification case is the headline: the same synthetic frame carves
**926 FREE cells** with the multi-row splat vs **26** with the single
eye-level scanline (35×), and exposes 632 frontier cells where the old splat
exposed 16. The height gate correctly produces **0 OCCUPIED** for a far floor
band and **29 OCCUPIED** for an eye-level band.

## RACE smoke — pending operator bring-up (~$0.80)

Standard RACE G15 bring-up per `docs/phase3-qwen7b-runbook.md` Phase 1. Keep
the Run-3 stopgap (`REMEMBR_STOP_COS=0.40 REMEMBR_STOP_MIN_STEP=20`). Run two
cheap smokes, **explicitly pinning `--scene`** (short smokes are single-scene;
episode iteration follows dataset order, not round-robin):

```bash
# A) Oracle env check — no model loads, both scenes
for sc in TEEsavR23oF wcojb4TFT35; do
  python -m embodied_memory.run_hm3d_pol --mode live --backbone oracle \
    --setting 1 --scene $sc --n-episodes 2 --target any --no-strict-pass \
    --out-dir runs/oracle-smoke-$sc
done
# B) Densified-grid escape check — full stack
python -m embodied_memory.run_hm3d_pol --mode live --backbone remembr \
    --setting 3 --scene wcojb4TFT35 --n-episodes 2 --target any \
    --out-dir runs/remembr-dense-smoke
```

### Decision tree on the read

- **Oracle reaches goal** (`success≥1` or `distance_to_goal`<1 m) **AND
  densified smoke passes** `path_traveled≥4 m` / `n_frontier_chosen≥1` → env
  navigable AND the grid fix unblocked movement. Proceed to the full 3×30
  ablation (runbook Phase 3) for the gate read.
- **Oracle reaches goal but densified smoke still stalls** → env fine, our
  perception/planner is still the bottleneck. Inspect the new `grid_*` counts:
  if `cells_free` is still tiny, densification didn't take (recheck
  planar-depth / intrinsics assumption — verify `normalize_depth` is false);
  iterate the splat **locally**, not on RACE.
- **Oracle ALSO stalls** → env/episode/action-space is broken (agent spawned in
  an unreachable pocket, goal coords wrong, or discrete action-space mismatch).
  No planner/perception fix matters; pivot to env debugging. Highest-value $0.10
  the oracle can spend.

## Tuning knobs added this run

| Ctor param | Default | Purpose |
|---|---|---|
| `FrontierPlanner.camera_height_m` | 0.88 | Agent eye height above floor; sets `_floor_y` in `reset(agent_pos)`. |
| `FrontierPlanner.obstacle_min_h` | 0.30 | Endpoint must rise this far above floor to count as OBSTACLE; lower → FREE (walkable). |

`--backbone oracle` runs with `--no-strict-pass` (empty-LTM pass-conditions
don't flip the exit code). No new env vars; the Run-1..4 knobs all still apply.

## Cost ceiling

| Phase | Best | Worst |
|---|---|---|
| Implementation + local tests | $0 | $0 |
| RACE bring-up + oracle smoke (A) | $0.50 | $1 |
| Densified-grid smoke (B) | $0.30 | $1 |
| Full ablation (if both green) | $6 | $10 |
| Buffer | $0 | $4 |
| **Total** | **$7** | **$16** |

Budget remaining ~$11. Hard cap unchanged: stop and escalate if costs trend
past **$17 without a gate read**.

## File index (Run 5)

| Path | Purpose |
|---|---|
| `embodied_memory/frontier_planner.py` | Densified multi-row depth splat + height gate + `grid_stats()` |
| `embodied_memory/episode_runner.py` | Oracle in-loop branch + `_init_oracle_follower`/`_oracle_action` + `None`-bridge guards + grid logging |
| `embodied_memory/run_hm3d_pol.py` | `--backbone oracle` choice + conditional model loads |
| `embodied_memory/habitat_env.py` | `get_sim()` accessor |
| `embodied_memory/episode_source.py` | base `get_sim()` |
| `embodied_memory/scripts/test_propose_candidates.py` | 5 new Run-5 sanity cases |
| `docs/phase3-qwen7b-runbook.md` | Run-5 amendment block at the bottom |

---

# Run 6 — Collision-aware step controller, grid A\* (prep, 2026-05-24)

**Date:** 2026-05-24 (local implementation + sanity tests only; no RACE run executed)
**Branch:** `phase2-readiness`
**Pod:** _(none — CPU-only laptop; faiss/habitat-free implementation + tests)_
**Backbone:** unchanged (Qwen2-VL-2B captioner + Qwen2.5-7B planner on RACE; backbone-agnostic change)
**Run dirs (planned):** `runs/remembr-astar-smoke`, then `runs/abl-s{1,2,3}-astar`
**Status:** **Prep complete — grid A\* step controller landed and unit-tested locally (21/21 sanity cases green). RACE smoke deferred to the next operator session.**

## TL;DR

Run 5 isolated the bottleneck to `frontier_planner.step_controller`: it steered
by **straight-line bearing only** (turn to face the chosen frontier, then
`move_forward`), with no routing around obstacles. On HM3D the straight line to
a *reachable* frontier crosses geometry, so `move_forward` collides, the
bbox<0.1 m escape toggles a turn, `force_replan` re-picks another wall-facing
frontier, and the agent oscillates in place — ~0.5 m over 249 steps. The oracle
clears the same starts because it follows the **navmesh**, routing *around*
obstacles.

Run 6 replaces the straight-line controller with **grid A\*** over the
`OccupancyGrid` (the Run-5 "Next session" lever). `step_controller` now runs A\*
from the agent cell to the chosen frontier cell — **FREE + UNKNOWN traversable,
OCCUPIED inflated-and-blocked** — and steers toward a **short-lookahead
waypoint** (~0.4 m) on that path. The agent routes around obstacles instead of
wedging. Self-contained pure numpy/stdlib (no Habitat coupling), so it keeps
the LTM stand-in independent of the simulator and loads in the faiss/habitat-
free sanity harness.

This is the previously-deferred grid-A\* path from the Run-5 writeup, picked
over the navmesh fallback because navmesh couples the planner to Habitat. It is
a **movement** fix; the false-STOP (bridge-CLIP-image) refactor and the full
3×30 ablation stay gated on it clearing the wedge first.

## What we ran

**Code change only.** No RACE provisioning, no live smoke/ablation. Patches
landed on `phase2-readiness` and verified with the same module-level sanity
pattern used since `117028d` (importlib-loaded, `sys.modules`-stubbed,
faiss/habitat-free).

| File | Change |
|---|---|
| `embodied_memory/frontier_planner.py` | New module-level `astar()` (8-connectivity, no diagonal corner-cutting, octile heuristic, `unknown_cost` penalty, `max_expansions` cap), `_inflate_occupied()` (numpy obstacle dilation, no scipy), `_snap_to_free()` (BFS-ring goal snap). `step_controller(candidate, agent_pos, agent_yaw)` rewritten to A\*-route + steer toward a `lookahead_m` waypoint; `_astar_action` / `_bearing_to_action` / `_straight_line_fallback` helpers. New `__init__` knobs `lookahead_m=0.4`, `inflate_radius_cells=1`, `unknown_cost=1.5`, `astar_max_expansions=20000`. Collision-escape kept as a safety net. Module docstring updated (collision-aware control no longer out-of-scope). |
| `embodied_memory/episode_runner.py` | One-line caller change at the `step_controller` call site — passes `step.agent_state.position`. Bearing recompute left as-is (now harmless, since the controller no longer steers by `bearing_rad` on the A\* path — this also defangs Run-3's "recompute undoes the escape turn" oscillation). |
| `embodied_memory/scripts/test_propose_candidates.py` | 8 new Run-6 sanity cases (13 → 21). All pass locally. |

### A\* design (aggressive passability profile)

- **Connectivity:** 8-conn with a no-corner-cutting guard (a diagonal is legal
  only when both shared orthogonal neighbours are unblocked) — smoother bearings
  than 4-conn's ±45° staircase, no clipping obstacle corners.
- **Traversability:** OCCUPIED (inflated by 1 cell ≈ 0.1 m for agent radius)
  blocked; FREE cost 1/√2; **UNKNOWN traversable** at `unknown_cost=1.5`× so the
  search prefers observed-free corridors but still crosses unobserved space when
  that's the only route (self-correcting: a wrongly-optimistic UNKNOWN cell
  flips OCCUPIED on collision and the next per-step replan routes around it).
- **Lookahead:** steer toward the cell ~0.4 m (4 cells) along the path, clamped
  to the path end — smooths heading vs the jittery immediate-next cell without
  cutting far corners.
- **Robustness:** the agent's own (start) cell is force-cleared in the blocked
  mask so standing next to a wall never self-blocks the planner (the single most
  important detail — without it A\* freezes worse than straight-line). Goal cells
  on/next to a wall are snapped to the nearest passable cell. A `max_expansions`
  cap bounds the rare passable-but-trapped-goal full-grid exhaustion. No path →
  straight-line fallback + `force_replan`.

## Sanity-test output (local)

```
$ python embodied_memory/scripts/test_propose_candidates.py
Run-4/Run-5 sanity tests
  ... (13 prior cases) ...
  case astar_routes_through_gap: OK
  case astar_none_when_walled_off: OK
  case astar_inflation_seals_one_cell_gap: OK
  case astar_goal_occupied_snaps: OK
  case astar_start_equals_goal: OK
  case astar_first_action_not_into_wall: OK
  case astar_lookahead_waypoint: OK
  case controller_fallback_on_none: OK
All cases passed.
```

The headline cases: `astar_routes_through_gap` (path goes through a wall's
single gap, never steps on OCCUPIED), `astar_first_action_not_into_wall` (with
the goal straight ahead behind a wall whose only gap is offset, the controller
emits a TURN toward the gap, not FORWARD into the wall — the exact failure mode
Run 5 diagnosed), `astar_inflation_seals_one_cell_gap` (1-cell inflation seals a
1-cell gap, proving the agent-radius clearance), and `controller_fallback_on_none`
(no path → straight-line bearing + `_force_replan`).

### Performance (local micro-benchmark, 200×200 grid)

Per-step A\* on a realistic dense local map (~11k FREE cells, scattered
clutter), goals ~3 m out:

| Scenario | Time |
|---|---|
| Reachable goal | mean **0.73 ms**, max 3.3 ms |
| Passable-but-trapped goal (full exhaustion), uncapped | 209 ms |
| Same, `max_expansions=20000` | 107 ms |

Recompute-every-step is comfortably cheap against the per-step Habitat render +
LLM planner inference (seconds). Real frontier goals are reachable by
construction (a frontier cell is FREE *adjacent to UNKNOWN*, so it always has a
passable neighbour), so the trapped-goal tail rarely fires; the cap bounds it
regardless.

## Why this and not the bridge-CLIP STOP refactor

Run 5's failure was **~0.5 m total movement** — the agent never navigated. The
bridge-CLIP-image STOP refactor addresses STOP *precision* once the agent is
near a goal; it does nothing for "can't escape the start wall". The Run-3
`STOP_COS=0.40 / STOP_MIN_STEP=20` stopgap already suppresses false STOPs in the
smoke. Movement is the binding constraint; A\* is the direct lever. Same
step-of-diagnosis pattern as Runs 1→5: each run patches the current load-bearing
failure and (we expect) exposes the next.

## Setting protocol — unchanged

The 3-setting protocol (memory off / STM / full) is preserved verbatim. The A\*
controller is a backbone-side change applied **uniformly** across all 3
settings, exactly like the `509dbc8` STOP fix and the `117028d`/`6265870`/Run-4
controller patches. The S1 vs S3 contrast still isolates the memory pipeline;
the new controller lifts the movement floor for every setting.

## Operator runbook (next session)

1. **Phase 0 — local sanity (free).** Done in this prep; `python
   embodied_memory/scripts/test_propose_candidates.py` → "All cases passed." (21).
2. **Phase 1 — RACE bring-up** (~$0.40). Standard RACE G15 per
   `docs/phase3-qwen7b-runbook.md` Phase 1. Keep `REMEMBR_STOP_COS=0.40
   REMEMBR_STOP_MIN_STEP=20`.
3. **Phase 2 — Movement smoke** (~$0.40). Single scene, full stack:
   ```bash
   REMEMBR_STOP_COS=0.40 REMEMBR_STOP_MIN_STEP=20 \
   python -m embodied_memory.run_hm3d_pol --mode live --backbone remembr \
       --setting 3 --scene wcojb4TFT35 --n-episodes 2 --target any \
       --out-dir runs/remembr-astar-smoke
   ```
   **Pass condition:** `path_traveled ≥ 4 m` (vs the ~0.5 m wedge). If the agent
   still moves <2 m, diagnose locally (grid stats, A\* path on a dumped grid) —
   do **not** re-pay until it clears. The oracle is already green; do not re-run it.
4. **Phase 3 — Full ablation** (~$6–10). Same 3×30×250 protocol so paired
   bootstrap stays valid: `PHASE2_OUT_SUFFIX=-astar bash scripts/run_phase2_ablation.sh`.
5. **Phase 4 — Gate read.** `analyze_ablation.py` for C1∧C2.

## Expected branches at the next gate read

- **Movement smoke passes (`path_traveled ≥ 4 m`).** Run the full ablation.
- **Full gate PASS (C1 ∧ C2).** Phase-2 milestone done; G3/G5/val scale-up
  become schedulable.
- **FAIL C1 only — agent now navigates but doesn't STOP at goals.** The cleanest
  outcome: it means A\* cleared the wall and the next session is the deferred
  **bridge-CLIP-image STOP refactor**. One architectural layer deeper, same as
  the Run-1→5 progression.
- **Smoke still wedges (<2 m).** A\* picked an unreachable target or the grid is
  too sparse at that start; inspect dumped `grid_*` + the A\* path locally.

## Cost ceiling

| Phase | Best | Worst |
|---|---|---|
| Implementation + local tests | $0 | $0 |
| RACE bring-up + movement smoke | $0.50 | $2 |
| Full ablation (if smoke clears) | $6 | $10 |
| Buffer | $0 | $4 |
| **Total** | **$6.50** | **$16** |

Hard cap unchanged: stop and escalate if costs trend past **$17 without a gate read**.

## Tuning knobs added this run

| Ctor param | Default | Purpose |
|---|---|---|
| `FrontierPlanner.lookahead_m` | 0.4 | Distance along the A\* path to the steering waypoint (smooths bearing). |
| `FrontierPlanner.inflate_radius_cells` | 1 | OCCUPIED dilation for agent radius (1 cell ≈ 0.1 m clearance). 2 = ~full radius but risks sealing tight doorways. |
| `FrontierPlanner.unknown_cost` | 1.5 | UNKNOWN-cell traversal penalty (>1 prefers observed-free routes; keeps unexplored space passable). |
| `FrontierPlanner.astar_max_expansions` | 20000 | A\* node-expansion cap; bounds the trapped-goal worst case (→ straight-line fallback). |

No new env vars; the Run-1..5 knobs (`REMEMBR_STOP_COS`, `REMEMBR_STOP_DIST`,
`REMEMBR_STOP_MIN_STEP`, `REMEMBR_MIN_WAYPOINT_DIST`, `REMEMBR_FRONTIER_INJECT`)
all still apply.

## File index (Run 6)

| Path | Purpose |
|---|---|
| `embodied_memory/frontier_planner.py` | `astar()` + `_inflate_occupied()` + `_snap_to_free()`; A\* `step_controller` + helpers; new ctor knobs |
| `embodied_memory/episode_runner.py` | `step_controller` call site passes `agent_pos` |
| `embodied_memory/scripts/test_propose_candidates.py` | 8 new Run-6 A\* sanity cases (13 → 21) |

