# Phase C — multi-scene 3-setting revisit ablation — design

**Date:** 2026-05-27
**Branch:** lifelong-revisit-eval
**Status:** approved design → implementation

## Problem

Phase 3 (Run 8) turned the revisit eval **GREEN** on a single-scene smoke
(`wcojb4TFT35`, chair+bed, n=6 warm pairs): warm soft-SPL S1 0.079 → S3 0.375,
paired Δ +0.296, 90% CI [+0.100, +0.517], p=0.002, first non-zero binary SPL,
memory fire-rate 0.833. That closed the question "does the hierarchical LTM ever
help?" — it does, when the indexed content is discriminative and a past sighting
is relevant.

But the smoke is **one scene, two categories, S1 vs S3 only**. Two gaps remain
before the finding generalizes:

1. **Single scene / single environment.** The research proposal's central claim
   (§3.1) is *cross-task, **cross-environment** lifelong accumulation*
   (跨任务、跨环境持续积累). One scene cannot show the effect is not scene-specific
   overfitting.
2. **No STM/LTM decomposition.** The smoke ran S1 (memory off) vs S3 (full). The
   proposal has four modules; S2 (STM-only) is needed to separate the STM module's
   contribution from the consolidation + hierarchical-LTM + rerank contribution
   (modules 2–4, the paper's novel part).

Phase C scales the revisit eval to **2 scenes × {chair, bed} × 3 settings**
(`wcojb4TFT35`, `TEEsavR23oF`), making it a single RACE command that emits a
Gate-A verdict with the S2 decomposition.

### Why this follows the paper

- **§3.1 / method summary** — the goal is reuse of historical experience in
  *future tasks* across tasks **and environments**. Revisit warm visits are
  future same-category tasks; multiple scenes exercise the cross-environment
  claim. The LTM persists across episodes within a process and recall is
  scene-filtered, so cross-scene cycling is sound.
- **Four modules → ablation settings.** S1 = all memory off (ReMEmbR-without-LTM
  baseline); S2 = STM only (module 1); S3 = full (modules 1–4). Then
  **S2−S1 isolates STM**, **S3−S2 isolates consolidation + hierarchical LTM +
  rerank** (the proposal's contribution), and **S3−S1** stays the headline
  full-system effect. Adding S2 makes the ablation *more* faithful to the
  proposal than the S1/S3 smoke.
- **Scope honesty.** Single-goal ObjectNav exercises the **fine layer**
  (§3.4 轨迹记忆, `z_i = Encoder(τ_i)`) — recalling a past sighting of the goal.
  The mid (success patterns) and coarse (affordances) layers are seeded but not
  meaningfully exercised by a single-goal task; this eval does not claim to test
  them. That scoping matches the existing Phase-1/2/3 reports.

## Scope (locked with the user)

- **Scenes:** `wcojb4TFT35`, `TEEsavR23oF` (the two val_mini scenes).
- **Categories:** `chair`, `bed` (both scenes have ≥3 same-category source
  episodes, so 3 reachable warm starts each; shared across scenes for a clean
  cross-scene comparison).
- **n_warm = 3** (matches the validated smoke).
- **Settings:** S1, S2, S3 (separate processes / out-dirs).
- **Size:** 4 cells × (1 cold + 3 warm) = 16 episodes × 3 settings = **48 episodes**,
  ~2 h on RACE, **12 warm pairs** for the paired delta (vs 6 in the smoke).

## Design — 5 changes

### 1. `habitat_env.py` — pin episode ordering (correctness)

In `_build_env`, after `get_config(...)` and inside the `read_write(config)`
block, set:

```python
config.habitat.dataset.episode_iterator_options.shuffle = False
```

(leaving `group_by_scene` at its default-true). The single-scene smoke happened
to yield cold-first order; a multi-scene `--scene all` run must **guarantee** that
within each `(scene, category)` group the cold seed episode is processed before
its warm visits, independent of habitat's version defaults — otherwise a warm
visit could run before the LTM holds its cold sighting and the visit-order labels
the analyzer assigns (by processing order) would be wrong. `group_by_scene`
processes one scene's episodes before the next; `shuffle=False` keeps each scene's
episodes in dataset order (cold first, by construction of the builder). Recall is
scene-filtered, so cross-scene cycling otherwise does not interfere.

Guarded so a habitat build lacking that config key does not crash the env
(try/except around the single assignment, matching the existing spl_guard
pattern at lines 127–133).

### 2. `race-revisit.sh` — generalize the driver to multi-scene, 3-setting

- New/changed args: `--scenes "wcojb4TFT35 TEEsavR23oF"` (default = both;
  replaces the singular `--scene`), `--categories "chair bed"` (default),
  `--n-warm 3`, `--tag revisit-c1`.
- **Build loop** — for each scene, resolve its val_mini source
  (`…/val_mini/content/<scene>.json.gz`) and call the **unchanged**
  `make_revisit_smoke.py --out-dir "$DS_DIR"` (one shared out-dir). The builder's
  `content/<scene>.json.gz` writes are additive across calls; its top-level
  `<name>.json.gz` is rewritten each call but only carries category maps + empty
  episodes, and both val_mini scenes share the standard ObjectNav category map
  (the scene-annotation map is unused — the HM3D semantic sensor is zeroed), so
  the rewrite is harmless. `DS_DIR` basename is constant → the top-level filename
  is stable.
- **n-episodes auto-count** — `--scene all` loads *all* scenes' episodes, so the
  default count must **sum** episodes across every `content/*.json.gz` in the
  shared dir, not count one file (the current single-file count would truncate the
  second scene).
- **Run loop** — `for S in 1 2 3` (adds S2), each `REMEMBR_STRICT=1 …
  --backbone remembr --setting $S --scene all --episodes-path "$DS"
  --target any --n-episodes "$N" --out-dir runs/${TAG}-s$S`.
- **Analyze** — pass all three out-dirs to `analyze_revisit.py`.
- All existing invariants preserved (documented in the script header):
  `--backbone remembr`, `REMEMBR_STRICT=1`, separate processes per setting,
  `--target any`, pre-test suite as scripts, abort-before-paid-run on any failure.

### 3. `analyze_revisit.py` — add the S2 decomposition

`paired_warm_delta` / `paired_cold_delta` already take arbitrary S-lists and key
on `(scene_id, episode_id)` (unique across scenes even when episode_ids repeat).
Add, **when an S2 run is present** in `by_setting`:

- warm **S2−S1** paired delta (STM-only effect),
- warm **S3−S2** paired delta (LTM-specific effect: consolidation + LTM + rerank),

reported alongside the existing warm **S3−S1** (primary) and the cold S3−S1
control. The **Gate-A a/b/c classification stays on S3−S1** (`classify_gate_a`
unchanged) — S2 deltas are reported as diagnostics, not a new gate. When no S2 run
is passed, output is identical to today (back-compatible).

A small refactor: factor the "print one warm/cold paired block" into a helper so
the three deltas (S2−S1, S3−S2, S3−S1) print uniformly; no change to the
bootstrap or the loaders.

### 4. `test_analyze_revisit.py` — cover S2 + multi-scene

- **S2 decomposition present:** build synthetic S1/S2/S3 runs; assert the S2−S1
  and S3−S2 blocks are computed and printed, and that the Gate-A verdict is
  **identical** to the S1/S3-only result (S2 must not change the gate).
- **Back-compat:** S1/S3-only input still produces the same report (no S2 block,
  no crash).
- **Multi-scene visit order:** two scenes each with a chair cold + warm sharing
  the episode_id `chair-warm-1`; assert visit-order is assigned per `(scene, cat)`
  (each scene's cold = order 0) and the warm pairing keys on `(scene_id,
  episode_id)` so the two scenes' identically-named warm episodes do **not**
  collide.

### 5. `test_make_revisit_smoke.py` — cover additive multi-scene build

Add a case: build scene A into a temp out-dir, then build scene B into the **same**
out-dir; assert both `content/A.json.gz` and `content/B.json.gz` exist afterward
(additive, second build does not clobber the first) and the top-level
`<name>.json.gz` re-loads with empty episodes + a category map. (Uses a tiny
in-memory source dict per scene — no Habitat.)

## Components touched

- `embodied_memory/habitat_env.py` — one guarded `shuffle = False` assignment.
- `embodied_memory/scripts/analyze_revisit.py` — S2 delta reporting + a print
  helper; loaders/bootstrap/`classify_gate_a` unchanged.
- `scripts/race-revisit.sh` — multi-scene build loop, `for S in 1 2 3`,
  multi-file n-episodes sum, `--scene all`.
- `embodied_memory/scripts/test_analyze_revisit.py`,
  `embodied_memory/scripts/test_make_revisit_smoke.py` — new cases.
- **Untouched:** `make_revisit_smoke.py` (build logic), `run_hm3d_pol.py`
  (`--scene all` already discovers from `content/`), `memory_bridge.py`,
  `episode_runner.py`, `spl_guard.py`, `text_encode_util.py`, the dialogue/MSC
  path.

## Testing

All sanity suites run locally (numpy/stdlib only, no Habitat):

```bash
python embodied_memory/scripts/test_analyze_revisit.py
python embodied_memory/scripts/test_make_revisit_smoke.py
python embodied_memory/scripts/test_spl_guard.py
python embodied_memory/scripts/test_text_encode_util.py
bash -n scripts/race-revisit.sh
```

The GREEN/RED Phase-C verdict itself comes from the RACE run — I cannot execute
Habitat/CUDA from this sandbox.

## Verification (on RACE)

```bash
cd <ltm on RACE> && git checkout lifelong-revisit-eval && git pull --ff-only
bash scripts/race-revisit.sh --tag revisit-c1     # defaults = both scenes, chair+bed, n-warm 3
```

Read the Gate-A block:
- **(a) GREEN** — warm fire-rate ≥ 0.25 AND warm S3−S1 > 0 → the effect
  generalizes across scenes; report Phase C, consider folding revisit into the
  standard harness.
- **(b)** fires but warm S3−S1 ≤ 0 → diagnose (wrong-instance recall / detour cost)
  before claiming generalization.
- **(c)** rarely fires → a cold seed didn't seat a usable sighting on one of the
  new scenes; inspect the new scene's cold pose / caption.

Also read the **S2 decomposition**: ideally S3−S2 > 0 (LTM adds value beyond STM)
and S2−S1 small (STM alone is near-neutral on single-goal ObjectNav), which would
attribute the gain to the hierarchical LTM specifically.

Cheap pre-check before the full ~2 h run: a 1-scene 1-category build
(`--scenes wcojb4TFT35 --categories chair`) to confirm the 3-setting path and the
S2 reporting wire up, then the full run.

## Out of scope (follow-ups)

- **Folding the revisit eval into the standard harness** (`analyze_ablation` /
  the val_mini 3-setting driver) — do only after Phase C confirms generalization.
- **Real object detector** for higher *binary* SPL — separate lever, unchanged
  here (soft-SPL remains primary; binary SPL is perception-bound at the 0.1 m
  success radius).
- **Larger matrix / more categories** (tv_monitor, plant, toilet) — the harness
  supports them via `--scenes`/`--categories`; deferred by the user's scope choice.
- **Mid/coarse layer evaluation** — needs a multi-goal or manipulation task; out
  of reach for single-goal ObjectNav.
