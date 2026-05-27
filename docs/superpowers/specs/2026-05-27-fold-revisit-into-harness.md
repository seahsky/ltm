# Spec — Fold the revisit eval into the standard ablation harness

**Date:** 2026-05-27
**Status:** approved
**Related:** `2026-05-27-phase-c-multiscene-revisit-design.md` (the eval this
folds in), `PHASE2_ABLATION_REPORT.md` Run 9 (Phase C result).

## Problem

Phase C closed with **Gate-A GREEN** — the LTM's warm-revisit benefit
generalizes across scenes (warm soft-SPL S3−S1 = +0.240, 90% CI [+0.073,+0.417],
p=0.008, n=12; S2 decomposition attributes the gain to the LTM). But the
visit-order revisit analysis still lives in a **separate script**
(`embodied_memory/scripts/analyze_revisit.py`), parallel to the standard
val_mini analyzer (`analyze_ablation.py`). There are two analyzer front doors
where there should be one.

**Key enabling fact (verified):** standard G4 runs *already contain warm
revisits* — the LTM persists across episodes within a process — so
`analyze_revisit.py` runs cleanly on the existing `runs/abl-s{1,2,3}-qwen` dirs
today. This integration is a **refactor of the entry point, not a re-run**; the
revisit logic itself is unchanged.

## Decisions (locked with the user)

1. **Analyzer shape — `--revisit` flag + shared library.** Add a `--revisit`
   mode to `analyze_ablation.py` (the one front door). The revisit logic stays
   in `analyze_revisit.py` as an imported library; `analyze_ablation.main()`
   **lazily imports** `analyze_revisit` only when `--revisit` is passed, then
   calls `load_revisit_run` + `print_report`. The lazy import avoids the
   circular import (`analyze_revisit` imports `analyze_ablation`'s
   bootstrap/loaders at module top). `analyze_revisit.py` **stays runnable
   standalone** (back-compat alias).

2. **Opt-in only.** Visit-order / Gate-A output appears **only** with
   `--revisit`. The standard `analyze_ablation` Phase-2 gate output is
   **unchanged** (no new block on every run). Rationale: on standard
   *interleaved* runs the data yields Gate-A verdict **(b)** (memory fires but
   warm S3−S1 ≈ 0); auto-printing a non-green verdict everywhere would mislead.
   Gate-A **(a)** GREEN only appears on the controlled-start dataset where warm
   memory is non-empty.

3. **Keep the two RACE drivers separate.** The controlled-start dataset build
   is genuinely revisit-specific, so `scripts/race-revisit.sh` stays its own
   driver. Only its **final analysis call** is repointed to
   `analyze_ablation --revisit`. `scripts/race-smoke.sh` is untouched.

## Acceptance criteria

- `python analyze_ablation.py --revisit <dirs>` produces output **byte-identical**
  to `python analyze_revisit.py <dirs>` on the merged G4 runs
  (`runs/abl-s{1,2,3}-qwen`) — equivalence diff is empty.
- `python analyze_ablation.py <dirs>` (no flag) prints the Phase-2 gate exactly
  as before — no revisit block.
- New `test_analyze_ablation.py` passes (4 cases: lazy-import contract,
  `--revisit` runs visit-order report, no-flag runs phase-2 gate, `--revisit`
  with S2 decomposition).
- Importing `analyze_ablation` does **not** pull `analyze_revisit` into
  `sys.modules` (the opt-in lazy-import invariant).
- All existing sanity suites stay green: `test_analyze_revisit.py`,
  `test_make_revisit_smoke.py`, `test_spl_guard.py`, `test_text_encode_util.py`,
  `test_episode_order.py`.
- `bash -n scripts/race-revisit.sh` clean; the driver's pre-test sanity block
  runs `test_analyze_ablation.py` (FATAL-on-fail) and its step [6/6] analysis
  call invokes `analyze_ablation.py --revisit`.

## Out of scope

- Real object detector for higher binary SPL (separate perception milestone).
- Widening the revisit matrix (tv_monitor/plant/toilet, more scenes) — driver
  already supports it via `--scenes`/`--categories`.
- Merging the two shell drivers or auto-running Gate-A on standard runs (both
  decided against above).
- Any change to the dialogue/MSC path (`dialogue_memory/`).
