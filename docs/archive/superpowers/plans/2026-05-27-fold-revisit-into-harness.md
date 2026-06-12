# Plan — Fold the revisit eval into the standard ablation harness

**Date:** 2026-05-27
**Spec:** `docs/superpowers/specs/2026-05-27-fold-revisit-into-harness.md`

Execution: subagent-driven TDD. Tests written first (red), then the dispatch
(green), then mechanical driver + doc edits, then controller-run verification
(incl. the equivalence diff on real run dirs).

## Task 1 — TDD: `test_analyze_ablation.py` (red) → `--revisit` dispatch (green)

### 1a. New `embodied_memory/scripts/test_analyze_ablation.py`
Standalone `case_*/main()/sys.exit` runner (NOT pytest), numpy/stdlib-only,
matching `test_analyze_revisit.py` conventions: `sys.path.insert(0, dirname)`;
inline `tempfile.TemporaryDirectory` fixtures; `assert` + per-case
`print("  case ...: OK")`; `main()` prints "All cases passed." and returns 0;
`sys.exit(main())`.

Reuse the temp run-dir fixture pattern from
`test_analyze_revisit.py::case_load_reads_episode_files`: write `summary.json`
with `{"ablation": {"setting": N}, "episodes": []}` + per-episode
`episode_*.json` carrying `scene_id`, `episode_id`, `target_category`,
`episode_idx`, `soft_spl`, `spl`, `success`, `n_steps`, `distance_to_goal`,
`n_memory_chosen`, `n_memory_candidates`, `decisions[]`. One fixture satisfies
**both** loaders (`load_run` and `load_revisit_run`).

Cases (order matters — lazy-import case must run first):
- `case_lazy_import_contract` (**runs first**): at module top import only
  `analyze_ablation`; assert `'analyze_revisit' not in sys.modules`. Pins the
  opt-in lazy-import — importing the standard analyzer must not pull the revisit
  module. Subsequent `--revisit` cases populate `sys.modules`.
- `case_revisit_flag_runs_visit_order_report`: `main(["--revisit", s1, s3])`,
  capture stdout. Assert visit-order markers present — `"visit groups"`,
  `"cold vs warm"`, `"WARM S3 - S1"`, `"Gate A verdict"` — and the phase-2
  marker `"phase 2 gate"` **absent**; return value 0.
- `case_no_flag_runs_phase2_gate`: `main([s1, s3])`. Assert `"phase 2 gate"`
  present and `"Gate A verdict"` **absent** (standard path regression).
- `case_revisit_with_s2_decomposition`: `main(["--revisit", s1, s2, s3])`.
  Assert `"WARM S2 - S1"` and `"WARM S3 - S2"` appear (delegation passes all
  dirs through).

Build the s1/s2/s3 fixture dirs so the loaders infer settings 1/2/3 and there
is at least one cold + one warm visit per category (so the warm-delta path and
S2 blocks actually execute). Markers verified against current source:
`analyze_revisit.print_report` emits "visit groups", "cold vs warm stratified",
"WARM S3 - S1", "WARM S2 - S1", "WARM S3 - S2", "Gate A verdict";
`analyze_ablation.print_phase2_gate` emits "phase 2 gate".

### 1b. `analyze_ablation.py` — add `--revisit` dispatch
In `main()` (~lines 395-420):
- Add argparse flag:
  ```python
  parser.add_argument("--revisit", action="store_true",
      help="Visit-order (revisit) analysis: cold/warm stratify, warm-paired "
           "soft-SPL delta + S2 decomposition + Gate-A verdict "
           "(delegates to analyze_revisit).")
  ```
- Right after the `len(args.run_dirs) < 2` check, branch before the standard path:
  ```python
  if args.revisit:
      import analyze_revisit  # lazy: avoids circular import (it imports us)
      runs = [analyze_revisit.load_revisit_run(p) for p in args.run_dirs]
      analyze_revisit.print_report(runs, args.bootstrap)
      return 0
  ```
- Leave the existing Phase-2 path untouched. `--bootstrap` is reused; note in a
  one-line comment that `--ci` does not apply to the revisit path
  (`print_report` hardcodes 90% CI, matching the standalone script).

### Done criteria for Task 1
`python embodied_memory/scripts/test_analyze_ablation.py` → all cases pass;
`python embodied_memory/scripts/test_analyze_revisit.py` still green.

## Task 2 — `scripts/race-revisit.sh`: repoint + extend sanity suite
- Step [6/6] (lines ~146-148): change the analysis call
  `python embodied_memory/scripts/analyze_revisit.py $OUT_DIRS`
  → `python embodied_memory/scripts/analyze_ablation.py --revisit $OUT_DIRS`,
  and update the `banner` text to match. Output identical; only the entry point
  moves.
- Pre-test sanity block (lines ~85-95): add a FATAL-on-fail line running
  `test_analyze_ablation.py`, same pattern as the others, so the `--revisit`
  dispatch is verified before any paid live run.
- `bash -n scripts/race-revisit.sh` must stay clean.

## Task 3 — Docs (CLAUDE.md)
Under "Next milestone" / "Running the ablation" / "Revisit harness": reflect
that the revisit analysis is now `analyze_ablation --revisit` (first-class
mode); note `analyze_revisit.py` remains a standalone alias. Keep the separate
dataset-build (`make_revisit_smoke.py` / `race-revisit.sh`) framing.
(PHASE2_ABLATION_REPORT.md harness-integration note deferred to the RACE
confirmation run.)

## Verification (controller, local — no Habitat)
1. Static: `bash -n scripts/race-revisit.sh`; `python -c "import ast;
   ast.parse(open('.../analyze_ablation.py').read())"`.
2. `python .../test_analyze_ablation.py` → all pass.
3. Regression: all five existing sanity suites green.
4. **Equivalence** (strongest): `analyze_revisit.py runs/abl-s{1,2,3}-qwen` vs
   `analyze_ablation.py --revisit runs/abl-s{1,2,3}-qwen` → `diff` empty.
5. Standard path: `analyze_ablation.py runs/abl-s{1,2,3}-qwen` prints the
   Phase-2 gate exactly as before.
