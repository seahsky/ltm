# scripts/archive — closed-arc reproduction drivers

These RACE drivers reproduce research arcs that are **CLOSED** (the outcome is
recorded and the lever is exhausted). They are kept for the exact reproduction
recipe — env vars, checkpoint flags, matrix config — that a reviewer or paper
author may need to re-run, not because they are part of the active pipeline.
Their Python analyzers / dataset builders / diagnostics are **still live** in
`embodied_memory/scripts/` (the drivers are the only archived layer).

| Driver | Arc | Outcome (see `PHASE2_ABLATION_REPORT.md`) |
|--------|-----|-------------------------------------------|
| `race-revisit-detector.sh`     | Goal-detector binary-SPL push (c1–c9)     | Run 11 — detector OFF strictly dominates; arc CLOSED |
| `race-oracle-ladder.sh`        | Binary-SPL / oracle-stop bottleneck       | Run 12 — termination is localization-bound; arc CLOSED |
| `race-train-scorer.sh`         | LTM importance **R** head (scorer)        | Run 13/14 — heuristic R at/near ceiling; lever CLOSED |
| `race-multion.sh`              | MultiON sequential ObjectNav (K=3 chains) | Run 15 — clean null (zero compounding); arc CLOSED |
| `race-train-predictor.sh`      | LTM surprise **U** head (predictor)       | Run 18 — trained U regresses warm; lever CLOSED |
| `race-train-utility-scorer.sh` | Goal-proximity **U** head                 | Run 19 — regresses (only head to hurt binary SPL); lever CLOSED |
| `race-room-clip.sh`            | Coarse-affordance CLIP room classifier    | Run 20 — CLIP-grounded but never chosen (inert/conservative); arc CLOSED |

## Path caveat — these are NOT runnable in place

Each driver computes its repo root as
`REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"`, which assumes the
script lives directly in `scripts/`. From `scripts/archive/` that resolves to
`scripts/`, not the repo root, so the `cd "$REPO_ROOT"` will land in the wrong
place. To re-run one: **move it back up to `scripts/` first** (or change the
`/..` to `/../..` in the `REPO_ROOT` line).

The active drivers stay in `scripts/`: `race-revisit.sh` (headline revisit
eval), `race-soundspaces-spike.sh` (audio / SoundSpaces), `race-wide-s2.sh`,
`race-cross-env.sh`, `race-benchmark-success.sh`, plus the infra scripts
(`race-setup.sh`, `race-smoke.sh`, `notify-run.sh`, `setup-vm.sh`).
