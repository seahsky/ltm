# docs/archive — design docs, plans, and runbooks for shipped / closed work

These documents describe features that **shipped** and research arcs that
**closed**. Their outcomes are recorded in `PHASE2_ABLATION_REPORT.md` (the live
running report) and summarized in `CLAUDE.md`. They are kept for design
rationale, not as active reference.

## Runbooks (superseded by `docs/race-box-runbook.md`)

- `phase2-race-runbook.md` — original RACE bring-up runbook.
- `phase3-qwen7b-runbook.md` — Qwen2.5-7B planner bring-up runbook (Runs 4–6).

## `superpowers/` — specs (design) + plans (TDD implementation) for shipped features

- `2026-05-25-memory-grounded-remembr-planner*` — memory-grounded ReMEmbR planner.
- `2026-05-25-navmesh-waypoint-controller-design.md` — navmesh `ShortestPathFollower` controller.
- `2026-05-27-fold-revisit-into-harness*` — folding the revisit eval into `analyze_ablation --revisit`.
- `2026-05-27-phase-c-multiscene-revisit*` — Phase-C multi-scene revisit ablation.
- `2026-05-28-goal-detector-binary-spl*` — goal detector for binary SPL (arc CLOSED, Run 11).
- `2026-06-05-race-email-notification-design.md` — RACE run-notification email (SHIPPED; the trio is live at `earshot/tools/notify/`, carried out of `scripts/` before the reset deleted it).

## Note on historical references

`PHASE2_ABLATION_REPORT.md` provenance/artifact tables written before
2026-06-12 reference these files at their **old** paths (`docs/…`,
`docs/superpowers/…`). Those are historical records and were intentionally left
unchanged; the files were relocated here on 2026-06-12.
