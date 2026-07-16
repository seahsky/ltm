# Anomaly-Response Build Plan — post-powered-matrix redesign (2026-07-12)

> **SUPERSEDED by `anomaly_response_buildplan_2026-07-16.md`.**
> Kept for provenance. Three rows below are known stale:
> **G0.2** lists a gate that had already fired HOLD (CLIP separation 0.020 vs the 0.05 bar, measured three times);
> the branch invariant says land on `lifelong-revisit-eval`, but `main` has been primary since 2026-07-12;
> and the whole plan is built on the n=64 controller census, which the 2026-07-16 session found to be measuring
> bed-triggered interrupts on partly cross-floor geometry (ADR-0003, ADR-0004).

Sequenced from the grilling session that checked the project against the 16-item requirement list.
The powered val matrix (`anommxv`, n=64) already gave two answers: the controller is a strong systems result (86% full investigate+resume), and the memory soft-SPL delta is an honest null at scale (+0.020, n=48, n.s.).
This plan builds the pieces that were on the requirement list but missing, and hardens the two claims the paper actually makes.

Everything follows the arc's discipline: **diagnose-first $0 gates before any paid run, byte-identical default paths (new behavior env-gated default-OFF), the two-env audio split (offline RIR render → O(1) live convolution), a single RIR grid, and one frozen LOCAL backbone for cross-quotability.**

## What the studies are

Two focused sub-studies share one dataset (decided in grilling — a full factorial is infeasible; the powered matrix already cost 17 h for 16 cells).

- **Study 1 — memory boundary.** `S1+ vs S3` across warm/cold × scenes/categories. Reviewer-proof strong baseline. Outputs: warm soft-SPL S3−S1+, Find-SR.
- **Study 2 — controller + audio.** Two A/Bs on a smaller cell set: realizable-vs-oracle localization, and room-normal-vs-anomalous discrimination. Outputs: Anomaly-response SR, discrimination (false/correct-interrupt), cost.

The **4 headline metrics**, each reported per setting: **Find-SR@1.0 m · Anomaly-response SR · soft-SPL · cost (steps + wall-clock).**
Design decisions behind them: `docs/adr/0001-realizable-anomaly-localization.md`, `docs/adr/0002-scene-conditioned-anomaly.md`; vocabulary in `CONTEXT.md`.

## Build-status legend

`RUN` already built, just needs a gate + a run · `WIRE` built, needs surfacing/arm · `EXTEND` build on an existing module · `NEW` genuinely new code.

---

## Phase 0 — $0 / offline go/no-go gates (run ALL first)

Each gate protects a later phase and prunes its axis if red. None spends live-episode GPU beyond a render/embed.

| Gate | Protects | Tool | Status | GREEN rule | If STOP |
|---|---|---|---|---|---|
| **G0.1 room-accuracy** | Phase 3 (scene-conditioning) | `diagnose_room_clip_cosines.py`, extend to report bathroom-vs-bedroom confusion on real HM3D frames | `EXTEND` (classifier + diagnostic exist) | CLIP separates the two rooms at ≥ ~0.75 accuracy | drop #16 to future work; keep context-free gate |
| **G0.2 frontier-separation** | S1+ strong baseline | `diagnose_clip_frontier_separation.py` (reuse the `caprl-gate` renders — no new render) | `RUN` (exists) | goal-containing frontiers separate from empties by a usable margin | S1+ ≈ S1 → the baseline defense is vacuous; fall back to "disclose weak baseline" |
| **G0.3 augmented-gate EER** | Phase 2 (aug + robust gate) | `diagnose_convolved_anomaly_calib.py`, extend with the augmentation set | `EXTEND` (convolved-gate diagnostic exists) | augmentation lowers EER on convolved audio vs the clean-calibrated delta | augmentation doesn't help → keep the recal delta (−0.2557), skip aug |
| **G0.4 energy-gradient climbability** | Phase 4 (realizable localization) | `NEW diagnose_energy_gradient.py` — replay the RIR grid, report spearman(energy, −dist) + count local minima on paths to source | `NEW` (small, pure grid/replay, ~12 TDD) | gradient is monotone-enough to climb (few local minima, spearman ≲ −0.4) | realizable localization is infeasible → keep oracle source, disclose as upper bound (per ADR-0001 fallback) |

Gate output convention: each prints `GATE_RESULT=GO|BORDERLINE|STOP` + the recommended parameters, same as `diagnose_convolved_anomaly_calib`.
Run them together; they are independent.

---

## Phase 1 — cheap wiring (low risk, unblocks metrics + the strong baseline)

Do this regardless of gate outcomes — it's the reporting spine.

- **P1.1 `WIRE` the 4 metrics.** Surface **Find-SR@1.0 m**, **Anomaly-response SR**, and **cost (steps + wall-clock)** as first-class fields in `summary.episodes` (`episode_runner` `RunSummary`) and the analyzer. soft-SPL is already there; the controller census (`diagnose_anomaly_controller.py`) already computes investigated/resumed/primary_completed, so Anomaly-SR is a projection of existing counters. TDD mirrors `test_summary_query_expanded`.
- **P1.2 `RUN` the S1+ semantic-frontier baseline arm.** Gated on **G0.2 GREEN**. The lever is built (`LTM_SEMANTIC_FRONTIER_BACKEND=clip`, `semantic_frontier_weight`); add it as an arm in `race-anomaly-response.sh` / the matrix driver so the headline delta becomes **S3 − S1+**. Guardrail from the backbone memory: the renorm branch is required or `n_memory_chosen` silently collapses — assert it fires.

Default path (no `semantic_frontier_weight`, no new metric flags) stays byte-identical; both drivers pre-verify the new tests.

---

## Phase 2 — audio augmentation + robust gate (gated by G0.3 GREEN)

Attacks the documented clean→convolved calibration cliff and the loud-bed false-fire.

- **P2.1 `NEW audio.augment_clip`.** Deterministic variant generator: added background at random SNR, reverb/room-size jitter, pitch- and time-shift, plus the RIR convolution itself. Pure function, seeded (no `Math.random`-style nondeterminism — vary by index), ~10 TDD. Lives beside `render_step_audio`, shares `diotic_collapse` so the calibration domain matches the live signal.
- **P2.2 `EXTEND` calibration.** Recalibrate `is_anomaly` delta/tau on the augmented+convolved set via the G0.3 diagnostic; thread the new recommended delta through `resolve_anomaly_gate_thresholds` as the anomaly-response default (replacing/validating the −0.2557/0.0341 recal).

Byte-identical when augmentation is unused; `is_anomaly` signature unchanged for the context-free path.

---

## Phase 3 — scene-conditioned discrimination (gated by G0.1 GREEN)

Makes keyword #16 real and gives discrimination a task-level job (the long-deferred C3 gap).

- **P3.1 `EXTEND is_anomaly` → room-conditioned.** Add a `ROOM_PRIOR: room → expected-sound-set` (NEW; distinct from the existing `CATEGORY_ROOM_PRIOR` which maps object→room). Wire `classify_room_clip` into the gate: fire iff the heard class is unexpected for the detected room. Env-gated `LTM_ROOM_CONDITIONED_ANOMALY`, default-OFF → context-free path byte-identical. ~12 TDD.
- **P3.2 `EXTEND make_anomaly_response_smoke` → two-rooms variant.** Same context-dependent clip (water/appliance hum, NOT alarm/glass/cry) placed room-NORMAL (agent must not interrupt) vs room-ANOMALOUS (must interrupt). Single RIR grid preserved. Needs the ambiguous-sound clip set fetched.
- **P3.3 discrimination A/B.** New metrics: **false-interrupt rate** (normal episodes, want 0) + **correct-interrupt rate** (anomaly episodes). Report the interrupt confusion matrix as a Study-2 diagnostic.

---

## Phase 4 — realizable localization (gated by G0.4 GREEN)

Replaces oracle-fed navigation with a genuine acoustic capability, A/B'd against the oracle upper bound (ADR-0001).

- **P4.1 `EXTEND` denser RIR render** if G0.4 says the 24-cell grid is too sparse to climb (`render_rir_grid --n-cells`).
- **P4.2 `NEW` energy-gradient INVESTIGATE mode.** Extend `anomaly_controller` INVESTIGATE: instead of point-goaling to the oracle xyz, step toward higher live binaural RMS, bias heading by the L/R level sign, and STOP on the energy peak confirmed by a visual detection of the anomaly object. Env-gated `LTM_REALIZABLE_LOCALIZATION`, default-OFF → oracle path byte-identical. The live RMS read is non-privileged (it's "how loud here", not the source coordinate) — assert no GT-distance read leaks in (the closed energy-STOP bug).
- **P4.3 A/B** realizable vs oracle on the Study-2 cells → the honest "reach-within-~1 m" number + the oracle upper bound side by side.

---

## Phase 5 — the two powered matrices

Gate-first per phase; only spend on axes that cleared Phase 0.

- **Study 1** (`race-anomaly-response-matrix.sh`, extended with the S1+ arm): S1+ vs S3, warm/cold, all feasible cells. Headline: warm soft-SPL **S3 − S1+**, Find-SR, module decomposition S2−S1+/S3−S2.
- **Study 2** (smaller GO cell set): realizable×oracle and room-normal×room-anomalous. Headline: Anomaly-response SR, discrimination confusion matrix, cost; realizable-vs-oracle gap.

Continue-on-failure + resumable (existing matrix-driver behavior); per-cell n≥3, pool for power; disclose the feasibility-selected sample bias (the powered-matrix caveat).

---

## Dependency order (critical path)

```
Phase 0 gates (all parallel, $0)
  ├─ G0.2 ──► P1.2 (S1+ run) ─────────────► Study 1
  ├─ G0.1 ──► P3.1 room gate ─► P3.2 ─► P3.3 ─┐
  ├─ G0.3 ──► P2.1 aug ─► P2.2 recal ─────────┼─► Study 2
  └─ G0.4 ──► P4.1 render ─► P4.2 climb ─► P4.3┘
P1.1 metric surfacing  ──────────────────────► both studies (do first)
```

Cheapest, highest-certainty first: **P1.1 → Phase 0 gates → P1.2/Study 1** (the memory-boundary result is nearly a re-run against the strong baseline).
The two genuine new builds (P2.1 augmentation, P4.2 energy-gradient) sit behind their gates so a red gate kills the build before it's written.

## Invariants (do not break)

- Byte-identical default paths — every new behavior env-gated, default-OFF, asserted by a static/regression test.
- Two-env audio split; single RIR grid (O(1) live convolution) — the same-sound/two-rooms design was chosen specifically to keep this.
- One frozen LOCAL backbone for the headline (Mistral-7B planner + 2B captioner) — no hosted-model drift in the scored arms.
- Candidate-proposer seam contract; S-settings differ only by `disable_ltm` (now also `semantic_frontier_weight` for S1+).
- Branch: the arc lives on `lifelong-revisit-eval`; land there (not main-only) before any RACE run.

## Open risks (carried from grilling)

- **Energy-gradient ceiling (~1 m, level-only sim).** If G0.4 is red, ADR-0001's fallback is oracle + disclosure; the controller headline survives either way.
- **CLIP room noise (~0.30 cosines).** G0.1 is the kill-switch for #16; if red, discrimination stays context-free and #16 becomes future work.
- **Cross-run noise > effect at small n.** Per-cell n=3; pool and quote CIs; the S1 noise floor bounds the consume-style A/Bs.
- **Feasibility-selected sample.** Study 1's cells are decouplable-geometry-biased; disclose (already a known powered-matrix caveat).
