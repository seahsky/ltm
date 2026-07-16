# Anomaly-Response Build Plan — post-VM-validation redesign (2026-07-16)

Supersedes `anomaly_response_buildplan_2026-07-12.md`.
Sequenced from the grilling session that read `runs/anomresp-bed-s{1,3}` (riftvm, commit `ae452a2`) against the code.

That run was a VM validation smoke, and its +0.4454 warm delta is **not quotable**: it is the same `TEEsavR23oF`/bed cell that already gave +0.4685 and +0.4144, at n=3, superseded by the powered null (+0.0197, n=48).
What the run actually delivered is three structural findings that invalidate the n=64 controller census.

## What changed since 2026-07-12

**Compute.** Three-month reserved **Tesla V100-SXM3-32GB** (`riftvm`). Compute is no longer the constraint; the ~9 weeks to a predicted (unverified) ~Sep 15 deadline is.
Same fp16 stack as the L4, so this is kernel-selection noise, not a precision confound. The rule is narrow: **any A/B lives entirely on one host.** Never put an L4 arm against a V100 arm.

**Goal.** Absolute numbers lead. The contribution is the controller as a working system with competitive Find-SR; the memory result is an honest negative characterized against a strong baseline.
"Prove S3 is helpful" is explicitly let go: the powered matrix already answered it at n=48, and a stronger baseline can only shrink a delta that is already zero.

**Three structural findings, all following from code and physics rather than luck, so all applying to every cell of the n=64 matrix:**

1. **The interrupt fires on the bed at step 0.** Every onset in both settings fired at step 0–10 against `t_anom=30`. The alarm never triggered anything, in any episode. See ADR-0004.
2. **The gate rejects nothing.** `n_audio_gate_rejected=0` in all eight episodes; CLAP labelled the vacuum `alarm` every time. Discrimination is 0 for 8. See ADR-0004.
3. **The source can be a floor away.** The picker prefers cross-floor sources; the grid is single-floor; `nearest` ignores y. The feasibility gate's verdicts came out exactly inverted. See ADR-0003.

**Consequence.** Every `anomresp-*` and `anommxv-*` dataset, grid and run is superseded. The n=64 census cannot be salvaged; it measures a vacuum cleaner.

## Build-status legend

`RUN` built, needs a run · `WIRE` built, needs surfacing/arm · `EXTEND` build on an existing module · `NEW` genuinely new code.

---

## Phase F — the three fixes (blocking for anything audio)

| Fix | What | Status |
|---|---|---|
| **F1 bed → noise floor** | Calibrate `bg_gain` against the bed, never hand-pick. Extend `diagnose_onset_calib` to read the bed through the same grid, emit `RECOMMEND_BG_GAIN`, and assert `bed_max < onset_rms` with margin. Driver FATALs if unmet. | `EXTEND` |
| **F2 floor-constrained picker** | `pick_anomaly_source` requires `\|source_y − goal_y\| < ~1.0 m` alongside the 3 m xz separation. Free in the builder from view_point positions; two-env split holds. | `EXTEND` |
| **F3 `nearest` floor guard** | `RIRGrid.nearest` gains `max_dy`, defaulting to **None = current behaviour** so audiogoal/revisit/objectnav stay byte-identical. Anomaly-response passes `max_dy≈1.0`; off-floor → no coverage → silence. `diagnose_anomaly_feasibility` inherits the same rule or it keeps greenlighting fictions. | `EXTEND` |

Prior results are unaffected on their merits: in the audiogoal and scale-up matrices audio was decorative (anomaly == goal, retrieval visual), so fabricated cross-floor audio could not have moved their soft-SPL.
The bug only bites where audio is load-bearing, which is anomaly-response and nothing else.

---

## Phase 0 — gates (run before the runs they protect)

| Gate | Protects | Status | GREEN rule | If STOP |
|---|---|---|---|---|
| ~~G0.2 frontier-separation~~ | ~~S1+~~ | **DELETED — already fired HOLD.** CLIP separation measured at 0.020 against the 0.05 bar, three independent times. Do not re-run a measured fact. | — | — |
| **BLIP-2 VRAM preflight** | S1+ / S3+ | `RUN` (`race-blip2-frontier.sh` self-verifies) | 7B + BLIP-2 ITM + captioner + CLIP (+ CLAP for anomaly-response) co-fit in ~31.7 GB usable | fall back to the Phi-3.5-mini planner and accept the cross-quotability break |
| **G0.1 room-accuracy** | the two-rooms discrimination arm | `RUN` (built, never run) | CLIP separates the room pair at ≥ ~0.75 | discrimination stays context-free; ADR-0002 becomes future work |
| **G0.3 augmented-gate EER** | the distractor gate | `EXTEND` — **re-scoped**: no longer defends the bed (ADR-0004), only the distractor | augmentation lowers EER on convolved audio | keep the recal delta, skip augmentation |
| **G0.4 energy-gradient climbability** | realizable localization | `RUN` (built, locally verified, never run) | few local minima, spearman ≲ −0.4 | keep oracle source, disclose as upper bound (ADR-0001 fallback) |

---

## The runs

**R1 — Table 1, backbone credibility. Start first; it is the only unblocked work.**
Plain HM3D ObjectNav, **full val split**, all 20 scenes, no anomaly, no decoupling, no feasibility gate, nothing selected.
Arms: **S1 (geometric frontier) vs S1+ (BLIP-2 ITM frontier)**.
Touches none of the three broken things, so it can occupy the V100 while Phase F lands.
Comparable to VLFM's SPL 0.304 and the ~0.43 published SOTA; it is the direct answer to "44% Find-SR looks weak".
Meshes and episodes are already on disk (`20 content files, 20 with a usable mesh`).
Guardrail from the backbone memory: the renorm branch is required or `n_memory_chosen` silently collapses. Assert it fires.

**R2 — Table 2, the contribution.**
Anomaly-response matrix on **rebuilt** datasets and grids, after Phase F. Arms **S1+ vs S3+**.
Headline: the controller census (Anomaly-response SR), Find-SR@1.0 m, soft-SPL, cost.
Selection bias is disclosed and no longer load-bearing, because R1 faces the field.

**R3 — Controller-and-audio study.**
Realizable-vs-oracle localization (after G0.4). Room-normal vs room-anomalous discrimination on the two-rooms variant (after G0.1).
Both mechanisms are built and TDD-green and have never been run.

## Calibrate the target

VLFM's own number is **SPL 0.304**; the best published HM3D ObjectNav SPL is ~**0.43** (VLingNav, 79.1 SR / 42.9 SPL on HM3D-v1).
The honest ceiling here is **Find-SR@1.0 m toward 0.6–0.7 with SPL around 0.3**, not 0.75.
Binary SPL 0.75 is ~1.75x the best number anyone has published; the 0.750 in the arc's notes is Run 12's **oracle-STOP** ceiling, which is the definition of not-realizable.

## Dependency order

```
BLIP-2 VRAM preflight ──► R1 (full-val ObjectNav, S1 vs S1+)   ◄── start today, unblocked
F1 + F2 + F3 ──► rebuild datasets + grids ──► R2 (S1+ vs S3+)
  ├─ G0.1 ──► two-rooms discrimination ──┐
  ├─ G0.3 ──► augmented distractor gate ─┼─► R3
  └─ G0.4 ──► realizable localization ───┘
```

## Invariants (do not break)

- Byte-identical default paths. Every new behaviour env-gated or parameter-defaulted to current behaviour, asserted by a regression test. F3 is the live example: `max_dy=None` keeps every existing task bit-identical.
- Two-env audio split; single RIR grid (O(1) live convolution).
- One frozen LOCAL backbone per table, one host per A/B.
- Candidate-proposer seam contract; settings differ only by `disable_ltm` (and `semantic_frontier_weight` for the `+` arms).
- **Branch: `main` is primary** (since 2026-07-12; `lifelong-revisit-eval` is a retained mirror). The 2026-07-12 plan's "land on lifelong-revisit-eval" invariant is stale.

## Open risks

- **Yield gets worse before better.** F2 removes cross-floor sources that are currently passing. 16/97 will drop. Cells that survive measure what we claim to measure.
- **The memory delta may go negative.** A VLFM-grade searcher raises S1+ while the wrong-instance over-fire still costs S3+. "Our LTM does not help, and against a strong baseline it slightly hurts" is the honest outcome to be ready for.
- **BLIP-2 VRAM is on the knife edge** (~31–33 GB estimated against ~31.7 usable, and the driver's budget predates CLAP). May need the KV cap trimmed.
- **Deadline unverified.** ICRA 2027 (Seoul, May 24–28) has no CFP posted. ~Sep 15 is extrapolated from ICRA 2026, not confirmed.
- **`onset_step` is invisible in `summary.json`.** Three findings hid behind a green exit code and a `CONTROLLER_RAN` verdict. Onset provenance belongs in the summary and in the controller census verdict, or this recurs.
