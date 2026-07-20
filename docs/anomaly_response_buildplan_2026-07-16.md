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
Guardrail from the backbone memory: the renorm branch is required or `n_memory_chosen` silently collapses. Assert it fires.

**R1 is BLOCKED on the val mesh split (2026-07-17, VERIFIED).** The buildplan's
"20 content files, 20 with a usable mesh" was **wrong**: only **2 of 20** val scenes
have a mesh on the VM (`TEEsavR23oF`, `wcojb4TFT35` — the val_mini pair). The download
defaults to `hm3d_minival_full` (10 scenes) + a `val -> minival` symlink, so the other
**18 val scenes have no `.basis.glb`** and crash at sim init (`ESP_CHECK ... No Stage
Attributes`). That is what killed `r1v1`: the pinned group-by-scene iterator put all 100
requested episodes on the first missing-mesh scene (`4ok3usBNeis`) → 0 completed → driver
FATAL in 3m53s. **Unblock:** download the full mesh split on the VM (needs the Matterport
token in `.env`, no GPU):
```
rm -f data/hm3d/scene_datasets/hm3d/val            # drop the val->minival symlink
HM3D_SCENE_GROUP=hm3d_val_full bash embodied_memory/scripts/download_hm3d.sh
```
**Guard added:** `inventory_hm3d_meshes.py` (+5 TDD) reports usable/missing meshes per
split; `race-r1-objectnav.sh` step **[4b]** now FATALs in seconds with this fix instead of
burning GPU on per-episode crashes. Until the download lands, the only runnable ObjectNav is
**val_mini (2 scenes)** — enough to de-risk the S1+/BLIP-2 path (D1), not enough to face VLFM.

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

---

## Grilling addendum — R1 smoke read (2026-07-17)

Grilling session over the `runs/r1nav-s1` "FAILED" email (`b78d2c0`, riftvm).
The run is **not** a failure: 30/30 episodes completed, `no_crash` passed, mean soft-SPL 0.1587.
The `❌ exit 1` is the memory-liveness pass-gate firing on a `--setting 1` memory-off run, where those gates can only fail.
It was a pipeline smoke, run as a raw `run_hm3d_pol` call — **not** the R1 driver — so it exercised only the S1 (geometric-frontier) arm on val_mini.

### Decisions

- **D1 — the smoke de-risked only the safe half.**
  It validated the shared substrate (env, episode loop over val scenes, soft-SPL, throughput) but never touched R1's novel surface: BLIP-2 ITM loading + co-fitting the 7B in VRAM, the S1+ arm producing non-flat spread, the vacuous-arm gate, the paired analysis.
  Next step is a **driver-level** smoke: `race-r1-objectnav.sh --tag r1smoke --split val_mini --n-episodes 20`.
- **D2 — the exit-code semantics are now setting-aware.**
  A memory-off `--setting 1` run passes on `no_crash` alone; memory-ON settings keep the strict full gate.
  This kills the false ❌ that trains the operator to ignore ❌ on baselines (the inverse of the open-risk-#5 alarm-fatigue failure). Landed (below).
- **D3 — R1 headlines native SPL@0.1 m (ring verified 2026-07-17).**
  The suspected ring gap was checked and does **not** exist: `race-r1-preflight.sh` read `success_distance: 0.1` from the canonical `objectnav_hm3d.yaml`, the standard HM3D ObjectNav ring VLFM's 0.304 / VLingNav's 0.429 report on.
  So the harness's native binary `spl` / SR@0.1 m are already cross-quotable and are R1's Table-1 headline — **no metric wiring is needed**.
  This **reverses** the earlier plan to add a 1.0 m `spl_1m` headline: a 1.0 m SPL is a *relaxed* ring that would OVERSTATE us against VLFM's 0.1 m number. The arc's "the benchmark uses 1.0 m" belief was wrong (it conflated the self-invented `success_1m` reach diagnostic with the benchmark).
  The residual R1 risk is now **capability, not metric**: native SPL@0.1 m is localization-bound (the smoke shows native mean SPL ≈ 0.05–0.15 << 0.304), and S1+ upgrades frontier choice, not STOP-localization, so it may not close the gap. D5 already commits us to shipping the honest number.
  See **ADR-0005** (rewritten with the verification); glossary term `Benchmark SPL` corrected in `CONTEXT.md`.
- **D4 — pre-register `w=0.5`** for the S1+ semantic-frontier weight.
  No sweep on val / val_mini (that is tuning on a subset of R1's own test set); tune on **train** only if at all, freeze, then run full val.
  Disclose in the paper that `w` was pre-registered, not fit.
- **D5 — R1 interpretation rule, fixed before the number is seen.**
  Report S1 and S1+ benchmark SPL as-measured against VLFM 0.304 / VLingNav 0.429.
  Claim "strong baseline" only if S1+ ≥ ~0.25 (within ~20 % of VLFM, allowing for our un-fine-tuned 2B stack); below that, R1 still ships with the baseline-strength language calibrated down to the measured value.
  R1 does **not** gate R2 either way. No post-hoc reframing.

### Code landed (2026-07-17, TDD)

- **Setting-aware pass-gate (D2).**
  `embodied_memory/pass_gate.py` — `required_pass_conditions(setting)` + `run_passed(...)` (pure, dependency-free).
  `run_hm3d_pol.main()` now gates the exit code on the setting-appropriate subset while printing per-condition PASS/FAIL honestly.
  Tests: `embodied_memory/scripts/test_pass_conditions.py` (8 cases, incl. the `r1nav-s1` regression; s2/None/no_strict_pass stay strict).
- **Benchmark-SPL math (now a RELAXED-ring diagnostic, not the headline).**
  `embodied_memory/metrics.py` — `compute_benchmark_spl(stopped, dist_at_stop, geodesic_optimal, path_len_taken, success_radius=1.0)` (pure).
  Tests: `embodied_memory/scripts/test_metrics.py` (8 cases).
  Retained as an optional ring-parameterized reach diagnostic; the ring check (D3) demoted it from the R1 headline. Verified locally by direct file-load and on the V100.
- **`race-r1-preflight.sh`** — $0 R1 preflight: pull, env, run the two test suites, read the SPL success ring, and verdict the 0.1-vs-1.0 m question. It answered D3.

### Ring verification result (2026-07-17, V100)

`race-r1-preflight.sh` → `success_distance: 0.1` on the canonical `objectnav_hm3d.yaml`; both test suites green (metrics 8/8, pass_gate 8/8).
**Outcome: native `spl`/SR@0.1 m ARE the VLFM-comparable Table-1 headline; no `spl_1m` wiring.** See D3 / ADR-0005.

### Next actions (unblocked)

1. **Driver-level smoke (D1):** `race-r1-objectnav.sh --tag r1smoke --split val_mini --n-episodes 20` — exercises S1+, BLIP-2, the vacuous-arm gate, the paired analysis.
2. **BLIP-2 VRAM preflight:** `race-blip2-frontier.sh --tag r1pre --skip-ab --planner Qwen/Qwen2.5-7B-Instruct`.
3. **Full-val R1** at `w=0.5`; report native binary SPL@0.1 m + SR@0.1 m vs VLFM 0.304 / VLingNav 0.429; interpret per D5.
   Face the capability risk (D3): if S1+ native SPL stays ≈ S1 (localization bound), the honest finding is that the frontier is not the bottleneck — STOP-localization is (cf. the closed L3 detector arc).

---

## Grilling addendum — R1 val_mini smoke read (2026-07-20)

Grilling session over the `r1spin2` run email (`45246a7`, riftvm, exit 0, 5h27m).
The capability risk D3 pre-registered materialized: the smoke settled the S1+ question and, with it, the 2026-07-16 "competitive absolute numbers" bet.
Durable decision record: **ADR-0006**; glossary updated in `CONTEXT.md` (S1+ demoted; new "R1 de-risk smoke vs Table 1" term).

### What the smoke showed

`r1spin2` is the **D1 de-risk smoke** on `val_mini` (2 scenes, 30 episodes), memory OFF both arms, 7B planner, anti-spin ON, 500-step budget — **not** Table 1 (full val is still blocked on the 18 missing meshes on riftvm).

1. **BLIP-2 semantic frontier (S1+) does not lift the number.** Paired SPL −0.0175 (CI [−0.053, 0], slightly hurts), soft-SPL +0.010 (n.s.), succ@1m identical 0.233. Vacuous-arm gate GREEN (13,405 scores, spread 0.45) ⇒ inert, not un-fired. 4th independent non-lift of a semantic frontier.
2. **The backbone is a weak searcher.** S1 SPL 0.031 / S1+ 0.014, soft-SPL ~0.14, SR@0.1 m 0.067/0.033, reach-within-1 m 0.233 — vs VLFM 0.304 and the plan target (SPL ~0.3, Find-SR@1m 0.6–0.7). ~10x under on SPL, ~3x under on the 1 m reach, at the full 500-step budget (capability, not timeout).
3. **Spin reduced, not dead.** With anti-spin ON: ep25 = 298 turns / 0 STOP; ep29 = forward 0 / replan_stuck 23 / unreachable_escape fired 10× yet stuck.

### Decisions

- **A — Full-val R1 still runs, reframed.** Purpose changes from "prove competitive" to "the honest Table-1 baseline the paper needs." Full val won't 10x a fixed backbone; you still need the 20-scene number (a 2-scene number is not quotable). Keeps **S1 vs S1+** so the BLIP-2 negative is powered.
- **B — S1+ demoted to a documented negative; geometric S1/S3 are the paper's spine.** Headline memory delta reverts to **S3 − S1**. **R2 drops the "+" arms** and runs S1 vs S3: S1+ ≈ S1 so "+" adds only overhead, it removes the BLIP-2 VRAM knife-edge on the V100, and S1/S3 stay cross-quotable to the whole prior arc. S1+ earns its keep only as the rebuttal to "re-run S1 with a real explorer" (VLFM's own head doesn't beat geometric here).
- **C — Retreat accepted; backbone/STOP-localization spend ruled out.** The contribution returns to the stable 06-30 anchor: controller-as-working-system + LTM honest negative, on a frozen, honestly-weak backbone whose absolute capability is out of scope. STOP-localization is the measured bottleneck but every realizable proxy failed and a real policy destroys the candidate-pool thesis — do not spend on it.
- **D — Bounded, diagnose-first spin fix into the frozen R1/R2 backbone.** Spin is a fairness + R2-integrity issue (a spinning agent aborts the investigate detour on the step cap → corrupts the controller census), *not* the ruled-out backbone spend. Diagnose on the existing per-episode JSONs first ($0), one targeted fix, land it in **both** R1 and R2 (one frozen backbone), or disclose the spin rate as a limitation.
- **E — The paper leads with the controller as a system.** The LTM negative is a strong supporting section; the weak backbone is disclosed as a scoping choice, not apologized for. Consequence: **R2 (re-earning the ADR-0003/0004-invalidated n=64 census on Phase-F data) is now the single most important run in the project.** Headline positive is Find-SR ≈ 0.44, defended on internal validity + detour-cost, not a leaderboard.

### Revised run config (supersedes the S3+ − S1+ headline in the body)

- **R1 (Table 1):** full val, **S1 vs S1+**, memory OFF, 7B, anti-spin ON, `w=0.5` pre-registered. Report SR/SPL@0.1 m vs VLFM/VLingNav + soft-SPL as a ring-independent companion. S1+ reported as the powered semantic-frontier negative.
- **R2 (contribution):** anomaly-response matrix on Phase-F-rebuilt data, **S1 vs S3** (no "+"). Headline = controller census (Anomaly-response SR, Find-SR@1 m, cost) + memory delta S3 − S1.

### Next actions (sequenced)

1. **Spin diagnosis ($0, no re-run):** `diagnose_spin.py` on the `r1spin2` per-episode JSONs on RACE — quantify the spin rate, split the ep25 (propose/turn-forever) vs ep29 (unreachable-escape-not-converging) signatures.
2. **Bounded spin fix** (if the diagnosis points to one), landed into the frozen anti-spin backbone; verify; else disclose the rate.
3. **Download the 18 missing full-val meshes on riftvm:** `HM3D_SCENE_GROUP=hm3d_val_full bash embodied_memory/scripts/download_hm3d.sh` (Matterport token in `.env`, no GPU; the `[4b]` mesh preflight FATALs until this lands).
4. **Full-val R1** (S1 vs S1+) on the fixed backbone → the honest Table-1 number.
5. **Phase F** (F1/F2/F3) → rebuild datasets + grids → **R2** (S1 vs S3) = the critical run.
