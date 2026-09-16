# `dream-2` is a null about a gate that could not open

**Status:** accepted (2026-09-16) for the two code changes below; the four measurements it pre-registers are unrun and are Sky's to book.
The changes are a units fix and a bounded cap, both in `memory/consolidate.py`, both covered by the mac suite.
Everything the ADR says about what to run next costs box time and is written as a pre-registration, not as a decision already taken.

`dream-2` ran the DREAM arm against `full` over 19 scenes, 282 episodes an arm, 4h45m, at commit `1706b6e`.
`full` reached the source on 94 of 282 (33.3%), `dream` on 92 (32.6%), net −2 over 56 discordant pairs, exact McNemar p = 0.8939.
Read as a result about memory, that is a clean null.
**It is not a result about memory.** The arm's `M^E` held 45 rows after 282 episodes, 38 of them written by a single episode, and the gate that was supposed to admit the rest could not open at the value it was set to.

## What the instrumentation said

`tools/dream_report.py` over the run:

| section | measurement |
|---|---|
| C | 275 of 282 episodes retained NOTHING; 45 rows written over the whole arm; `rows_added` max 38 in one episode |
| C | `M^P` rows per episode: min 0, median 1, max 2 |
| B | `M^E` key pairwise cosine, median of the per-episode means: 0.7243; median max 1.0000 |
| A | `omega^E` moved by more than 0.05 on 236 of 281 episodes; `M^K` carried zero weight on all 281 |
| D | 1 of 282 episodes started against an empty `M^E`; largest `M^E` any episode saw: 45 rows |

Section D is the one that says `2b04fd9`'s chain WORKED.
`dream-1` started all 19 scenes from empty; `dream-2` started one.
The chain is not the defect.

## The defect: eq. 10 degenerated to eq. 12

`I_j = alpha*C_j + beta*U_j + gamma*N_j` (eq. 10), all three weights at 1.0, retained on `I_j > eta` with `eta = 0.5` (eq. 13).

`_shares` normalised `C_j` and `U_j` into shares of an episode total, so `sum_j C_j = 1` exactly and the mean share is `1/J`.
`N_j = 1 - max cos(h_j, m)` (eq. 12) is a cosine and carries no such factor.
A 250-step episode segments into `J` between 21 and 84 at the shipped `coherence` 0.99, `min_segment` 3, `max_segment` 12 — verified by executing `segment_trajectory`, and corroborated by the run's own `rows_added` max of 38 being that episode's `J`.
So `mean(C_j + U_j)` was between 0.024 and 0.095 against an `N_j` of order 1.

**`I_j` was a novelty score with a rounding error added, and `eta` was a pure novelty threshold.**

That threshold was priced against the one regime in which it cannot bind.
`tools/ablation_sweep.sh` recorded the reasoning before the run: the box's real `I_j` ran 1.0000 to 1.2901 *with an empty memory*, where every `N_j` is 1.0 by construction, "so 0.5 retains everything on episode one and starts to bite as soon as `M^E` has rows to be unlike."
It did not start to bite. It shut.
With `M^E` non-empty its keys sat at 0.909 to 0.989 cosine, putting `N_j` in [0.011, 0.091], so `max I_j` was about 0.38 against a threshold of 0.5.
The run's own `dream_importance_min` of 0.0128 is a nearest-neighbour cosine of 0.9872, measured.

The first episode of a chain therefore writes its whole trajectory and every later episode writes nothing.
`novelty`'s docstring argued for returning 1.0 rather than 2.0 on an empty memory precisely to avoid letting "the very first episode's segments outscore every later segment on novelty purely because the store was empty, fill `M^E` with one episode, and then threshold out everything that followed."
Returning 1.0 changed the margin and not the mechanism: `1.0 > eta` is all that is required, for any `eta` below 1.

### What was ruled out

Four competing explanations were checked against the code and all four are refused.

- **Segmentation.** Healthy. `J` in [21, 84], and the flooding episode's 38 rows sit inside it.
- **Consolidation gated on success.** It is not gated. The call site is guarded only by `if dream is not None and dream_state is not None`; `reached` is a field stored on the row per eq. 15, not a condition. `contribution`'s docstring refuses the gate in writing. 282 episodes consolidated, 92 reached, 7 retained.
- **The store dropping rows.** `extend` is tuple concatenation with no dedup or cap; `load_memory` raises on a version mismatch, a short row, or a missing file rather than returning a partial store; `rows_added` is a length difference off the two stores.
- **Scene order.** Both the sweep and the reader sort the same labels with `sorted()`, so the first row printed is the first executed.

## Two further defects this found, neither fixed here

**1. The divert override removes memory from the phase the headline measures.**
`task/plan.py::_rank` sorts on "is this the anomaly divert" BEFORE `Score`, so whenever a divert candidate is in the pool the divert is the pick and eq. 26 is never read.
A target is named on every diverting step, so DREAM's memory is arithmetically inert across the whole detour — the stage 4 to stage 5 transition that `source_reached` measures.
This is deliberate and argued at `plan.py:36-44`; what was never priced is the consequence.
Memory's only route to the headline is indirect: it steers pre-interrupt SEARCH, which sets the pose at `t_anom`, which sets whether onset fires at all and whether the 120-step detour budget suffices.
**The mechanism the paper is about is switched off during the phase the paper's headline measures, and what remains is a second-order effect whose sign nobody has measured.**

**2. `dream-2` is a two-variable contrast.**
`task/runner.py:880-884` asserts that `pick_plan` reduces to `pick_waypoint` when memory is empty, so a run whose `M^L` never fills picks what the pre-DREAM agent picks.
That is true only at `lambda_feasibility = 0`, which is what the test pinning it uses.
`dream-2` ran `lambda_feasibility = 0.5` and `S_feas` is not constant across candidates.
So 92 against 94 is `full` against a feasibility term *and* a memory term, and since the memory was near-empty all night the likelier reading is that it measured the feasibility term at near-zero effect.

## The changes

**1. `C_j` and `U_j` become multiples of the episode's mean segment, not shares of its total.**
`_shares` is renamed `_mean_multiples` and divides by `total / J` instead of `total`.
`mean_j(C_j) = 1.0` exactly for any `J`, so the terms are stationary in episode length and commensurate with a cosine.
Everything the share form was chosen for survives: both stay unitless so no metre-to-cosine constant appears, `alpha` and `beta` stay constants as eq. 10 requires, and no knob is added.

**This is an implementation bug fix, not a deviation to defend in print.**
The module docstring states the provenance: the paper gives eq. 10, 12 and 13 and leaves everything else open, with `C_j` and `U_j` described in one clause each and no formula for either.
Eq. 10 as published specifies no normalisation at all.
The `1/J` was introduced here, for a stated and reasonable motive, with an unnoticed side effect.
Removing it moves the implementation toward the paper, not away from it.

**2. `retain` gains `max_kept`, a per-episode cap on eq. 13, exposed as the `max_retained` knob.**
This one IS a deviation and it is a bounded one.
Eq. 13 has no cap. The cap exists because eq. 12 hands an empty memory `N_j = 1.0` for every segment, so the first episode of any chain clears any `eta` below 1 outright regardless of what `C` and `U` are scaled to.
The rescale does not touch that and would otherwise just write more near-duplicate segments from the same trajectory.
It is inert whenever fewer than `max_kept` segments clear `eta`, ties go to the earlier segment so the result is deterministic in its inputs, and `runner.py` now records `dream_segments_over_eta` on every episode so the audit says when the cap bound rather than leaving a reader to infer it from a row count.

### What was NOT chosen

- **Lowering `eta` alone.** Self-defeating. Novelty is monotone decreasing in memory size — `N_j` maxes over a superset — so a lower threshold writes more rows earlier, drops `N_j` faster, and re-closes the gate at a lower level. It buys a longer transient.
- **A quantile or top-k rule.** It fixes the scale mismatch and the non-stationarity in one move and makes the retention rate a knob you set rather than a number you discover overnight. But eq. 13 is one of the three equations the paper gives in full, and a quantile rule is not that equation. Held in reserve for if the rescale re-runs and still collapses; it would need its own ADR and a paragraph in print.
- **Scaling `N_j` down.** Inverts the problem: it suppresses the one term the paper actually specifies to accommodate two it does not.

## `eta` is now unpriced, and that is the point

After the rescale `mean(C_j + U_j)` is 2.0 by construction, so the sweep moves to `eta = 2.0`, which reads as "this segment carried more than an average segment's worth".
`max_retained` is 12.
**Both are first choices and neither is calibrated.**
A synthetic probe over 250-step trajectories differing only in whether progress arrives evenly or in bursts had 9 to 14 of 21 segments clearing `eta` 2.0, and 0 to 3 of the same fixtures clearing 3.0.
That spread over fixtures that similar is itself the finding: the distribution is not predictable off-box.
It is also why the cap is 12 rather than 8 — a cap of 8 bound on every synthetic episode tried, and a cap that always binds IS the retention rule with `eta` as decoration, which is the top-k deviation this ADR declined to ship.

What has changed is that an `eta` sweep is now meaningful for the first time: `I_j` has a scale that depends on neither `J` nor how full the memory is.

## Pre-registration for what comes next

There is no in-repo pre-registration for the DREAM arm at all — no ADR, and it is absent from `PHASE2_ABLATION_REPORT.md` — unlike ADR-0018 for the matrix.
Nothing so far is a forking path: `2531eba` pre-registered the chain branch and both `eta` pathologies verbatim, and the sweep script says the knobs are first choices the run exists to price.
This section closes that gap. In order, and the first three are cheap:

1. **Price `eta`.** One scene. Read `dream_segments_over_eta` from `dream_report` and set `eta` so its median sits BELOW `max_retained`. If the cap binds on most episodes, the cap is the retention rule and `eta` is decoration.
2. **Instrument eq. 26 and run one scene.** `as_measurements` exists, is exported, and its only importer is a test that asserts the five names and discards the values, so none of eq. 26's numbers is on any `dream-2` audit. The counter that decides everything is `plan_pick_differs_from_no_memory` restricted to `divert is None`. **If it comes back near zero on non-divert steps, retention is settled as the wrong fix and nothing further is needed.**
3. **Run the `--dream-lambda-memory 0.0` control** with `lambda_feasibility` held at 0.5. This is what makes `dream-2` interpretable and it is worth more than a retention-fixed headline arm, because without it the headline arm carries the identical confound.
4. **Only then, a headline arm — as build-then-freeze.** `2b04fd9`'s chain took the number of independent memory realisations from 19 to 1, which violates the independence `episode_diff`'s McNemar assumes in a way the tool cannot detect. Build the memory in a pass at a different `--seed` with `--dream-memory-out` only, then evaluate with `--dream-memory-in` and no `-out`. No code change, and it is the only design under which `power.py`'s MDE table is true.

**The primary outcome for 3 and 4 is `dtg_source_final`, not binary reach.** It is continuous, already computed, already printed, and does not require crossing the 1 m ring, so a Channel-A effect shows up in it at far smaller `n` than the headline needs. Paired against a `lambda_memory 0.0` arm it answers sign, which a rate cannot.

**`k_experience` stays small.** `memory_consistency` averages `_leg_agreement` over the retrieved hits, so a diverse store averaged over large `k` converges toward the same value for every candidate, and a term constant across the pool cannot move an argmax. `dream-2`'s store was degenerate enough that this did not bite; a fixed one will. Measured over synthetic pools, discriminability degrades about 40% from `k=1` to `k=20`.

## What this ADR does not claim

**The fix is a precondition for a fair test, not a treatment.**
No Find-SR prediction is registered for it, and the honest reasons are three.
`dream-2`'s null was measured with `omega` already moving on 236 of 281 episodes and `M^P` non-empty, so the downstream mechanism was live and the outcome still did not move.
The repo's base rate for this class of fix is 0 for 6: every threshold or weight retuned in response to an inert mechanism — the coarse-affordance CLIP thresholds, the importance-head calibration, the detector c7 radius, the detector c9 agreement radius, the Step-2 consume gate — worked mechanically and none converted into a behavioural win.
And `matrix-2` measured an episodic store *hurting* a right prior by 10.3 points, so "the memory filled up, therefore SR should rise" is the exact shape of argument this tree's closed arcs refute.

`dream-2` may not be quoted as a test of the memory.
The memory it tested was one episode's segments duplicated 45 ways, scored by a gate that could not open, feeding a term that is switched off during the phase the headline measures.
Its p = 0.8939 is quotable only against the sharp null that the memory changed nothing, under which the chain is inert and the violation is anti-conservative.
Any effect-size bound read off it is withdrawn.
