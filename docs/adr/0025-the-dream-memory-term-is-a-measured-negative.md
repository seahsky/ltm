# ADR-0025: the DREAM memory term is a measured negative

Status: accepted, 2026-09-17
Supersedes nothing. Builds on ADR-0024, which diagnosed `dream-2` and pre-registered the sequence this record closes the third step of.

## What happened

`dream-3` ran on 2026-09-16: three arms, 19 scenes, 15 episodes each, `n = 282` per arm, 7h41m, commit `bb661b0`, at the `eta = 4.5` that `eta-2` priced.

| arm | source reached | rate |
|---|---|---|
| `full` | 86 / 282 | 30.5% |
| `dream` | 79 / 282 | 28.0% |
| `dream-nomem` | 94 / 282 | 33.3% |

`dream-nomem` is `dream` with `lambda_memory = 0.0` and every other knob identical, built by substitution into `DREAM_KNOBS` so the two cannot drift apart in a second knob.

**The memory term costs 15 episodes.**
Exact McNemar on the pre-registered contrast, `dream` against `dream-nomem`: 28 discordant pairs won by the control, 13 by the memory, over 41 discordant of 282 paired, **p = 0.0275**.
The scene-level sign test over the same table is 11 up, 3 down, 3 tied, **p = 0.0574** (computed by hand from the driver's own per-scene nets, not by a tool in this repo).
Both tests point the same way; one clears 0.05 and one does not, and ADR-0016's rule is that both are reported.

**The decomposition is exact and it is what the run was built for.**
`full` → `dream` is −7 episodes; `full` → `dream-nomem` is +8; `dream` → `dream-nomem` is +15, and −7 = +8 − 15.
So ADR-0024's third defect is now a number rather than an argument: `dream-2`'s two-variable contrast was a small positive feasibility term masking a larger negative memory term, and the near-zero it reported was the sum of the two.

## Why this is a negative result and not another null

`dream-1` and `dream-2` could not test the memory: the first threw away nineteen memories, the second could not open the retention gate at all.
`dream-3` has none of those excuses, and the difference is worth stating as a table.

| precondition | `dream-2` | `dream-3` |
|---|---|---|
| rows retained per episode | 0 on 275 of 282 | median 4, min 0, max 12 |
| episodes that retained nothing | 275 of 282 | 24 of 282 |
| `max_retained` cap bound | not reached | 3 of 282 |
| largest `M^E` | 45 rows | **1192 rows** |
| `M^P` per episode | median 1 | median 5 |
| `omega^E` moved past the 0.05 floor | flat | **270 of 281** |
| episodes starting from an empty `M^E` | 1 of 282 | 1 of 282 |

Every precondition the previous two runs failed, this one meets.
The gate opens, the threshold thresholds, the memory grows across all nineteen scenes, the pattern level populates, and the weighting moves on 96% of episodes.
**The mechanism was demonstrably live when it failed**, which is what separates this from the nulls before it.

## The harm is uniform, and that says what kind of harm it is

`placement_report` splits every arm by whether the anomaly was placed at a semantic anchor or geometrically.

| arm | anchored (n=134) | geometric (n=148) |
|---|---|---|
| `full` | 31 (23.1%) | 55 (37.2%) |
| `dream` | 27 (20.1%) | 52 (35.1%) |
| `dream-nomem` | 35 (26.1%) | 59 (39.9%) |

Turning the memory term on costs 8 anchored episodes and 7 geometric ones: −6.0 and −4.7 points, the same direction and nearly the same size on both branches.

This is the informative part.
A memory that had learned a real room-level regularity would help on anchored episodes, where the source sits where the regularity says it should, and do nothing or harm on geometric ones where the learned rule does not apply.
A memory that had learned a *wrong* regularity would show the mirror image.
Undifferentiated degradation across both is neither: it is the shape of noise added to an argmax.

That is consistent with a mechanism already on the record.
`memory_consistency` averages `_leg_agreement` over the `k_experience` retrieved hits, and `dream-3`'s store sits at 0.7261 mean pairwise cosine with a median maximum of 0.9993 — many near-duplicate keys.
A term that is nearly constant across candidates, plus sampling noise, cannot improve an argmax and can only perturb it.
Discriminability measured over synthetic pools degrades about 40% from `k = 1` to `k = 20`.

**This remains an inference from an outcome.**
ADR-0024 step 2 was not run before `dream-3`, so eq. 26's own numbers are on none of its audits.
The counters that settle it directly landed afterwards (PR #128): `plan_pick_differs` over `plan_eligible_steps`, with the denominator excluding every step where the divert override was in force, the pool held one candidate, or the retrieval returned nothing.

## What this ADR does not claim

**The magnitude is not reportable.**
5.3 points sits below the 6.71-point MDE of a single run at `n = 282`, and `repeat-1` measured a standard deviation of 7.7 episodes on the *difference* between byte-identical reruns.
McNemar's p is a valid test against the sharp null for this pair of runs and needs no flip-rate estimate, but a replication could land differently.
`eta-1` and `eta-2` are the local demonstration: same command, same seed, same scene, and they disagreed on 3 versus 2 reached and on an `I_j` maximum of 16.06 versus 23.24.

**The arms differ in one thing that was not intended.**
The sweep chains `M^E` only for the arm literally named `dream`, so `dream-nomem` built a memory per scene and discarded it — 96 rows at most against 1192.
With `lambda_memory = 0` that memory cannot enter the plan score, so the contrast remains "memory term on" against "memory term off" and the control's store size is irrelevant to it.
What the run cannot answer is whether a *smaller* or per-scene memory with the term on would do better.
That is a third arm, not a reinterpretation of this one.

**Nothing here is a claim about DREAM as published.**
This tree's DREAM is an implementation of eq. 8-27 with two documented deviations (the `retain` cap, and `_mean_multiples` in place of the shares that carried a hidden `1/J`), a `M^K` level that was never populated, and a divert override that removes eq. 26 from the entire detour.
A negative result about this implementation on this task is not a negative result about the architecture.

## What is decided

**Report DREAM as a documented negative. Do not retune it further.**

The base rate is the argument.
Every threshold or weight this repo has retuned in response to an inert mechanism — the coarse-affordance CLIP thresholds, the importance-head calibration, the detector c7 radius, the detector c9 agreement radius, the Step-2 consume gate, and now eta — has worked mechanically and none has converted into a behavioural win.
That is 0 for 7, and `dream-3` is the first where the mechanism was verifiably live when it failed, which makes the eighth attempt less promising rather than more.

Two things would change this conclusion, and only two:

1. **Section G says the term is LIVE.** If eq. 26 changed picks on a material fraction of eligible steps, the harm is steering rather than noise, and a steering term that is reliably wrong is a term whose sign can be investigated. If it says INERT with a zero `S_mem` spread, the fix is `k_experience` or the store's diversity and never `lambda_memory` or `eta`, and the arc closes.
2. **A repeat pair reverses the sign.** One more `dream` plus `dream-nomem` at the same knobs, roughly five hours, is what turns the magnitude from credible-direction into reportable. If the direction flips, the effect was inside the apparatus and this ADR is withdrawn.

Both are cheap and both are pre-registered here before either is run.

**What gets written up.**
`full` at 30.5% remains the baseline of record for this sweep, and `dream-3` is reported as a three-arm decomposition with the memory term negative, the feasibility term small and positive, and the magnitude of each unresolved at one run per arm.
The `dream` arm is not quoted as a headline number in either direction.
