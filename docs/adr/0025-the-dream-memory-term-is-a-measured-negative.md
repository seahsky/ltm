# ADR-0025: the DREAM memory term is a measured negative, and its reach is 0.5% of decision steps

Status: accepted, 2026-09-17
Builds on ADR-0024, which diagnosed `dream-2` and pre-registered the sequence this record closes.

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
The scene-level sign test over the same table is 11 up, 3 down, 3 tied, **p = 0.0574** (computed by hand from the driver's own per-scene nets; no tool in this repo computes it).
Both tests point the same way; one clears 0.05 and one does not, and ADR-0016's rule is that both are reported.

**The decomposition is exact.**
`full` → `dream` is −7 episodes; `full` → `dream-nomem` is +8; `dream` → `dream-nomem` is +15, and −7 = +8 − 15.
ADR-0024's third defect is now a number rather than an argument: `dream-2`'s two-variable contrast was a small positive feasibility term masking a larger negative memory term, and the near-zero it reported was the sum of the two.

## Why this is a negative result and not a third null

`dream-1` and `dream-2` could not test the memory: the first threw away nineteen memories, the second could not open the retention gate at all.

| precondition | `dream-2` | `dream-3` |
|---|---|---|
| rows retained per episode | 0 on 275 of 282 | median 4, min 0, max 12 |
| episodes that retained nothing | 275 of 282 | 24 of 282 |
| `max_retained` cap bound | not reached | 3 of 282 |
| largest `M^E` | 45 rows | **1192 rows** |
| `M^P` per episode | median 1 | median 5 |
| `omega^E` moved past the 0.05 floor | flat | **270 of 281** |

Every precondition the previous two runs failed, this one meets.
**The mechanism was demonstrably live when it failed**, which is what separates this from the nulls before it.

## What eq. 26's own counters say, and what they refute

`eq26-1` ran ADR-0024's step 2 on 2026-09-17: one scene, 15 episodes, `eta = 4.5`, commit `e2d3d84`, 7m29s, with the counters that landed in PR #128.

```
steps that RANKED a pool:                                    2028
  of those, divert in pool:                                  1611   (79.4%)
  ELIGIBLE (no divert, pool > 1, retrieval answered):          392   (19.3%)
PICKS THE MEMORY CHANGED:                        11 of 392 eligible   (2.81%)
S_mem spread across the pool, per-episode mean:  median 0.1053
margin to the runner-up in rank order:           mean 0.2465, worst -0.2861
```

**The verdict is LIVE**, so a difference between this arm and a `lambda_memory = 0` control is attributable to steering rather than to the cost of retrieval.
Three things follow, and the first corrects this ADR's own earlier reading.

**1. The "noise on the argmax" reading is refuted in its specific form.**
`dream-3`'s harm was uniform across the placement split — −6.0 points anchored (n=134) and −4.7 geometric (n=148) — and that shape was read here as a term gone constant across candidates, the failure `k_experience` predicts when `memory_consistency` averages `_leg_agreement` over a degenerate store.
Section G measures `S_mem` spread at a median of **0.1053**, not zero.
The term varies across candidates and does change picks.
It is not constant, and `k_experience` is not indicted by this measurement.

**2. Its reach is 0.5% of decision steps.**
Eleven changed picks over 2028 ranked steps is 0.54%, or **0.73 changed picks per episode**.
A term that alters fewer than one waypoint choice per episode is a thin channel through which to explain a 5.3-point swing in Find-SR.
It is not impossible — one redirected waypoint early in an episode can change the whole trajectory — but it makes the alternative reading materially more likely: that `dream-3`'s difference is largely apparatus variance.
The supporting number is the discordance rate itself. `dream` and `dream-nomem` disagreed on **41 of 282 episodes, 14.5%**, and `repeat-1` measured **16.2%** disagreement between byte-identical reruns.
The two arms behave no more differently than one arm does from itself.
The entire signal is the *asymmetry* of those disagreements, 28 against 13, not their number.

**3. The divert override excludes eq. 26 from four steps in five.**
1611 of 2028 ranked steps had a divert in the pool, where `_rank` decides structurally and eq. 26 is never read.
ADR-0024 recorded this defect and left its sign unmeasured; **79.4% is that measurement.**
The memory is switched off for the large majority of the steps at which a waypoint is chosen, and the whole of the detour that the headline metric measures.
The same section prices the override for the first time: a worst-case margin of **−0.2861** means the divert outranked a candidate scoring that much higher under eq. 26.

**The caveat on all three.**
`eq26-1` is one scene with a 73-row memory, against `dream-3`'s 1192-row chained store.
Retrieval hits differ with store size, so the 2.81% rate is indicative for that condition and is not the rate `dream-3` ran at.
Re-running the counters inside a chained sweep is the way to close that gap, and it is not worth a night on its own.

## What this ADR does not claim

**The magnitude is not reportable.**
5.3 points sits below the 6.71-point MDE of a single run at `n = 282`, and `repeat-1` measured a standard deviation of 7.7 episodes on the *difference* between byte-identical reruns.
McNemar's p is a valid test against the sharp null for this pair of runs and needs no flip-rate estimate, but a replication could land differently.
`eta-1` and `eta-2` are the local demonstration: same command, same seed, same scene, disagreeing on 3 versus 2 reached and on an `I_j` maximum of 16.06 versus 23.24.

**The arms differ in one thing that was not intended.**
The sweep chains `M^E` only for the arm literally named `dream`, so `dream-nomem` built a memory per scene and discarded it — 96 rows at most against 1192.
With `lambda_memory = 0` that memory cannot enter the plan score, so the contrast remains "memory term on" against "memory term off" and the control's store size is irrelevant to it.
What the run cannot answer is whether a *smaller* or per-scene memory with the term on would do better.

**Nothing here is a claim about DREAM as published.**
This tree's DREAM is an implementation of eq. 8-27 with two documented deviations (the `retain` cap, and `_mean_multiples` in place of shares that carried a hidden `1/J`), an `M^K` level that was never populated, and a divert override that removes eq. 26 from 79.4% of ranked steps.
A negative result about this implementation on this task is not a negative result about the architecture.

## What is decided

**Report DREAM as a documented negative. Do not retune it.**

The base rate is the argument.
Every threshold or weight this repo has retuned in response to an inert mechanism — the coarse-affordance CLIP thresholds, the importance-head calibration, the detector c7 radius, the detector c9 agreement radius, the Step-2 consume gate, and now `eta` — has worked mechanically and none has converted into a behavioural win.
That is 0 for 7.

**One thing remains, and it is pre-registered here.**
A repeat pair of `dream` and `dream-nomem` at the same knobs, roughly five hours, is what decides whether −5.3 points is an effect or apparatus variance.
Section G has made that the *only* open question rather than one of two: the mechanism is live, so a reproduced negative is a real negative, and a reversed sign means the effect was never there.
If the sign holds, DREAM is reported as a measured negative with a mechanism (steering, on 0.5% of decision steps, in the wrong direction).
If it reverses, this ADR is withdrawn.

**And if a structural lever is ever wanted, it is the divert override and not the memory knobs.**
Eq. 26 is read on 19.3% of ranked steps.
No setting of `eta`, `max_retained`, `k_experience` or `lambda_memory` changes that number, because `_rank` decides it before any of them are consulted.
That would be its own ADR, its own control arm, and a re-run of the whole ablation table, and nothing in `dream-3` argues it is worth doing.
