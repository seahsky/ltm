# ADR-0025: the DREAM memory term is live, measured, and inert

Status: accepted, 2026-09-17.
Supersedes this record's own first version, accepted earlier the same day, which called the memory term a measured negative on the strength of one run.
The repeat pair that version pre-registered is what withdrew it, and both runs are kept below.
The file was renamed from `0025-the-dream-memory-term-is-a-measured-negative.md`; the number is unchanged.

Builds on ADR-0024, which diagnosed `dream-2` and pre-registered the sequence this record closes.

## The claim that was withdrawn, and what withdrew it

The first version of this ADR read: *"A repeat pair of `dream` and `dream-nomem` at the same knobs, roughly five hours, is what decides whether −5.3 points is an effect or apparatus variance. If the sign holds, DREAM is reported as a measured negative with a mechanism. If it reverses, this ADR is withdrawn."*

`dream-4` ran that pair on 2026-09-17: 19 scenes, 15 episodes each, `n = 282` per arm, 5h40m, commit `fcca4b7`, at the same `eta = 4.5`.

| run | `dream` | `dream-nomem` | delta | discordant | exact McNemar | scene sign test |
|---|---|---|---|---|---|---|
| `dream-3` | 79 / 282 (28.0%) | 94 / 282 (33.3%) | **−5.3 pts** | 13 vs 28 of 41 | **p = 0.0275** | 11 / 3 / 3, p = 0.0574 |
| `dream-4` | 89 / 282 (31.6%) | 91 / 282 (32.3%) | **−0.7 pts** | 14 vs 16 of 30 | **p = 0.8555** | 7 / 6 / 4, **p = 1.0** |

The sign did not reverse.
It collapsed, which is the branch the pre-registration did not name, and it resolves the same way: the effect did not replicate, so the negative is withdrawn.

**The decisive number is the movement of one arm against itself.**
`dream` went from 79 to 89 reached across the two runs, on the same knobs, the same seed, the same nineteen scenes: **10 episodes, 3.5 points**.
The two arms differ by 2 episodes in `dream-4`.
One arm moved five times further against its own repeat than the arms moved against each other, and `repeat-1` predicted exactly this: a standard deviation of 7.7 episodes on the difference between byte-identical reruns.
`eta-1` against `eta-2` is the same demonstration at one scene.

**The pooled test is not quoted here, and that is deliberate.**
Pooling gives 27 against 44 over 71 discordant pairs, p = 0.0568.
Those are the same 282 episodes measured twice, not 564 independent pairs, so McNemar's independence assumption is violated in the anti-conservative direction.
It is worth one sentence of direction and it is not a result.

## What the two runs jointly establish

**The mechanism is live.**
`eq26-1` measured eq. 26's own counters on 2026-09-17: one scene, 15 episodes, `eta = 4.5`, commit `e2d3d84`, 7m29s.

```
steps that RANKED a pool:                                    2028
  of those, divert in pool:                                  1611   (79.4%)
  ELIGIBLE (no divert, pool > 1, retrieval answered):          392   (19.3%)
PICKS THE MEMORY CHANGED:                        11 of 392 eligible   (2.81%)
S_mem spread across the pool, per-episode mean:  median 0.1053
margin to the runner-up in rank order:           mean 0.2465, worst -0.2861
```

These are counters and not a contrast, so `dream-4` does not touch them.
The memory term varies across candidates, it changes the argmax, and it does so on 11 of 392 eligible steps.

**Its reach is 0.54% of ranked steps.**
Eleven changed picks over 2028 is **0.73 per episode**.

**Its effect on Find-SR is not measurable at `n = 282`.**
Two runs, one at p = 0.0275 in one direction and one at p = 0.8555, over an apparatus whose byte-identical reruns disagree on 16.2% of episodes.

Those three facts are consistent and they are the finding: **a memory term that demonstrably steers, on fewer than one waypoint choice per episode, moves the outcome by no amount this design can resolve.**
That is a stronger result than the negative would have been.
A negative at 0.73 changed picks per episode was always going to be hard to defend as causal; "live, measured, and inert" is defensible, and it is what the two runs say together.

**The precondition table still stands**, and it is what separates this from `dream-1` and `dream-2`.

| precondition | `dream-2` | `dream-3` |
|---|---|---|
| rows retained per episode | 0 on 275 of 282 | median 4, min 0, max 12 |
| episodes that retained nothing | 275 of 282 | 24 of 282 |
| `max_retained` cap bound | not reached | 3 of 282 |
| largest `M^E` | 45 rows | **1192 rows** |
| `M^P` per episode | median 1 | median 5 |
| `omega^E` moved past the 0.05 floor | flat | **270 of 281** |

`dream-1` threw away nineteen memories and `dream-2` could not open the retention gate at all, so neither tested anything.
This pair did.
**The mechanism was verifiably live before the null**, which is the first time that has been true in this repo for any of the seven levers below.

**The divert override is measured for the first time.**
1611 of 2028 ranked steps had a divert in the pool, where `_rank` sorts on is-divert before `Score` and eq. 26 is never read.
ADR-0024 recorded this defect and left its sign unmeasured; **79.4% is that measurement.**
The memory is switched off for four ranked steps in five, including the whole of the detour that the headline metric measures.
The worst-case margin of **−0.2861** prices it: a divert outranked a candidate scoring that much higher under eq. 26.
This is the most likely single explanation for the inertness, and it is a structural property of the ranking rule rather than of any memory knob.

## What this ADR does not claim

**It does not claim the memory term helps, and it does not claim it harms.**
`dream-3`'s −5.3 points is withdrawn as an effect estimate.
`dream-4`'s −0.7 points is not an effect estimate either, and neither is their average.
The honest statement is an absence: no difference this design can resolve, with an MDE of 6.71 points at `n = 282` for a single run.

**It is not a bound on the effect.**
Two runs do not give one, and `repeat-1`'s variance is why.
An effect smaller than roughly 5 points would be invisible to every run in this arc.

**The arms differ in one thing that was not intended.**
The sweep chains `M^E` only for the arm literally named `dream`, so `dream-nomem` built a memory per scene and discarded it, 96 rows at most against 1192.
With `lambda_memory = 0` that store cannot enter the plan score, so the contrast is still "memory term on" against "memory term off".
What neither run can answer is whether a *smaller* or per-scene memory with the term on would do better.

**`eq26-1` is one scene with a 73-row memory**, against `dream-3`'s 1192-row chained store.
Retrieval hits differ with store size, so 2.81% is indicative for that condition and is not the rate the sweeps ran at.

**Nothing here is a claim about DREAM as published.**
This tree's DREAM is an implementation of eq. 8-27 with two documented deviations (the `retain` cap, and `_mean_multiples` in place of shares that carried a hidden `1/J`), an `M^K` level that was never populated on any episode of any arm, and a divert override that removes eq. 26 from 79.4% of ranked steps.
A null about this implementation on this task is not a null about the architecture.

## What is decided

**Report DREAM as a documented null with a measured mechanism. Do not retune it.**

The base rate is the argument.
Every threshold or weight this repo has retuned in response to an inert mechanism has worked mechanically and none has converted into a behavioural win: the coarse-affordance CLIP thresholds, the importance-head calibration, the detector c7 radius, the detector c9 agreement radius, the Step-2 consume gate, and now `eta`.
That is 0 for 7.
The difference here is that the mechanism was proven live first, so this entry is evidence about the design rather than about the tuning.

**No further DREAM runs are booked.**
The repeat pair was the last open question and it is answered.
A third run would be a third draw from a distribution whose spread is already measured at 7.7 episodes, and it would need three repeats an arm, 14h15m, to reach an MDE of 3.87 points that nothing predicts the effect exceeds.

**If a structural lever is ever wanted, it is the divert override and not the memory knobs.**
Eq. 26 is read on 19.3% of ranked steps.
No setting of `eta`, `max_retained`, `k_experience` or `lambda_memory` changes that number, because `_rank` decides it before any of them are consulted.
That would be its own ADR, its own control arm, and a re-run of the whole ablation table.
Nothing in `dream-3` or `dream-4` argues it is worth doing.

## One defect this pair exposed

`dream-4` exited 1 after a clean 5h40m.
Every gate was green, all 564 episodes were on disk, and the readout reported "no baseline to quote against" because the sweep had no directory called `full`.
It had not been asked for one: `--arms "dream dream-nomem"` is the invocation this ADR's first version pre-registered by name.
The driver called its own pre-registered command a failure.

Fixed in the same change as this record.
The readout now picks a reference arm (`full` when the sweep ran it, otherwise the first arm asked for), prints which one it chose, and is red only when an arm that *was* requested is missing from disk.
`tests/mac/test_ablation_readout.py` holds both halves of that rule.
