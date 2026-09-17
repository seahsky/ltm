# ADR-0027: eq. 26 is anti-selective between divert candidates, and the override was right

Status: accepted, 2026-09-18.
Closes ADR-0026 on its own third branch.

`propose-1` is the first result in this arc that reproduced and cleared its own measured noise.
It says the opposite of what ADR-0026 predicted.

## The run

`propose_sweep.sh --tag propose-1`, four arms, 19 HM3D val scenes, 15 episodes each, `n = 282` per arm, roughly eleven hours, commit `eabc93f`, all 19 scenes complete in the prior pass.
`propose-a`/`propose-b` and `replace-a`/`replace-b` are the same command run twice, so the apparatus is priced inside the run.

| arm | source reached | rate |
|---|---|---|
| `replace-b` | 93 / 282 | 33.0% |
| `replace-a` | 84 / 282 | 29.8% |
| `propose-b` | 65 / 282 | 23.0% |
| `propose-a` | 60 / 282 | 21.3% |

**Making the prior earn the pick cost 26 episodes, 9.2 points.**

| contrast | conditional, stage 4 to 5 | Find-SR@1m |
|---|---|---|
| repeat 1, `replace-a` against `propose-a` | net −25 over 41, **p = 0.0001** | net −24 over 46, p = 0.0005 |
| repeat 2, `replace-b` against `propose-b` | net −28 over 44, **p = 0.0000** | net −28 over 54, p = 0.0002 |
| within `replace`, a against b | net +4 over 10, p = 0.3438 | net +9 over 21, p = 0.0784 |
| within `propose`, a against b | net +0 over 16, **p = 1.0000** | net +5 over 27, p = 0.4421 |

**The within-arm rows are why this one is reportable.**
The same command twice disagreed by 9 episodes in `replace` and 5 in `propose`.
The contrast is 26, about 2.9 times that floor, and it landed at the same magnitude in both repeats.
Every earlier comparison in this repo rested on a single run of each arm against `repeat-1`'s historical 16.2% flip rate; `dream-3` and `dream-4` are what that costs.
Measuring the repeat inside the run is the one structural lesson of the DREAM arc, and this is it spent.

The conditioning holds. Pairs dropped for not reaching `INVESTIGATE_ENTERED` were 91 against 87 in repeat 1 and 81 against 78 in repeat 2.
Close, as they must be if stage 4 is upstream of the treatment, which it is: the prior is consulted inside `is_diverting`.

## The mechanism was live, and that is what makes this a finding

```
propose-a   eligible 3764   ranked first 1981   52.6%
propose-b   eligible 3837   ranked first 1902   49.6%
```

**Eq. 26 preferred the memory's waypoint on about half of every step it was offered on.**
ADR-0026's fourth branch does not fire.
`eq26-1` measured 2.81% of eligible steps changed on a frontier pool; between two divert candidates the term is not marginal at all, it decides roughly half the picks.

So the difference above is attributable to the ranking rather than to retrieval cost, a suppressed rail, or an arm that never ran.

## What it means

`replace` always takes the memory's waypoint.
`propose` takes it about half the time and the acoustic estimate the rest.
**Both pure policies beat the mix**, and the mix is 9.2 points below the better one.

A selector that is merely uninformative cannot do that.
Splitting between two policies at random lands between them; landing *below* both requires the selector to prefer the worse option more often than chance.
**Eq. 26's `S_mem` is anti-selective between divert candidates.**

`plan.py:30-42` records why the structural override exists: the old tree let a maximal frontier tie the divert, the tie-break handed the pick away, and the anomaly interrupt became advisory.
The override was built on the judgement that eq. 26 could not be trusted to rank a divert.
**This run is the measurement behind that judgement, and it says the judgement was right.**

The reason is visible in the code and was written down before this ran. `plan.py:44-50`: there is no shared space between `h^av`, a fused audio-visual embedding, and a candidate, which is a point on the navmesh, so the only honest comparison left is geodesic length and straightness. A selector that can only compare walk shapes is being asked which of two places holds a sound. It has no access to the question.

## What this ADR does not claim

**It does not claim memory helps.**
`replace` at 29.8% and 33.0% sits on the baseline of record (`full` at 30.5% in `dream-3`, 33.3% in `dream-2`), but those are different runs and `repeat-1` is exactly why a cross-run comparison is not reportable here.
There is still **no NONE arm** (ADR-0023 records this), so "the semantic prior is worth anything at all" remains unmeasured.
What is measured is that eq. 26's selection between the prior's waypoint and the acoustic estimate is worse than taking the prior unconditionally.

**It does not clear the rail.**
The 6.0 m rail suppressed **more than it admitted**: 4255 railed against 3813 emitted in `propose-a`, 4366 against 3842 in `propose-b`, about half of all consultations.
Half the memory's opinions never reached the pool, and the mechanism was still live at 50% on the rest.
A rail-removed arm is the obvious next question and it is not being run; see below.

**A small navmesh loss is recorded rather than folded in.**
49 emitted proposals in `propose-a` and 5 in `propose-b` were filtered by `reachable_pool` before ranking, so `emitted` exceeds `eligible`. Not material at this scale, but the two numbers are not interchangeable.

## What is decided

**Keep the structural divert override. Do not use eq. 26 to rank divert candidates. Retire the proposer.**

ADR-0026 pre-registered this branch: *"conversion down: the semantic prior is actively misleading even when checkable. Retire."*
The measurement is stronger than the branch anticipated, because it is not the prior that is misleading, it is the selector.

**The rail-removed arm is NOT booked**, and the base rate is the argument.
Every threshold this repo has retuned in response to a result has worked mechanically and none has converted into a behavioural win, which is 0 for 7 (ADR-0025).
A selector that prefers the worse waypoint about half the time at a 6.0 m rail has no mechanism by which a 50 m rail makes it selective; it would simply be offered more chances to be wrong.

**The code stays in, switched off.**
`--memory-proposes` defaults to False, which is `replace`, which is what every earlier result was measured under.
The arm is a control that can be re-run against any future selector, and deleting it would mean rebuilding it to ask the same question again.

**What this makes worth doing, if anything in memory is:** give `S_mem` access to the question.
A term that can compare only walk geometry cannot choose between two places, and no rail, weight or store size changes that.
That is a schema change to `m^E` (ADR-0026 scoped one and declined it), it is a paper deviation, and it needs its own record.
Nothing in `propose-1` argues it is worth doing before the apparatus questions below are closed.

## The defect this run exposed in the readout

`propose-1`'s own mechanism check reported every counter at zero over 1128 episodes whose audits were on disk throughout.
The check was an inline Python heredoc globbing `<scene>/episodes/<N>/audit.json` while the writer writes `<scene>/episodes/ep0000.audit.json`.
That is `dream-1`'s failure and `pilot-1`'s, committed inside the change whose comment cites both.
Fixed in PR #135: `tools/propose_report.py`, a tested module, which separates "no episode on disk", "counters not recorded", "emitted nowhere", "emitted but navmesh-filtered" and the fourth-branch ratio, and exits nonzero when nothing was readable.
Without that fix this run reads as an inert mechanism, which is the exact wrong conclusion.
