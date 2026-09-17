# ADR-0026: memory proposes a waypoint and is ranked, instead of replacing one unchecked

Status: proposed, 2026-09-17.
Pre-registration. Nothing has been built and nothing has been run.

Follows ADR-0025, which closed the DREAM arc as live-but-inert, and ADR-0023, which recorded the matrix defects.
This record exists because that closure identified a structural lever and declined to book it, and because reading the steering path afterwards found a second one that no ADR had named.

## The two facts this rests on

**1. Every memory mechanism this repo has built is a re-ranker, except one, and that one is a replacer.**

Eight levers have been tried.
Seven reorder a pool that a geometric proposer generated: the S2 DOA rerank, coarse-affordance, the importance head, the Step-2 consume gate, the temporal-context head, query expansion, and DREAM's eq. 26.
None of them proposes a waypoint.
`agent/proposers.py:69-72` records that a proposer was always the intended seam and was never built: ADR-0008's "memory plugs in here later", and "a `SOURCE_MEMORY` proposer would emit the same type into the same pool".

The eighth is different and is the one that matters here.
`task/runner.py:1527-1528` does this, once per episode, while diverting:

```python
if memory_prior is not None:
    target = memory_prior.target
```

That is not a re-rank.
**The memory's guess REPLACES the acoustic estimate outright**, with no comparison, no ranking and no fallback.
`_divert_candidate(target, pose)` is then built from whatever won, and since there is exactly one investigate candidate (`DIVERT_CANDIDATE_ID`, `runner.py:830`) the structural override in `_rank` makes it the pick unconditionally.

**2. Replacement can only add variance, and that is what was measured.**

`matrix-2` found the episodic store cost **10.3 points against a right prior** and gained 3.7 against a wrong one, an interaction of about −14.0 points.
Replacement explains the shape exactly.
When the memory is right it is redundant, because the acoustic cue already points at the source.
When it is wrong it is unchecked, because nothing outranks it.
A mechanism that can only be redundant or catastrophic has a negative expected value, and no amount of tuning the store changes that.

## What the current wiring makes impossible

**Eq. 26 never acts on the detour.**
`_rank` (`task/plan.py:289-290`) sorts on is-divert before `Score`, and the divert class has exactly one member, so there is nothing for eq. 26 to rank.
`eq26-1` measured the consequence: a divert was in the pool on **1611 of 2028 ranked steps, 79.4%**.

**And the override must not be lifted.**
`plan.py:30-42` records the regression: the old tree scored a divert 1.0, a maximal frontier tied it, the tie-break handed the pick away, and the anomaly interrupt became advisory.
`tests/mac/test_task_plan.py:371` is the arm holding that line.
So the memory term is excluded from four ranked steps in five by a rule that exists for a good reason and has to stay.

**This is where the episodes are lost.**
The funnel (`report/audit.py:58-63`) puts `INVESTIGATE_ENTERED` at stage 4 and `SOURCE_REACHED` at stage 5.
`dream-4`'s smoke gates show most failures sitting at stage 4: the agent diverts and does not arrive, at per-scene rates between 7% and 53%.
The detour arc reached the same place from the other side, finding arrival spent and the residual deficit to be exploration, with `ep 14` walking 16.74 m to close 0.36 m on a source 2.49 m away.

So the memory term is switched off precisely where the headline is decided, and the one memory mechanism that does reach that phase is the one with no check on it.

## The decision

**Memory proposes a competing investigate candidate. It does not replace the acoustic one, and eq. 26 chooses between them.**

Concretely, at `runner.py:1527-1528`, instead of overwriting `target`, emit a second `Candidate` with `source=SOURCE_INVESTIGATE` and its own `candidate_id`, and put both into the pool.

Three things follow, and each one is the point.

**The interrupt stays mandatory.**
Both candidates are in the divert class, so `_rank` still sorts every divert ahead of every frontier.
`test_the_divert_survives_a_memory_that_hates_it` asserts `scored[0].candidate.source == SOURCE_INVESTIGATE`, which a second divert satisfies, so the regression arm passes unchanged.
This is verified by reading the assertion, not assumed.

**Eq. 26 gets work to do on the detour, for the first time.**
The divert class now has two members on the steps where memory has an opinion, which is inside the 79.4% that has always been closed to it.

**The semantic prior becomes checkable.**
It has to win on `Score` rather than win by assignment.
That is the direct test of the replacement hypothesis above: if replacement is why `matrix-2` lost 10.3 points, competition should recover most of it, because a wrong prior now loses to the acoustic estimate instead of overwriting it.

### The safety rail is part of the decision, not an implementation detail

A memory waypoint whose geodesic distance from the acoustic estimate exceeds a configured bound is **not emitted at all**.
Without it a wrong prior is unbounded and this becomes `matrix-2`'s failure mode wearing new notation.
The bound is typed on `RunConfig` (ADR-0008, no environment flags) and its value is a first choice that this run prices, in the language `ablation_sweep.sh:266` already uses for the DREAM knobs.

### What is deliberately NOT in scope

**M^E is not extended.**
`ExperienceEntry` (`memory/longterm.py:213-218`) carries `context`, `trajectory`, `target_concept`, `outcome`, `sound_concept` and `room_concept`, and h^traj is four scalars: `path_length_m`, `net_displacement_m`, `straightness`, `mean_abs_turn_rad`.
**There is no position anywhere in M^E**, so DREAM's own store cannot propose a point and nothing here pretends otherwise.
The proposal comes from the semantic store, which already holds points.
Adding a spatial component to m^E is a larger change, a paper deviation, and a separate record.

**The divert override is not touched.**

**No arc-shaped waypoint derived from h^traj.**
It was considered and rejected: h^traj can express how much a past detour bent but not which way, it shares no mechanism with the decision above, and a null from it would say nothing about this one.

## The pre-registration

The contrast is `memory-propose` against `memory-replace`, at identical knobs, differing in that one line.
`memory-replace` is today's behaviour and is the control of record.
Both arms get `--clap` and the same store; the substitution pattern is `dream-nomem`'s, which is the only design in this repo that has isolated a term cleanly.

**Two repeats per arm, minimum.**
`dream-3` measured −5.3 points at p = 0.0275 and `dream-4` measured −0.7 at p = 0.8555 on the same contrast, and `repeat-1` measured a 16.2% flip rate on byte-identical reruns with SD 7.7 episodes on the difference.
A single run of this is not reportable and will not be reported.

**The primary outcome is stage 4 to stage 5 conversion**, not Find-SR.
`SOURCE_REACHED` given `INVESTIGATE_ENTERED` is the transition this change acts on, it is already recorded on every audit, and it removes the stage 2 and stage 3 attrition that Find-SR carries as noise.
Find-SR@1m is reported beside it, always, and both tests are reported for both (ADR-0016).

**The branches, fixed in advance.**

| result | reading | what happens |
|---|---|---|
| conversion up, reproduced across both repeats | competition is the fix; replacement was the defect | write it up; re-open the memory line |
| conversion flat, both repeats | the prior carries nothing the cue does not already have | **the memory line is retired** |
| conversion down | the semantic prior is actively misleading even when checkable | retire, and `matrix-2`'s +3.7 on a wrong prior is re-read as noise |
| eq. 26 ranks the memory candidate first on under 5% of eligible steps | the safety rail or the store is suppressing the mechanism | not a result about memory; fix and re-run before reading anything else |

**The retirement branch is the point of writing this down.**
The base rate on memory levers in this repo is 0 for 7 (ADR-0025).
This is the eighth, and the only reason to expect a different outcome is structural: it is the first that is not a re-ranker and the first that can act during the detour.
That is a reason to run it once, properly, and it is not a reason to keep going afterwards.
**If stage 4 to 5 conversion does not move across two repeats, memory as a lever on this task is closed**, and the write-up is the negative described in ADR-0025 plus this arm as its final control.

## What must ship with it

Both arms, per ADR-0014: the proposer firing and being ranked first on some step, **and** the forced case where memory has no opinion, which must reduce byte-identically to today's single-divert behaviour.
A counter on every audit for how often the memory candidate was emitted, how often it was suppressed by the rail, and how often eq. 26 ranked it first.
`dream-1` wrote its central quantity to disk and no reader could print it, and `eta-1` repeated that inside the fix for it.
The readout ships in the same change as the mechanism, or the run cannot be read.
