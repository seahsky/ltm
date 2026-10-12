# ADR-0031: in silence, the detour returns to where it was loudest

Status: **proposed**, 2026-10-02.
Amends ADR-0016's cast from the step the agent's own offset detector fires. Before that step the surge, the scan, the cast, the arrival rule and `is_rising` carry unchanged.
**Gated on an off-box replay.** No arm ships and no night is booked until the replay's pre-registered branch says so.
**The gate's numbers below are proposed and are Sky's to set before the replay reads a run.**

`legs-1` read NULL (PR #172): a cast that acts on its finished legs does not move Find-SR.
ADR-0029's NULL branch names the next candidate: the silent half of the detour, and where the sound was loudest before it stopped.
This ADR proposes that change and the replay that prices it first.

## What the detour does in silence today

The source sounds for `sounding_steps` = 60 steps (ADR-0017, `WindowPolicy.FIXED_STEPS`), and the detour's budget is `investigate_max_steps` = 120.
A detour that does not arrive spends about half of its budget after the source has stopped.

After the offset step and the cue's tail, every reading is the bed.
`leg_replay` measured the cue constant to machine precision on every silent leg, so `is_rising` never fires there and no leg reader can decide.
The cast keeps its cycle (ADR-0016): 6 scan turns, then one turn and 8 forwards, alternating.

**The silent cast starts wherever the offset found the agent, and it uses nothing the sounding half learned.**
`ControllerState` carries the plateau count and, for `READ_LEGS`, the last leg. The runner trims `energy_history` to what `is_rising` reads.
No field holds where the agent was when the sound was loudest.

## The evidence, and how thin it is

- **The silent half is large.** 3,083 of 6,666 completed legs were walked on the bed alone at TC 1 (PR #149), and 3,144 of 5,680 at TC 0 (ADR-0029's pre-registration).
- **It reaches little.** SWS on `legs-1` is 23 of 271 for `full` and 30 of 268 for `full-b`. SWS counts every episode that ran past its offset step, and 72 of `full`'s 271 had reached the source before it. Among the rest the rate is about 23 of 199, 11.6% (estimate from the run's printed counts).
- **`pilot-2` priced the offset itself.** A source that never stops reached 40.8% against 34.2% for the windowed one, net −24 of 365, exact McNemar p 0.0027. That is 6.6 points.
- **`oracle-2` measured the whole gap** at 65.2 points, on the same criterion (ADR-0028).

**One of these argues against the lever, and it is stated here before any replay.**
`pilot-2`'s 6.6 points is under the 7.30-point MDE of a 282-episode night at the TC-0 flip rate.
It is what the cue is worth in the second half under today's controller. It is not a bound on a controller that does something different there.
It was also measured at TC 1, on 365 episodes of an earlier build, so it does not transfer as a number.
It still says the first expectation is a small lever, and the gate below is written so that a small lever reads STOP.

**A place handed to the silent phase has never been priced against the cast.**
ADR-0018's memory cells replace the cast's probe with a recalled place once the source is silent. Every `matrix-2` cell did that, and no arm ran without it.
What `matrix-2` measured is the episodic store on top of that replacement: 10.3 points lost against a right prior (ADR-0026). That replacement explains the loss is ADR-0026's hypothesis, not a measurement.
One part of it holds by construction: a place the agent goes to and then stands at reaches nothing. So this ADR does not replace the cast. It moves where the cast runs.

**What is not measured, and what the gate measures:** where the agent is when the source stops, where it was loudest, and how far each is from the source.
No reader has asked that of any run. The records carry what it needs on every step: `measured_rms`, `position`, `geodesic_to_source`, `source_playing`.

## The decision

A new typed arm on `RunConfig`, off by default. Three parts, each from the agent's own signals only.

1. **The agent keeps its loudest place.** It keeps the highest mean of `cue_phase_folds` consecutive readings, and its own position at one step of that window.
   - `cue_phase_folds` is the clip's loop length in steps, recorded by the calibration, 5 at the shipped defaults. A mean over one whole loop cancels the loop's phase.
     `is_rising` uses `RISING_WINDOW` = 5, a constant set for a different reason. The two match at the shipped defaults by coincidence, which `controller.py` already names.
   - Windows hold detour readings only. The first window ends at the detour's step `cue_phase_folds`, and the last ends at the step before the detector fires.
   - The position is the one at index `cue_phase_folds // 2` of the window, counting from 0. That is the middle step for an odd count.
   - A later window replaces the kept one only if its mean is strictly higher.
   - A scan turn changes the level at a fixed position. So the loudest place can be a heading and not a place. Nothing corrects for that, and the gate's accuracy test measures the net result.
2. **The agent decides the source has stopped when the level is back at the bed.** The agent heard the room before the sound: `pre_onset_rms` is its own last reading before the window opened.
   The detector fires on the first detour step where the last `cue_phase_folds` detour readings are each within `pre_onset_rms_tol` (5%, relative) of that level. Readings from before the detour do not count.
   - It fires once in a detour.
   - It is not "under `onset_rms`". `onset_rms` sits between the bed and the source's low level across the calibration poses, so a sounding source heard from far away reads under it. That rule would call far silent.
   - A source too faint to lift the reading 5% over the bed still reads as silent. The gate measures how often that happens.
3. **When the detector fires, the agent routes to the loudest place, and the cast continues there.**
   - If the agent is within 1.0 m of the loudest place, horizontal distance between its own positions, nothing changes. The arm is then `full`, step for step.
   - Otherwise the loudest place is a divert waypoint and the follower routes to it. The runner already hands a divert to the follower for the oracle arm and for the memory prior. This arm's condition is its own detector, not the runner's `sounding` flag, which is the task's ground truth.
   - During the return the waypoint steers, and the cue rule does not: no surge and no cast action is applied. The arrival rule still ends the detour.
   - The return ends when the agent is within that 1.0 m. The plateau count is held during the return, and the cast continues from the count it had when the detector fired. No scan is repeated.

**One variable.** The arm changes where the silent cast runs and nothing else.
A different search at the loudest place (a spiral, a bounded sweep) is a second variable, and `dream-2` is the record of what a two-variable contrast costs (ADR-0024).

The agent does not know whether the loudest place is nearer the source than where it stands. It returns whenever the two are 1.0 m apart.
So the arm can lose reach, and the gate prices the losses with the gains.

## The gate: an off-box replay, pre-registered

**The tool.** `earshot/tools/silent_replay.py`, to be built: tested, read-only, no GPU, minutes.
The loudest-place tracker and the offset detector land with it as pure functions in `agent/controller.py`, and the replay imports them, so the gate prices the functions the arm would run.
Nothing in the controller's rule calls them until the gate reads BUILD. ADR-0029 did the same with `leg_t` and `leg_verdict`.

**The data.** Five renders of `full`'s rule at TC 0, each over the same 282 episodes:

```
python -m earshot.tools.silent_replay runs/no-tc/no-tc runs/regate-a/no-tc runs/regate-b/no-tc runs/legs-1/full runs/legs-1/full-b
```

- The `read-legs` arms are not read. Their legs were chosen by a reader, and this gate prices a change to `full`.
- **The five are not one code version.** `no-tc` ran at 8e6f16d, the two `regate` renders at 32d3989, and `legs-1` at c8e2348, after the preset flip (#169) and `READ_LEGS` (#170). ADR-0030 refused a code difference for its re-gate. This gate accepts it and checks the rule in place of the commit: each detour's actions are rebuilt and checked step by step against the recorded `realizable_action`, as `leg_replay` does, and an episode that disagrees is never read.
- The five share their episodes, so the pooled counts are episode-renders and not independent episodes. The per-run tests below are the guard.

**A run is refused, by name,** if any of these holds:

- its recorded arm is not `full`'s on `leg_replay`'s four `FULL_ARM` fields, or its recorded `temporal_coherence` is not false;
- no step carries `geodesic_to_source`, `position` or `realizable_action`;
- its calibration records `cue_phase_folds` under 1. A window of zero readings would fire the detector on its first step.

**An eligible episode** meets all of these:

- it entered INVESTIGATE, and its first detour step has a route to the source;
- it carries `pre_onset_rms` and its window's `offset_step`;
- the detector fired on a detour step;
- the source was not reached at or before that step;
- the step the detector fired on and the loudest place both have a route to the source.

An episode whose detector fired while `source_playing` was still true is eligible, because the arm would act there. It is also counted as a **false fire**.
Episodes that fail a condition are counted per condition and printed. They are never dropped without a count.
`legs-1` suggests 185 to 199 eligible per render (estimate: 271 past the offset, 72 reached before it, and up to 12 that never heard the onset).

All distances are `geodesic_to_source`, which is analyst-only and never reaches the controller.

| symbol | what it is |
|---|---|
| `d_off` | route to the source at the step the detector fired |
| `d_L` | route to the source at the loudest place |
| `d_min` | the smallest route to the source at any detour step up to and including the step the detector fired |
| `d_end` | route to the source at the detour's last step |

An episode is **informative** when `d_L` and `d_off` differ by 1.0 m or more. That is a difference in route to the source, and it is not the 1.0 m of the decision's part 3, which is the agent's distance to its own loudest place.

**It reports, per run and pooled:**

1. **Where silence finds the agent.** The distribution of `d_off`, with counts in five bands: under 2 m, 2 to 3, 3 to 5, 5 to 8, 8 m and over.
2. **What the silent cast does from there.** `d_end − d_off`, and the share of episodes that reached the source after the detector fired, by the same bands.
3. **What the loudest place knows.** `d_L − d_off`, the count of informative episodes, and among them the share where the loudest place is the nearer one.
4. **How far loudest is from nearest.** `d_L − d_min`, which is zero or more by construction.
5. **The offset detector.** Its delay in steps after the record's `offset_step`, and the false fires as a share of episodes whose detector fired on a detour step.
6. **The return's length.** The forward steps the agent took between the loudest place and the step the detector fired. It is a path the agent can retrace, so it bounds the return's distance. It does not bound the return's steps: the follower turns 30° a tick, and turning round alone costs about six.
7. **The price.** Defined below.

**The price.** The reach curve is one logistic fit, `p(d)`, of "reached the source after the detector fired" on `d_off`.
It is fitted once, on the eligible episodes of all five runs that are not false fires.
For each eligible episode the price is `p(d_L) − p(d_off)` if the episode would return (part 3's 1.0 m, on the recorded positions), and zero if it would not.
A run's price is the sum over its eligible episodes, in points of 282. **The pooled price is the mean of the five run prices.**
The price counts `SOURCE_REACHED`, which under the oracle detector is the same episode set as Find-SR@1m.

The fit is monotone in distance, so a loudest place that is farther prices as a loss with no extra rule.
If the fitted curve does not fall with distance, every return to a nearer place prices at zero or under, and the gate reads STOP.
The banded table of item 2 is printed beside the fit so a misfit can be seen. It is not used to correct the price.

**The price is optimistic, and every named bias points that way:**

- it does not subtract the return's steps from the budget;
- it assumes a cast that arrives at the loudest place reaches as often as a cast that silence found at that distance. The agent has already been at the loudest place once without arriving;
- episodes that silence finds near the source are likely the easier ones, so the curve credits distance with what the episode's difficulty did;
- a false fire is priced on the silent curve. The source still sounds there, and the return takes the agent off a live cue;
- the curve is fitted on the episodes it prices.

A price that fails while optimistic fails.
One bias can point the other way: if the true curve is much steeper near the source than a logistic allows, the fit underprices short returns. The banded table shows it if so.

**The branches**, checked in order. The first that applies is the reading.

| branch | condition | what happens |
|---|---|---|
| **NOT_RUN** | a run is refused; or a run has fewer than 100 eligible episodes; or fewer than 30 pooled episodes reached the source after the detector fired, so the curve cannot be fitted; or false fires are more than 10% of the detector's fires in any run | not a result about the loudest place. The cause is found and fixed first. A detector that cannot tell silence needs a new rule, and a new rule is an amendment made before any other number is read |
| **STOP** | the pooled price is under **7.30 points**; or the price is zero or negative in any run; or the loudest place is the nearer one in under **75%** of informative episodes pooled, or under **70%** in any run; or any run has fewer than 30 informative episodes, so its accuracy cannot be measured | the lever closes with no box time spent. Report items 1 and 4 stay as findings |
| **BUILD** | anything else | the arm is built as "The decision" states it. The night's branches are written into this ADR before it is booked |

7.30 points is the MDE of one 282-episode night at the 19.1% TC-0 flip rate (`power.mde_paired`). An optimistic price under it names an effect the night could not resolve.
75% and 70% are ADR-0029's accuracy thresholds, kept so the two gates read on one scale.
**The 7.30, the 75% and 70%, both 1.0 m values, the 5% tolerance, the floors of 100 and 30, and the 10% are judgment, not derivation.**

**What `legs-1` changed about this gate.** ADR-0029's gate priced a verdict's accuracy, passed at 94.0% and 77.2%, and the night was a null.
An accurate signal is necessary and it did not convert. So this gate has a reach test beside its accuracy test, built from what the silent cast is measured to reach.

**A reading the gate gives whichever branch it takes.** Report items 1 and 4 split the gap by half of the detour.
If most eligible episodes have `d_min` far from the source, the detour never came near while the source sounded, and the loss is in the sounding half. No change to the silent half can reach those episodes.
If `d_min` is near and `d_off` is far, the agent came near and left. That is the case this ADR is for.

## The night, if the gate passes

```
nrun bash earshot/tools/ablation_sweep.sh --tag <fresh-tag> --arms "full <arm> full-b <arm>-b"
```

Four arms, the repeat in the same run, as `legs-1` ran. About 7 h 36 m at `legs-1`'s rate.
The primary outcome is Find-SR@1m. SWS is the secondary, because the arm acts only after its detector fires.
**Liveness before any null:** each episode records whether the detector fired, whether the agent returned, and how many steps the return took. An arm that never returned is NOT_RUN.
The night's branches and its MDE are written here after the replay and before the booking. They are not written now, because the replay sets what a plausible effect is.

## Consequences

- A typed arm on `RunConfig`, mapped to plain parameters at the runner's call site, as `CastPolicy` is. `agent/` still does not import `config`.
- `ControllerState` carries the loudest window's mean and its position, the detector's run of bed-level readings, and whether a return is in progress. It is state for the reason `plateau_steps` is: the runner trims `energy_history`.
- The controller needs the agent's pre-onset level, which the runner holds on its onset state today and does not pass in.
- The record carries the step the detector fired and the loudest place, so a replay rebuilds the arm exactly.
- `silent_replay.py` lands first, in its own PR, with both arms of the detector tested: a far, faint, sounding trace that must not read as silent, and a silent one that must (ADR-0014).

## We chose this over

- **A second night of `READ_LEGS`, or a LOUDER-only arm.** ADR-0029 names the one-branch arm under its LOSS branch. The night read NULL, and both contrasts are under one point.
- **A recalled place from memory.** ADR-0018's stores answer the silent phase with a category's location. `matrix-2` is the matrix of record for that, and ADR-0025 closed the DREAM term.
- **A local search at the loudest place.** A second variable. It is the next step if the gate reads BUILD and the night shows the return works.
- **A gradient fitted over the sounding trajectory.** It could name a place beyond the loudest one. It needs more state and a fit with its own failure modes, and the replay's item 4 says first whether the trajectory came near at all.
- **A longer sounding window.** That changes the task, not the agent.

## What this cannot reach

An episode whose detour never came near the source while it sounded has no near place to return to.
The 12 episodes with no navmesh route cap every arm at 270 of 282.
The 65.2-point gap is a ceiling on knowing where the source is, and `pilot-2` says most of it is not in the silent half.
