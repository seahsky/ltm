# ADR-0029: the cast reads its own legs

Status: **accepted**, 2026-09-21 (PR #146), with the gate's thresholds as proposed.
Amends ADR-0016's cast. The surge, the scan, the arrival rule and `is_rising` carry unchanged.
**Gated on an off-box replay.** No controller code ships and no night is booked until the replay's pre-registered branch says so.
**The gate ran on 2026-09-21 and read STOP.** `READ_LEGS` does not ship; see "The result".
**ADR-0030 re-reads this gate on legs rendered with `temporalCoherence` off**, over three renders, with the thresholds above unchanged.
**That re-gate read BUILD at T_LEG 1.0 (2026-10-01), so `READ_LEGS` was built on the flipped render.**
**The night ran on 2026-10-01 (`legs-1`) and read NULL.** `READ_LEGS` closes as a lever on Find-SR, and `full`'s cast policy does not change; see "The night's result".

`oracle-2` measured the headroom: Find-SR@1m is 33.3% for `full` and 98.5% for the same controller handed the source coordinate, on the same criterion, over the same episodes (ADR-0028).
The follower, the budget, the navmesh and the arrival rule ran unchanged in both arms, so the 65.2 points are localization.
This ADR proposes the first change aimed at that gap since the gap was measured.

## What the cast does today

When the cue is dead, `realizable_investigate_step` hands the tick to `cast_action` (ADR-0016):

- **scan**: `SCAN_STEPS` = 6 turns, one full revolution at `investigate_probe_turn_deg` 60°;
- **cast**: one turn, then `CAST_STEPS` = 8 forwards, which is one probe's reach, `investigate_probe_m` 2.0 m;
- **alternate**: each new leg turns the other way from the last, by `(index // period) % 2`.

**Nothing reads a finished leg.**
The agent walks two metres along one heading, and the next leg turns by a rule that looks only at the leg's ordinal.
A leg that got louder and a leg that got quieter produce the same next move.
The only reader of the cue during a leg is `is_rising`, and it acts only by interrupting the leg with a surge.

## A correction to how this lever was named

ADR-0028 and the `oracle-2` report called this "comparing level at leg endpoints 1 to 2 m apart instead of across one 0.25 m step".
The second half describes `eps-1`'s *measurement unit*, not the current rule, and the difference changes the argument.

`is_rising` compares the mean of the last five readings with the mean of the five before, against a bar of `max(eps, s·√(2/5))`, where `s` is the SD of the ten pooled readings.
Its baseline is already up to ten steps long. What it lacks is not length. Two properties matter:

- **It pools turns with forwards.** A turn changes the measured level without changing the distance, and its own docstring leaves "whether the contamination is biased rather than symmetric" as the next thing to measure.
- **Its noise estimate contains the signal it is testing for.** `s` is taken over readings that carry the trend. On a noiseless linear climb over the ten readings the bar is already 38% of the gap (`s` = 3.03 steps' rise, × √0.4 = 1.91, against a gap of 5). That does not block a clean climb, but it spends part of the noise budget on the thing being detected.

So the lever is not "a longer baseline". It is a reader that uses one heading, measures its noise off the trend, and acts on what the leg found.

## The evidence, and how thin it is

- `oracle-pilot` (2026-09-19): median `sig/sc` **1.94** over plateau windows spanning a median **1.66 m**.
  `sig/sc` is `|slope| × d_span / residual SD`, with the slope fitted against the TRUE distance to the source.
  Three limits: **one scene, 15 episodes**; the trajectories were the oracle arm's, which approach the source by construction, the best case for a gradient; and the slope is absolute, so a window that got quieter while approaching scores the same.
- The same run: one forward buys `rise/eps` **0.02** at 3–5 m, negative beyond 5 m, in the cue domain.
- `eps-1`'s 0.61–0.86 of local scatter per step is from before ADR-0019 split the readout, so it prices a different quantity and is not evidence either way here.

**Suggestive, not established.** The gate below re-measures it at scale, on the realizable arm's own legs, before anything ships.

## The decision

A third `CastPolicy` value, `READ_LEGS` (`--cast-policy read_legs`, sweep arm `read-legs`).

**At the end of each cast leg, fit the cue level against the agent's own displacement along the leg**, over the readings taken at the leg's heading.
The slope's t-statistic against a critical value `T_LEG` gives one of three verdicts:

| verdict | the next leg |
|---|---|
| **LOUDER** | keeps the heading: no turn |
| **QUIETER** | reverses the heading |
| **INCONCLUSIVE** | turns as today, alternation unchanged |

**Why a fit and not two endpoints.** Two endpoints is the fit at n = 2 and throws away the readings between them.
The fit uses every reading on the leg, and its residual is measured off the fitted line, so the trend does not enter its own noise estimate.
One heading means no turn enters either.

**Why the agent's own displacement and not the step count.** A leg that collides or that the follower shortens covers less than two metres, and a fit against step index would call that leg flat.
The pose is agent-estimable: the realizable arm already reads it to place every probe.
No source coordinate and no route enters, so the arm stays realizable (ADR-0001).

**Why INCONCLUSIVE is today's move.** A reader that never clears `T_LEG` is byte-identical to `full`.
So the contrast against `full` is one variable: whatever the new arm gains or loses comes from the verdicts that fired, and the audit counts how many did.
It does not mean the arm cannot do worse. Wrong verdicts can, which is what the gate prices.

**What is not touched.** `is_rising`, the surge, the scan, `SCAN_STEPS`, `CAST_STEPS`, the arrival rule.
A leg interrupted by a surge gets no verdict: the surge has already acted on the cue.

### It has a threshold, and that has to be said

The base rate for thresholds retuned in response to a result is 0 for 7 in this repo (ADR-0025).
`T_LEG` is a threshold. The difference is when and how it is set:
it is chosen **off-box, on three runs already on disk, before the arm runs once**, it has to reproduce in each run separately, and it is written into the code before the night and not touched after.
The night tests the branch, not the number.

## The gate: an off-box replay, pre-registered

**The tool.** `earshot/tools/leg_replay.py`, tested, read-only, no GPU.
For every completed cast leg in a realizable arm's traces, it computes the verdict over a grid of `T_LEG` values and grades it against `Δroute`, the change in `geodesic_to_source` from the leg's first pose to its last.
`Δroute` is analyst-only and never reaches the controller.

A leg is **informative** when `|Δroute| ≥ 0.5 m`. A leg that ran across the source's bearing has no right answer, so it is scored separately and not graded.
Legs are segmented from the recorded `realizable_action`. The audit records position but not yaw, so a leg's heading comes from its displacement.
A run that does not carry `realizable_action`, `geodesic_to_source` and `CalibrationRecord.cue_render_scatter` is refused by name, never reconstructed silently.

It reports, per run and pooled:

1. **per-scene plateau `sig/sc`** off `detour_report`'s own `_plateau_rows`, which is the at-scale re-measurement of the 1.94;
2. **decisive rate**: the share of informative legs that get LOUDER or QUIETER;
3. **accuracy**: the share of decisive verdicts whose sign matches `Δroute`, split by verdict and by the route distance at the leg's start;
4. **false-decisive rate** on uninformative legs;
5. how often `is_rising` fired on those same legs, which is what the current reader already recovers.

**The data.** `abl-2/full`, `oracle-1/full` and `oracle-2/full`: 846 episodes of the same arm at identical behaviour, three independent renders.
All three runs postdate the three fields by date (`realizable_action` 2026-08-07, `geodesic_to_source` 2026-08-08, `cue_render_scatter` 2026-08-31); the tool verifies it per run.

**The branches**, written before the replay runs:

- **BUILD.** A single `T_LEG`, chosen on the pooled data, gives decisive accuracy **≥ 75%** at a decisive rate **≥ 25%** of informative legs, and keeps accuracy ≥ 70% in each of the three runs separately. Implement `READ_LEGS` at that value.
- **ONE BRANCH.** If only LOUDER or only QUIETER passes, ship that branch and leave the other INCONCLUSIVE.
- **STOP.** Anything else. The cue over a two-metre leg is not recoverable on the agent's own axis, and the lever closes with no box time spent.

50% is chance. 75% is a verdict right three times in four, and fewer than a quarter of informative legs changes too little of the cast to matter.
**Those two numbers are judgment rather than derivation, and they are the one decision in this ADR that is Sky's to set before the replay runs.**

### The gate, operationalized before the replay read a run

Building the replay forced four details the branches above leave open.
Each is fixed in the commit that builds the tool, before it has read a single run, so none of them can be chosen by looking at the result.

- **The grid.** `T_LEG` is chosen from 1.0, 1.5, 2.0, 2.5, 3.0, 4.0 and 6.0, and from nothing else.
- **Each branch is judged alone.** BUILD needs LOUDER and QUIETER each right at least 75% pooled and at least 70% in every run, and the two together decisive on at least 25% of informative legs. ONE BRANCH needs one branch to pass those accuracy tests and to be decisive on at least 25% of informative legs by itself. The branch text above reads BUILD on the pooled accuracy of both verdicts, but ONE BRANCH only makes sense if each branch is judged separately. Pooling would also let a right LOUDER branch carry a coin-flip QUIETER branch over 75%, and QUIETER is the verdict that reverses the agent.
- **Where several values pass, the most decisive is chosen.** It changes the most of the cast at the accuracy the gate demands. Ties go to the smaller `T_LEG`.
- **An accuracy that could not be measured fails.** A branch that fired on no informative leg in one run has no accuracy in that run, and it does not pass.

**The fifth report item was wrong, and the replay reports a replacement.** "How often `is_rising` fired on those same legs" is zero on every leg the replay grades, by construction. A leg gets a verdict only if it reaches its next turn, and a surge would have ended it first.
The replay reports instead how many completed legs walked 0.5 m or more toward the source with no surge.
The current reader missed each of those legs, and a right LOUDER verdict is what `READ_LEGS` would add there.

**`leg_t` and `leg_verdict` are in `agent/controller.py` already**, and so far only the replay calls them.
The replay imports them rather than keeping its own copy, so the gate prices the function the arm will run.
`leg_verdict` takes `t_leg` with no default, so the arm cannot run at a value the replay did not choose.

**What the replay cannot say.** The legs in `full` were walked under blind alternation. A reading rule changes which legs get walked, so the replay prices the *verdict*, not the sweep.
It is the same caveat `detour_report` prints on its arrival ceiling: a ceiling, not a prediction.

## The result

`leg_replay` over `abl-2/full`, `oracle-1/full` and `oracle-2/full`, 2026-09-21: **STOP**.
The full readout is in `PHASE2_ABLATION_REPORT.md` under "leg replay".

- **The replay is exact:** 81,945 of 81,945 detour steps agree with the recorded rule, and no leg is unverified. 6,666 legs completed, and 4,817 are informative.
- **LOUDER is right and rare.** At T 1.0 it is 92.3% right pooled and 90.4% in the worst run, and it fires on 324 informative legs, 6.7%.
- **QUIETER is near chance:** 57.1% to 61.7% at every value where it fires on more than 32 legs.
- **The STOP depends on the second operational detail above.** Judged on accuracy alone, LOUDER would be ONE BRANCH at T 1.0. That detail was fixed before the run, so the STOP stands. A LOUDER-only arm needs its own pre-registration.
- **The leg reads direction, over a downward trend.** At T 1.0, approaching legs read LOUDER 11.4% and QUIETER 22.6%. Receding legs read LOUDER 1.1% and QUIETER 35.6%.
- **Hypothesis, not measured:** `full`'s source sounds for 60 steps and the detour budget is 120, so legs after the offset get quieter whichever way they walk. `source_playing` on the same records settles it.
- **The 1.94 re-measured at scale:** a median plateau `sig/sc` of 1.50 over 1,227 windows in 19 scenes, with 70.6% of windows louder nearer the source.
- **The sounding split (PR #149, not a gate) keeps the STOP.** Restricted to legs read while the source sounded, LOUDER fires on 14.4% and QUIETER is 59.3% right, so neither branch passes.
  The gate's denominator counted 2,163 informative legs read on the bed alone, where the cue is constant and no reader can decide. That understated every decisive rate, and it did not change the branch.
- **The windowed source explains only the 573 legs that span the offset**, which read QUIETER at chance. Sounding legs carry direction (median t −0.37 approaching, −1.38 receding) on top of a shift of about −0.9 that the offset does not explain. The loop phase is the candidate cause, and it is not measured.
- **The loop is not the pull (PR #151, not a gate).** A reader that pairs each reading with the one a whole loop later cancels the loop, and the pull grows: 63.7% of approaching sounding legs and 92.6% of receding ones get quieter.
  The loop was the 9-reading fit's noise: median t on receding legs goes from −1.38 to −6.86 once it is cancelled.
  The cause of the pull is open. Grading the same legs on straight-line distance to the source is the next read-only check.
- **The grading axis is not the pull either (PR #153, not a gate).** Graded on the horizontal straight line, 88.4% of legs that opened it and about two thirds of legs that closed it got quieter. Route and line agree on 92.8% of legs.
  The level falls along a leg whatever the geometry. The render preset ships `temporalCoherence: 1`, which ticket 01 flagged for discrete motion and nothing has A/B'd against a leg. That is a hypothesis, not a finding. `PHASE2_ABLATION_REPORT.md` names the two checks that test it.
- **The level falls standing still (PR #155, not a gate).** In the detour's first scan, where the agent turns in place, 83.7% of 184 reads fall, by a median 11.1% per loop, against 76.7% and −9.0% on walking legs.
  So the fall is mostly a trend in time, not motion, and the motion hypothesis above is contradicted. Two caveats are open: complete scans exclude those a surge cut, and only 453 of 979 complete sounding scans stood still.
  A controlled probe on the box, standing still with `temporalCoherence` 1 and 0, is the next check.
- **The pull is `temporalCoherence` (`no-tc`, 2026-09-23, not a gate).** `full` and `full` with the preset key off ran on the same night over the same 282 episodes. With it off, sounding legs read +1.56 approaching and −1.91 receding (same phase: 84.6% of approaching legs read up), and the pull is gone. `full` on that night read −0.41 and −1.29 again.
  This grid over the `no-tc` legs reads **BUILD at T 1.0**: LOUDER 95.8%, QUIETER 79.9%, decisive on 32.9%. `full` on the same night reads STOP, with QUIETER at 55.9%.
  **This is not the gate.** It is one render against the three this ADR names, and the arm was chosen after the STOP. Reading these thresholds on legs rendered with the preset off is a new pre-registration. Find-SR did not move (98 against 102 of 270, p 0.67), which is what a controller that does not read legs predicts. `PHASE2_ABLATION_REPORT.md` has the tables.

The night below is not booked.

## The night, if the gate passes

```
nrun bash earshot/tools/ablation_sweep.sh --tag <fresh-tag> --arms "full read-legs full-b read-legs-b"
```

**Four arms, the repeat measured in the same run**, about 7 h 12 m at `oracle-2`'s rate.
`propose-1` is why: it was the first result in this arc to clear its own noise, because the repeat ran beside it.
`full-b` and `read-legs-b` carry exactly their originals' flags (PR #170).
The night renders with `temporalCoherence` off, the preset since ADR-0030, and `read-legs` refuses any other render.

The primary outcome is **Find-SR@1m**: `episode_diff` at its default stage, `SOURCE_REACHED`, which under the oracle detector is the 1.0 m confirm. The secondary is stage 4 → 5 conversion (`--given-stage INVESTIGATE_ENTERED`), because the reader acts inside the detour.
**Liveness before any null.** Each READ_LEGS episode records `legs_read`, `legs_louder`, `legs_quieter`, `legs_inconclusive` and `legs_unread`, and `window_report` prints them before anyone reads a delta.
DREAM's arc is the reason: its memory term was the first lever in this repo proven live before its null (ADR-0025), after `dream-1` and `dream-2` had both returned nulls about a mechanism that never ran.

### The night's pre-registration (2026-10-01, before booking)

Written after ADR-0030's re-gate read BUILD and before the night is booked, as this section required.

**What this n can resolve.** Each arm holds 282 paired episodes. At the 19.1% flip rate ADR-0030 measured at this render, the paired MDE is **7.30 points** per contrast (80% power, alpha 0.05, `power.mde_paired`). A smaller effect is not resolvable by this night, however it is read. The driver prints its own figure before it starts; that one is authoritative.

**What the replay says is plausible, and what it does not.** At T_LEG 1.0 the reader decided on about 30% of completed legs over three renders (1,433 decisive verdicts on 4,234 informative legs, plus 19.5% of 1,263 uninformative ones). Silent legs, 3,144 of the 5,680 completed, never get a verdict. The replay prices verdicts on legs walked blind. It does not price how reach changes when the agent acts on them, which is what this night measures. No effect size is predicted here.

**The six reads**, printed one line each at the end of the driver's readout (`night_reads` in `ablation_sweep.sh`):

| # | read | what it is |
|---|---|---|
| 1 | `full -> read-legs` | contrast A |
| 2 | `full-b -> read-legs-b` | contrast B, the repeat of A |
| 3 | `full -> full-b` | the baseline against itself: tonight's noise |
| 4 | `read-legs -> read-legs-b` | the reader against itself: tonight's noise with the reader on |
| 5 | `full -> read-legs`, given INVESTIGATE_ENTERED | contrast A, stage 4 → 5 |
| 6 | `full-b -> read-legs-b`, given INVESTIGATE_ENTERED | contrast B, stage 4 → 5 |

Each contrast is an exact McNemar over its own 282 pairs, and each is read on its own. The two are not pooled: they share episodes, so pooling would count each episode twice. The scene-level sign test is reported beside every contrast and breaks no tie (ADR-0016).

**The branches, fixed here.** They are checked in order, and the first that applies is the reading.

| result | reading | what happens |
|---|---|---|
| **NOT_RUN.** `window_report` exits 2, or in either READ_LEGS arm LOUDER plus QUIETER is under 10% of `legs_read` | the reader barely acted, at about a third of the replay's rate or less | not a result about reading legs. Find why the live arm's legs differ from the replay's, and fix it, before any delta is read |
| **GAIN.** Contrasts A and B both net positive, both exact p < 0.05, and each net larger than both repeat nets (reads 3 and 4) | acting on a finished leg converts the direction it carries into reach | `READ_LEGS` becomes `full`'s cast policy. The ablation table is re-run on the new baseline before any component is quoted against it |
| **LOSS.** A and B both net negative, both exact p < 0.05 | wrong verdicts cost more than right ones buy. QUIETER is wrong on about 23% of its legs, and on 41% of legs that span the offset | `READ_LEGS` as built is retired. A LOUDER-only arm is the remaining shape, and it needs its own pre-registration |
| **DIRECTION ONLY.** A and B have the same sign, and at least one misses p < 0.05 or does not clear the repeat nets | the sign reproduces, the size is not resolved at this n | not reported as a gain or a loss. Both contrasts and both repeats are reported as measured. A second night is Sky's call, not automatic |
| **NULL.** Anything else: A and B disagree in sign, or both are inside the repeat nets | the leg knows the direction and acting on it does not move reach | `READ_LEGS` closes as a lever on Find-SR. The silent half of the detour is the next candidate (where was the sound loudest before it stopped), and it needs its own ADR |

**The secondary reads (5 and 6) qualify a branch and never pick one.** They are only sound if the two arms drop about the same number of pairs for not reaching INVESTIGATE, because the reader acts after that stage. `episode_diff` prints both drop counts. A clear gap means the conditioning has failed, and only the unconditional reads are quoted.

**Three things this night cannot say.**

- Which verdict did the work. LOUDER and QUIETER act together. A per-verdict attribution needs a one-branch arm, and that is a separate night.
- Whether the gain or loss lives in the sounding half of the detour. `legs_*` are per episode, not per sounding state.
- Anything about TC 1. Every arm renders at TC 0, and so does `full`. The ablation table of record (`abl-2`) is at TC 1, and nothing tonight is quoted against it.

### The night's result (`legs-1`, 2026-10-01)

Four arms at commit c8e2348, 282 episodes each over 19 scenes, 7 h 36 m, every gate green: **NULL**.
The tables are in `PHASE2_ABLATION_REPORT.md` under "ADR-0029's night".

- **The reader was live.** It said LOUDER or QUIETER on 439 of 1,620 legs in `read-legs` (27.1%) and on 447 of 1,619 in `read-legs-b` (27.6%), against a 10% floor and about 30% in the replay. NOT_RUN does not apply.
- **The two contrasts differ in sign.** `full -> read-legs` is net +2 over 64 discordant pairs (exact McNemar p 0.90). `full-b -> read-legs-b` is net −2 over 52 (p 0.89).
- **Both are inside the night's own noise.** `full -> full-b` is net +9 over 55 (p 0.28), and `read-legs -> read-legs-b` is net +5 over 55 (p 0.59).
- **The scene level agrees with neither sign.** A is 7 scenes up and 6 down (exact sign test p 1.00), and B is 4 up and 9 down (p 0.27).
- **The conditioned reads change nothing.** Given INVESTIGATE_ENTERED, both contrasts keep every discordant pair: +2 over 64 and −2 over 52, on 270 pairs. The drops are 12 against 11 and 12 against 10, so the conditioning is sound.
- **Find-SR@1m** is 95, 97, 104 and 102 of 270 for `full`, `read-legs`, `full-b` and `read-legs-b`.
- **The flip rate is 19.5%** in both repeats, against the 19.1% the MDE was computed from. The reader added no measurable variance.

NULL is the last branch, and both of its conditions hold.
**`READ_LEGS` closes as a lever on Find-SR.** `CastPolicy.READ_LEGS` stays in the tree as an arm, and `full` keeps the alternating cast.
An effect under the 7.30-point MDE is not ruled out; the two estimates are +0.7 and −0.7 points.

The gate and the night do not contradict each other.
The replay priced the verdict at 94.0% and 77.2% on legs walked blind. The night priced what acting on it is worth, which is what the gate said it could not price.
Why a correct verdict does not convert is not measured. LOUDER and QUIETER acted together, so one helping and the other costing would read as this null, and no arm here separates them.
The next candidate is the one the NULL branch names: the silent half of the detour, under its own ADR.

## Consequences

- `CastPolicy.READ_LEGS`, and `cast_action` gains the last verdict as an argument. The two pure functions it calls, `leg_t` and `leg_verdict` in `agent/controller.py`, landed with the replay.
- `ControllerState` carries the current leg's readings and positions. The runner trims `energy_history`, and a leg is longer than the window. This is the same reason the plateau counter lives on the state (ADR-0016).
- **The reversal needs checking on the fake world before the box.** A rule turn becomes a probe offset, and the follower, not the rule, turns the body. So the realized heading change of "reverse" has to be asserted as 180° in a test, not assumed.
- `oracle-2` inferred that the run-to-run noise enters through the audio-driven steering. A reader that acts on more of the audio may add variance, which is a second reason the repeat runs in the same night.

## How it was built (2026-10-01)

ADR-0030's re-gate read BUILD at T_LEG 1.0. Building the arm forced six details this ADR left open. Each is fixed in the code before the night, and none was chosen by looking at a night.

- **The verdict reads the gate's leg.** At a leg's opening, the controller fits the last `1 + CAST_STEPS` readings and the positions they were taken at. While the plateau runs unbroken, those are the leg's readings from the one after its turn through the opening tick, which is `leg_replay`'s `rows[start + 1 : start + LEG_PERIOD + 1]`. The runner hands the controller this tick's `measured_rms` and `position`, the values the record keeps. `along_leg` moved from `leg_replay` into `agent/controller.py`, and the replay imports it. A fake-world run checks that the replay, rebuilt over the arm's own record, reaches the arm's verdict on every leg it read.
- **QUIETER is a new action, `ACT_REVERSE`, and the reversed leg holds its heading.** The probe is placed afresh every tick from the body's yaw, and the follower turns the body 30° a tick. A reverse probe placed once is lost on the next tick. So the reversed leg's forwards aim along the reversed heading until the leg ends, and the follower turns the body round. On the fake world the walk after a reversal points 180° from the leg before it.
- **The leg a reversal opened is not read.** Turning round takes about six of its nine readings, which mixes turns with forwards: the shape this ADR built the reader to avoid, and not one the gate graded. Its opening records `unread` and casts as today.
- **LOUDER is a forward at the body's yaw.** At a leg's end, the body faces along the leg it walked.
- **The verdict goes on the record.** `StepRecord.leg_verdict` is set on the tick a verdict chose the action, and `detour_report.rule_action` takes it, so a replay rebuilds a READ_LEGS run exactly. `leg_replay.load_run` still refuses any run that is not `full`'s rule: legs walked under a reader are not the gate's population.
- **`T_LEG` is `controller.READ_LEGS_T_LEG = 1.0`**, a constant with no flag. A test pins it to the gate's grid.
- **The arm refuses any render but `temporalCoherence` off, said explicitly.** That value was chosen on legs rendered without it, and on legs rendered with it the gate read STOP. `runner.check_read_legs_render` raises before the environment probe, so a `read-legs` run on the old preset, or on a record whose value is `null`, fails in seconds and never reaches a night.

**Liveness.** Each READ_LEGS episode records `legs_read`, `legs_louder`, `legs_quieter`, `legs_inconclusive` and `legs_unread` (the names this ADR's night section gives as `n_*`). `window_report` prints them per arm and exits 2 if a reading arm said louder or quieter on no leg. The driver reads that exit code, so the night ends red rather than quoting a reader that did not act.

**The arms.** `read-legs` is `--cast-policy read_legs`. `full-b` and `read-legs-b` carry exactly their originals' flags, and `full-b` keeps the baseline's per-scene smoke bar.

## We chose this over

- **A smaller epsilon.** Rejected by measurement (`eps-1`).
- **`RISING_SIGMAS = 0`** (`earshot/loosen-the-bar`). Confounded twice already (ADR-0016's second amendment). It also tunes how often the pooled test fires, not whether the cast uses what it walked through.
- **Longer legs alone.** More signal per leg and still no reader. The replay's breakdown of legs by length shows whether length is what limits a verdict.
- **A two-leg finite-difference gradient.** More information from two headings, and more state to get wrong. It is the next step if single-leg verdicts are accurate but too rarely decisive.
- **The ILD magnitude**, which `audio/lateral.py` discards. A different cue with its own record.
- **A stochastic cast.** ADR-0016 rejected it, and `oracle-2` located the apparatus noise in the audio steering.

## What this cannot reach

The 12 episodes with no navmesh route cap every arm at 270 of 282, 95.7%.
The reader acts only in the cast phase, after a full scan, so a detour that dies before its first complete leg is unchanged.
And 98.5% is a ceiling on knowing where the source is. Nothing here says a realizable reader gets near it.
