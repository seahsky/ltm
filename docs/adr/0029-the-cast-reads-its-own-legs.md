# ADR-0029: the cast reads its own legs

Status: **accepted**, 2026-09-21 (PR #146), with the gate's thresholds as proposed.
Amends ADR-0016's cast. The surge, the scan, the arrival rule and `is_rising` carry unchanged.
**Gated on an off-box replay.** No controller code ships and no night is booked until the replay's pre-registered branch says so.
**The gate ran on 2026-09-21 and read STOP.** `READ_LEGS` does not ship; see "The result".

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

The night below is not booked.

## The night, if the gate passes

```
nrun bash earshot/tools/ablation_sweep.sh --tag <fresh-tag> --arms "full read-legs full-b read-legs-b"
```

**Four arms, the repeat measured in the same run**, about 7 h 12 m at `oracle-2`'s rate.
`propose-1` is why: it was the first result in this arc to clear its own noise, because the repeat ran beside it.
Two arms (about 3 h 36 m) would test against a flip rate from other nights, and the MDE at `n = 282` is **6.36 points** on the current 14.5% flip rate.
`full-b` and `read-legs-b` are the same flags under a second name, which `ablation_sweep.sh` does not have yet.

The primary outcome is **Find-SR@1m**. The secondary is stage 4 → 5 conversion (`episode_diff --given-stage INVESTIGATE_ENTERED`), because the reader acts inside the detour.
The night's own branches are written into this ADR **after** the replay and **before** the night, because the replay's decisive rate says how large an effect is plausible.

**Liveness before any null.** The audit carries `n_legs`, `n_louder`, `n_quieter` and `n_inconclusive` per episode, so the night can say whether the reader fired before anyone reads a delta.
DREAM's arc is the reason: its memory term was the first lever in this repo proven live before its null (ADR-0025), after `dream-1` and `dream-2` had both returned nulls about a mechanism that never ran.

## Consequences

- `CastPolicy.READ_LEGS`, and `cast_action` gains the last verdict as an argument. The two pure functions it calls, `leg_t` and `leg_verdict` in `agent/controller.py`, landed with the replay.
- `ControllerState` carries the current leg's readings and positions. The runner trims `energy_history`, and a leg is longer than the window. This is the same reason the plateau counter lives on the state (ADR-0016).
- **The reversal needs checking on the fake world before the box.** A rule turn becomes a probe offset, and the follower, not the rule, turns the body. So the realized heading change of "reverse" has to be asserted as 180° in a test, not assumed.
- `oracle-2` inferred that the run-to-run noise enters through the audio-driven steering. A reader that acts on more of the audio may add variance, which is a second reason the repeat runs in the same night.

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
