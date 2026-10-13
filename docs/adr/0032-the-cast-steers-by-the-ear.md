# ADR-0032: the cast steers by the ear

Status: **accepted**, 2026-10-13 (PR #178), with the night's numbers as proposed.
**Built, and the fake-world checks pass** (2026-10-13). See "How it was built". The night is not yet booked.
Amends ADR-0016's cast at one place: the odd legs' turn. The surge, the scan, the even legs, the arrival rule and `is_rising` carry unchanged.
**Gated on fake-world checks, then a pre-registered night.** The night's numbers below are proposed and are Sky's to set before it is booked.

ADR-0031's replay put the loss in the sounding half: of the detours that had not arrived when the source stopped, 44.8% never came within 5 m of it by route before the offset detector fired (PR #175).
`sounding_replay` then read why (PR #177): the detours had the time, and the agent heard the right side.
This ADR proposes that the cast stop turning away from what it hears.

## What the cast does with the ear today

`lateral_sign` is the interaural level sign, read every step: +1 the source to the right, −1 left, 0 nothing.
The cast reads it in two places (`agent/controller.py`, `cast_action`):

- **the scan:** `SCAN_STEPS` = 6 ticks, each turning toward the side the sign names;
- **every leg's opening:** `turn_right = lateral_sign > 0`, then flipped when the leg's ordinal is odd.

**So even legs turn toward the heard side, and odd legs turn away from it.**
While the sign holds steady, the turns alternate. That is ADR-0016's design: following a stable sign on every leg traces a closed polygon, and around a source a closed polygon is an orbit. The argument was not measured.
With a sign 0, the turns alternate left and right by ordinal alone.

An earlier draft of this ADR, and the report section PR #177 added, said the legs after the first ignore the sign. That was wrong, and a fresh-context review found it; both are corrected in this PR.

## The evidence

- **The ear is right 71.8%** of 64,647 sounding steps, 71.4% to 72.2% in every run (PR #177). It is 66.6% at 8 m and over, and 66.9% where the route is more than 1.2 times the straight line. The heading behind each grade was rebuilt from 69,971 pairs of forwards, and none disagreed.
- **So an odd leg turns the wrong way about 72% of the time it hears a side** (inference: the odd-leg openings are a subset of the steps graded, and nothing selects them by accuracy).
- **The detours had the time.** Only 85 of 1,324 (6.4%) could not have arrived while the source sounded, charging no turn and no wall. 357 of the 442 that never came within 5 m had the time.
- **The detour falls off at 8 m.** From 8 to 12 m, 315 of 316 detours had time and 11 arrived.
- **The sign is worth reading.** `abl-2`'s `no-cue` arm reads every sign as 0: the scan turns left six times and the legs alternate by ordinal alone. It reached 59 of 282 against `full`'s 101 (20 gained, 62 lost). That prices every use of the sign together, at TC 1, and does not separate the scan's aim from the legs' turns.
- **In silence the ear says nothing.** The bed is the same samples in both ears (`audio/bed.py`), so once the cue's reverb tail has gone the sign is 0.

**One result argues against acting on an accurate signal.** ADR-0029's leg verdicts read 94.0% and 77.2% right, and acting on them moved nothing (`legs-1`).
This ADR does not assume accuracy converts. The night prices reach.

## The decision

A new `CastPolicy.EAR_LEGS`, off by default, from the agent's own signals only.

**When the sign at a leg's opening names a side, the leg turns toward it, odd leg or even.** When the sign is 0, the leg turns as `full`'s does.

- Even legs are unchanged: they already turn toward the heard side.
- An odd leg with a sign of +1 or −1 turns toward that side, where `full` turns away.
- An odd leg with a sign of 0 turns as `full`'s does, right after a left. So once the cue's tail has gone, every opening is `full`'s, and the cast in silence does not trace a polygon of left turns.
- The sign is the single reading at the opening tick, the one `full` already reads. **One variable:** the odd-leg flip, when the ear has an answer.

The cue carries the room's reverb for `cue_tail_steps − 1` steps past the offset step, so the first openings after the source stops can still hear a side. The arm can differ from `full` there and nowhere else in silence.

**The turn.** A turn action places a probe 60° off the body's heading, 2 m out, and the follower routes to it. On the fake world the follower answers with one 30° turn, and the next tick's forward aims along the new heading. That the box's follower does the same is inferred, not shown. The record carries the follower's action at every opening, so the night measures it.

**The orbit.** ADR-0016 flipped the odd legs to prevent one, and this arm removes the flip whenever the ear answers. As a first-order inference: at a sign right with probability p, a leg's expected correction is 30° × (2p − 1), about 13° at 71.8%. A source abeam at r metres moves its bearing by about 115°/r over a 2 m leg. So the arm cannot keep a source abeam inside roughly 9 m. That bound is pessimistic, because it assumes the agent walks across the bearing and not toward it. Whether the arm homes or circles is what the fake-world checks and the night measure.

## Before the night

No off-box read. The arm acts on the same single reading `sounding_replay` graded at every sounding step, at the same render, and a floor on its accuracy would decide nothing the 71.8% has not.
What is not known is the motion: whether turning toward the ear on every leg homes or circles. A replay cannot answer that, because the arm changes which legs get walked.

So, on the fake world, before the box:

1. **Homing.** A source 6 m to one side, a sign that is right: the arm's distance to the source falls across legs, and the test prints it beside `full`'s.
2. **Circling.** The same at 2 m, and with a sign right 72% of the time by a fixed sequence: the test prints the path length per metre closed for both arms. It does not assert a value.
3. **Silence is `full`.** With the sign 0 at every opening, the arm's actions equal `full`'s tick for tick.
4. **The record rebuilds the arm.** `detour_report.rule_action` takes the policy, and `leg_replay` and `detour_report` agree with the arm's recorded `realizable_action` at every step.

## The night, pre-registered

```
nrun bash earshot/tools/ablation_sweep.sh --tag <fresh-tag> --arms "full ear-legs full-b ear-legs-b"
```

Four arms, the repeat in the same run, as `legs-1` ran. About 7 h 36 m at `legs-1`'s rate. All four render at TC 0.

**What this n can resolve.** 7.30 points per contrast at the 19.1% TC-0 flip rate (`power.mde_paired`). The driver prints its own figure before it starts, and that one is authoritative. `legs-1` measured 19.5%.

**Liveness first.** Each `EAR_LEGS` episode records the openings where it turned toward the ear and `full` would have turned away, and the follower's action at each of them. `window_report` prints both per arm.

**The six reads**, as `legs-1` printed them: `full -> ear-legs` (contrast A), `full-b -> ear-legs-b` (contrast B), the in-run repeats `full -> full-b` and `ear-legs -> ear-legs-b`, and A and B given INVESTIGATE_ENTERED.
Each contrast is an exact McNemar over its own 282 pairs and is read on its own. The scene-level sign test is reported beside each and breaks no tie.

**The branches**, checked in order. The first that applies is the reading.

| result | reading | what happens |
|---|---|---|
| **NOT_RUN.** `window_report` exits 2; or in either ear arm no opening turned against `full`'s rule; or a reader's rebuild of either ear arm disagrees with its record on any step | the arm did not run as specified | not a result. Find the cause first |
| **GAIN.** A and B both net positive, both exact p < 0.05, and each net larger than both repeat nets, signed | steering by the ear converts into reach | `EAR_LEGS` becomes `full`'s cast policy. The ablation table is re-run on the new baseline before any component is quoted against it |
| **LOSS.** A and B both net negative, both exact p < 0.05 | turning toward the ear on every leg costs reach; the circling ADR-0016 argued for is the first candidate | `EAR_LEGS` is retired. `detour_report`'s metres walked per metre closed, per arm, says whether it circled |
| **DIRECTION ONLY.** A and B have the same sign, and at least one misses p < 0.05 or does not clear the repeat nets | the sign reproduces and the size is not resolved | reported as measured. A second night is Sky's call |
| **NULL.** Anything else, including a net of 0 in either contrast | the ear knows the side and steering by it does not move reach | `EAR_LEGS` closes as a lever on Find-SR |

As in ADR-0029, GAIN must clear the repeat nets and LOSS need not: a loss at p < 0.05 in both contrasts is enough to retire an arm.
The secondary reads qualify a branch and never pick one, under ADR-0029's drop-count rule.
**The liveness rule, the 7.30, and the repeat-net rule are judgment carried from ADR-0029, not derivation.**

**What the night cannot say.** Whether a larger turn, or a side voted over the whole leg, would do more: the arm changes the odd-leg flip and nothing else. Anything about TC 1.

## How it was built (2026-10-13)

- **The arm is one condition in `cast_action`.** The odd-leg flip is skipped when `ear_legs` is set and the sign is nonzero. `ear_overrides` names exactly the openings where that changes the answer, and a test checks it against `cast_action` over every opening and sign.
- **The decision marks the ear's turn.** `ControllerDecision.ear_turn` is set on the tick the cast chose a turn that `ear_overrides` names, and never on a surge or an arrival. The runner counts these as `ear_turns`, and as `ear_turns_followed` when the follower's own action on that tick was the same turn.
- **The record rebuilds the arm.** `detour_report.rule_action` takes `ear_legs`, and `leg_replay` and `detour_report` read it off the record's `cast_policy`. `window_report` runs `leg_replay.episode_legs` over each ear arm's episodes and prints THE EAR: the turns, the turns followed, and the steps rebuilt. It exits 2 if the arm never turned against `full`'s rule or a step does not rebuild.
- **The driver** gains `ear-legs` and `ear-legs-b` (`--cast-policy ear_legs`), and its closing six reads run for whichever four-arm night is on disk.

**The fake-world checks.**

1. **Homing.** In open floor with a right sign, the arm arrives within 60 ticks from every one of 240 random bearings at 2, 4, 6, 8 and 10 m. `full` arrives from 2 m and from none further. Its zigzag runs about 15° off the scan's aim, the bisector of the two alternating headings, and passes the source just outside the ring (inference from the walks: at 6 m abeam its nearest pass is 1.04 m).
2. **A noisy ear, printed** (sign errors drawn independently each tick, 240 bearings per row):

   | sign right | arrives within 60 ticks from | 2 m | 4 m | 6 m | 8 m | 10 m |
   |---|---|---|---|---|---|---|
   | 71.8% | `ear-legs` | 45.8% | 36.7% | 34.6% | 33.3% | 28.3% |
   | | `full` | 45.8% | 16.2% | 11.2% | 7.1% | 5.8% |
   | 66.6% | `ear-legs` | 37.9% | 29.6% | 18.3% | 21.7% | 17.9% |
   | | `full` | 37.9% | 16.2% | 12.1% | 5.0% | 6.2% |

   The arm walks less path in every row from 4 m out, because more of its walks end at the ring. Circling is not measured separately here.
3. **Silence is `full`.** With the sign 0, the arm's actions equal `full`'s for 120 ticks at every distance.
4. **End to end on the fake world**, a source 8 m abeam: the arm made 2 ear turns, the follower followed both, and it reached the source at step 41. `full` came no nearer than 2.22 m. `leg_replay` rebuilds the arm's record at every step, and rebuilt as `full` it does not.

**What these cannot say.** They have no walls, no climb and no surge. The errors are independent, while a real room's errors are likely tied to the pose. They price the cast's geometry, not the night.

## Consequences

- `CastPolicy.EAR_LEGS`. `cast_action` takes the policy as a flag: with it set and a nonzero sign, the odd-leg flip is skipped. With it unset, the function is unchanged.
- No new state. The sign at the opening tick already reaches `cast_action`.
- `detour_report.rule_action` takes the flag, so `leg_replay`, `detour_report` and the night's liveness check rebuild the arm from its record.
- The runner records the opening ticks where the arm turned against `full`'s rule, as `legs_*` were recorded for `READ_LEGS`.
- `controller.py`'s docstring said the later legs alternate "regardless of the sign". That was wrong and is corrected in this PR.

## We chose this over

- **A side voted over the whole finished leg.** Nine readings at different poses may be wrong less often together than one. It is a second variable, it needs the leg buffer `READ_LEGS` keeps, and it is the next step if this arm's turns are right but too few.
- **A bearing from the scan's sign flips.** Turning through a revolution, the sign flips once facing the source and once facing away, and the direction of each flip tells them apart. A bearing to about 30°, at the cost of a revolution each time. More state and more steps.
- **Re-scanning before every leg.** Six turn ticks per two-metre leg cut the sounding window's travel by about 40%.
- **A larger turn per leg.** A second variable.
- **The ILD magnitude.** `audio/lateral.py` keeps only the sign.
- **A longer sounding window.** That changes the task, not the agent, and 93.6% of detours had time.

## What this cannot reach

The 85 detours out of time while the source sounded, and the 12 episodes per run with no navmesh route.
Detours that start 12 m or more away are out of time in 43.5% of cases.
`pilot-2` measured 90 to 99% of detour steps plateaued, at TC 1; if that holds at TC 0, the cast is most of every detour, and this arm acts on half its legs.
The 98.5% ceiling is knowing where the source is. A sign right two times in three does not give that.
