# ADR-0028: the localization ceiling arm arrives on the baseline's own test

Status: accepted, 2026-09-20. **Measured 2026-09-20 (`oracle-2`): the first pre-registered branch fired** — see "The result" at the end.
Adds `Localization.ORACLE_MATCHED`. Does not change `Localization.ORACLE`.

`oracle-1` was the first `--localization oracle` run in the history of this repo.
It measured a +173-episode gain and a Find-SR@1m that went the other way, and the reason is that the two arms were never scored on the same question.

## The run

`ablation_sweep.sh --tag oracle-1 --arms "full oracle-loc"`, 19 HM3D val scenes, 15 episodes each, `n = 282` per arm, 3 h 30 m, exit 0, commit `72b8184`.

| arm | source reached | rate | Find-SR@1m | source SPL |
|---|---|---|---|---|
| `full` | 93 / 282 | 33.0% | 93 / 270 | 0.251 |
| `oracle-loc` | 266 / 282 | **94.3%** | **2 / 270** | 0.007 |

`episode_diff`: 282 paired, 93 both, 16 neither, **173 oracle-only, 0 full-only, exact McNemar p = 0.0000**, gains in 19 of 19 scenes.
Every pair agreed on `source_xyz` before it was subtracted.

Both columns come off the same 282 episodes.
A 61-point gain on one and a 34-point loss on the other is not noise and it is not a surprise once the criteria are read.

## The cause, verified in code

The two arms did not share an arrival test.

- `ORACLE` arrived when horizontal distance to the source fell inside `ControllerConfig.investigate_arrive_radius_m`, **1.5 m**.
- `REALIZABLE` arrives on the detector's confirm, which under `Detector.ORACLE` is `DetectorConfig.oracle_radius_m`, a **geodesic 1.0 m**, set at Find-SR's primary ring so the agent stops where the metric counts it.
  Since the `arrive-2` change that is the whole test: `realizable_investigate_step` returns `ACT_STOP` on `visual_confirm` alone, with no plateau conjunct.

Both write `FunnelStage.SOURCE_REACHED`.
`compute_source_spl` then scores success as `source_reached and dist_at_reach <= success_radius`, and `success_radius` is 1.0.
So the oracle agent stopped the instant it crossed 1.5 m, resumed, and never earned the criterion the baseline is quoted on.

`full`'s own identity is the proof: 93 reached and 93 Find-SR@1m, the same 93.
Where arrival **is** the 1.0 m ring, the two numbers are one number.

**Consequence: `oracle-1`'s +173 is arrival criterion and information mixed in a proportion that run cannot separate.**
No "localization is worth N points" claim comes off it.
`oracle-loc` is a ceiling on its own criterion and on nothing else.

## What survives the confound, and it is the valuable half

This rests on the oracle arm alone rather than on the difference, so the mismatch does not touch it.

`detour_report runs/oracle-1/oracle-loc --across-scenes`: 266 reached, 16 abandoned, and **12 of those 16 had no navmesh route to the source at any step of the detour** (`mv2HUxq3B53` 1, `p53SfW6mjZe` 4, `qyAac8rV8Zk` 6, `wcojb4TFT35` 1).
With the coordinate handed over and a route available the agent reaches 1.5 m in **266 of 270, 98.5%** — four routable failures in the whole sweep.

**So the follower, the step budget and the navmesh are not what holds `full` at 33.0%.**
That is a localization failure, and it is the first time this arc has had evidence for the claim rather than an argument.

Two more counts come free.
`refused` is 0 and `in-ring` is 0 in all 19 scenes, so no abandoned oracle episode stood inside the ring and was scored as not arriving: the `arrive-2` refusal lever is spent in this arm.
And the 12 unrouted episodes **cap any Find-SR at 270 of 282, 95.7%**, structurally, under any controller — the detector's view-point list for the anomaly object is seeded with the source position alone, and a `None` distance reads as not detected.

`full` at 33.0% against `abl-2`'s 35.8% is a 2.8-point gap, inside `repeat-1`'s measured 3.0-point apparatus noise, so the two sweeps agree and HM3D v0.2 did not move underneath them.

## The decision

Add a third arm. `ORACLE_MATCHED` is `ORACLE`'s steering with `REALIZABLE`'s arrival test:

- handed the true source position, injected into the pool as the divert candidate, routed to by the navmesh follower, exactly as `ORACLE` is;
- arriving on `visual_confirm`, the **same expression** the realizable arm STOPs on, off the **same** detector query, computed once in the step loop and handed to both arms.

It is one pure function, `task/runner.oracle_arrived`, which selects the test per arm and raises on an arm it does not know.
The controller is untouched: it takes `arrived_at_source` as a plain `bool` and may not import `config` (ADR-0013), so the arm selects the test and the controller only obeys the answer.

An unknown arm raising rather than inheriting the ring is deliberate, and `oracle-1` is why.
An arm scored on 1.5 m does not look broken. It looks excellent.

### This is not a threshold retune

The base rate for those in this repo is 0 for 7 (ADR-0025), and the next reader will file this with the seven unless it is written down here in these words.

No threshold moves.
1.5 m keeps its value and keeps its arm; 1.0 m keeps its value and keeps its arm.
What changes is that a **ceiling** is now scored on the criterion its **baseline** is scored on.
A retune hopes a number will behave better. This makes two arms comparable, and its failure mode is a null that means something rather than a null that means nothing.

### `ORACLE` stays exactly as it was

`oracle-1` is on disk and its audits carry `localization_arm: "oracle"`.
Redefining that value would make the run unreproducible from this tree and would make one field mean two things — `dream-2`'s shape, where a control and its treatment could not be told apart afterwards.
`oracle-loc` also still answers a question of its own: run beside the matched arm it prices the last 0.5 m directly.

## The next run, pre-registered

```
nrun bash earshot/tools/ablation_sweep.sh --tag <fresh-tag> --arms "full oracle-loc-matched"
python -m earshot.tools.episode_diff runs/<tag>/full runs/<tag>/oracle-loc-matched
```

Two arms, 19 val scenes, 15 episodes, about 3 h 30 m on `oracle-1`'s measured rate.
`full` runs **in the same night** and is the control: `repeat-1` measured 16.2% of outcomes flipping on byte-identical reruns, worth 3.0 points, and this contrast does not take its control from a previous sweep.
Adding `oracle-loc` as a third arm costs about 1 h 45 m more and buys the ring decomposition; it is optional, because `oracle-1`'s arms pair against this one episode for episode.

The primary outcome is **Find-SR@1m**, not stage 5, because stage 5 is the number the arrival mismatch corrupted.
`full`'s 33.0% is the comparator. Pre-registered branches, on the matched arm's Find-SR@1m:

- **Above about 85%.** Localization is the whole gap. The cast-leg-endpoint change — comparing level at leg endpoints 1 to 2 m apart instead of across one 0.25 m step, which `detour_report`'s `sig/sc` of 1.94 over 1.66 m plateau spans says is recoverable — becomes the lever worth a night, and it attacks the measured cause rather than a threshold.
- **Between about 45% and 85%.** Localization is the largest single term and the arrival rule costs the rest. Both are levers and the gap between this arm and `oracle-loc`'s 94.3% prices the second one without another run.
- **Near `full`'s 33.0%.** The diagnosis inverts: knowing exactly where the source is buys nothing, so the confirm is the limiter and not the search. That would rule the cast-leg lever out before it is built, which is the cheapest thing this run can do.

Nothing here is quotable below 4 points, and 95.7% is the ceiling on every branch.

## What this does not settle

The confirm is **geodesic** and `dist_at_reach` is **horizontal**, which is the axis mismatch already documented in `metrics.py`.
It runs the safe way for this arm — a route is never shorter than the line it spans, so a confirm at 1.0 m geodesic implies 1.0 m horizontal, so every matched arrival is inside the ring Find-SR scores.
It is still two axes, and a future detector whose confirm is not a distance test (`Detector.CAPTION`) breaks the implication rather than weakening it.

`detour_report`'s unvalidated-reconstruction branch (PR #139) tested for the arm *name* `oracle`, and that predicate is widened in the same change that adds the arm.
`realizable_action` is written in `step_controller`'s `if realizable:` branch, so every non-realizable arm is blind to it, `oracle_matched` as much as `oracle`.
Matching on the name would have sent an `oracle_matched` run down "re-run to arm the check", which is a night spent arriving back at the same zero — the cost that branch exists to prevent.
The test is now "no realizable episode in this run", and two oracle-family arms run together read as unarmable rather than merely mixed, because splitting them produces two unarmable halves.

## The result

`ablation_sweep.sh --tag oracle-2 --arms "full oracle-loc-matched"`, 19 HM3D val scenes, 15 episodes each, `n = 282` per arm, 3 h 36 m, exit 0, commit `66927ed`, every smoke gate green.

| arm | source reached | **Find-SR@1m** | source SPL | steps/ep | SWS |
|---|---|---|---|---|---|
| `full` | 90 / 282 = 31.9% | **90 / 270 = 33.3%** | 0.253 | 192.2 | 0.102 |
| `oracle-loc-matched` | 266 / 282 = 94.3% | **266 / 270 = 98.5%** | 0.926 | 178.4 | 0.266 |

`episode_diff`: 282 paired, 90 both, 16 neither, **176 matched-only, 0 full-only**, exact McNemar p = 0.0000, gains in **19 of 19 scenes** against a sign-test threshold of 15.
With every discordant pair on one side the p is arithmetic rather than estimated: two-sided exact is 0.5^175, about 2e-53.

**The first branch fired: localization is the whole gap.**
65.2 points of Find-SR@1m separate the baseline from a ceiling scored on the baseline's own criterion, in the same night, over the same episodes.
The follower, the step budget, the navmesh and the arrival rule ran unchanged in both arms. Only the source coordinate differs, and it is worth everything between 33.3% and 98.5%.

**The arm did what it was built to do, and the record proves it rather than the argument.**
This ADR predicted that a 1.0 m geodesic confirm implies a 1.0 m horizontal reach, so every matched arrival lands inside the ring Find-SR scores.
Measured: **266 reached and 266 Find-SR@1m**, the same identity `full` shows at 90 and 90.
Both arms now report one number twice. The confound `oracle-1` carried is gone.

**Tightening the ring cost nothing, and that settles what `oracle-1` measured.**
`oracle-loc` at 1.5 m reached 266 of 282. `oracle-loc-matched` at the 1.0 m confirm reached 266 of 282.
`detour_report --across-scenes` splits the 16 abandoned **identically** across the two runs — `5cdEh9F2hJL` 2, `mv2HUxq3B53` 3, `p53SfW6mjZe` 4, `qyAac8rV8Zk` 6, `wcojb4TFT35` 1 — with the same 12 unrouted and the same four routable failures (two each in `5cdEh9F2hJL` and `mv2HUxq3B53`).
So the same episodes reach under either test; the 1.5 m arm stopped short of the ring and the 1.0 m arm walked the last half metre.
`oracle-1`'s 2 of 270 was a measurement artefact and nothing else. It was never a harder task.
The identity is at the scene grain, from two readouts; `episode_diff runs/oracle-1/oracle-loc runs/oracle-2/oracle-loc-matched` would put it at the episode grain and has not been run.

**The arrival rule is spent.** `refused` 0 and `in-ring` 0 in every scene: no episode stood inside the ring and failed to arrive.
**The structural ceiling is nearly attained.** 12 of 282 have no navmesh route to the source at any step, capping every arm at 270 of 282, 95.7%. The matched arm reads 266 of 282, 94.3%.

**The baseline did not drift.** `full` at identical behaviour read 35.8% in `abl-2`, 33.0% in `oracle-1` and 31.9% here: 3.9 points over three sweeps, consistent with `repeat-1`'s 3.0-point flip noise.
`oracle-2`'s code differs from `oracle-1`'s only in `oracle_arrived`, which returns `False` for `REALIZABLE` exactly as the inline expression did, and in `source_class` on the audit, which is a record field.
So `episode_diff runs/oracle-1/full runs/oracle-2/full` is a free, current flip-rate measurement on the baseline of record, replacing `repeat-1`'s from the `arrive-2` era. It has not been run either.

**What this does not show.** A ceiling bounds; it does not promise.
Nothing here says a realizable method can approach 98.5%.
The anomaly object's view-point list is seeded with the source position, so once the oracle agent arrives the confirm is near-certain. That is the design — it isolates "can the agent get there" from "does it know where there is" — and it means goal detection is not exercised in either arm, which is the oracle-STOP disclosure the smoke prints on every scene.

**What the branch names next.** The cast-leg-endpoint change: compare level at leg endpoints 1 to 2 m apart instead of across one 0.25 m step.
`detour_report`'s `sig/sc` of 1.94 over 1.66 m plateau spans says the cue is recoverable at that baseline, and `eps-1` proved that a smaller epsilon over one step cannot recover it.
It is the first lever in this arc with a measured ceiling behind it, and it attacks the measured cause rather than a threshold.
It is not booked here. It needs its own record and an off-box price first.
