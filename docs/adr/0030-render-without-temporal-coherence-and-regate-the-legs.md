# ADR-0030: render without `temporalCoherence`, and re-gate the legs on that render

Status: **proposed**, 2026-09-23.
Reopens ADR-0029's gate on legs rendered with the preset key off, under that ADR's thresholds.
Changes `spec.ACOUSTICS_PRESET`, which ticket 06 set and which every run since the clean room has used.

## What was measured

`no-tc` (2026-09-23, commit 8e6f16d, `PHASE2_ABLATION_REPORT.md` has the tables) ran `full` and `full` with `--temporal-coherence off` on the same night, over the same 282 episodes.

- **Find-SR@1m did not move.** 98 against 102 of 270, 22 gained and 26 lost, exact McNemar p 0.6655; scenes 7 up and 10 down, sign p 0.629.
- **The preset key buys no speed at this preset.** 0.0348 audio s per step with it off, 0.0350 with it on. Ticket 06's "about 10%" was measured at the stock 5000 rays and depth 200.
- **The pull is the preset key.** With it off, sounding legs read +1.56 approaching and −1.91 receding, and 84.6% of approaching legs read up (same phase). `full` on the same night read −0.41 and −1.29 (PR #150's pull), and −9.0% per loop on walking legs (the figure PR #156 first read on the older runs).
- **ADR-0029's grid reads BUILD on the `no-tc` legs** at T 1.0: LOUDER 95.8%, QUIETER 79.9%, decisive on 32.9%. `full` that night reads STOP, with QUIETER at 55.9%.
- **What it cost.** The climb floor (`cue_render_scatter`) rose 22% at the median. 10 episodes never heard the onset, and all 10 failed in `full` as well. SWS fell from 0.133 to 0.099.

`walk-1` had already shown where the loss is: at the same poses along the same walks, the preset key cuts the rise on approaching legs from +16.2% to +3.1% per loop, in the renderer's own IR.
Ticket 01's parameter sheet records the key as off by default and as needing continuous motion. The agent moves in discrete 0.25 m steps and discrete turns.

## Decision 1: re-gate the legs on the render with the key off

ADR-0029's gate is re-read over three renders of the `no-tc` arm.

**The data.** `runs/no-tc/no-tc`, which was read before this ADR, and two fresh renders of the same arm over the same episodes:

```
nrun bash earshot/tools/ablation_sweep.sh --tag regate-a --arms no-tc
nrun bash earshot/tools/ablation_sweep.sh --tag regate-b --arms no-tc
```

About 1 h 52 m each, which is half of `no-tc`'s 3 h 43 m for two arms (estimate).
They run before Decision 2 lands, because Decision 2 removes the `no-tc` arm name.

**The three renders must be the same code.** The driver runs `git pull --ff-only` itself, so a change merged between the nights would reach one render and not another.
Each run's `provenance.txt` records its commit. Before the read, `git diff 8e6f16d <commit> -- earshot/` must hold comments and docs only, for both fresh renders. Anything else is NOT_RUN.

**The renders must be independent.** If `episode_diff` finds 0 discordant pairs between any two of the three, the renderer repeated itself, the guard below is vacuous, and the gate is NOT_RUN.
`repeat-1` measured 16.2% of outcomes flipping at TC 1, so this is not expected. It has not been measured at TC 0.

**The read**, with `leg_replay` as it stands on `main` when this ADR merges and not changed before it runs:

```
python -m earshot.tools.leg_replay runs/no-tc/no-tc runs/regate-a/no-tc runs/regate-b/no-tc
python -m earshot.tools.leg_replay runs/regate-a/no-tc runs/regate-b/no-tc
```

**Everything in ADR-0029's gate carries unchanged:** the grid of `T_LEG`, the 75% pooled and 70% per-run accuracy for each branch judged alone, the 25% decisive rate, the most-decisive-passing choice with ties to the smaller value, and an unmeasurable accuracy failing.
Sky set those numbers before the first replay read anything. Changing them now would be choosing them by the result.
Where ADR-0029's text and `leg_replay.py` differ, the code reads: its ONE BRANCH choice breaks a tie on the decisive rate by accuracy before `T_LEG` (`leg_replay.py:838-841`), and ADR-0029 names only `T_LEG`.

**One guard is added, because one of the three renders has been seen.** The `no-tc` arm was picked after ADR-0029 read STOP, and its first render is the one that read BUILD.
So the verdict must also hold on the two fresh renders without it.
At the `T_LEG` the three-render read chooses, over `regate-a` and `regate-b` pooled, each passing branch is at least 75% right.
The decisive rate is judged over the same two renders pooled, as ADR-0029 judges it: at least 25% of informative legs for the two branches together under BUILD, and for the one branch alone under ONE BRANCH.

**How to read the guard off the second command.** Ignore the verdict it prints for itself: it chooses its own `T_LEG` and says it is not the gate.
Read its grid row at the three-render `T_LEG`. Each branch's accuracy is its right/fired. The combined decisive rate is printed. A branch's decisive rate alone is its fired count over the informative count printed once above the grid.
Nothing new has to be built.

**The branches**, fixed here, before either fresh render exists:

- **BUILD.** Both branches pass the three-render gate and the guard. Implement `READ_LEGS` at the chosen `T_LEG` (ADR-0029), on the Decision 2 render. ADR-0029's four-arm night follows, and its own branches are written before it is booked, as that ADR requires.
- **ONE BRANCH.** Exactly one branch passes both. Ship that branch and leave the other INCONCLUSIVE, as ADR-0029 says.
- **STOP.** Anything else. `READ_LEGS` stays closed. Decision 2 stands without it.

**In practice the outcome is BUILD or STOP.** On the render already read, neither branch is decisive on 25% by itself at any `T_LEG`: at T 1.0 LOUDER fires on 212 of 1,415 informative legs (15.0%) and QUIETER on 254 (18.0%), and both rates fall as `T_LEG` rises. So a guard that fails on either branch reads STOP, as ADR-0029's rules already imply.

**Not the sounding subset.** `leg_replay` also prints the grid over sounding legs alone, where `no-tc` reads higher (95.7% and 92.4% at T 1.0). That split was chosen after the first STOP, so the gate reads the whole grid, as ADR-0029's did.

**The rerun flip rate at the new render comes free.** The three renders pair by episode, so each of

```
python -m earshot.tools.episode_diff runs/regate-a/no-tc runs/regate-b/no-tc
```

and its two siblings measures how many outcomes flip on a byte-identical rerun with the key off.
The key keeps state between renders, so whether it added to `repeat-1`'s 16.2% has not been measured. ADR-0029's night needs this rate to size its MDE.

## Decision 2: the key goes to 0, whatever the gate reads

`ACOUSTICS_PRESET["temporalCoherence"]` becomes 0.

- It buys no speed at this preset, and Find-SR does not move detectably either way.
- It hides the approach along a leg, and the approach along a leg is what the controller reads.
- Ticket 06 admitted it on a gradient check over static poses, which could not have seen that. Ticket 01 had flagged it for discrete motion.

**What the key does keep, measured once:** 10 onsets of 282, all in episodes that failed in both arms, and a higher SWS (0.133 against 0.099, never tested for significance).
Neither reaches Find-SR on this evidence. Both are recorded here so that the flip is not later read as free.

A STOP in Decision 1 does not reverse this. It would say the leg reader does not pay, not that a render that hides approaches should stay.

**THE RECORD MUST SAY WHICH RENDER RAN.** Every run without `--temporal-coherence` records `run_config.audio.temporal_coherence` as `null`, which has meant "the preset", which has been 1. (`runs/no-tc/no-tc` records `false`.)
Flip the preset and leave the record alone, and `null` means 1 on every old run and 0 on every new one, with nothing on disk to tell them apart.
So the change that flips the key makes every new record a bool:

- `__main__.py`'s `--temporal-coherence` default stops being `None`, and `config_from_args` always writes a bool. Today it overwrites the `AudioConfig` default with `None` whenever the flag is absent, so changing `AudioConfig` alone would not be enough.
- `AudioConfig.temporal_coherence` defaults to `False`, and the preset entry becomes 0.

From then on, `null` means only "before ADR-0030", which is 1.
No reader in the tree resolves a `null` today: `hold_probe` and `leg_probe` set the key per arm, and `flip_report` reads the ray count only.

**The `no-tc` arm goes, and `tc-on` replaces it.** After the flip, `--temporal-coherence off` is the default, so `no-tc` would be byte-identical to `full`. That is `dream-2`'s shape: a control differenced against itself and reported as a clean null.
`tc-on` (`--temporal-coherence on`) is the arm that reproduces the pre-flip render, so a later night can price the difference again against its own `full`.

## What this changes, and what it does not

- **Every number before this ADR stands, as a number at TC 1.** The ablation table of record (`abl-2`), `matrix-2`, `oracle-2` and the DREAM arc were all measured with the key on, and they stay quoted that way.
- **A new comparison needs its control at the new render.** Every arm in this repo is already read against a `full` from the same night, and after the flip that `full` renders with the key off.
- **The ablation arms are not re-run by this ADR.** `no-climb`, `no-cue` and `anechoic` each touch what the climb hears, so their rows at TC 0 may differ. A claim about a component at the new default needs that component's arm at the new default.
- **The onset loses 10 episodes of 282 at TC 0 on this evidence.** All 10 also failed at TC 1, so Find-SR does not see them, but they are attrition at stage 2 that `full` did not have. `window_report` prints them as CENSORED.

## Order

1. Merge this ADR. Nothing else ships with it.
2. Run `regate-a` and `regate-b` on today's code, and read the gate. That is one night and no code.
3. The flip: preset to 0, the explicit record, and `tc-on` in place of `no-tc`, with their tests. One PR, whatever step 2 read, and not merged before both renders have run. It touches:
   - `earshot/audio/spec.py` (the preset and its comment) and `earshot/audio/config.py` (the default);
   - `earshot/__main__.py` (the flag's default, its help text, and `config_from_args`);
   - `earshot/tools/ablation_sweep.sh` (the arm, its reason and the header) and `earshot/tools/placement_report.py` (`ABLATION_ARMS`);
   - the "tc1 = the shipped preset" labels in `earshot/tools/hold_probe.py` and `earshot/tools/leg_probe.py`;
   - `tests/mac/test_audio_spec.py`, `test_config.py` and `test_ablation_sweep_driver.py`, which pin the key at 1, a `None` default, and `no-tc` as the only arm carrying the flag;
   - `CLAUDE.md`.
4. If step 2 read BUILD or ONE BRANCH: `READ_LEGS` (ADR-0029), then that ADR's night, with its branches written first.

## What this cannot settle

- The legs were walked under blind alternation. A reader that acts on them changes which legs get walked, so the gate prices the verdict and not the sweep. This is ADR-0029's caveat and it carries.
- Why `walk-1` could not reproduce the recorded fall at TC 1: the run has a whole episode of render history behind each leg, and the probe had 10 walk-in steps. `no-tc` establishes the cause. It does not measure how the history depth enters.
- Whether Find-SR moves at TC 0 under a controller that reads legs. `no-tc` measured it under one that does not.
