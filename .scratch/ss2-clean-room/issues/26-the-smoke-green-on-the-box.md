# 26 — The smoke, green on the box

Type: task
Status: open
Blocked by: 25

## Question

Build `task/smoke.py` and get task spec §8's nine assertions green on the RACE V100.

This is the destination as the map named it: one HM3D episode, end to end, with SoundSpaces 2.0 audio rendered live in the simulator at every step.

## The nine assertions

1. **Audio is live and every-step.** Render count equals step count exactly.
2. **The audio context is sound.** Ticket 12's guard armed and green through ticket 16's verified invariants: mesh vertex floor cleared, no swallowed spec keys, canary armed on **every** render.
3. **The IR is real.** Non-silent, scene-dependent, trimmed to actual decay rather than fixed-width.
4. **Provenance did not raise** (§3.1).
5. **The full loop ran.** SEARCH → onset → INVESTIGATE → CHECK → RESUME → a legitimate termination. CHECK and RESUME must **both** be reached.
6. **A report was emitted** with §5.1's schema fully populated.
7. **Per-step audio wall-clock recorded and inside a stated ceiling.** Set **generously**, not at ticket 06's 27.2 ms — ticket 06 measured 2.3x pose variance against ticket 04 on the same scene, so a tight bound fails for a reason that is not a regression.
8. **`env_check.py` passed** (ticket 17).
9. **Hermeticity** — deferred to ticket 27, because it is a re-run rather than an assertion.

The smoke verifies audibility at its own start pose once, with a calibration render, so it is deterministic (§2.5's one exception).

## Two decisions, not bookkeeping

**The primary find-task is NOT required to succeed.** The destination says the agent *runs* its primary find-task. Requiring success would gate an irreversible deletion on a backbone measured at 0.031 benchmark SPL with ~45% explore-timeout — the capability ADR-0006 retreated from claiming. What is required is that the primary loop runs and terminates legitimately.

**The smoke runs the realizable arm**, not the oracle one. An oracle-localization smoke leaves the entire live-audio path unexercised in the one episode that exists to prove it, and the sound really would be a stopwatch. The oracle arm is retained as a **bisection tool**: on failure, running it isolates audio from controller in one step.

The risk is taken deliberately — a realizable climb that cannot reach an audible same-floor source is a finding better had *before* deleting the old tree.

## Required disclosure

The smoke runs an **oracle STOP**, so it does not exercise goal detection at all.
`diagnose_spin` decomposed the 0.031 benchmark SPL as stop_miss ~50% + explore_timeout ~45% + success ~5%, and an oracle STOP deletes the stop_miss half outright.
**Smoke find numbers will look far better than 0.031 for a reason that must be disclosed rather than enjoyed, and they are not capability numbers.**

## Also settled here

Task spec §9's builder numbers, set against measurement rather than argued: `investigate_max_steps` (currently 40), the bed level constant and the audible band the calibration sweeps, criterion 7's wall-clock ceiling, and the tolerance on §3.1's pre-onset RMS assertion.

And the box confirmation `guarded_observe()` has never had — it is new code from ticket 20 whose fakes have never met the binary, which is exactly the shape of ticket 12's warning shot (27 green tests, then a raise on the first real spec).

## Watch for

`docs/race-box-runbook.md` — the self-update gotcha lives in 33 drivers and has already cost a 10-hour run. Read §8's footguns before the first box trip, not after.
