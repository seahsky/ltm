# 26 — The smoke, green on the box

Type: task
Status: resolved (2026-08-05)
Blocked by: 25 (resolved)

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

## Comments

### 2026-08-05 — `task/smoke.py` exists, and the first run's four findings are closed

Not resolved: §8's criteria are not green *on the box*, because no session on this Mac can
run one. What changed is that the gate now exists, the four defects PR #32 handed forward
are fixed, and one of them turned out to be structural.

**The gate.** `earshot/task/smoke.py` — `judge()` is pure (records in, verdict out), so
ticket 19's third row applies and every criterion has a red path tested with injected
records: 33 new Mac tests, each criterion in both directions. Criterion 9 is **NOT_RUN by
construction** rather than omitted, so no run can read green while hermeticity is
outstanding. `judge_run_dir` reads the artefacts with the real readers, which is what
caught its own first bug — it read `run_paths()[0]` as the env-report *file* when it is the
run *directory*, and would have reported criterion 8 NOT_RUN on the box for a reason that
has nothing to do with the environment.

**1. The funnel over-credited.** `_funnel_stage` read each flag independently on the
premise *an episode that resumed necessarily investigated*, which the abort path falsifies.
The nesting is enforced now. This was criterion 5's own measurement, so the over-credit was
the gate asserting a loop that did not run.

**2. The detour re-entered forever.** SEARCH's guard was `onset_fired and not
state.investigated`, and an abort correctly leaves `investigated` False. The abort is now
terminal for the interrupt: one detour per episode, and the sub-budget is the whole
detour's rather than one attempt's.

**3. §9's sub-budget, set against measurement: `investigate_max_steps` 40 → 120.** 40 was
never enough and the re-entry hid it — the fake stall-turn climb needs **59 steps** and
passed under 40 only because a second attempt started from a pose the first had improved.
`test_it_still_reaches_the_source` was passing on the strength of the bug it was supposed
to be independent of; with the abort terminal it is now the budget's guard.

**4. The collision flag is recorded, and deliberately not consumed.** `StepRecord` gains
`collided` and `displacement_m` (+ `EpisodeAudit.forward_summary`), because nothing
separated a forward that moved from one that hit a wall. The rule does **not** read it, and
that is measured rather than assumed: `allow_sliding` is False, `heard_signal` takes no
pose, so a collided forward repeats the reading exactly and ADR-0011's stall branch already
turns. A collision branch was built, then reverted when trajectories came back
byte-identical with the flag read and ignored across four wall geometries.

### The finding that changed the arm

Chasing that flag produced the real one. **The realizable climb livelocks against any
obstacle.** `move_forward` was its only translation and the energy gradient decided where
forward pointed, so: blocked forward → flat reading → turn → gradient turns it back →
collide. Measured ending pressed flat against the wall at exactly `(0.00, -1.00)` with
**zero lateral movement**, 119 of 123 forwards colliding, unchanged by tripling the budget,
and identical in a geometry where moving 1 m sideways would have cleared it. No sequence of
that rule's actions can go around anything. The first box episode's *never line-of-sight,
`min_d2source` 3.19 m* is this.

The gap was never the oracle coordinate — it was the **planner**. During INVESTIGATE the
agent bypassed `_steer` entirely, while the oracle arm injected a waypoint and used the
navmesh follower. So the climb now names a **place**: `realizable_investigate_probe` turns
the carried rule's action into a point `investigate_probe_m` (2.0) along the heading, offset
`investigate_probe_turn_deg` (60, deliberately wider than the simulator's 30-degree turn so
a probe does not snap back onto the same obstacle), injected as the same `SOURCE_INVESTIGATE`
divert the oracle arm uses. **The arm stays realizable** — heading from live binaural energy
and the lateral sign, distance a constant, map the agent's own; no source coordinate enters,
and the two arms still name different fields so "which arm ran" is readable off a decision.
The realizable arm now **raises** without a pose rather than falling back to blind stepping.

**A Mac cannot show this working.** `_task_fakes.FakeWorld.follower` steers in a straight
line, so the livelock persists there for the fake's reasons; routing is a navmesh capability
and `tests/box/test_investigate_route_box.py` exercises it (ADR-0014), with a control arm
that walks the old straight line into the wall so a green cannot come from an unwalled pair.

628 Mac tests (581 → 628), ruff clean. Five plants verified red for the right reason: the
funnel un-nested, the abort re-entry, the discarded collision flag, the 40-step budget, and
the env-report path.

**Hands to the box.** Nothing here has executed against habitat-sim. The trip is
`python -m earshot.audio.clips --out-dir data/anomaly_audio` once, then
`python -m earshot --run-dir runs/ss2-first-episode --n-episodes 1 --max-steps 250`, then
`python -m earshot.task.smoke --run-dir runs/ss2-first-episode` — and
`tests/box/test_investigate_route_box.py` **before** trusting any of it, since the whole fix
rests on the follower routing.

---

## Comment — 2026-08-05, the box's second and third episodes

Two more runs on the V100, two more defects, and both were in the **builder** rather than
in the agent. Each was invisible in the artefact: every number the run recorded was
correct, and criterion 5 failed for a reason nothing in the record named.

### Run 2 (`runs/ss2-ep1`) — the source was a storey below the start

The detour worked. 62 forwards, **0 collisions**, 15.5 m walked — the probe-routed climb
doing exactly what the previous comment built it to do. It still never reached the source:
`min_d2source` 1.88 m, `source_is_visible` false at all 153 steps, and the measured RMS
*falling* 0.0407 → 0.0121 over 120 steps. It got quieter while walking cleanly.

The frame was the first suspect and the box **refuted** it — `test_audio_box` pins the
lateral cue agent-frame (ILD +0.081 facing, −0.065 turned) and `test_agent_frame_box`
measures 0.00° of error at all four yaws. So a geometry probe
(`probes/check_episode_geometry.py`) was written instead of a third theory, and it read
**CROSS-FLOOR**: agent start y +2.064, primary anchor and source both y −0.536.

ADR-0010's floor rule measured `max_dy_m` against the **primary anchor only**. The source
sat *at* the goal's level — `|anchor − source|` 0.000, the rule satisfied exactly — and a
full storey below where the agent begins. Legal by the builder's own test, unwinnable in
practice, and a greedy energy climb cannot take stairs. **Fix (`63eda9c`): the rule covers
both anchors.** `t_anom` is why it must be both rather than either — the anomaly fires
mid-episode, so the agent may be on the start's floor or the goal's, and only a source
within reach of both is climbable either way. The side effect is the right one: in a
cross-floor episode the two anchors are further apart than `max_dy_m`, nothing qualifies,
and the episode is **skipped with a reason** instead of running as a silent null.

ADR-0010 predicted the wrong symptom, which is why this survived: it expected walking into
a wall while the energy rose through the ceiling. The real shape has nothing to see.

### Run 3 (`runs/ss2-ep2`) — the anomaly arrived on the last step

The floor rule worked: episode 0 was skipped with the message naming its 2.60 m start-to-
anchor gap, and a same-floor episode ran. Then:

    step 30: onset at RMS 0.250589 (threshold 0.013833)
    step 30: primary goal reached — STOP
    funnel: ONSET_FIRED

The find took 30 steps. `t_anom` was 30. The source started sounding on the last step of
the episode, so there was no search left to interrupt, and completion wins that tie in
SEARCH.

`t_anom = 30` was tagged `fake` with the reason *"low enough that a 500-step episode has
room for the detour and the resume"*. That reasons about the step **budget**; under an
oracle STOP the binding constraint is the **find**, because the episode ends when the agent
reaches its goal. A number chosen against 500 was spent on an episode that lasted 31. This
is the third fake constant this ticket has had falsified by measurement, after
`investigate_max_steps` 40 and the single-anchor floor rule.

**Fix (`f8c6923`): derive it per episode.** `dataset.derive_t_anom` takes the straight-line
xz distance from the start to the nearest primary view point, subtracts the oracle STOP
radius (the part of the route never walked), and divides by the forward stride. Every
approximation leans the same way — a straight line is never longer than the navmesh route,
no step covers more than one stride — so the quotient is a genuine **lower bound on the
earliest step the find can end on**, and half of it puts the onset strictly inside the
search by an argument rather than by a guess about a scene. `T_ANOM_FLOOR_STEPS` (3) wins
when the goal is within arm's reach; that episode is degenerate rather than mis-timed, and
§2.5 says it shows as a funnel stage rather than a screen.

`RunConfig.t_anom` becomes a **pin**: `None` derives, an integer forces (`--t-anom` for an
experiment holding the onset fixed). Two consequences followed and both are improvements:

- The audit records the effective `t_anom` beside `source_xyz`, because `funnel_stage` sits
  in the same record and is *computed* from it. It went on `EpisodeAudit` rather than
  `OnsetRecord` because a structural test caught the second projection — `OnsetRecord` is
  `OnsetState` plus exactly one audit-owned field, and `t_anom` is a property of the episode
  as built, not of the onset as measured.
- Smoke criterion 4 **reconciles** the pin against the record rather than looking one up.
  Where a run pinned a value, the two must agree; where it did not, the record is the only
  place the bound exists. The old shape would have gone quiet on exactly the runs the smoke
  performs.

`FORWARD_STEP_M` and `ARRIVAL_RADIUS_M` are copies — `task/dataset.py` may not import `sim`
and a Mac cannot load it anyway (torch at module scope) — so a test reads both defaults out
of their own source with `ast`. Drift there is silent and one-directional: a longer stride
or a smaller radius puts the onset back outside the find.

### What was deliberately NOT changed

The controller pins **completion over the interrupt** in SEARCH
(`test_completing_takes_priority_over_an_onset`, ticket 23). Flipping it would also have
turned run 3 green — the agent would divert, climb, and resume onto the bed it was standing
beside. It is left alone: the pin is a deliberate ticket-23 decision, the task spec's §4
does not speak to the tie, and changing the agent's semantics to make a gate pass is the
move this ticket keeps catching in other forms. The scheduling fix is the honest one, and
it also handles the case the precedence flip cannot — a find that ends *before* the source
ever sounds.

### State

662 Mac tests (628 → 662), ruff clean, every fix plant-verified red for the right reason.
Criterion 5 has not yet been *tested* — runs 2 and 3 both failed upstream of the climb, on
geometry and on timing. Run 4 is the first honest test of the probe-routed detour.

---

## Resolution — 2026-08-05, `runs/ss2-ep3`

**Criteria 1–8 PASS on the RACE V100. Criterion 9 is NOT_RUN by construction and belongs to
ticket 27**, which is how this ticket defined its own bar. The destination sentence is now a
measured fact: one HM3D episode ran end to end with SoundSpaces 2.0 audio rendered live in
the simulator at every step, the agent ran its find-task, an alarm fired, it investigated,
resumed, and reported.

    1. PASS  audio live and every-step          168 renders / 168 loop steps
    2. PASS  audio context sound                335370 verts (submitted 335362), canary seen
    3. PASS  the IR is real                     shape (2, 13909), peak 0.5949
    4. PASS  provenance did not raise           onset step 4, 4 pre-onset readings, t_anom 4
    5. PASS  the full loop ran                  funnel stage 6 (PRIMARY_RESUMED)
    6. PASS  a report was emitted               all 9 of §5.1's keys
    7. PASS  audio wall-clock inside its ceiling max 0.0653 s, mean 0.0553 s, ceiling 0.5 s
    8. PASS  env_check passed                   5 probe(s), all pass

`onset 4 → INVESTIGATE 4 → RESUME 8 → primary goal 167`, 89 forwards, **0 collisions**,
22.25 m walked.

### The detour reached the right instance

Criterion 5's flag is not enough on its own: arrival in the realizable arm is plateau **plus**
visual confirm, and the confirm is `detector.detects("chair")`, which fires on *any* chair
view point within a metre. §4.2 anticipated this by making "reached the source" a measured
distance, and the measurement settles it: **`dist_at_stop` 0.486 m** against a
`source_separation_m` of 3.216 m, with `min_d2source_m` 0.117 m. It stopped at the source
chair, not a different one.

The honest caveat is the leg's length. The alarm fired at step 4 with the agent already
`start_pose_anomaly_rms` 0.190 from a standing start, so the detour was four steps and about
a metre. The builder decouples the source from the **goal** (≥ 3 m in xz) and says nothing
about the **start**, so a source next to the agent's opening pose is legal. The claim that
the probe-routed detour can route *around* something rests on
`tests/box/test_investigate_route_box.py` — 0.11 m behind a wall in 29 steps, 0 collisions,
against a control arm that walks the old straight line into it — and not on this episode.

### §9's builder numbers, now measured rather than argued

- **`investigate_max_steps`** 40 → 120 held: the detour used 4 of it.
- **Criterion 7's ceiling** 0.5 s is ~7.6x the measured max (0.0653 s) and ~9x the mean
  (0.0553 s) over 168 live renders. Generous on purpose, and now with a number behind the
  word. Live-every-step costs 9.29 s of the episode's 19.48 s wall clock.
- **§3.1's pre-onset tolerance** never fired: the pre-onset spread is 0.001 .. 0.001 over 4
  readings, the bed level exactly, which is ADR-0009's unrendered bed doing what it promised.
- **The calibration** separated by 44.5 dB over 16 poses; `onset_rms` 0.0130 against an
  onset measured at 0.215.
- **`derive_t_anom`'s bound was tight on geometry and loose on the agent.** The episode's
  geodesic `start_end_distance_m` is 3.129 m, so the earliest step the find could have ended
  on was 8 and the derivation chose 4. The find actually ended at **167** — the agent walked
  22.25 m to cover 3.13 m (`soft_spl` 0.103). So the conservatism that made this work is the
  explorer's inefficiency, not route tortuosity, and the bound is doing its job at the
  geometric end where the argument lives.

### One defect found in the artefact, not fixed here

**`source_is_visible` has been false at every step of every box episode** — 31, 153 and now
168 steps, including at 0.117 m from the source and at the 0.486 m STOP. That is not
credible as geometry. The arithmetic in the run's own log points at the cause: the listener
transform sits at y 3.564 while the source sits at y 2.064, **exactly 1.5 m apart**, because
the source is a goal **view point** — a navigable pose on the floor — and the listener is at
sensor height. A ray from eye level to a point lying on the floor plane hits the floor at or
just before its endpoint, so the probe can never return true for a source placed this way.

This is inference from one arithmetic coincidence, not a measurement. It matters because
§3.3 calls the field "the best available diagnostic for why a gradient climb stalled" and
§3.2 records it at every step, so if this is right the artefact carries 168 dead booleans —
the same shape as `source_dy_m` reading 0.000 while the real drop was 2.6 m.

Not fixed here because the obvious fix is **not** a bug fix: raising the source off the floor
would make the ray meaningful *and* change every acoustic number this map has measured, and
a real alarm is not on the floor anyway. That is a task-design decision about §2.1's source
placement, and it goes to the map rather than being done quietly on the day the smoke went
green. The separating measurement is cheap — query `sourceIsVisible()` with the listener
placed at the source position; still false means the probe is inert rather than occluded.

### State

662 Mac tests, ruff clean, `tests/box/` green (43/45 with the frame PINNED agent-frame).
Four defects closed on the box across four runs, each invisible in the artefact it produced:
the funnel over-credit, the detour re-entry, the single-anchor floor rule, and the constant
`t_anom`. Three of the four were fake constants or rules falsified by a measurement.

Hands to **ticket 27**: the same run, green again, with `embodied_memory/` and
`dialogue_memory/` moved out of the repo.
