# Anomaly-response task spec, on live in-sim audio

Decided 2026-08-04 in the grilling session on ticket 09 of the `ss2-clean-room` map.
This is the task specification for the clean-room tree: what the robot is asked to do, what it hears, what it may use to decide, what it emits, and what counts as the smoke being green.

It is the source of truth for the task.
The agent architecture is ADR-0008.
The reset order is ticket 10.
The glossary is `CONTEXT.md`, and its language is binding here.

Four ADRs were written alongside this spec and supersede the four that encoded grid-era workarounds:

| New | Supersedes | Subject |
|---|---|---|
| ADR-0009 | ADR-0004 | One positioned source, unrendered diotic bed, no habitat-sim fork |
| ADR-0010 | ADR-0003 | Same-floor source as builder policy, not an audio guard |
| ADR-0011 | ADR-0001 | Realizable localization: agent-frame lateral sign, measured ceiling |
| ADR-0012 | ADR-0002 | Captioner-provided room label, CLIP stays dropped |

ADRs 0005 to 0008 are untouched.
ADR-0008's four deferrals to ticket 09 are all discharged here, CLIP included, so nothing in it reopens.

## 1. The task

One episode is one mission: the **primary find-task** plus the **anomaly response**.

The agent is given an ObjectNav goal category at episode start and searches for it.
At step `t_anom` an anomaly source begins playing.
When the agent's own measured loudness crosses the onset threshold it interrupts the search, investigates the source, checks what is there, resumes the primary find-task, and emits a report.

Nothing about the task changed with the renderer.
What changed is that several of its constraints were workarounds for a precomputed RIR grid, and those are re-derived below.

## 2. Audio

### 2.1 Sources

**Exactly one positioned source per episode: the anomaly** (ADR-0009).
habitat-sim is built stock at the SHA ticket 17 pins.
No multi-source patch, no fork.

The **background bed** is a fixed-level diotic signal generated directly and mixed after rendering.
It never touches the RIR, so it is position-invariant by construction.
This is what `CONTEXT.md` has always defined it as, and it is what the grid-era implementation was not.

The **room-normal distractor** is an across-episode arm, exactly as ADR-0002 specified: the same sound placed where it is room-normal in one episode and room-anomalous in another.
It is never simultaneous with the anomaly, so it needs no second source.

### 2.2 Source placement

The anomaly source sits on the primary goal's floor, `|Δy| < ~1.0 m`, checked in the dataset builder (ADR-0010).
This is task-scope policy, not a fabrication guard: there is no grid, no `nearest`, and no silent snapping to defend against.
There is no runtime guard, and none is needed.

The xz separation rule that decouples the source from the primary goal carries unchanged.

### 2.3 Levels and the calibration gate

The bed level is a **chosen constant**, not a calibrated one.
It is our signal, unrendered and position-invariant, so there is nothing to calibrate it against.
`bg_gain` is retired outright.

`onset_rms` is **derived from measurement**:

1. Render the anomaly at a spread of poses across the intended audible band.
2. Take the bed level `B` and the distribution of anomaly RMS over those poses.
3. Set `onset_rms` strictly between `B` and the low percentile of the anomaly distribution.
4. Report the separation between the two distributions as the gate number, in the pattern of ticket 13's EER and the CapRL separation gate.

If the distributions overlap the gate **fails**, and the correction is `globalVolume`, never a hand-nudged threshold.
`globalVolume` is measured at 1.0 on our branch (ticket 04), not the 0.25 ticket 01 quoted from `main`.

The old `onset_rms` of 0.065 is meaningless under this renderer and is not carried.

### 2.4 Why an absolute threshold is sound now

ADR-0004 established that no absolute RMS threshold could work, because a 1.4x temporal step sat inside an 8x spatial swing.
That argument depended on the bed being rendered.
With an unrendered bed, the pre-onset signal is flat at `B` at every pose in every scene, and the spatial swing lives entirely in the post-onset term.
The temporal step detector ADR-0004 called the technically correct fix stays unbuilt, because it is no longer needed.

### 2.5 Audibility is not screened at build time

Onset fires when the agent's own live signal crosses the threshold.
The grid-era `AUDIBLE` / `OUT_OF_COVERAGE` cell check has nothing to check, and pre-screening would reintroduce offline rendering by the back door.

The consequence is deliberate and is carried in the metrics rather than hidden: `t_anom` is when the source starts **playing**, not when the agent hears it, so an agent that never gets close enough produces an episode with no onset.
That attrition is visible as a stage in the funnel (section 6).

The smoke is the one exception.
It verifies audibility at its own start pose once, with a calibration render, so it is deterministic.

## 3. Onset and provenance

### 3.1 The invariant

`onset_step < t_anom` is **structurally impossible** under ADR-0009: the bed is below threshold and position-invariant, the anomaly contributes exactly zero before `t_anom`, and there is no second source.
This is the `anommxv` break, closed by construction rather than by a heuristic.

Because it cannot fail for the reason it was invented to catch, the check changes job.
It becomes an **asserted invariant** that **raises**, not a diagnostic that is read afterwards:

- pre-onset measured RMS equals the bed level within tolerance
- `onset_step >= t_anom`

A violation means the bed level drifted or the source started early.
Both are silent-fabrication bugs of exactly the kind this map keeps finding, so they stop the run rather than being logged.
This is ticket 12's discipline applied to the signal instead of the mesh.

### 3.2 The per-step record

Recorded at **every** step, not windowed around the onset:

- measured RMS
- lateral sign
- whether the source is playing
- `sourceIsVisible()`
- the action taken

The action is there so that a rotation-driven rise in RMS is distinguishable from a translation-driven one after the fact (ADR-0011).

### 3.3 `sourceIsVisible()` is analyst-only

It is free (one `RLRA_TraceRayAnyHit`), it is the primitive the deferred non-LOS-but-audible seed design needs, and it is the best available diagnostic for why a gradient climb stalled.

It is computed from the ground-truth source position.
**The controller must never read it.**
Feeding it to the decision rule would plant a hidden oracle inside the arm ADR-0001 built specifically to avoid one.

## 4. The anomaly response loop

### 4.1 Localization

The realizable arm is the default and is what the smoke runs (ADR-0011).
The agent reaches the source from live binaural RMS, the inter-aural level sign, and visual confirmation, never from an oracle coordinate.

The greedy rule in `realizable_investigate_step` is unchanged: rising loudness means forward, peak-or-plateau plus visual confirm means STOP, a stall turns toward the louder half-plane.

Two things around it change.

**The lateral sign is agent-frame and must be pinned by a test.**
The grid rendered at identity listener yaw, so the cue was world-frame and the fusion arc compensated with `heard == -right(world-bearing)`.
Live rendering uses the agent's real transform, so the same function returns an agent-frame cue.
The code does not change and the convention inverts underneath it.
Carried across with the old compensation, the controller turns the wrong way on every stall, and it looks like a mediocre climb rather than a bug.

**The rotation-versus-translation conflation is instrumented, not fixed.**
`turn_left` now changes RMS without changing distance.
Section 3.2's record makes this decidable on data later.

### 4.2 Arrival and resume

Arrival is agent-estimable: peak-or-plateau plus visual confirm.
The oracle `investigate_arrive_radius_m` of 1.5 m is not an arrival criterion in this arm and survives only in the oracle arm.

"Reached the source" stops being a defined radius and becomes a **measured distance-at-STOP**, reported as a distribution (section 6).
ADR-0001's asserted ~1 m ceiling was a grid-resolution artefact and is retired.

Resume is unchanged: restore primary state, force a re-query, return to SEARCH.

### 4.3 Classification and normality

CLAP performs open-set normal-versus-anomaly classification on the heard clip.
Whether the sound is anomalous is conditioned on the room it is heard in (ADR-0012), and the room label comes from the Qwen2-VL-2B captioner the tree already carries, behind a provider seam.
CLIP does not return.

The smoke does not exercise this: there is one sound and it is the anomaly by construction.
It earns its place in R2's distractor arm, which is out of scope for the `ss2-clean-room` map.
ADR-0002's $0 room-classifier accuracy gate carries across the substitution and must clear before anything depends on the label.

## 5. Outputs

Two artefacts, split on whether the information is agent-estimable.

### 5.1 The agent's report

The agent's testimony.
Constructible from agent-estimable signals **only**, with an **identical schema in both localization arms**:

- `primary_completed`
- `heard_at_step` (the onset step)
- `room` (from the captioner provider)
- `anomaly_class` (from CLAP)
- `stopped_at_pose` (the agent's own pose at STOP)
- `visual_confirm_object` (from the detector, or absent)
- `investigate_aborted`
- `resumed`
- `n_benign_ignored`

`source_xyz` leaves the report.
The oracle arm's privilege shows in its trajectory and its audit record, never in its testimony.

This directly answers the "the sound is just a stopwatch, the coordinate is handed to the agent" objection: if a report cannot be constructed without ground truth then the arm is not realizable, and that is now checkable by reading the schema.

The report stays **structured, with no generated text**.
ADR-0008 dropped the LLM, and nothing in the report was ever prose.

### 5.2 The episode audit record

Everything privileged or diagnostic:

- ground-truth source position
- distance-at-STOP
- the `sourceIsVisible()` history
- section 3's provenance assertions and the measured pre-onset bed RMS
- the calibration separation margin and the threshold in force
- the funnel stage this episode reached
- per-step audio render wall-clock

It lands in the same place as `AudioContextReport` (map requirement 1c) and `env_report.json` (map requirement 8).
Those three should be one location, not three.

## 6. Metrics

Restated against `CONTEXT.md`.

**Find-SR** carries unchanged: 1.0 m primary, 0.1 m as the localization-bound diagnostic.
`success_1m` stays a diagnostic and is never a success number.

**Anomaly-response SR is replaced by a staged funnel**, with a count at each stage and no single headline fraction:

1. episodes run
2. `t_anom` reached
3. onset fired
4. investigate entered
5. source reached (CHECK)
6. primary resumed

The denominator for the loop is stage 2.
Section 2.5's audibility attrition is visible at stage 3 rather than absorbed into an aggregate.
A single fraction would silently mix "never heard it" with "heard it and failed to reach it", and this project's record has more than one case where an aggregate hid the mechanism.

**soft-SPL is computed but not headlined.**
`CONTEXT.md` defines it as the primary science metric for the memory delta, and memory is out of this build by the map's chart-time decision, so nothing here consumes it.
It is computed anyway, because `metrics.py` carries verbatim and the follow-on memory effort inherits it already wired.

**Benchmark SPL is computed and never cross-quoted from this map.**
ADR-0005 makes native `spl` at 0.1 m the cross-quotable number, ADR-0006 turned absolute numbers off as the lead, and ADR-0008 established that an oracle STOP deletes the stop_miss half of the 0.031 decomposition.
Quoting anything this map produces against VLFM's 0.304 would overstate.

**Two metrics are new:**

- **distance-at-STOP** to the anomaly source, as a distribution, replacing ADR-0001's asserted ceiling
- **per-step audio render wall-clock**, reported every run

The second exists because ticket 06's 27.2 ms at the `cheap_preset` is the measurement the whole feasibility claim rests on, and it should be auditable on every run rather than trusted from one sweep.

**Cost** stays steps-to-complete plus wall-clock, as `CONTEXT.md` defines it.

## 7. The carry line

Ticket 10 left the audio module carry line to ticket 09.
It is not file-shaped.
**Neither `audio.py` nor `audio_task.py` carries as a file**, in the same way ticket 07 rewrote the frontier proposer from 1129 LOC to roughly 300 rather than porting it with dead branches attached.

**Carries, re-homed into a new audio module:**

- the CLAP open-set normal-versus-anomaly gate, its prompt banks and its calibration (calibration gate ran GO at perfect separation, EER 0.00)
- `resolve_anomaly_clip` and the ESC-50 fetch
- `normalize_clip`
- `lateral_sign`, with section 4.1's frame convention pinned by a test
- `ROOM_PRIOR`, which ADR-0012 keeps alive

**Dies with the grid:**

- `render_step_audio`, the `fftconvolve` lookup, `cached_source`, `RIRGrid` resolution
- `diotic_collapse`, because the bed is generated diotic at source rather than collapsed from a render

**Dies on the record, not on judgement:**

- `should_audio_stop`, because the audio-energy STOP arc is closed as ground-truth-privileged and capped at roughly 1 m by grid resolution
- `estimate_doa`, because `CONTEXT.md` says to avoid DOA framing on the grounds that the near-zero time-difference cue is not what drives this, and its ITD branch is engine-weak
- `audio_target_for_retrieval` and `gate_retrieval_target`, because they are memory-facing and memory is out of this build

**Rewritten rather than ported:**

- the onset detector, which reduces to a one-shot threshold on live RMS
- its calibration, whose shape section 2.3 changed entirely
- the per-step orchestration in `process_audio_step`, which now drives a live render instead of a grid lookup

**Does not carry:** `perception.py` (506 LOC), per ADR-0012.

## 8. Smoke-green acceptance criteria

The gate ticket 10's irreversible deletion commit hangs off.
All of the following, as checkable assertions:

1. **Audio is live and every-step.** Render count equals step count exactly.
2. **The audio context is sound.** Ticket 12's guard armed and green through ticket 16's verified invariants: mesh vertex floor cleared, no swallowed spec keys, canary armed on every render.
3. **The IR is real.** Non-silent, scene-dependent, trimmed to actual decay rather than fixed-width.
4. **Provenance did not raise** (section 3.1).
5. **The full loop ran.** SEARCH, onset, INVESTIGATE, CHECK, RESUME, then a legitimate termination. CHECK and RESUME must both be reached.
6. **A report was emitted** with section 5.1's schema fully populated.
7. **Per-step audio wall-clock recorded and inside a stated ceiling.** Set generously, not at ticket 06's 27.2 ms: ticket 06 measured 2.3x pose variance against ticket 04 on the same scene, so a tight bound would fail for a reason that is not a regression.
8. **`env_check.py` passed** (ticket 17).
9. **Hermeticity.** The same run, green again, with both old trees moved out of the repo (ticket 10).

**The primary find-task is not required to succeed.**
The destination says the agent runs its primary find-task, not that it completes it.
Requiring success would gate an irreversible deletion on a backbone measured at 0.031 benchmark SPL with roughly 45% explore-timeout, which is the capability ADR-0006 retreated from claiming.
What is required is that the primary loop runs and terminates legitimately.

**The smoke runs the realizable arm.**
An oracle-localization smoke would leave the entire live-audio path unexercised in the one episode that exists to prove it.
The oracle arm is retained as a bisection tool: if the smoke fails, running it isolates audio from controller in one step.

### Required disclosure

The smoke runs an **oracle STOP**, so it does not exercise goal detection at all.
`diagnose_spin` decomposed the 0.031 benchmark SPL as stop_miss around 50%, explore_timeout around 45%, success around 5%.
An oracle STOP deletes the stop_miss half outright.
**Smoke find numbers will look far better than 0.031 for a reason that must be disclosed rather than enjoyed, and they are not capability numbers.**

## 9. Left to the builder

Deliberately not decided here, because they are numbers to set against measurement rather than decisions:

- `investigate_max_steps`, the detour sub-budget, currently 40
- the bed level constant, and the audible band the calibration in section 2.3 sweeps
- the wall-clock ceiling in criterion 7
- the tolerance on the pre-onset RMS assertion in section 3.1
