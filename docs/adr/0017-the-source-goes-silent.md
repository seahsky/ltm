# The source goes silent, because a sound that never stops leaves memory nothing to do

**Status:** accepted (2026-08-20, from the grilling session on the direction change).
**Retires the anomaly-response loop** as the task: the interrupt, the resume, the CLAP normality gate and the room-normal distractor arm all stop being the thing under test.
**The task spec** `docs/anomaly_response_task_spec.md` is superseded on §1, §4.2 and §4.3; §2, §3 and §5 carry almost unchanged.
**ADR-0009, 0010, 0011, 0015 and 0016 all carry.** The renderer, the single positioned source, the unrendered bed, the same-floor builder policy, the realizable climb and the cast are untouched.

One episode is now: a source sounds for a bounded **sounding window**, goes silent, and the agent must reach it.
The primary find-task is the sound source itself.
There is no interrupt, because there is nothing to interrupt.

## What forced it

The direction changed to sound-source finding with a memory stack, and the first question a memory has to answer is what it is for.

Against a source that sounds continuously, the answer is nothing.
`arrive-2` reached 40.3% of 365 episodes and `yield-2` reached 46.0%, both with **no memory in the tree at all** — the live binaural climb plus the cast does that on its own.
A memory arm added to a continuously-sounding task would be measured against a baseline that already has the cue, in a task where the cue is sufficient.
That is not a hard experiment, it is an unmeasurable one: every cell of the intended generalization matrix would be solved the same way, by climbing.

Silence is what creates the question.
After the offset step there is no gradient, so what the agent does next is entirely a function of what it knows — this scene's layout, or what that class of sound usually comes from.
Those are the two memories ADR-0018 builds, and the silent phase is the only place either of them can pay.

## Why the field's convention rather than ours

SAVi (Chen et al., CVPR 2021) and SAVN-CE (CVPR 2026) both draw an onset and a duration per episode; SAVN-CE uses onset uniform on [0, 5] s and duration Gaussian with mean 15 s.
Adopting it costs nothing and buys **SWS**, "success when silent" — the published fraction of episodes completed after the acoustic event ended.
That is a cross-quotable metric measuring exactly what this decision creates, and re-deriving it under a local name would forfeit the comparison for no gain.

## We chose this over

**A continuously sounding source (AudioGoal / AV-Nav).**
Rejected on the argument above: the existing climb already solves it at 40–46% with no memory, so the matrix has no contrast to show.

**Keeping the ObjectNav primary and treating the sound as a cue.**
Rejected because it leaves two success criteria in one episode.
Every delta then splits across a find and a response, which is the ambiguity `CONTEXT.md` already warns about under **Mission** — "mission complete is not a single number" — and it would put the memory effect on the wrong side of it.

## What it costs, stated rather than discovered later

**Built work is retired.** The interrupt, the resume, the once-per-episode detour budget, the CLAP normality gate and the never-built distractor arm stop being the subject.
The controller's climb and cast survive; the state machine around them does not.

**The renderer needs a reverb tail.** `clips.render_through_ir` convolves the whole clip through the current pose's IR every step, which models a source sustaining forever and is fine while `playing` is monotone.
Cut to a window and the silence arrives as a hard step to the bed with no tail, which is unphysical.
SAVN-CE names this as SoundSpaces 2.0's own shortcoming and fixes it with a preallocated accumulation buffer that adds each step's convolution into a running signal.
About 50 lines, pure NumPy, no Habitat coupling, CC-BY-4.0 and portable by hand.
**This is not yet built** and no sounding-window run may report an SWS before it is.

**The cast becomes the default, not the exception.** ADR-0016's cast fires when the rising predicate reads false.
After the offset step the field collapses to the bed, so the predicate reads false at every remaining step by construction.
Everything ADR-0016 measured about plateau windows was measured on a live cue; none of it transfers to a silent one, and the cast's behaviour in silence is unmeasured.

**"Onset" now means two things.** In this tree `onset_step` is the step the AGENT's RMS crossed threshold; in the literature "onset" is the step the source started.
`CONTEXT.md` carries the tree's meaning and says "sounding window opens" for the other.
The collision is recorded rather than resolved because renaming `onset_step` would invalidate every audit record on disk.

---

## Amendment, 2026-08-31: the buffer is built, and the way it was read was wrong

**The decision above stands.** The sounding window, the offset step, the silent phase and SWS are unchanged, and so is every argument for them.
Three of this ADR's own statements are amended here: "this is not yet built", the reverb tail it asked the buffer for, and what `AudioConfig.step_seconds` is.
The repair has its own record, **ADR-0019** (`docs/adr/0019-the-cue-readout-is-one-step-wide.md`), which carries the split and its argument in full; this section says what changed about the claims made above.

### The buffer is in, and the SWS bar moved out of that paragraph and into code

`earshot/audio/tail.py` is the preallocated accumulation buffer this ADR named as missing.
On a sounding step it emits `hop = round(step_seconds * sample_rate)` samples of the LOOPED clip, convolves them through THAT step's IR, and adds the result into a running buffer laid out forward in time.
It is pure NumPy and imports no Habitat, so ADR-0013's edge holds: `audio/` still does not import `sim`.

"No sounding-window run may report an SWS before it is" is no longer a sentence in an ADR, which is a thing a run can ignore.
`silent_phase_tally` raises `TailNotActiveError` on any eligible episode whose record carries no active tail; `SilentPhaseTally.__post_init__` refuses at construction a tally whose `n_tail_active` is short of its own denominator, so no path reaches the `sws` property with the counts wrong; and `tail_verified` is what separates `measured` from `measured_tail_unverified` in the artefact.
A run of episodes without a tail therefore produces no SWS at all: not a 0.0, and not a partial rate over the episodes that did have one.

**The box half is NOT_RUN and is therefore red.** 1334 mac tests pass and `ruff check earshot/` is clean, both re-verified on this Mac, but `bash earshot/tools/box_gate.sh` has not been run on the V100 since the split. By this tree's own convention that is not a green.

### What the buffer got wrong, and the control that settled it

The buffer had ONE readout: the last `N = len(clip)` samples.
`N` is 5 s at the shipped defaults and a step is 1 s, so the number the agent called its instantaneous RMS was a **five-second moving average**, and the decay this ADR asked for above was the analysis window emptying rather than the room.

The control that settles it is an anechoic 1-sample IR, a room with literally no reverberation, on the same clip and the same buffer.
Measured on this Mac at the box's numbers (`N = 220500`, `hop = 44100`, synthetic IR `L = 72300` at RT60 0.8 s, 5 s white noise), post-offset readout as a fraction of the settled level:

```
clip readout, room       0.9000  0.7806  0.6401  0.4599  0.1090   ->  0 at fold 7
clip readout, anechoic   0.8944  0.7735  0.6309  0.4475           ->  0 at fold 5
                delta    0.0056  0.0071  0.0092  0.0124
```

A room with no reverberation reproduces the curve to within 0.56 to 1.24 points.
That is not a badly chosen IR, it is structural: ticket 06 measured `L = 72300` (1.64 s) against a 4.0 s cap while the read window is 5 s, so `L < N` in every configuration this tree can produce and the clip readout's tail can never be reverberation-dominated.

### The fix, and what it cost

One buffer, two readouts (ADR-0019).
The **cue readout** is the last `hop` samples, what arrived during THIS step; it feeds `rms`, `lateral_sign` and the onset detector, and therefore the controller and the calibration.
The **clip readout** is the last `N` samples, unchanged, and feeds CLAP and nothing else.

The same control, on the same buffer and the same folds:

```
cue readout, room        0.2438                                   ->  0 at fold 3
cue readout, anechoic    0.0000                                   ->  0 at fold 1
```

24.4 points apart on the offset step, and `cue_tail_steps` collapses 3 to 1 under the control.
The cue tail is reverberation, and the anechoic control no longer reproduces it.

**A more reverberant scene was rejected as the fix** before the split was chosen: the anechoic control shows the clip tail is the window emptying at any `L`, and `L < N` is a property of the design rather than of the scene.
**Re-deriving the CLAP domain was rejected as out of scope**: ADR-0018's bank of record was measured on clip-length waveforms, so `heard_clip_window` hands CLAP exactly what the old readout returned, still deferred until the clip window is full.

What it costs, stated rather than discovered later:

**Every SWS number produced between this ADR and the split is on the old readout and is not comparable to one produced after it.**
`post_offset_audible_steps`, `n_tail_audible` and `heard_within_window` all get stricter and their values fall.
Hearing the source on its tail was a `clip_tail_steps` opportunity and is now a `cue_tail_steps` one: 7 steps to 3 at the box's numbers, 6 to 2 at the runner fixture (confirmed against the shipped `TailState`).
The fall is the correction, not a regression, and `SoundingWindowRecord.cue_tail_steps` being present is the reliable marker of which domain a record's trace is in.

**`StepRecord.measured_rms` changes domain under an unchanged name.** It is the one field the split could not rename, for the same reason `onset_step` is not renamed above: every audit record on disk carries the old meaning.
`tools/detour_report.py`'s band slopes and residuals are suspended until re-run, and a run mixing pre- and post-split episodes must never be pooled.

### What the fix did NOT fix

**The rising blocked pairs are the accumulator's, not the readout's.**
Re-measured on `test_task_runner`'s wall fixture with three arms in one run: the pre-ADR-0017 whole-clip renderer, the clip readout (what this ADR shipped), and the cue readout.

```
                                pre-ADR-0017   clip readout   cue readout
blocked forwards, whole episode          27             22            26
first turn at step                       13             17             5
blocked forwards before that turn         9             13             1
rising blocked pairs                      0             14            14
adjacent blocked pairs                   23             19            21
held-pose reading spread / max        0.000         0.0038         0.548
first nonzero lateral_sign at step       14             18             6
collided / forward                    0.871          0.846         0.867
```

The four-step stall lag this ADR introduced does not shrink under the split, it inverts: the cue arm turns at step 5, eight steps before the pre-ADR-0017 renderer and twelve before the clip readout.
The 14 rising blocked pairs do not move at all. Whatever the accumulator does to `is_rising` against a wall, it survives the readout change, and the arm that shows that is the middle column.

**`climb_eps` is still the wrong size, in the other direction, and now for a different reason.**
This ADR's estimator was 1.91x to 3.55x too large because it sampled independent single renders against a loop that averaged correlated ones.
Independently re-measured at the box's numbers, `single/cue` comes out **0.44** and **0.94** on white noise and **0.002 / 0.009** on a 0.6 s transient, against `single/clip` of 3.00 to 3.14: the cue arm sits AT or ABOVE the pre-ADR-0017 arm, so the averaging model behind the epsilon is wrong for a second time.
The mechanism is that the cue arm carries the loop's phase and the clip arm cannot see it: with a perfectly deterministic renderer the clip arm's SD/level is ~1e-15 while the cue arm's is 3.11e-03 for white noise and **2.014** for a 0.6 s transient.
For a bursty clip, `climb_eps(cue_render_scatter)` hands `is_rising` a floor of about twice the level itself and the climb would never fire.
Nothing is gated on it here: `cue_phase_crest` and `cue_phase_min_ratio` ride on the calibration record so the case is identifiable, and correcting it would be a policy change riding in an audio commit.

**The lateral cue changed character and nothing compensates.** The first nonzero `lateral_sign` moves from step 14 (pre-ADR-0017) and 18 (clip) to 6 (cue), and the cue is `LATERAL_AMBIGUOUS` on a quiet fold because the bed is diotic. Unmeasured beyond the table above.

**`metrics["start_pose_audible"]` remains cross-domain.** It compares a bare whole-clip render against an accumulator-derived `onset_rms`. The split does not make it worse, because `onset_rms` does not move, and it does not fix it either.

### `AudioConfig.step_seconds` was invented here, and it is `provenance: fake`

Nothing in this tree mapped a simulator step to a span of seconds before this ADR.
`AgentSpec` is 0.25 m and 30 deg per step; the accumulation buffer cannot exist without a duration, because each step's convolution has to be written at an offset and that offset IS this number.

1.0 s, chosen and not derived.
Deriving it from 0.25 m at a walking pace would imply a measurement of a speed nobody took and would read as `provenance: source`.
Round, because it is also the cross-quote: 500 steps is 500 s, so SAVN-CE's 15 s mean duration reads as 15 steps with no arithmetic.
**Rejected: 0.25 s** (0.25 m at ~1 m/s), because it makes the clip ramp four times as long and a ramp that wide eats any candidate window. That rejection was made against the clip ramp and it survives the split for a narrower reason: the clip ramp still bounds the CLAP deferral.

**The split raises this number's stakes rather than settling them.** `hop` IS the cue window, so every quantity the controller now reads is a function of `step_seconds`: `cue_tail_steps = ceil((hop + L - 1)/hop)`, `phase_folds = N // gcd(N, hop)`, and the bed cross-term, which scales as `1/sqrt(hop)` and is therefore `sqrt(N/hop) = sqrt(5)` larger than the clip readout's at the defaults.
Before the split, `step_seconds` set a ramp and a tail; after it, `step_seconds` is the sensor's width.
It is still `provenance: fake`, and the ramp and tail numbers in its comment block are the CLIP readout's (5 and 7 at the box's numbers) rather than the cue's (1 and 3).

### Still open: how long the window should be

Unchanged by the split and still undecided.
The default is `WindowPolicy.FIXED_STEPS` at 60 steps and **60 is `provenance: fake`** with no sweep behind it.
The mechanism keeps all four policies reachable (`CONTINUOUS`, `FIXED_STEPS`, `BUDGET_FRACTION`, `DRAWN`), and all three bounded defaults resolve to the same 60 steps at `max_steps = 500` on purpose, so the first policy comparison is not confounded by a level change riding along with a variance change.

The evidence that answers it is what the first sweep records: the `onset_step - t_anom` distribution and the `heard_within_window` rate, both on the metrics bag and therefore in `audit.json`.
**The split changes what both of those numbers mean**, so the sweep has to be run after it, and a sweep run before it cannot be pooled with one run after.
`onset_delay_steps` no longer carries the 0-to-4 step upward bias the clip readout's five-fold fill imposed, because `CUE_RAMP_STEPS` is 1 by construction; and `heard_within_window` reads the shorter tail described above.

### Still open: the controller's stall lag, recorded rather than resolved

The table above is the record, not the resolution.
Compensating for the stall in the controller now would put a policy change inside an audio commit, and this repo's rule is that a claim that X broke because of a change needs the arm where the change is absent.

Two arms are needed and neither has been run on the box.
The three-arm wall fixture is the arm where the READOUT is absent, and it is in-process and cheap.
**`WindowPolicy.CONTINUOUS` is the arm where the WINDOW is absent**, and it is the one that separates an audio change from a policy change on a real run: it removes the offset step while leaving the accumulating renderer in place, so a funnel delta measured against it is attributable to the window rather than to the renderer and the readout together.
It is kept as the named control arm for exactly this, and until it is run, a controller change and a readout change are one delta.

One hazard is new and is recorded rather than fixed.
The cue reading cycles with the loop: at a held pose its spread is 54.8% of its maximum against the clip readout's 0.38%.
At the shipped defaults `RISING_WINDOW == phase_folds == 5`, so each of `is_rising`'s two averaging windows spans exactly one loop period and the phase cancels in the mean **by arithmetic coincidence**.
Change `step_seconds`, or use a clip whose length is not a whole number of hops, and the windows beat against the period and the climb can read a phase artefact as a gradient.
The bound reaches the record as `metrics["sounding_phase_folds"]`.
