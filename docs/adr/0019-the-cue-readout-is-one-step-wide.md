# The cue readout is one step wide, because a five-second moving average is not a reverb tail

**Status:** accepted (2026-08-31).
**Amends ADR-0017.** The accumulation buffer, the sounding window, the offset step and SWS all carry unchanged.
What changes is how the buffer is READ: one buffer, two readouts, and the agent reads the one that is a step wide.
**Amends what ADR-0017's own numbers mean.** `SoundingWindowRecord.tail_steps` and `ramp_steps` keep their names and their arithmetic and change role; every `StepRecord.measured_rms` written after this is a different domain from every one written before it.
**ADR-0018 is untouched by construction.** CLAP still receives the clip-length readout, still deferred until the clip window is full.

## The defect

ADR-0017's buffer had one readout: the last `N = len(clip)` samples.
`N` is 5 s at the shipped defaults and a step is 1 s, so the number the agent called its instantaneous RMS was a **five-second moving average**, and its decay after the offset step was the analysis window emptying rather than the room.

The control that settles it is an anechoic 1-sample IR -- a room with literally no reverberation -- on the same clip and the same window.
Measured on this Mac at the box's numbers (`N = 220500`, `hop = 44100`, synthetic IR `L = 72300` at RT60 0.8 s, 5 s white-noise clip), post-offset readout over the settled level:

```
clip readout, room       0.9000  0.7806  0.6401  0.4599  0.1090  ->  0 at fold 7
clip readout, anechoic   0.8944  0.7735  0.6309  0.4475          ->  0 at fold 5
```

The room buys **0.56 to 1.24 points** over the first four steps.
A room with no reverberation reproduces the curve the tree was calling a reverb tail.

This is structural rather than a badly chosen IR.
Ticket 06 measured `L = 72300` (1.64 s) against a 4.0 s cap while the read window is 5 s, so `L < N` in every configuration this tree can produce, and the clip tail can never be reverberation-dominated.

Two further consequences, both measured before the change:

* **The controller read pose HISTORY.** A reading was a function of the last `clip_tail_steps` = 7 poses. On `test_task_runner`'s wall fixture the accumulator took 13 blocked forwards before its first turn against the pre-ADR-0017 renderer's 9, and produced 14 rising blocked pairs where the pre-ADR-0017 arm produces 0.
* **`climb_eps` was measuring the wrong quantity.** The estimator sampled independent single renders (SD 3.490e-04) while the loop averages `clip_tail_steps` correlated ones (SD 1.830e-04, lag-1 autocorrelation 0.804 against 0.022) -- a factor of 1.91, and 3.55 under a second noise model.

## The decision

One buffer, two readouts.

```
clip readout   buffer[:, :N]              (2, N)     unchanged, CLAP only
cue  readout   buffer[:, N - hop : N]     (2, hop)   what arrived DURING this step
```

The **cue readout** feeds `rms`, `lateral_sign` and the onset detector, and therefore the controller and the calibration.
The **clip readout** feeds CLAP and nothing else.

The cue readout at step *j* holds exactly the samples that arrived during step *j*, so its tail is driven by the IR width `L` rather than by `N`.
Measured, same buffer, same folds:

```
cue readout, room        0.2438                     ->  0 at fold 3
cue readout, anechoic    0.0000                     ->  0 at fold 1
```

24.4 points apart on the offset step, and `cue_tail_steps` collapses 3 -> 1 under the control.
**The cue tail is reverberation and the anechoic control no longer reproduces it.**

The three consequences, after:

* the fill ramp is gone -- `CUE_RAMP_STEPS` is 1 by construction, so `onset_delay_steps` carries none of the 0-to-4 step upward bias the clip readout's `ceil(N/hop)` = 5 fold fill imposed;
* a reading is a function of the last `cue_tail_steps` poses -- 3 at the box, 2 at the runner fixture -- rather than of the last 7;
* `climb_eps` is measured in the domain `is_rising` compares.

## Decision 1 -- two beds, never one sliced

The split gives the runner two signal lengths, so it builds two beds: `bed_signal(hop, bed_rms)` for `heard_step` and `bed_signal(len(clip), bed_rms)` for `heard_clip_window`, each RMS-normalised at its own length.

Slicing the last `hop` samples off the clip-length bed was rejected on arithmetic.
A slice of `n` Gaussian samples carries a relative RMS error of about `1/sqrt(2n)`, and `n` is `hop`, which is a free parameter (`step_seconds` x `sample_rate`).
Measured against the fixed `BED_SEED`: the worst disjoint hop-slice deviates **0.3107%** at the shipped hop of 44100, **6.7906%** at the runner fixture's hop of 441, and **17.7320%** at the tail fixture's hop of 100.
`AudioConfig.pre_onset_rms_tol` is 0.05, so a slice raises `ProvenanceError` outright at two configurations this tree ships tests at and spends 78% of the tolerance at a third -- and the cost scales the wrong way, because the smaller the step the worse it gets.
`bed.py` promises that normalising after generation is what makes that tolerance a bound on DRIFT rather than a slack budget for sampling noise; slicing spends exactly what it promised not to.

Cost: one extra `2 x hop` float32 buffer per episode, and a stated non-property -- the two beds share `BED_SEED` but are each scaled by their own RMS, so they are **not sample-aligned** and nothing may compare their samples. Nothing does.

## Decision 2 -- the calibration aggregates over the loop's phases, quadratically

`sweep_cue_rms` measures the quadratic mean of the cue RMS over all `phase_folds = N // gcd(N, hop)` loop phases at each pose.

The decisive argument is an exact identity:

```
sqrt( mean_over_phase_folds( rms(cue_j)^2 ) )  ==  rms(steady_state_render(ir, clip, hop))
```

The `phase_folds` cue windows are disjoint, consecutive, and tile the settled period an integer number of times.
Measured at ratio **1.000000000000** in five configurations: the tail fixture (800/100/512), the runner fixture at both IR widths (2205/441/64 and 2205/441/900), and the box's numbers (220500/44100/72300) on both a white-noise clip and a 0.6 s transient.

So **`onset_rms` does not move**, and the whole change is reviewable as one number that must not move beside several that must.

Rejected: the MAXIMUM over phases, which raises the threshold by the crest factor (2.2361 for a 0.6 s transient on a 5 s loop, measured) in a clip-dependent way, makes every historic threshold unpriceable, and suppresses detection at every pose but the loudest fold.
Rejected: the MINIMUM, which puts a bursty clip's low percentile at the bed and fails the 6 dB gate for a source that is plainly audible.
Rejected: one arbitrary phase, which is the defect this ADR names.

The honest cost is recorded rather than gated: `cue_phase_crest` and `cue_phase_min_ratio` ride on the calibration record, and the 6 dB gate is deliberately NOT tightened here.
Tightening it would be a second change, and it would make four of ESC-50's five classes unusable -- the argument `tail_is_active`'s docstring already makes about the same clips.

## The naming, and what stays still

| thing | before | after | why |
|---|---|---|---|
| readout, clip width | `tail_readout` | `clip_readout` | "the tail" now means two lengths; no alias, deliberately |
| readout, cue width | -- | `cue_readout` | (2, hop), a copy and never a view |
| per-step composition | `heard_step(..., bed=)` | `heard_step(..., bed_cue=)` | the rename is what makes `mix_bed`'s shape refusal fire |
| CLAP composition | (`heard_step`'s return) | `heard_clip_window(state, bed_clip=)` | called once per episode, not once per step |
| state, clip tail | `TailState.tail_steps` | `TailState.clip_tail_steps` | a bare `tail_steps` is the symbol a future edit picks up by mistake |
| state, cue tail | -- | `TailState.cue_tail_steps` | `ceil((hop + L - 1)/hop)` |
| state, clip fill | `TailState.source_fill` | `TailState.clip_source_fill` | unprefixed beside prefixed neighbours is where a reader takes the wrong window |
| state, clip ramp | (runner's inline `ceil`) | `TailState.clip_ramp_steps` | one definition of the ramp, which the runner's own comment demanded |
| cue ramp | -- | `CUE_RAMP_STEPS = 1` | a constant, not a property and not a record field |
| calibration scatter | `render_scatter` | `cue_render_scatter` + `clip_render_scatter` + `single_render_scatter` | renamed rather than redefined -- see below |
| **record** `analysis_window_samples` | -- | unchanged | the buffer's read window; NOT what the controller reads |
| **record** `tail_steps` | -- | name and arithmetic unchanged, ROLE changed | the CLIP readout emptying; bounds CLAP, keeps its `tail_is_active` clause |
| **record** `ramp_steps` | -- | name and arithmetic unchanged, ROLE changed | the CLIP fill; bounds the CLAP deferral, no longer corrects `onset_delay_steps` |
| **record** `cue_tail_steps` | -- | NEW, `Optional[int]` | criterion 4's fence post; the first field that is evidence the acoustics did work |

Nothing is renamed on disk, so no `audit.json` is reinterpreted.
The asymmetry between the state's `clip_tail_steps` and the record's `tail_steps` is real, and the one assignment that crosses it carries a WHY comment.

`render_scatter` is renamed rather than redefined because its written definition -- "the spread of the reading the climb compares" -- stayed true across the change while the reading itself changed length.
That is exactly what would have let the domain move under a stable name.
The legacy key is ambiguous across two eras and its disambiguator is `single_render_scatter`'s presence: with it, the record is post-ADR-0017 and its `render_scatter` is the clip-loop number; without it, the record is pre-ADR-0017 and it is the whole-clip number.

## What this deliberately does NOT decide

**The CLAP domain.**
ADR-0018's bank of record -- anchor recall 0.911 / 0.895 over two 27-minute box runs -- was measured on clip-length waveforms.
Re-deriving it is not in scope, so `heard_clip_window` hands CLAP exactly what `heard_step` used to return, still deferred until `clip_source_fill >= 1.0`.

**Whether the controller should compensate for the loop's phase.**
The cue reading cycles with `phase_folds`, and at the shipped defaults `RISING_WINDOW == phase_folds == 5`, so each of `is_rising`'s two averaging windows spans exactly one loop period and the phase cancels in the mean **by arithmetic coincidence**.
At any other `step_seconds`, or any clip length that is not a whole number of hops, the windows beat against the period and the climb can read a phase artefact as a gradient.
Nothing here compensates: an audio change and a policy change in one commit is a confound.
The bound reaches the artefact as `metrics["sounding_phase_folds"]`, and the arm that settles it is a paired sweep.

## What it costs, stated rather than discovered later

**Every SWS number produced between ADR-0017 and this change is on the old readout and is not comparable to one produced after it.**
`post_offset_audible_steps`, `n_tail_audible` and `heard_within_window` all get stricter, and their values fall.
Measured on the mac fixtures: the anechoic delta fake's `post_offset_audible_steps` goes to 0 -- an honest hard cut, correctly reported -- and an episode whose agent was 0.25 m outside the arrival ring at the offset step no longer closes it, where under the clip readout it did.
What carried it was the analysis window emptying.

**`StepRecord.measured_rms` changes domain under an unchanged name**, and it is the one field this change could not rename.
`SoundingWindowRecord.cue_tail_steps` being present is the reliable marker of which domain a record's trace is in.
`tools/detour_report.py`'s band slopes and residuals should be treated as suspended until re-run, and a run mixing pre- and post-split episodes must never be pooled.

**The lateral cue changes character and nothing compensates.**
`lateral_sign` now reads (2, hop) instead of (2, N): this step's arrival rather than a five-pose average stale by up to four steps.
Measured on the wall fixture, the first nonzero `lateral_sign` moves from step 18 (clip) to step 6 (cue).
It is `LATERAL_AMBIGUOUS` on a quiet fold, because the bed is diotic and contributes exactly zero ILD, and `_turn_toward` scans left on ambiguous.

**The cross-term with the bed grew by sqrt(5).**
The sweep measures the source alone while the runner reads `mix_bed(readout, bed)`; that term scales as `1/sqrt(n)` in the window length, so at `hop` instead of `N` it is `sqrt(N/hop)` larger. Unmeasured after the split; the box prints it.

**The pre-registered scatter ordering is already doubted.**
`single_render > cue_render > clip_render` was the prediction, on the model that the cue readout averages `cue_tail_steps` renders where the clip readout averages `clip_tail_steps`.
Measured on this Mac at the box's numbers, it does not hold: 100.0% of a 72300-sample RT60-0.8 s IR's energy sits inside the first hop, so the cue window holds essentially ONE render and averages nothing.
Worse, the cue arm carries the loop phase, which the clip arm cannot see: with a **perfectly deterministic** renderer the clip arm's SD/level is ~1e-16 while the cue arm's is 3.1e-03 for 5 s of white noise and **2.01** for a 0.6 s transient.
For a bursty clip, `climb_eps(cue_render_scatter)` would hand `is_rising` a floor of about twice the level itself, and the climb would never fire.
That is not corrected here -- it is measured, recorded as `cue_phase_crest` / `cue_phase_min_ratio`, and it is the single most valuable number the first box run after this change can produce.

## Scope this makes newly possible and which is NOT taken

With `cue_tail_steps` on the record, `tail_is_active` could gain a fifth clause requiring `cue_tail_steps > 1` -- refusing an SWS on a scene whose room does not outlive one step.
Not here.
It would be a new refusal in a commit that already moves a domain, and it would silently disqualify scenes rather than measure them.
The honest first move is a sweep reporting the distribution of `cue_tail_steps` across scenes.
