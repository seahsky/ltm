# LTM-Embodied Agent — Anomaly-Response Evaluation

Glossary for the audio-cued anomaly-response experiment (ICRA-2027 arc).
The task: an agent runs a primary find-task; an anomaly sound interrupts; it investigates the source, resumes, and reports.
This file is a glossary, not a spec — implementation lives in `earshot/`.

## Language

### Task and mission

**Primary find-task**:
The ObjectNav goal the agent is given at episode start (reach a target object category).
Drives the SPL/success ground truth.
_Avoid_: main goal, objective.

**Anomaly response**:
The interrupt behavior — hear an abnormal sound, go to its source, check it, then resume the primary find-task.
_Avoid_: audio goal, sound goal (retired framing where sound == target).

**Abandoned investigation**:
An anomaly response where the onset fired and the detour was entered, but the agent resumed the primary find-task having never reached the source.
A failure of the anomaly response, not a third outcome: it counts against Anomaly-response SR.
The step sub-budget expiring is how the detour ends, not why it ended: `detour-1` measured every abandoned detour as already plateaued when the budget cut it off, so a definition that names the budget names a mechanism the data refuted.
_Avoid_: resume, abort (both name the transition, neither says the source was never reached); reading a "RESUME" log line as evidence the source was found; naming the step budget as the cause.

**Mission**:
One episode = the primary find-task PLUS the anomaly response. "Mission complete" is not a single number — see Find-SR and Anomaly-response SR.

### The four headline metrics (per setting S1/S2/S3)

**Find-SR**:
Fraction of episodes where the agent completes the primary find-task (reaches the target within the success radius). Reported at 1.0 m primary; 0.1 m as a localization-bound diagnostic.
_Avoid_: success rate (ambiguous — always qualify find vs anomaly).

**Anomaly-response SR**:
Fraction of episodes where the controller runs the full loop — onset detected → investigated (reached source) → resumed the primary. Reported separately from Find-SR.
_Avoid_: mission success, controller success (say which loop states are required).

**soft-SPL**:
Graded path efficiency; the primary science metric for the memory delta (warm paired S3−S1).
_Avoid_: SPL (unqualified — binary SPL is a separate, localization-bound number).

**Benchmark SPL**:
The harness's native binary `spl`: STOP called within `success_distance` (**verified 0.1 m** in the canonical `objectnav_hm3d.yaml`) of a goal viewpoint, weighted by the geodesic path ratio.
This IS the standard HM3D ObjectNav ring VLFM's 0.304 and VLingNav's 0.429 are measured at, so native `spl` / SR@0.1 m are R1's cross-quotable Table-1 headline (ADR-0005), with no metric wiring needed.
A 1.0 m SPL is a RELAXED reach diagnostic, NOT the benchmark: quoting it against VLFM would overstate. `success_1m` (closest approach < 1.0 m at any step) is STOP-INDEPENDENT and never a success number.
_Avoid_: calling a 1.0 m SPL "benchmark"; the retired belief that "the benchmark uses 1.0 m"; success_1m as a success rate; soft-SPL for R1 (VLFM/VLingNav do not report it).

**R1 de-risk smoke vs Table 1**:
Table 1 (a.k.a. R1) is the **full-val** (20-scene) ObjectNav baseline, the only number quotable against VLFM 0.304 / VLingNav 0.429. A **val_mini** (2-scene) run is the **de-risk smoke**: it exercises the S1+ path, the vacuous-arm gate, and the paired analysis, but a 2-scene absolute number is indefensible and never enters the table.
_Avoid_: calling a val_mini run "R1" or "Table 1" (the driver's `R1 / Table 1 for split=val_mini` banner was a mislabel, corrected 2026-07-20; ADR-0006).

**Cost**:
Steps-to-complete (primary) and/or wall-clock. The efficiency-of-effort axis distinct from soft-SPL.
_Avoid_: time (say steps or wall-clock).

### Settings and warm/cold axes

**S1 / S1+ / S2 / S3 / S3+**:
Memory-off geometric-frontier baseline / memory-off **semantic-frontier** baseline / STM-only / full LTM on the geometric frontier / full LTM on the semantic frontier.
The semantic frontier is a **BLIP-2 ITM** match probability (VLFM, ICRA-2024), NOT the CLIP goal-cosine: CLIP is measured flat on HM3D sim renders (separation 0.020 against a 0.05 bar, three independent measurements) and is retired.
**S1+ is a documented negative, not the strong baseline** (ADR-0006): the BLIP-2 semantic frontier is inert-to-harmful (r1spin2 paired SPL −0.0175, soft-SPL +0.010 n.s.), the 4th independent non-lift of a semantic frontier. So the paper's spine is the **geometric** backbone, the headline memory delta reverts to **S3 − S1**, and **R2 drops the "+" arms**; S1+ survives only in R1 as a powered negative ("no cheap explorer, including VLFM's own head, beats geometric here").
_Avoid_: baseline (unqualified — say which of S1/S1+); calling S1+ "the strong baseline"; "Study 1" as a name (it reads as S1 — see the study names below).

**Memory-boundary study / Controller-and-audio study**:
The two sub-studies. Named, never numbered: "Study 1" and "S1" are different things and the collision has already caused a misread.

**Realizable localization**:
Reaching the anomaly source using only live binaural energy-gradient climb + L/R level sign + visual confirmation — no oracle source xyz. A/B'd against the oracle-source arm (the disclosed upper bound). Ceiling ~1 grid cell (~1 m), sim is level-only.
_Avoid_: DOA (the near-zero time-difference cue is not what drives this).

**Warm vs cold (seen axis)**:
Warm = the agent mapped this scene on a prior silent pass (visual LTM has it). Cold = first visit.

**Heard vs not-heard (audio axis)**:
Whether the anomaly sound was heard/stored on a prior visit. Currently a CLOSED negative (audio-memory value redundant with vision) — not a live experimental axis.

### Anomaly detection

**Scene-conditioned normality**:
Whether a heard sound is anomalous depends on the room it is heard in (running water is normal in a bathroom, anomalous in a bedroom). Grounded by the CLIP zero-shot room classifier + a hand-authored `ROOM_PRIOR` (room → expected-sound set); the sound is an anomaly iff it is unexpected for the detected room.
_Avoid_: context-free gate (the retired audio-only `is_anomaly`).

**Room-normal distractor**:
A discrete benign sound that is normal for the current room and must be IGNORED (no interrupt). Its correct rejection is what makes scene-conditioned discrimination load-bearing rather than decorative.
This is the ONLY thing discrimination is claimed on (ADR-0004).
_Avoid_: background bed (the continuous diotic noise floor — a different thing).

**Background bed**:
The continuous diotic noise floor present in every scene. It must sit BELOW `onset_rms` at every grid cell, so it is never the interrupt trigger (ADR-0004).
`bg_gain` is calibrated against the bed, never hand-picked.
_Avoid_: mixture (say bed, or say bed + anomaly).

**False interrupt**:
An onset that fired on anything other than the anomaly. Diagnosed by **onset provenance**, not by the interrupt count.

**Onset provenance**:
`onset_step` compared to `t_anom`. An onset before `t_anom` cannot be the anomaly, because the anomaly is not playing yet.
This is the one check that distinguishes a working interrupt from a vacuum cleaner, and it is invisible in `summary.json` — it lives in the `[audio] onset @step` log line.
_Avoid_: reading `n_audio_onset_fired` as evidence the anomaly was heard (it counts onsets, not causes), and reading `n_audio_gate_rejected=0` as "the gate had nothing to reject" (onset is one-shot, so 0 means the gate ACCEPTED the first over-threshold tick).

### The realizable climb

**Render scatter**:
The ray-traced renderer disagreeing with itself — the spread of received RMS across repeated renders at ONE fixed pose, holding distance, geometry and clip constant.
It is the noise floor any "did it get louder?" test has to clear, and it is measured per episode rather than assumed.
It is NOT the calibration sweep's spread: those poses sit at different distances, so their spread is the distance gradient, which is the very thing a rise is being read for.
_Avoid_: render noise (unqualified — say scatter at a fixed pose, or say the gradient); reading an unmeasured scatter as zero (unmeasured is null, and the run then falls back to the pre-`detour-2` threshold).

**Surge / cast**:
The two things the climb does. **Surge** is a forward step taken because the cue rose. **Cast** is what it does when the cue is dead: a turn, then a committed run of forward steps, with successive legs alternating direction so the sweep cannot close into an orbit.
The distinction is load-bearing because they are driven by different things — a surge is evidence, a cast is the absence of it — and because the agent had no cast at all until `eps-1` measured what that cost (ADR-0016).
_Avoid_: search, wander (neither says whether the agent had a cue); reading a forward step as evidence the agent heard something.

**Plateau window**:
A maximal run of consecutive detour steps over which the climb's own rising predicate reads false.
Reconstructed from the readings the controller actually used, so it is the controller's own verdict rather than a distance band a reader chose.
Its LENGTH is the unit that matters: a hail of one-step windows and a handful of long ones are different mechanisms with different fixes.
_Avoid_: stall, plateau (unqualified — say the window, and say how long it was).

**Floor-constrained source**:
The anomaly source sits on the primary goal's floor (`|Δy| < ~1.0 m`). Off-floor sources produce fabricated audio, because the RIR grid is rendered on one floor and `nearest` resolves by xz (ADR-0003).
