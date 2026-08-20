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

### The sounding window (sound-source finding)

Terms resolved in the 2026-08-20 grilling session, when the task pivoted from anomaly response to sound-source finding.
The section is OPEN: only the decisions listed here are settled, and the anomaly-response language above is not yet retired.

**Sounding window**:
The contiguous run of steps over which the positioned source emits.
It opens at `t_anom` and closes after a chosen duration, so the source is silent for the rest of the episode.
This is the pivot's load-bearing change: a source that sounds forever is reachable by gradient climb alone, which leaves memory nothing to do and makes all four generalization cells score alike.
The convention is the field's — SAVi (Chen et al., CVPR 2021) and SAVN-CE (CVPR 2026) both draw an onset and a duration per episode rather than sounding continuously.
_Avoid_: onset (it already means something else here — see below); playback, clip length (the recording's own duration is not the window).

**Offset step**:
The first step at which the source no longer emits, i.e. the step the sounding window closes.
_Avoid_: onset for either end of the window; "the sound ends" without saying which step.

**Onset step** (unchanged, and the collision is the point):
In this tree `onset_step` is the step the AGENT's own measured RMS crossed threshold — when it HEARD the source, not when the source STARTED.
`t_anom` is when the source starts.
The field's vocabulary uses "onset" for the source's start, so a paper sentence and a log line can use the same word for different steps.
Carry the tree's meaning and say "sounding window opens" for the other.
_Avoid_: importing SAVi's "onset"/"offset" pair wholesale onto `onset_step`; reading a cross-quoted onset time as a threshold crossing.

**Silent phase**:
The steps after the offset step, during which the only live signal is the background bed.
It is where both memories are supposed to pay, because there is no cue left to climb.
_Avoid_: silence (the bed still sounds); "after the sound" (say the offset step).

**Inferred goal class**:
The agent is told NOTHING at step 0.
The goal category is what CLAP returns for the clip the agent heard, so audio-to-class is inside the agent's own loop rather than handed to it.
This is what makes the not-heard column non-vacuous: told the category outright, the sound's identity never matters and heard and not-heard score alike by construction.
It is also the claim against MAGNet, whose 21-way head is welded into its tensor shapes and cannot take a 22nd class without retraining, while an open-set text encoder takes one more prompt and no training at all.
_Avoid_: goal category (unqualified — say whether it was inferred or given); treating a given category as a weaker version of the same task.

**CLAP separation gate**:
The measurement that must clear before anything reads the inferred class: CLAP's discrimination on clips rendered through a real reverberant IR at the distances episodes actually pose, not on dry recordings.
It exists because the capability is unproven here rather than merely untested — the one arc that ran it had the gate reject 0 of 8, which is what a gate that discriminates nothing also looks like.
Its number is a separation, in the pattern of the EER and CapRL gates, and a failure is fixed at the audio path, never by lowering the bar.
_Avoid_: reading a CLAP call that returns a finite logit as evidence the gate passed; depending on the inferred class before the gate has a number.

**Episodic LTM**:
The scene-keyed store: what was seen where, on a prior visit to THIS scene.
It is what the seen axis tests, and it pays in both heard and not-heard columns of the seen row.
It cannot pay in an unseen scene, and that is a property of the design rather than a shortfall — the archived stack made the same commitment structurally by hard-filtering its fine layer to the current scene.
_Avoid_: LTM (unqualified — say episodic or semantic; they are different stores answering different cells).

**Semantic LTM**:
The scene-AGNOSTIC store: sound class to object category to a spatial prior, accumulated across prior visits to OTHER scenes.
It is the only thing that can fill the unseen-and-heard cell, so it is what makes the matrix a factorial rather than three arms and a duplicate.
Its associations must be LEARNED from the prior passes, never handed over as a table: the dataset's own sound-to-object mapping is placement ground truth and giving it to the agent would test the author's prior instead of the agent's memory.
_Avoid_: prior, world knowledge (neither says the agent acquired it); reading a hand-authored placement table as if the agent had learned it.

**STM**:
Within-episode state, rebuilt every episode and never persisted.
Its job is the silent phase: carry the energy history and the lateral-sign trace from the sounding window so a bearing survives the offset step.
It is the counterpart to the cross-visit stores, and the axis it serves is time-within-episode rather than either generalization axis.
_Avoid_: memory (unqualified — STM crosses no episode boundary and settles no cell of the matrix).

**Sound-room mapping**:
The dataset's table from a sound class to the ROOM its source is placed in — a flush in the bathroom, an alarm in the bedroom, a TV audience in the living room.
The room resolves to an HM3D ObjectNav category for PLACEMENT (bathroom is the toilet, bedroom the bed, living room a sofa, tv_monitor or chair), so the object is where the source physically sits and the room is what the semantic store learns.
It is PLACEMENT GROUND TRUTH and analyst-only, fenced off in the same way `sourceIsVisible()` is: handing it to the agent turns the unseen-and-heard cell into a measurement of the author's table rather than the agent's memory.
**It was OBJECT-level until 2026-08-20 and the `clapsmoke-3` gate refuted that**: the `plant` anchor scored 0.383 with 187 of 480 rows landing on `toilet`, because water sounds predict a room with plumbing and a houseplant is not one. Rooms are the level the sounds encode.
_Avoid_: sound-object mapping (superseded — the anchor is a room); sound prior, class prior (both read as something the agent holds); using it anywhere in the controller.

**Sounding class vocabulary**:
The set of ESC-50 classes an episode can draw its source clip from, each anchored to one of three rooms and placed at that room's HM3D object.
The vocabulary is built at the CLASS level rather than being the anchors themselves: three rooms cannot be split heard-from-not-heard, but thirteen classes over three rooms can.
A class needs a room to belong to. Outdoor sounds have none, so `chirping_birds`, `crickets` and `rain` moved to `ABSENT_CLASSES` rather than being deleted — a sound whose source is not in the house is the hardest honest negative the forced-failure arm can have.
_Avoid_: goal category as a synonym for sound class (the mapping is many-to-one); assuming a class exists because ESC-50 has it — it must have a room.

**Sound-room affinity**:
How strongly a sound class implies one ROOM rather than any other.
It is the membership test for the sounding class vocabulary, and it binds harder than CLAP accuracy does: a flush implies a bathroom and an alarm a bedroom, but coughing and vacuuming happen in every room, so a vocabulary carrying them asks the semantic store to learn noise and it will correctly fail to.
A class that CLAP separates perfectly is still disqualified if its affinity is flat — `coughing` scored 1.000 in `clapsmoke-3` and is out.
**Grades are never derived from measured recall.** Fitting the ground truth to the classifier would make the matrix circular: the semantic store would be scored against a table built from the very model it depends on. `mouse_click` keeps its grade at 0.017 recall, and the separation gate is what cuts it.
_Avoid_: sound-object affinity (superseded); audibility or CLAP separability as the membership test (both are about the signal, affinity is about what the signal predicts); re-grading a class because the gate scored it badly.

**Pruned vocabulary**:
The sounding class vocabulary AFTER the separation gate has cut the classes CLAP cannot resolve to a ROOM through reverb at the distances episodes pose.
The candidate set going in is deliberately generous, including the weak anchors, so the surviving set is a measured artefact rather than an author's judgement.
Three independent cuts, each reported with its own count: too few rows, too little separation, wrong affinity.
The separation cut reads **anchor recall, not class recall** (ADR-0018, from `clapgate-1`): a class confused for a sibling of the same room still sends the agent to the right room, so cutting on class recall prices a cost the task never pays.
_Avoid_: fixing the vocabulary before the gate has run; reporting the pruned set without the candidate set it was cut from; quoting a survivor count without saying which recall the cut read.

**Success when silent (SWS)**:
The published fraction of episodes the agent completes by reaching the goal AFTER the sounding window closed (Chen et al., CVPR 2021, §5).
Adopted verbatim so it is cross-quotable against SAVi and SAVN-CE, and because it is the one standard metric that isolates what the silent phase tests.
_Avoid_: re-deriving it under a local name; reporting it without SR beside it.

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
Whether the sound CLASS has an LTM entry from a prior visit. Class-level, not recording-level: heard means an entry exists, not-heard means CLAP must place the class open-set from its name alone.
**REOPENED 2026-08-20**, having been a closed negative (audio-memory value redundant with vision) since the `write_audio_event` A/B.
The old null was measured on a single-goal harness where the agent could SEE the goal, so a stored audio event never carried anything vision was not already supplying.
Under the sounding window the source is silent for most of the episode, so in the silent phase there is nothing left for a stored audio-place association to be redundant WITH. That is the mechanical reason the null does not transfer, and it is the only thing licensing the reopening.
Reopening costs a prior visit per episode, which couples this axis to the seen axis: both need the same prior pass.
_Avoid_: recording-level "unheard" (that is a CLAP robustness question, not a memory one, and is the SECONDARY axis); citing the old redundant-with-vision null against the silent phase without saying the harness differed.

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

**Refused arrival**:
An abandoned episode that stood inside the detector's arrival ring (`DetectorConfig.oracle_radius_m`, 1.0 m geodesic) and was scored as never reaching the source.
It was possible because the stop rule read `visual_confirm AND not rising`: confirmation is a pure function of distance, so an abandoned in-ring episode proves the rising test was true at every in-ring step — the agent believed it was still climbing and walked back out.
The count decomposes a headline delta exactly, because a reach requires a confirm which requires being in the ring: **ring entries = reached + refused**, so a change either finds more entries (exploration) or converts more of them (arrival).
_Avoid_: near miss, almost reached (neither says the agent was inside the ring the detector uses); reading a refusal count off records that carry only the horizontal axis, which is at most the geodesic and so reads more steps as in-ring than the ring holds.

**Unrouted source**:
An episode whose source the navmesh could never reach — `find_path` returned nothing at every step, which the audit records as `distance_axis == "horizontal"` on a run that wrote routes.
It is unwinnable rather than failed: the detector asks the same pathfinder about the same target and reads a `None` distance as not-detected, so no confirm can fire whatever the controller does.
The builder screens xz separation and `|Δy|` (ADR-0010, ADR-0015) but never asks whether the source is REACHABLE, so these sit in the headline's denominator as losses no policy could have avoided — 23 of the 365 built episodes, identical across arms.
_Avoid_: unreachable (says nothing about which of the two pathfinder queries failed); dropping them from a denominator without printing the count you dropped.

**Floor-constrained source**:
The anomaly source sits on the primary goal's floor (`|Δy| < ~1.0 m`). Off-floor sources produce fabricated audio, because the RIR grid is rendered on one floor and `nearest` resolves by xz (ADR-0003).
