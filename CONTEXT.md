# LTM-Embodied Agent — Anomaly-Response Evaluation

Glossary for the audio-cued anomaly-response experiment (ICRA-2027 arc).
The task: an agent runs a primary find-task; an anomaly sound interrupts; it investigates the source, resumes, and reports.
This file is a glossary, not a spec — implementation lives in `embodied_memory/`.

## Language

### Task and mission

**Primary find-task**:
The ObjectNav goal the agent is given at episode start (reach a target object category).
Drives the SPL/success ground truth.
_Avoid_: main goal, objective.

**Anomaly response**:
The interrupt behavior — hear an abnormal sound, go to its source, check it, then resume the primary find-task.
_Avoid_: audio goal, sound goal (retired framing where sound == target).

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

**Cost**:
Steps-to-complete (primary) and/or wall-clock. The efficiency-of-effort axis distinct from soft-SPL.
_Avoid_: time (say steps or wall-clock).

### Settings and warm/cold axes

**S1 / S1+ / S2 / S3**:
Memory-off geometric-frontier baseline / memory-off **semantic-frontier** baseline (CLIP goal-cosine frontier term, the reviewer-proof strong baseline) / STM-only / full LTM. The headline memory delta is **S3 − S1+**.
_Avoid_: baseline (unqualified — say which of S1/S1+).

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
_Avoid_: background bed (the continuous diotic noise floor — a different thing).
