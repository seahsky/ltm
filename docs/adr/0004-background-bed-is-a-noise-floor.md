# The background bed is a noise floor, not an interrupt trigger

**Status:** accepted (2026-07-16, grilling session). Reverses the Gate-0b R1 rule.

Gate-0b's R1 rule required a `bg_gain` at which the convolved background bed **clears** `onset_rms`, on the reasoning that otherwise the CLAP anomaly gate never runs and the reject-half of the mixture is vacuous.
That rule chose `bg_gain=1.0` and rested the whole design on the gate rejecting the bed.
It does not.
We decided to **drop `bg_gain` until the bed sits below the onset threshold everywhere on the grid**, making it a genuine noise floor, and to **relocate the discrimination claim to the room-normal distractor** (ADR-0002's two-rooms variant), where the agent's INVESTIGATE decision actually depends on it.

The rule was not merely unmet, it is unmeetable.
In `runs/anomresp-bed-s{1,3}` every onset in both settings fired at step 0–10 against `t_anom=30`, so the alarm never triggered a single interrupt in any episode; CLAP labelled the vacuum `alarm` every time; `n_audio_gate_rejected` was 0 in all eight episodes.
The cause is structural, not a threshold that needs tuning.
The bed is rendered through the *same* RIR grid from the *same* normalized level, so `bed(x) ≈ alarm(x)` at every cell, and the calibration that sets `onset_rms=0.111` to make the alarm audible at 4 m is computed on the alarm alone and is therefore guaranteed to be cleared by the bed at step 0.
Worse, before `t_anom` the signal is `bed(x)` and after it is `bed(x)+alarm(x)`, roughly a 3 dB step, while position swings energy nearly 8x across the grid: **no absolute RMS threshold can detect a 1.4x temporal step inside an 8x spatial swing.**
Gate-0b's own GO was recorded as weak (`delta=-0.2557`, "CLAP nearly inverts on convolved audio"), and the mix1 note predicted this exact false-fire. Both were right.

We chose this over (a) keeping `bg_gain=1.0` and fixing the discriminator, which Gate-0b already tested from both directions (text EER 0.094/0.175, audio-prototype 0.293/0.178 — both fail, leaving only the coin-flip of G0.3's augmented recalibration), and (b) replacing absolute-threshold onset with a temporal step detector, which is the technically correct fix but is unbuilt and confounded by agent motion.

This is also what the project already believed twice over.
`CONTEXT.md` defines the background bed as "the continuous diotic noise floor" and explicitly marks it as *a different thing* from the room-normal distractor; a noise floor that fires the interrupt is not a noise floor.
The C3 review concluded a month earlier that the continuous bed "does NOT make the agent's INVESTIGATE decision discrimination-dependent — for TASK-level load-bearing discrimination add a DISCRETE benign distractor onset."

**Consequences.**
The interrupt moves from step 0 to `t_anom`, mid-search, which is the task the paper claims to study; `SEARCH -> interrupt -> RESUME` becomes real rather than `interrupt -> investigate -> then search`.
The CLAP gate becomes decorative on the bed and the discrimination claim is dropped from it entirely.
`bg_gain` is calibrated, never hand-picked: the onset calibration must read the bed as well as the anomaly and assert `bed_max < onset_rms` with margin, or this reverts silently.
Discrimination now rests on ADR-0002's two-rooms arm, which is built and TDD-green and has never been run.
