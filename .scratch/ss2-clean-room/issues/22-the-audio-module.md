# 22 — The `audio/` module

Type: task
Status: open
Blocked by: 20 (resolved 2026-08-04)

## Question

Build every module under `earshot/audio/` except the guard (which lands in ticket 20), against task spec sections 2, 3 and 4.3.

`audio/` does **not** import `sim` — the sensor handle and the `observe` callable are injected (ADR-0013), which is what keeps this whole ticket Mac-testable except where the binary is the subject.

## What to build

- **`spec.py`** — `audio_sensor_spec(cfg)`, the **only** `AudioSensorSpec()` call site in the tree, routing through `apply_audio_config` + `assert_no_swallowed_keys`. A bare `setattr` anywhere else re-opens the `py::dynamic_attr` trap.
- **`sensor.py`** — `AudioSensorHandle`, which arms the guard in its constructor (requirement 1b: construction and arming cannot be separated, because the mesh upload is lazy). Exposes `set_source(xyz)` and `source_is_visible()`.
- **`bed.py`** — the fixed-level diotic bed, generated directly and mixed **after** rendering (ADR-0009). It never touches the RIR, so it is position-invariant by construction. `diotic_collapse` does not carry.
- **`onset.py`** — a one-shot threshold on live RMS, plus §3.1's provenance **assertions that raise**: pre-onset RMS equals the bed level within tolerance, and `onset_step >= t_anom`. Not a diagnostic read afterwards.
- **`calibration.py`** — §2.3's sweep: render the anomaly across the audible band, take the bed level and the anomaly RMS distribution, set `onset_rms` strictly between them, and report the separation as the gate number. **Overlap fails the gate**, and the correction is `globalVolume` (measured 1.0 on our branch), never a hand-nudged threshold.
- **`clips.py`** — `resolve_anomaly_clip`, the ESC-50 fetch, `normalize_clip`.
- **`lateral.py`** — `lateral_sign`, with §4.1's frame convention pinned. The grid rendered at identity yaw so the cue was world-frame; live rendering uses the agent's real transform and the same function now returns an **agent-frame** cue with no code change. Carried across with the old compensation, the controller turns the wrong way on every stall.
- **`clap.py`** — the open-set normal-vs-anomaly gate and its prompt banks (calibration ran GO, EER 0.00).
- **`normality.py`** — the `RoomLabeler` protocol, `ROOM_PRIOR`, `NullRoomLabeler` (the smoke) and `CaptionerRoomLabeler` (ADR-0012, takes an injected `Captioner`).
- **`config.py`** — `AudioConfig`: bed level, the audible band to sweep, the pre-onset RMS tolerance. **Not `onset_rms`** — that is derived at run start.

## Does not carry

`render_step_audio`, the `fftconvolve` lookup, `cached_source`, `RIRGrid`, `diotic_collapse`, `should_audio_stop`, `estimate_doa`, `audio_target_for_retrieval`, `gate_retrieval_target`.
Neither `audio.py` nor `audio_task.py` carries as a file (task spec §7).

## Done when

Mac tests green for onset, calibration, bed and clips. `lateral.py`'s frame convention has a **box** test in `tests/box/` — a fake cannot settle which frame the real renderer produces, and this is the one assumption in this ticket that a green Mac suite licenses nothing about.

## Watch for

`sourceIsVisible()` is **analyst-only** (§3.3). It is computed from the ground-truth source position, and the controller must never read it — feeding it to the decision rule plants a hidden oracle inside the arm ADR-0011 exists to avoid.
