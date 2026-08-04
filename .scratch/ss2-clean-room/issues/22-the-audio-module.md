# 22 — The `audio/` module

Type: task
Status: resolved
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

## Answer

**Built: ten modules under `earshot/audio/`, 121 new Mac tests (114 → 235, all green under the `earshot-mac` 3.9.25 env), `ruff check earshot/` clean, and a four-test box file whose headline is the frame convention.** Every module is numpy-only at import time and imports no simulator, so the whole layer tests on this laptop — which is ADR-0013's injection rule paying out exactly as ticket 18 predicted.

**Five corrections, all found by building, and one of them is the ticket's own premise.**

**(a) `audio/spec.py` cannot construct an `AudioSensorSpec`, and this ticket's first line asked it to.** ADR-0013 calls it "THE only `AudioSensorSpec()` call site", but `AudioSensorSpec` is a habitat-sim type and the same ADR reserves those for `sim/world.py` — `test_layering.py` fails on the import. There is no third option: `task/` cannot name one either. So construction and configuration split — `sim.world.audio_spec_parts()` returns a **bare** spec plus the Binaural enum member, `audio/spec.py` is the only place that **configures** one. The property the requirement was protecting survives *intact and slightly stronger*: `py::dynamic_attr` swallows unknown keys silently, so the validator has to sit on the one path that writes fields, and a bare constructor call writes none. ADR-0013 amended in place; `sim/world.py`'s audio-blindness claim narrowed to match (it now names two habitat-sim types and reads neither).

**(b) The task spec is written about a signal, and the sensor returns an IR.** §3.1 asserts "pre-onset measured RMS equals the bed level", and ADR-0009 mixes the bed *after* rendering — so the quantity being compared is a **received signal**, while `get_sensor_observations()` gives an impulse response. Ticket 06's gradient work read IR energy directly, which is fine for a monotone climb and wrong as a threshold domain: calibrate on IR energy, threshold on a received signal, and the two differ by the clip's own level — a silent unit error whose symptom is a threshold that never fires. So the heard signal is `conv(IR, clip) + bed` throughout, and `calibration.sweep_anomaly_rms` measures through the same `render_through_ir` the runner will. **This also answers where §7's "rewritten rather than ported" `process_audio_step` lives**, which ADR-0013's tree named no module for: `clips.render_through_ir` plus `bed.heard_signal`, rather than a new file. `heard_signal` takes **no pose argument** — that absence is the whole difference from `render_step_audio`, which needed one to pick a grid cell and could fabricate from it.

**(c) §3.1's second invariant was going to be dead code.** `onset_step >= t_anom` cannot fail inside the per-step fold — the control flow makes an early onset unrepresentable — and a branch that cannot be reached is not a check. It moved to `assert_provenance`, whose subject is the **recorded** state: an `OnsetState` assembled by any other caller, which is what an analyst would quote. That function also carries a third check ticket 16 taught and the spec does not list: with `t_anom > 0` and zero pre-onset readings, invariant 1 never ran, so it is **unverified rather than satisfied**. Same discipline as the log canary, and it raises for the same reason.

**(d) The silent synthetic fallback does not carry.** `build_anomaly_clip` returned a seeded broadband burst whenever the clip path was missing or unreadable — so a run whose ESC-50 staging had failed produced a plausible episode in which CLAP, calibrated on real recordings at EER 0.00, classified a noise burst. `load_anomaly_clip` **raises**; `synthetic_burst` still exists and has to be asked for by name. This is the map's recurring failure class and the whole of the fix is one `raise`.

**(e) A pybind detail that would have failed on the box and passed here.** Reading a `def_readwrite` enum field back gives a re-wrapped object, not the interned class attribute, so `spec.channelLayout.type is binaural` would reject a spec that took the value **correctly**. The check is `==`, and a Mac test uses a deliberately re-wrapping fake so the comparison cannot quietly revert.

**Two judgement calls worth naming.** `is_anomaly`'s room conditioning moved out of `clap.py` into `normality.is_anomalous_here`, so each module holds one kind of evidence — cosines there, a hand-authored prior here — and "the room overrode the audio" is a line in the audit record rather than a branch inside a scoring function; the calibrated `(delta, tau)` are now the **defaults** rather than `(0.0, 0.0)`, which is why the old plain path needed two extra flags to get a working gate. And the `Captioner` protocol is declared at its consumer rather than imported from `vlm.py`: ADR-0013 permits `audio` → `vlm`, but the concrete connector drags torch and transformers, and this layer's entire Mac surface depends on importing neither. Ticket 23/25's `vlm.py` satisfies it by shape.

**One invariant armed early, in ticket 21's shape.** §3.3's "the controller must never read `sourceIsVisible()`" had nothing enforcing it — the realizable arm would still climb and its report would still validate, so the arm would stop being realizable in silence. `tests/mac/test_analyst_only.py` scans for both spellings (attribute *and* string constant, since `getattr(sensor, "sourceIsVisible")` is the same reach with the AST hidden) outside `audio/`, `report/` and `task/`. `agent/` does not exist yet, so it scans nothing — which is why its second test pins that the name it searches for is still the one `sensor.py` exposes, so a rename cannot turn it into a search for nothing. Verified by planting a violation: it fires.

**The numbers §9 left to the builder**, each tagged with provenance in `config.py` and set generously per ADR-0014: `bed_rms` 1e-3 (a **choice**, whose validity the calibration gate decides — §2.3 says there is nothing to calibrate a bed against), the swept band 1–8 m over 16 poses, `pre_onset_rms_tol` 5 % relative, `MIN_SEPARATION_DB` 6.0 and `ANOMALY_LOW_PERCENTILE` 10 in `calibration.py`. `sample_rate` is the branch's measured 44100 rather than the old tree's 48000, which is not cosmetic: **ESC-50 is 44.1 kHz, so the standard path never resamples** and `load_anomaly_clip`'s resample branch stays unexercised. The threshold is placed at the **geometric** mean of bed and anomaly-percentile — the arithmetic midpoint of 0.001 and 0.1 sits 34 dB above one and 6 dB below the other, which is not "between" in the sense that matters.

**What the box must settle, and what a red there means.** `tests/box/test_audio_box.py` renders one fixed source from one position twice — facing it, then turned 180 degrees. The agent frame predicts the sign **flips**; the world frame predicts it does not. That pair is decisive and no fake can produce it. Its failure message says so explicitly: a red is the **finding** — live rendering behaves as the grid did and ticket 23's controller needs the `heard == -right(world-bearing)` compensation back — **not a test to fix by inverting `lateral_sign`**. A third assertion separates the other way this can go wrong: agent-frame but inverted means ear 0 is the right channel, which is a two-constant edit rather than a controller change. The same file measures the real spec taking the preset, the received signal's domain and IR width against ticket 06's "trimmed to decay, not to `maxIRLength`", and the **per-step bill** (guarded render + convolution) that smoke criterion 7 wants audited every run rather than trusted from one sweep. `box_gate.sh` picks it up by discovery; its header now names three suites and points at the frame verdict as the line to read first.

**Honest limits.** A green Mac suite here is evidence about our own logic and nothing else. Untested until the box: every real-binding behaviour, and in particular whether `setAudioSourceTransform` takes at all — ticket 16 already found one call on this branch (`RLRA_SetListenerHRTF`) that returns `Success` over a failed load, and the source-transform handler is `ESP_ERROR` into a `void`, so its only channel is the guard's log scan on the next render. `CaptionerRoomLabeler` ships live but **unmeasured**: ADR-0002's $0 room-classifier accuracy gate carries across the captioner substitution and has not been run, which is safe only because the abstain contract means an unmeasured labeller cannot turn into a decision, and because the smoke runs `NullRoomLabeler`. CLAP's `(delta, tau)` were measured against a grid render convolved offline; the domain *should* match a live IR convolved the same way, but that is an inference and the first run to exercise the gate re-measures it — nothing in the smoke depends on the numbers. And ticket 21's `test_world_box.py` still assembles its own minimal audio spec rather than going through `audio/spec.py`, so it measures a configuration nothing runs; harmless, since its subject is `World`, and the new box file exercises the real path.
