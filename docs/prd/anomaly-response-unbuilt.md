# PRD — Anomaly-response: unbuilt ADR-0001 + ADR-0002 work

_Realizable anomaly-source localization (ADR-0001) and scene-conditioned anomaly detection (ADR-0002). Scope = everything the two ADRs decided that is not built yet. The two kill-switch gates (G0.4 climbability, G0.1 room-accuracy) and the 4-metric surfacing are already built; this PRD covers the mechanisms behind them._

Labels: `ready-for-agent`

## Problem Statement

Our anomaly-response evaluation makes two claims a reviewer can dismiss today.

First, the agent is supposed to "go to the anomaly source," but it navigates to the source's **oracle** ground-truth coordinate handed to it in episode metadata. The sound is only a stopwatch — there is no genuine acoustic localization, so Anomaly-response SR is measured on privileged information.

Second, the anomaly detector decides "is this sound abnormal?" from the audio alone, with no awareness of the room. It is calibrated on clean clips but runs on room-impulse-response-convolved, room-varied, background-mixed audio, so it needs a hand-tuned decision margin and still false-fires on a loud benign background. And because it fires the same way regardless of context, the agent's interrupt decision never actually depends on discrimination — the detector is not load-bearing on the task.

Both gaps were pre-registered: ADR-0001 (realizable localization) and ADR-0002 (scene-conditioned anomaly detection). We have already built the cheap kill-switch gates for each — `gradient_climbability`/`energy_gradient_verdict` (G0.4) and `room_pair_accuracy`/`room_gate_verdict` (G0.1) — and surfaced the 4 headline metrics into the run summary. What is missing is the actual mechanism each gate protects.

## Solution

**Realizable localization (ADR-0001).** The agent reaches the source using only agent-estimable signals: it climbs the live binaural loudness gradient, biased by the left/right inter-aural level sign, and confirms arrival visually — with no ground-truth source coordinate. This runs as an A/B arm against the oracle-source arm (the disclosed upper bound), so we can quote a genuine "reach-within-~1 m" number for the anomaly response. The acoustic ceiling is roughly one grid cell (~1 m, the sim is level-only), which is accepted; if the rendered grid is too sparse to climb, we render a denser grid.

**Scene-conditioned anomaly detection (ADR-0002).** Whether a heard sound is an anomaly depends on the room it is heard in (running water is normal in a bathroom, anomalous in a bedroom). The gate classifies the room (existing CLIP room classifier), looks up what is normal there (a hand-authored `ROOM_PRIOR`), and fires only on the unexpected. To make that decision load-bearing rather than decorative, the dataset places one **ambiguous** sound in a room where it is room-normal (agent must **not** interrupt) versus room-anomalous (must interrupt) — the room flips the verdict on the same clip. **Clip augmentation** calibrates the gate on the distribution it actually hears at runtime, removing the clean→convolved calibration cliff and the loud-bed false-fire.

Every new behavior is env-gated and default-OFF, so the objectnav / audiogoal / revisit paths stay byte-identical.

## User Stories

1. As a Researcher, I want the agent to reach the anomaly source using only agent-estimable acoustic and visual signals, so that Anomaly-response SR is not measured on the oracle source coordinate.
2. As a Researcher, I want a realizable-localization arm A/B'd against the oracle-source arm, so that I can report the realizable "reach-within-1 m" number beside the oracle upper bound.
3. As the Agent, I want to climb the live binaural loudness gradient toward louder cells, so that I approach the source without being told where it is.
4. As the Agent, I want to bias my heading by the left/right inter-aural level sign, so that I turn toward the source half-plane when the gradient is ambiguous.
5. As the Agent, I want to confirm arrival by visually detecting the anomaly object at the loudness peak, so that I stop at the source rather than at an arbitrary loud cell.
6. As a Reviewer, I want a guarantee that the realizable arm never reads the ground-truth source distance or coordinate, so that the localization claim is credible.
7. As a Researcher, I want the realizable-localization behavior gated behind an env flag that is off by default, so that the oracle path and all non-anomaly tasks stay byte-identical.
8. As a Researcher, I want to render a denser room-impulse-response grid when the climbability gate says the grid is too sparse, so that the loudness gradient is actually climbable.
9. As a Researcher, I want to run the climbability gate on a rendered grid before spending live compute on the realizable arm, so that I do not pay for an un-climbable field.
10. As a Reviewer, I want the ~1 m localization ceiling disclosed as a property of the level-only simulator, so that strict-radius failure is not read as a bug.
11. As the Agent, I want a bounded investigate budget so that when the source is an unreachable or too-distant detour, I abort the divert and resume the primary find-task.
12. As a Researcher, I want the anomaly gate to judge a heard sound relative to the room it was heard in, so that the same sound can be normal in one room and anomalous in another.
13. As a Researcher, I want a hand-authored room→expected-sound prior as the ground truth for normality, so that scene-conditioning works despite HM3D having no room-type ground truth.
14. As the Agent, I want to classify the current room from the keyframe with the existing CLIP room classifier, so that I can decide whether a heard sound is expected here.
15. As the Agent, I want to fire the interrupt only when the heard sound class is unexpected for the detected room, so that I ignore room-normal sounds and chase room-anomalous ones.
16. As a Researcher, I want a same-sound / two-rooms dataset variant that places one ambiguous sound where it is room-normal versus room-anomalous, so that the interrupt decision genuinely depends on the room.
17. As a Researcher, I want the discrimination reported as a false-interrupt rate (room-normal episodes) and a correct-interrupt rate (room-anomalous episodes), so that scene-conditioning is a measured outcome, not an assertion.
18. As a Researcher, I want the ambiguous-sound set (water, appliance hum) distinguished from the unambiguous anomaly set (alarm, glass, cry), so that the two-rooms test uses sounds whose normality actually depends on context.
19. As a Researcher, I want the room-conditioned gate gated behind an env flag, default-OFF, so that the context-free gate and non-anomaly tasks stay byte-identical.
20. As a Researcher, I want the scene-conditioning arm skipped when the room-accuracy gate (G0.1) is red, so that I do not build on a room classifier that cannot separate the two rooms.
21. As a Researcher, I want to augment the anomaly and benign clips (background mix at a target SNR, reverb / room-size jitter, pitch-shift, time-shift, and the room-impulse-response convolution the live path applies), so that the gate can be calibrated on the distribution it actually hears at runtime.
22. As a Researcher, I want the augmentation to be deterministic given its spec, so that a calibration run is reproducible.
23. As a Researcher, I want the CLAP anomaly gate recalibrated on the augmented + convolved distribution, so that the clean→convolved calibration cliff disappears and I do not hand-tune the decision margin.
24. As a Researcher, I want the recalibrated gate to reduce the loud-bed false-fire, so that the agent does not interrupt on a deafening benign background at the start pose.
25. As the Agent, I want augmentation to preserve the clip's shape and sample rate, so that the augmented clip flows through the render/convolution path unchanged.
26. As a Maintainer, I want augmentation exposed as one pure function on the audio module, so that both the live path and the calibration diagnostic use the same primitive (domain match).
27. As a Researcher, I want the room-accuracy gate's `main()` to render frames at object view_points and emit a machine-parseable `GATE_RESULT`, so that I can run G0.1 on a GPU host as a go/no-go before the scene-conditioning study.
28. As a Researcher, I want the two studies (memory-boundary and controller+audio) to share one dataset, so that a full factorial does not blow up the compute budget.
29. As a Reviewer, I want the audio to be causally necessary for the interrupt (onset-gated), so that the anomaly response is not an always-on behavior dressed up with sound.
30. As a Maintainer, I want every new mechanism to assert a byte-identical default path with a static/regression test, so that turning a flag off provably restores prior behavior.
31. As a Researcher, I want the realizable-localization decision expressed as a pure function of `(energy_history, lateral_sign, visual_confirm)`, so that it is tested without the simulator.
32. As a Researcher, I want the room-normality decision expressed as a pure function of `(sound_class, detected_room, ROOM_PRIOR)`, so that it is tested without CLAP or the simulator.
33. As a Researcher, I want the anomaly-response driver to run the realizable-vs-oracle and room-normal-vs-anomalous A/Bs on a smaller cell set (Study 2), so that each claim stays powered without a combinatorial blow-up.

## Implementation Decisions

**Shared: audio augmentation (P2.1, new pure seam).**
- Add one pure function `audio.augment_clip(clip, sample_rate, spec)` returning a deterministic augmented waveform. `spec` selects transforms and magnitudes: background mix at a target SNR, reverb / room-size jitter, pitch-shift, time-shift, and (optionally) the RIR convolution the live path already applies. No RNG, no CLAP, no simulator.
- The sub-transforms stay internal to the audio module and are exercised through `augment_clip` (one public seam). Augmentation shares the existing diotic-collapse / render primitives so the calibration domain matches the live signal.
- Consumer (G0.3): extend the convolved-anomaly calibration diagnostic to score the augmented + convolved distribution and emit the recommended decision margin; wire that margin as the anomaly-response gate default (replacing the hand-tuned value).

**ADR-0002: scene-conditioned anomaly detection.**
- Extend `audio.is_anomaly` with optional room-conditioning: when enabled, it takes the detected room and a `ROOM_PRIOR` (room → expected-sound set) and fires iff the heard class is unexpected for that room. New env flag, default-OFF → the context-free path is byte-identical.
- Add `ROOM_PRIOR` as a new hand-authored table (room → expected-sound set). This is distinct from the existing category→room prior; it is the ground truth for normality (HM3D has no room-type ground truth).
- The room label comes from the existing CLIP room classifier (`classify_room_clip`) — no new perception.
- Extend `make_anomaly_response_smoke` with a same-sound / two-rooms variant: one ambiguous clip placed at a room-normal source (no-interrupt episode) versus a room-anomalous source (interrupt episode), on a single RIR grid (the O(1) live-convolution invariant is preserved — we deliberately reject a simultaneous 2-source distractor).
- New metrics: false-interrupt rate (room-normal episodes) and correct-interrupt rate (room-anomalous episodes), surfaced beside the existing controller census.
- Complete the room-accuracy gate (G0.1) `main()`: render frames at object view_points, build `(ground-truth room from category prior, predicted room)` pairs, feed the already-built `room_pair_accuracy` / `room_gate_verdict`, print `GATE_RESULT`. Sim integration; the pure logic is built and tested.

**ADR-0001: realizable anomaly-source localization.**
- Extend the `anomaly_controller` state machine's INVESTIGATE mode with a realizable-localization branch, selected by a new env flag, default-OFF (→ the oracle path is byte-identical). In this branch the investigate target is derived from agent-estimable signals, not the oracle source xyz.
- Add a pure decision helper `realizable_investigate_step(energy_history, lateral_sign, visual_confirm) → action/STOP`: step toward higher live binaural loudness, bias heading by the left/right level sign, STOP when loudness peaks AND the visual detector confirms the anomaly object.
- Hard constraint: the realizable branch must never read the ground-truth source distance or coordinate. The live loudness read ("how loud here") is non-privileged; a static check asserts no GT-distance read leaks in.
- Add a denser-grid path via `render_rir_grid`'s cell-count control, used when the climbability gate (G0.4, built) reports the grid is too sparse.
- Run the realizable arm A/B against the oracle-source arm; report both the realizable "reach-within-1 m" number and the oracle upper bound.

**Experiment structure.** Two focused sub-studies share one dataset: Study 1 (memory boundary, `S1+ vs S3`) and Study 2 (controller + audio: realizable-vs-oracle and room-normal-vs-anomalous A/Bs). Augmented-gate calibration is held fixed (used everywhere, not an axis). Each phase runs its cheap gate first (G0.1/G0.3/G0.4) before any paid run.

## Testing Decisions

- **Test external behavior at the highest seam, not implementation details.** The three new pieces are each a pure function; tests assert their input→output behavior, so they survive refactors and need neither CLAP nor the simulator.
- `audio.augment_clip`: assert each transform changes the clip as specified (background mix raises energy toward the target SNR; time/pitch shift transform the signal; shape and sample rate preserved), that composition is order-stable, and that the same spec is deterministic. Prior art: the audio-primitive tests and the byte-identity-first tests around the existing render/convolution path.
- Room-conditioned decision: assert the pure `(sound_class, detected_room, ROOM_PRIOR)` verdict — unexpected-for-room fires, expected-for-room abstains — and that the context-free path is byte-identical when the flag is off. Prior art: the existing `is_anomaly` calibration tests and the room-gate tests (`room_pair_accuracy`/`room_gate_verdict`).
- Realizable-investigate decision: assert the pure `(energy_history, lateral_sign, visual_confirm)` helper — climbs toward higher energy, STOPs only on peak-plus-visual-confirm, and never consults a GT distance — plus a static check that the realizable branch reads no ground-truth source field. Prior art: `test_anomaly_controller` (the controller is already a pure, fully-tested state machine).
- Dataset variant: assert the same-sound / two-rooms construction (one clip, two placements; room-normal vs room-anomalous), and that the default (no two-rooms flag) path is byte-identical. Prior art: `test_make_anomaly_response_smoke` and the construction-issues checks.
- Every flagged mechanism carries a default-OFF byte-identity regression test (static or golden), mirroring the existing anomaly-wiring / active-goal no-op tests.
- The sim-integration glue (G0.1 `main()`, the A/B drivers) is verified on a GPU host, not by unit tests — those are `GATE_RESULT` / run-summary assertions, consistent with the existing gate diagnostics.

## Out of Scope

- **Sub-1 m acoustic localization.** The simulator's binaural cue is level-only (the time-difference cue is stripped), so the localization ceiling is ~1 m; strict success at 0.1 m to the source stays out of reach and is not a goal here.
- **A learned scene-conditioned model.** Normality is a hand-authored `ROOM_PRIOR`; training a classifier on `(audio, scene)` labels is deferred (prior trained heads all regressed).
- **A simultaneous two-source distractor.** The room-normal distractor is exercised via the same-sound / two-rooms behavioral A/B on a single grid; a second concurrent RIR source (which would break the single-grid O(1) invariant) is out of scope.
- **A non-oracle, range-capable direction-of-arrival write.** The oracle-source arm remains as the disclosed upper bound; a triangulated or learned range model is future work.
- **Hosted-model backbones for the headline.** The scored arms stay on the frozen local backbone (cross-quotability); hosted models are offload/diagnostics only.
- **The memory-boundary study's mechanics (S1+, S2/S3).** Those are separate plan items; this PRD is the controller+audio (Study 2) mechanisms plus the shared augmentation/gate work.

## Further Notes

- **Gate-first.** Each mechanism sits behind a cheap $0/offline gate: G0.1 (room accuracy) protects scene-conditioning, G0.3 (augmented-gate EER) protects augmentation/recalibration, G0.4 (climbability, built) protects realizable localization. A red gate prunes its axis before any paid run; if G0.4 is red, ADR-0001's fallback is oracle + disclosure and the controller headline survives either way.
- **Default-OFF byte-identical is the load-bearing invariant.** Every new behavior is env-gated and default-OFF; the objectnav / audiogoal / revisit paths must remain byte-identical, asserted per mechanism.
- **Robustness risks to watch.** Scene-conditioning leans on the CLIP room classifier reliably separating the two rooms (cosines are noisy ~0.30 — the G0.1 gate is the kill-switch). Realizable localization reopens an arc a prior review closed as near-impossible in this sim; energy-gradient is the one cue with a measured positive signal, but the ~1 m ceiling is hard.
- Design source: `docs/adr/0001-realizable-anomaly-localization.md`, `docs/adr/0002-scene-conditioned-anomaly.md`, `docs/anomaly_response_buildplan_2026-07-12.md`; glossary in `CONTEXT.md`.
