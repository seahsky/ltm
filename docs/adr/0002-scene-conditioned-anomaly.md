# Scene-conditioned anomaly detection with a same-sound/two-rooms test

**Status:** accepted (2026-07-12, grilling session)

Anomaly detection was context-free: CLAP scored the clip against fixed anomaly/normal prompt banks with no knowledge of the room. We decided that whether a sound is anomalous should depend on the room it is heard in (running water is normal in a bathroom, anomalous in a bedroom).

**Grounding.** The room is named by the already-built CLIP zero-shot room classifier; normality is a hand-authored `ROOM_PRIOR` (room → expected-sound set); the gate fires iff the heard class is unexpected for the detected room. HM3D has no room-type ground truth, so the hand-authored prior is the ground truth for normality.

**Making it load-bearing.** A room-conditioned gate that always fires is behaviorally identical to the context-free one. So the dataset uses a **same-sound / two-rooms** behavioral A/B: one context-dependent sound placed where it is room-normal (agent must NOT interrupt) versus room-anomalous (must interrupt). The room flips the verdict on the same clip; the metric is false-interrupt rate (normal episodes) and correct-interrupt rate (anomaly episodes). Single RIR grid — the O(1) live-convolution invariant is preserved (we rejected a simultaneous 2-source distractor for this reason).

**Consequences.** Requires *ambiguous* sounds (water, appliance hum), not the unambiguous alarm/glass/cry set, and leans on the CLIP room classifier reliably separating the two rooms (cosines ~0.30 — a robustness risk, gated by a $0 room-classifier-accuracy check before any live run). Clip augmentation (SNR/reverb/pitch/time-shift + the RIR convolution) is used to calibrate the gate so it is robust on convolved, room-varied audio.
