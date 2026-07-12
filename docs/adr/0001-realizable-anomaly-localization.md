# Realizable anomaly-source localization via energy-gradient climb

**Status:** accepted (2026-07-12, grilling session)

The anomaly-response controller previously navigated to the source by reading its oracle ground-truth xyz from episode metadata — defensible only as an upper bound, and a clear reviewer target ("the sound is just a stopwatch; the coordinate is handed to the agent").
We decided to build a **realizable** localization arm: the agent reaches the source using only live binaural RMS (climb the "getting louder" energy gradient, measured spearman ≈ −0.45 vs distance), the inter-aural level L/R sign for heading, and the visual detector/captioner to confirm arrival — then A/B this against the oracle-xyz arm.

We chose this over (a) keeping oracle-only (honest but a reviewer target and no real audio capability) and (b) lateral-sign-only DOA (no range signal, degenerates to a biased random walk) and (c) onset-pose visual-search-only (collapses into the already-closed "audio redundant with vision" negative).

**Consequences.** This reopens an arc a prior 7-agent review closed as near-impossible in this sim. The binaural cue is level-only (ITD stripped), so the localization ceiling is roughly one RIR grid cell (~1 m); binary success at 0.1 m to the source stays unreachable, and reach-within-~1 m is the honest metric. Energy-gradient climb needs a denser RIR render than the current ~24-cell grid to be climbable. The realizable arm is a multi-day build; the oracle arm stays as the disclosed upper bound.
