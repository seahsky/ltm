# 09 — Re-derive the anomaly-response task spec on live audio

Type: grilling
Status: open
Blocked by: 06 (02, 08 resolved)

## Question

The task given to the robot is unchanged, but several of its constraints were workarounds for the precomputed grid.
Which of them survive live in-sim rendering, and what is the task spec now?

## Why it matters

Four ADRs encode constraints that exist *because* audio was a baked grid:

- **ADR-0003, floor-constrained source.** The source must sit on the goal's floor, because the grid renders one floor and `nearest` resolves by xz, so an off-floor source fabricates audio. Live rendering removes the grid entirely. With `transmission` enabled (ticket 01) a source a floor away produces real attenuated audio. This constraint is a candidate for retirement, and retiring it changes what the task tests.
- **ADR-0004, background bed as noise floor.** `bg_gain` was calibrated against the bed so it never triggers the interrupt. Under live rendering the level calibration point is `globalVolume`, and the old `onset_rms=0.065` is meaningless. This must be recalibrated from scratch, and if ticket 02 says single-source, the bed needs a mechanism at all.
- **ADR-0001, realizable localization.** The stated ceiling was "~1 grid cell (~1 m)" because the grid was the resolution limit. Continuous receiver positions remove that ceiling. The energy-gradient climb may now be genuinely climbable rather than quantised.
- **ADR-0002, scene-conditioned normality** and the room-normal distractor. Depends on ticket 02's source-count answer.

There is also unfinished business the reset should not silently inherit: the `anommxv` headline was invalidated by three structural breaks, one of which was the interrupt firing on the background bed at step 0 rather than on the alarm. Onset provenance has to be designed in, not bolted on.

## What would resolve it

A grilling session, after 02, 06 and 08 land, covering:
- Source placement rules, and whether the floor constraint is retired, kept as a simplification, or replaced by a transmission-aware rule.
- How the bed and the room-normal distractor are produced under the source-count reality.
  Ticket 02 turned this into a real choice rather than a forced workaround: habitat-sim exposes one source, but the engine is natively multi-source with **one IR per source**, and a ~40-line wrapper patch reaches it at zero extra renders per step. So the options are (a) patch and make the bed and distractor genuine positioned sources, (b) post-render diotic bed with no distractor mechanism, (c) sequential re-render at Nx cost. Decide this here; ticket 06's source-count sweep supplies the price of (a).
  Weigh in particular that per-source IRs make **onset provenance structural** — the `anommxv` break where the interrupt fired on the bed and was read as the alarm becomes impossible by construction, rather than something a heuristic has to disentangle from a summed signal. That is a strong argument for (a) independent of cost.
- Whether `sourceIsVisible()` (a single-ray LOS test the branch exposes for free) becomes a first-class per-step annotation, given the deferred "non-LOS but audible" seed idea depends on exactly this distinction.
- Level calibration: how `globalVolume`, the bed level and the onset threshold are set, and by what measurable gate rather than by hand.
- Onset provenance as a first-class output, not a log line.
- Investigate and resume criteria: what counts as reaching the source now that positions are continuous.
- The metric set, restated against `CONTEXT.md` (Find-SR, anomaly-response SR, benchmark SPL, cost).
- Which ADRs are superseded, and by what.

Deliverable: the task spec for the new tree, plus superseding ADRs for whichever of 0001 to 0004 no longer hold.

## Note added by ticket 08 (resolved 2026-08-01) — the dataset is fixed, and one option is closed

**HM3D, `minival` for the smoke, acoustic materials permanently off** (`docs/adr/0007-hm3d-stays-mp3d-out-of-scope.md`). MP3D is out of scope, so any task-spec option that leaned on material-dependent room character is closed before this ticket starts.

Two things this ticket can now take as given:

- **The acoustic world is uniform-absorption.** Room-scale RT60 variation survives via `V/S`; furnishing-dependent variation does not exist. A task spec must not depend on the agent distinguishing a carpeted room from a tiled one by ear.
- **The `val_mini` constraint is gone.** Ticket 05 measured HM3D `val` mesh coverage at 20/20, not the 2/20 that forced earlier work onto `val_mini`. The smoke stays on `minival` because it is small and ticket 04 already loaded a scene from it, but a task spec is free to assume full `val` is available downstream.

Ticket 06 remains the live blocker: whether audio renders live at every step, or the spec has to be written against a throttled variant.

## Note added by ticket 07 (resolved 2026-08-01) — three things land in this ticket's lap

The agent is decided (`docs/adr/0008-clean-room-agent-architecture.md`): a candidate-pool frontier explorer, one `GoalDetector.detects(obj)` seam serving both the primary STOP and the anomaly CHECK, the anomaly controller ported near-verbatim, no LLM and no CLIP in the agent.

07 deliberately did **not** decide the following, because they are task-spec questions and deciding them there would have decided this ticket by accident:

- **The ADR-0002 room classifier.** 07 removed CLIP from the agent, and the room classifier is now the *only* route by which CLIP returns to the tree. If scene-conditioned normality survives ticket 02's multi-source answer, this ticket owns whether CLIP comes back for it, and on what evidence — CLIP is measured flat on HM3D sim renders (separation 0.020 against a 0.05 bar, three times), though that was frontier value at distance, not room typing.
- **Whether the controller's localization policy is amended for live audio.** 07 ported it verbatim on purpose. Its `(energy_history, lateral_sign, visual_confirm)` inputs were shaped by the precomputed grid — quantised energy, ADR-0001's ~1 m ceiling — and continuous receiver positions may make the gradient climb genuinely climbable. Amending it is this ticket's call, not a port decision.
- **The report's content.** `build_report` returns a structured dict today (`primary_completed`, `investigated`, `investigate_aborted`, `resumed`, `anomaly_class`, `source_xyz`, `n_benign_ignored`). Dropping the LLM costs nothing here, since nothing in it was ever generated text. Whether the destination's "and reports" wants more than this is a task-spec question.

One thing this ticket must **state** rather than assume, for the smoke-green criteria: the smoke runs an **oracle STOP**, so it does not exercise goal detection at all. `diagnose_spin` decomposed the 0.031 benchmark SPL as stop_miss ~50% + explore_timeout ~45% + success ~5%, and an oracle STOP deletes the stop_miss half outright. Smoke find numbers are therefore not capability numbers and must not be quoted as such.
