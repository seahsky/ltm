# Map: SoundSpaces 2.0 clean room

Label: `wayfinder:map`
Effort: `ss2-clean-room`
Charted: 2026-07-31

## Destination

A clean-room package at a new root in this repo, running on the RACE V100 only, where one HM3D episode runs end to end with SoundSpaces 2.0 audio rendered **live in the simulator at every step**.
The agent runs its primary find-task, an anomaly sound fires, it investigates the source, resumes, and reports.
The map is done when that smoke is green on the box and the old `embodied_memory/` and `dialogue_memory/` trees are deleted.

## Notes

Domain: embodied AI, Habitat simulator, HM3D scenes, geometry-based acoustic simulation.
The task glossary is `CONTEXT.md`.
Treat its language as binding: primary find-task, anomaly response, Find-SR, benchmark SPL, onset provenance, background bed, room-normal distractor.

Decided at chart time. Do not relitigate without saying so out loud:

- **Clean-room rebuild, not a port.** New top-level package in this repo; the old trees are deleted only once the smoke is green.
- **Audio renders live in-sim every step.** No precomputed RIR grid, no offline render, no `fftconvolve` lookup.
- **RACE V100 is the only execution environment.** This Mac is edit-and-push only, because `libRLRAudioPropagation.so` is a prebuilt Linux-x64 binary needing GLIBC >= 2.29.
- **Memory is out of the build.** The STM and LTM calculation carries across as the reference implementation for the follow-on effort, but nothing in this map depends on it working.
- **Execution is in scope**, not just planning. Wayfinder's plan-don't-do default is overridden here.

Two constraints that are already load-bearing and were not obvious:

- `data/`, `runs/`, `models/` are gitignored.
  The clean-room rebuild is a source-tree operation and does not touch 1.2 GB of HM3D or any downloaded weights.
- The old two-env split existed because the audio build (Python 3.9, numpy < 1.24, a 2022-era habitat-sim branch) could not hold the VLM stack.
  "Purely SoundSpaces 2.0" means collapsing that split, so proving the collapse is possible is the gate the whole map hangs off. **Gate PASSED 2026-08-01 — see ticket 04.**

Skills to consult per session: `/grilling`, `/domain-modeling`, `/research`, `/prototype`.

## Decisions so far

- [01 — SoundSpaces 2.0 parameter sheet](issues/01-parameter-sheet.md) — 23 acoustics knobs documented; the four that decide whether live-every-step is affordable are `irTime`, `indirectRayCount`, `threadCount`, `temporalCoherence`, and the two that could retire ADR-0003's floor constraint are `transmission` and `diffraction`. **Partly superseded: read against `main`, not our branch — see 02 and 11.**
- [02 — Can one audio sensor render simultaneous sources?](issues/02-simultaneous-sources.md) — No in habitat-sim (one source, hardcoded index 0), but **yes in the engine underneath**: `RLRA_AddSource` / `RLRA_ClearSources` / per-source IRs keyed `(listenerIndex, sourceIndex)`, one `RLRA_Simulate` for all. A ~40-line wrapper patch to files ticket 04 already compiles reaches it at zero extra renders per step, and per-source IRs make onset provenance structural. Cost of N sources is unmeasurable from source (closed `.so`) and moves to ticket 06.
- [11 — Reconcile the parameter sheet against the branch we actually build](issues/11-parameter-sheet-branch-reconcile.md) — `irTime` → **`maxIRLength`**; `updateDt`, `dumpWaveFiles`, `writeIrToFile`, `outputDirectory` gone; `enableMaterials` moved to the spec and defaults **false**; `directRayCount` + an HRTF basis are new; channel layouts narrowed to Mono/Binaural/Ambisonics. `transmission`, `diffraction`, `temporalCoherence` and the ray counts all survive, so 06 and 09 stand. **Every numeric default is unverified** — they live in the closed `.so`, so 04 must print them. Unknown config keys are silently swallowed (`py::dynamic_attr`), so the new tree's wrapper must validate keys and check every `RLRA_Error`.

- [04 — One-env feasibility: can the audio build hold the rest of the stack?](issues/04-one-env-feasibility.md) — **GREEN, the two-env split is dead.** One env (`ss2`) holds habitat-sim(audio, `RLRAudioPropagationUpdate @ 4f61e321`) + torch 2.0.1/cu117 on the V100-32GB + the CLAP stack, and the `numpy<1.24` pin held through every layer. The audio sensor renders a non-silent IR in a real HM3D scene. **The 23-knob parameter sheet is now MEASURED, not quoted** — `transmission` defaults **ON**, `enableMaterials` **False**, `maxIRLength` 4.0 (no `irTime`), and the `dynamic_attr` trap lives on the **spec** only. Two cracks: **CLAP cannot instantiate** (transformers 4.57 disabled its torch backend against torch 2.0.1 → ticket 13), and the single render was **0.60 s**, which is ticket 06's *unaffordable* case rather than its tolerable one.

- [05 — RACE box inventory](issues/05-race-box-inventory.md) — **Nothing blocks the map.** GLIBC **2.39** (the map's load-bearing assumption, never previously measured), 4 cores, V100-32GB, **680 GB disk free**, and `ss2` intact so ticket 06's sweep can run. Three findings travel: **HM3D val mesh coverage is 20/20**, not the 2/20 that forced earlier work onto `val_mini` (→ 08); **the box already runs torch 2.8.0+cu128 with CUDA on this V100**, so ticket 13's "bump torch" is measured rather than argued — though that env's numpy 1.26.4 sharpens the risk (→ 13); and **no MP3D is on the box at all** (→ 08). Loose ends for 10: a suspected ~9.3 GB duplicate in `data/`, and ~24 GB of VRAM held by something unaccounted for.

- [08 — Scene dataset: stay on HM3D or move?](issues/08-scene-dataset.md) — **HM3D stays; materials permanently off; MP3D out of scope** (ADR `docs/adr/0007`). MP3D is the better acoustic dataset and loses anyway, because the room character it buys is a property no result consumes — and the materials-off path is SoundSpaces' *own* HM3D reference configuration. Corrects two of this ticket's own arguments: cross-quotability is **weaker** than stated (ADR-0006 retired competitive absolute numbers, so it is the secondary reason, not the load-bearing one), and the "2 of 20 meshes" argument for moving is **dead** (05: val coverage is 20/20). Knock-ons: 12 loses its probe and keeps its guard, 06 runs the HM3D arm only, semantic annots stay on 10's keep list, nothing is downloaded.

- [03 — Do acoustic materials resolve on HM3D?](issues/03-materials-on-hm3d.md) — Materials are matched by **substring** against the Habitat semantic category name (only 13 of the 30 shipped materials are even reachable). For HM3D: **no by default, degraded at best**, behind three independent gates — `enableMaterials` is constructed `false`, plain HM3D has no semantic scene, and v0.2's *texture-based* semantics appear to hand the audio sensor an **empty mesh** (new ticket 12). The degraded path is confirmed as **no material database at all**, and it is what SoundSpaces itself runs on HM3D. Acceptable for us: the gradient's load-bearing terms are geometric, so uniform absorption costs **contrast, not structure**. Also: geometry uploads **once per context**, not per step (good news for 06).

- [07 — What is the rebuilt agent?](issues/07-what-is-the-rebuilt-agent.md) — A **candidate-pool frontier explorer with a detector seam and no LLM in the loop** (ADR `docs/adr/0008`). Proposers → scorer → waypoint → navmesh follower, with memory later plugging in as another proposer. One `GoalDetector.detects(obj)` serves *both* the primary STOP and the anomaly CHECK, oracle implementation for the smoke and caption-grounded for R2. Frontier proposer rewritten to ~300 LOC (A\* and the steering fallbacks were already dead on the live path; the semantic head is ADR-0006's negative); anomaly controller ported near-verbatim. **Dropped: the 7B planner** (`n_remembr_chosen ≈ 0`, frees ~15 GB), **CLIP**, and **the env-flag surface** — the clean room carries behaviour, not flags, and asserts the invariant instead. Costs to disclose: the paper stops claiming a ReMEmbR extension, the smoke does not exercise goal detection, and an oracle STOP deletes ~50% of the measured failure mass so smoke numbers are not capability numbers.

## Not yet specified

- **The new package's module layout and seams.**
  What the simulator wrapper, audio sensor wrapper, controller, and runner look like as deep modules. Waits on 09 (task spec).
  **The agent-side seams are now settled by 07** (ADR-0008) and are inputs here, not open questions: the proposer→scorer→waypoint pool, the `GoalDetector` interface, and the anomaly controller as a pure decision function that overrides the *pick*. What is left is the runner and the two wrappers around them.
  Four concrete requirements to carry in, all now settled and none of them waiting on 07 or 09:
  1. **Loud invariant assertions at context creation** (03): non-empty audio mesh, key validation, every `RLRA_Error` checked — both the empty-mesh trap and the swallowed-key trap fail silently while still producing plausible audio.
     **Now owned by ticket 12**, which 08 rescoped from a probe into exactly this guard. This entry stays as the specification; 12 is where it gets built and tested.
  2. **The key validator goes on `AudioSensorSpec` and nowhere else** (04, measured not predicted): the spec swallows unknown keys, `acousticsConfig` raises.
  3. **No fixed-width IR buffer** (04): the IR is trimmed to actual decay, not to `maxIRLength` — 1.64 s came back against a 4.0 s cap, so width is scene- and pose-dependent.
  4. **The runner drives `habitat_sim` directly and does not need habitat-lab** (04, verified on the box), so the new tree owns three small pieces habitat-lab used to supply — ObjectNav `.json.gz` episode loading, `sim.make_greedy_follower()` steering, and the SPL/SoftSPL arithmetic. Only the first has any weight.
  5. **One box-only fact the episode loader must settle** (08): whether `objectnav_hm3d` v1 loads against `hm3d_basis.scene_dataset_config.json` or requires `hm3d_annotated_basis.scene_dataset_config.json`. The old `habitat_env.py` reaches for the annotated one, which is suggestive, not proof. Not its own ticket — it is one line inside the loader — but it is why ticket 10 keeps the 9.3 G of semantic annotations rather than deleting them.
- **How the STM/LTM calculation is carried across.**
  Copied, vendored, or imported; what interface it sits behind; whether the consolidation math is lifted verbatim. Waits on the package layout.
  **That it is carried is no longer open** — 07 puts it on ticket 10's carry list, along with three other pieces that must move before the deletion (the caption-grounded detector stack, the SPL/soft-SPL arithmetic, and the ObjectNav `.json.gz` loader). Only the *how* is still fog.
- **Smoke-green acceptance criteria.**
  What exactly counts as "one episode end to end". Waits on 09.
  07 hands it one thing it must state rather than assume: **the smoke runs an oracle STOP, so it does not exercise goal detection at all**, and its find numbers are not comparable to the 0.031 benchmark SPL (an oracle STOP deletes the stop_miss ~50% of that failure mass outright).
- **Test strategy for a Linux-only stack from a Mac.**
  Which layers stay pure enough to unit-test locally, and what has to be a box-only integration test.
  Dropping habitat-lab (above) helps here rather than hurting: episode loading becomes a gzipped-JSON parse the Mac can unit-test, where `habitat.Env` was box-only by construction.
  07 adds two more Mac-testable layers by construction: the **anomaly controller** (a pure function over `(energy_history, lateral_sign, visual_confirm)`, no simulator) and the **navmesh reachability filter** (its `snap_point` / `geodesic` callables are injected). The SPL arithmetic is a third.
- **How far the clean room is willing to fork habitat-sim.**
  02 found the first patch worth carrying (multi-source), and 04 built patch-capable and confirmed the trap is real on the binary (`multi-source surface: none`) while shipping **stock, no patches applied**. If more follow, the tree owns a habitat-sim fork with a maintenance cost and a reproducibility story, which is a different commitment from "we build upstream with a flag". Revisit once 06 has priced multi-source and 09 has said whether the patch is taken.
  **Refined by 06 (2026-08-01): 06 prices the sequential *upper bound* only, not the patched cost.** Concurrent sources need the patch, and the patch decision was routed through 06's sweep — a loop, broken by ordering. The patch only earns a rebuild in the narrow band where one render fits the budget and three sequential renders do not; outside that band the fork question is decided on onset-provenance merit alone, not on cost. So this entry no longer waits on a patched measurement.
  03 found a second, much smaller candidate: `RLRA_WriteIRMetrics` (RT60, EDT, DRR, C80, C50, D50, TS per frequency band) exists in the engine but is **not bound to Python** on this branch. ~~Ticket 12 only takes it if the cheaper OBJ-colour proxy is ambiguous.~~ **Withdrawn by 08 (2026-08-01):** it was only ever a tie-break for ticket 12's material-characterisation probe, and that probe is dropped with materials. So multi-source is once again the *only* patch candidate, and the fork question is decided on onset-provenance merit alone.

## Out of scope

- **The experiment matrix.**
  R1/Table 1, R2, the anomaly-response ablation, any S1/S2/S3 run. This map ends at one green episode, not at numbers.
- **Rebuilding the memory stack.**
  Deferred to a follow-on effort by explicit decision. Only the STM/LTM calculation carries across as reference.
- **The dialogue/MSC path.**
  `dialogue_memory/` is deleted as part of the reset; the MSC benchmark arc is not resumed.
- **Running the simulator on this Mac.**
  Structurally impossible (Linux-x64-only audio binary), not merely deprioritised.

- **MP3D, and any other scene dataset.**
  Ruled out by [08 — Scene dataset: stay on HM3D or move?](issues/08-scene-dataset.md) (ADR `docs/adr/0007`): the acoustic-material fidelity MP3D buys is a property no result in this experiment consumes, and moving would cost the whole prior record, a fresh multi-GB download, and a redrawn destination.
  Ruled out **unconditionally**, not parked behind ticket 06's gradient number — a flat gradient would be a source-placement or gain problem, since materials are off in the MP3D reference configuration too.
  This also closes the **split** option (HM3D for benchmark numbers, MP3D for an audio-realism figure) and any audio-realism demonstration.

- **Acoustic materials, on any dataset.**
  Same decision. `enableMaterials` stays `false` and the new tree carries no material-database path.
  The consequence is a stated limitation, not a hidden one: uniform absorption, room-scale RT60 variation preserved via `V/S`, furnishing-dependent variation absent.
