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
  "Purely SoundSpaces 2.0" means collapsing that split, so proving the collapse is possible is the gate the whole map hangs off.

Skills to consult per session: `/grilling`, `/domain-modeling`, `/research`, `/prototype`.

## Decisions so far

- [01 — SoundSpaces 2.0 parameter sheet](issues/01-parameter-sheet.md) — 23 acoustics knobs documented; the four that decide whether live-every-step is affordable are `irTime`, `indirectRayCount`, `threadCount`, `temporalCoherence`, and the two that could retire ADR-0003's floor constraint are `transmission` and `diffraction`. **Partly superseded: read against `main`, not our branch — see 02 and 11.**
- [02 — Can one audio sensor render simultaneous sources?](issues/02-simultaneous-sources.md) — No in habitat-sim (one source, hardcoded index 0), but **yes in the engine underneath**: `RLRA_AddSource` / `RLRA_ClearSources` / per-source IRs keyed `(listenerIndex, sourceIndex)`, one `RLRA_Simulate` for all. A ~40-line wrapper patch to files ticket 04 already compiles reaches it at zero extra renders per step, and per-source IRs make onset provenance structural. Cost of N sources is unmeasurable from source (closed `.so`) and moves to ticket 06.
- [11 — Reconcile the parameter sheet against the branch we actually build](issues/11-parameter-sheet-branch-reconcile.md) — `irTime` → **`maxIRLength`**; `updateDt`, `dumpWaveFiles`, `writeIrToFile`, `outputDirectory` gone; `enableMaterials` moved to the spec and defaults **false**; `directRayCount` + an HRTF basis are new; channel layouts narrowed to Mono/Binaural/Ambisonics. `transmission`, `diffraction`, `temporalCoherence` and the ray counts all survive, so 06 and 09 stand. **Every numeric default is unverified** — they live in the closed `.so`, so 04 must print them. Unknown config keys are silently swallowed (`py::dynamic_attr`), so the new tree's wrapper must validate keys and check every `RLRA_Error`.

- [03 — Do acoustic materials resolve on HM3D?](issues/03-materials-on-hm3d.md) — Materials are matched by **substring** against the Habitat semantic category name (only 13 of the 30 shipped materials are even reachable). For HM3D: **no by default, degraded at best**, behind three independent gates — `enableMaterials` is constructed `false`, plain HM3D has no semantic scene, and v0.2's *texture-based* semantics appear to hand the audio sensor an **empty mesh** (new ticket 12). The degraded path is confirmed as **no material database at all**, and it is what SoundSpaces itself runs on HM3D. Acceptable for us: the gradient's load-bearing terms are geometric, so uniform absorption costs **contrast, not structure**. Also: geometry uploads **once per context**, not per step (good news for 06).

## Not yet specified

- **The new package's module layout and seams.**
  What the simulator wrapper, audio sensor wrapper, controller, and runner look like as deep modules. Waits on 07 (what the rebuilt agent is) and 09 (task spec).
  Ticket 03 added a concrete requirement to carry in: the wrapper needs **loud invariant assertions at context creation** (non-empty audio mesh, key validation on `AudioSensorSpec` specifically, every `RLRA_Error` checked), because both the empty-mesh trap and the swallowed-key trap fail silently while still producing plausible audio.
- **How the STM/LTM calculation is carried across.**
  Copied, vendored, or imported; what interface it sits behind; whether the consolidation math is lifted verbatim. Waits on the package layout.
- **Smoke-green acceptance criteria.**
  What exactly counts as "one episode end to end". Waits on 09.
- **Test strategy for a Linux-only stack from a Mac.**
  Which layers stay pure enough to unit-test locally, and what has to be a box-only integration test.
- **Whether the geometric frontier searcher is rebuilt or replaced.**
  ADR-0006 retreated to the geometric spine after four non-lifts of a semantic frontier, but a clean room reopens the question. Waits on 04 (what models can even run in the one env).
- **How far the clean room is willing to fork habitat-sim.**
  02 found the first patch worth carrying (multi-source), and 04 now builds patch-capable. If more follow, the tree owns a habitat-sim fork with a maintenance cost and a reproducibility story, which is a different commitment from "we build upstream with a flag". Revisit once 06 and 09 have said whether the multi-source patch is actually taken.
  03 found a second, much smaller candidate: `RLRA_WriteIRMetrics` (RT60, EDT, DRR, C80, C50, D50, TS per frequency band) exists in the engine but is **not bound to Python** on this branch. It would settle acoustic questions directly instead of by proxy. Ticket 12 only takes it if the cheaper OBJ-colour proxy is ambiguous.

## Out of scope

- **The experiment matrix.**
  R1/Table 1, R2, the anomaly-response ablation, any S1/S2/S3 run. This map ends at one green episode, not at numbers.
- **Rebuilding the memory stack.**
  Deferred to a follow-on effort by explicit decision. Only the STM/LTM calculation carries across as reference.
- **The dialogue/MSC path.**
  `dialogue_memory/` is deleted as part of the reset; the MSC benchmark arc is not resumed.
- **Running the simulator on this Mac.**
  Structurally impossible (Linux-x64-only audio binary), not merely deprioritised.
