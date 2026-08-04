# 18 — The new package's module layout and seams

Type: grilling
Status: open
Blocked by: none (09 resolved 2026-08-04)

## Question

What are the modules of the clean-room package, where do the seams fall, and what is the new root called?

The agent side is settled by ticket 07 (ADR-0008) and the task side by ticket 09 (`docs/anomaly_response_task_spec.md`).
Both are **inputs here, not open questions**.
What is left is the runner, the two wrappers (simulator, audio sensor), and the fixed points a dozen resolved tickets have already nailed down.

## What is already decided and must be accommodated

Ticket 07 / ADR-0008 fixes the agent seams: the proposer → scorer → waypoint pool, the `GoalDetector.detects(obj)` interface serving both the primary STOP and the anomaly CHECK, and the anomaly controller as a pure decision function that overrides the pick.

Ticket 09 fixes the task seams: the audio module (one live-rendered source plus a synthesized diotic bed), the onset detector and its calibration, the room-label **provider seam**, the split between the agent's report and the episode audit record, and the per-step audio record.

## The requirements this ticket inherits

Accumulated on the map's "Not yet specified" section by tickets 04, 08, 10, 12, 15, 16 and 17, and graduated here when ticket 09 resolved.

1. **The audio guard owns the first render, and the entry point is a fixed point in the layout.**
   Ticket 12 built `probes/audio_guard.py` and ticket 16 verified it on the binary; it ports verbatim.
   The layout owes it four things.
   (a) `pin_habitat_logging()` runs **before** anything imports `habitat_sim`, so whichever module touches the simulator first is a designed fixed point, not an accident.
   (b) The guard is **not** at context creation, because the mesh upload is lazy (`newInitialization_` is consumed by the first `runSimulation`) — it *owns* the audio sensor's first render, so whichever module constructs the sensor also arms it. They cannot be separated.
   (c) `AudioContextReport` is per-episode audit output and needs somewhere to land.
   (d) **The guard redirects fd 1 *and* fd 2 around the render it owns** (ticket 16: habitat splits its log across both; `ESP_DEBUG` writes to stdout). So nothing else in the process may write to stdout during that window — no progress print, no `tqdm` bar, no stdout logging handler. That is a property of where the runner puts its own output.

2. **Every path that configures an audio spec goes through the key validator.**
   Ticket 12 built `apply_audio_config` + `assert_no_swallowed_keys`; ticket 04 measured that the `py::dynamic_attr` trap is real and lives on the **spec** only.
   A bare `setattr` anywhere else re-opens it.

3. **No fixed-width IR buffer.**
   The IR is trimmed to actual decay, not to `maxIRLength`.
   Confirmed at three independent poses: 1.64 s (04) against a 4.0 s cap, 1.506 s (15), 1.26 s (16, `[2, 55637]` at 44.1 kHz).
   Ticket 16 adds a consumer detail: the observation is **not** a numpy array, so `getattr(obs, "shape")` reads `None` and anything wanting a shape must `np.asarray` it or walk the nesting.

4. **The runner drives `habitat_sim` directly; habitat-lab is dropped** (ticket 04, verified on the box).
   The new tree owns three pieces habitat-lab used to supply: ObjectNav `.json.gz` episode loading, `sim.make_greedy_follower()` steering, and the SPL / SoftSPL arithmetic.
   Only the first has any weight.

5. **One box-only fact the episode loader must settle** (ticket 08): whether `objectnav_hm3d` v1 loads against `hm3d_basis.scene_dataset_config.json` or requires `hm3d_annotated_basis.scene_dataset_config.json`.
   The old `habitat_env.py` reaches for the annotated one, which is suggestive, not proof.
   Not its own ticket — it is one line inside the loader — but it is why ticket 10 keeps the 9.3 GB of semantic annotations.

6. **A `reference/` directory outside the lint, test and import surface** (ticket 10).
   `<newroot>/reference/memory/` holds ~3,400 LOC of deliberately-inert vendored code, including `memory_bridge.py`.
   It is vendored **broken**: it imports `faiss` and `sentence-transformers`, and its interface is built against the deleted `episode_runner` and the env-flag surface ADR-0008 removed.
   If it can fail CI or be imported by accident, vendoring it was a mistake.

7. **The live ports have named homes** (ticket 10): the SPL arithmetic (`metrics.py`, 55 LOC, verbatim), an ObjectNav `.json.gz` loader extracted out of `habitat_env.py`, a `detects()`-shaped detector module shipping `OracleDetector` + `CaptionDetector` (OWLv2 dropped, L3 snap-gate kept), and the three `notify_*` files as-is.

8. **Three fixed points from the env pin** (ticket 17): `<newroot>/tools/bootstrap_ss2.sh` (the rebuild recipe, **moved** out of `.scratch/probes/oneenv_gate.sh`, not copied), `<newroot>/tools/ss2-constraints.txt` (9 exact pins + habitat-sim at SHA `4f61e321`), and `<newroot>/<pkg>/env_check.py` (the assertion, **importable** so bash and the runtime share one implementation).
   `assert_env()` joins `pin_habitat_logging()` at requirement 1(a)'s pre-`import habitat_sim` entry point.
   `env_report.json` joins `AudioContextReport` at 1(c), so that location now has two claimants and should be one place.
   Knock-on outside the layout: `docs/race-box-runbook.md` §3's "a record of what worked once, not a lockfile" goes stale the moment the constraints file exists.

9. **No lazy-loading seam, and the layout should not grow one** (ticket 15).
   The full stack measures **5.547 GiB co-resident against 31.73 GiB usable**, so every model is constructed eagerly at startup and held.
   Live audio costs **0.000 GiB** (RLR is CPU-side), so the audio sensor never competes with anything.
   This is a *removed* requirement: a layout that had to defer or evict models to fit is a materially different and worse design, ruled out on measurement.
   Resource pressure here is CPU and wall-clock (ticket 06), and module boundaries should reflect that.

10. **Ticket 09 adds a third claimant to the audit location** and one new seam.
    The **episode audit record** (ground-truth source position, distance-at-STOP, `sourceIsVisible()` history, provenance assertions, funnel stage, per-step audio wall-clock) lands beside `AudioContextReport` and `env_report.json`.
    The **room-label provider** is a seam whose implementation defers to R2; the layout fixes the seam now.
    The **agent report** is a separate artefact from the audit record and must not be able to reach ground truth — ticket 09 made that checkable by schema, and the layout should keep it checkable.

## What would resolve it

A grilling session producing the module tree, the seam list, and the new root's name, written down as an ADR or a layout document the build follows.

Note that ticket 10's phase 1 (vendor + port) cannot start until the root exists, so this ticket is on the critical path to the deletion commit.
