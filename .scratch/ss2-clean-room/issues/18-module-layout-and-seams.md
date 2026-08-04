# 18 — The new package's module layout and seams

Type: grilling
Status: resolved
Assignee: Sky
Blocked by: none (09 resolved 2026-08-04)
Resolved: 2026-08-04 — see the Answer section. ADR: `docs/adr/0013-clean-room-module-layout.md`. Surfaced tickets 20–27.

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

## Answer

**The root is `earshot/`, the root *is* the package, and the load-bearing decision is an edge that is absent: neither `audio/` nor `agent/` imports `sim`, so `import habitat_sim` appears in exactly one file in the tree.**

Full tree, layer graph, seam list and rationale: `docs/adr/0013-clean-room-module-layout.md`.
What follows is what this ticket decided beyond restating the ADR — the corrections it owes other tickets, and what it hands on.

### The root is the package, and this ticket's own notation was wrong

`<newroot>/<pkg>/env_check.py` was inherited from the fog patch, not decided.
It does not survive the facts: this repo has **no `pyproject.toml`, no `setup.py`, no lint config and no CI**, so everything resolves from the repo root and a nested package costs either a two-level import prefix or a `sys.path` insertion.
Nothing here is ever pip-installed — ticket 17 pins an *environment*, it does not build a distribution — so the separation buys nothing.

Requirement 8's `<newroot>/<pkg>/env_check.py` therefore collapses to `earshot/env_check.py`, which is what both `bootstrap_ss2.sh` and the runtime reach as `python -m earshot.env_check`.

`earshot` shares no prefix with `embodied_memory`, which matters during the window when both trees exist: a stale import fails loudly instead of reading as a typo.

### Three corrections this ticket owes, all in the direction of less work

**1. Requirement 1(d) is narrower than this ticket stated.**
It reads "nothing else in the process may write to stdout during that window — no progress print, no `tqdm` bar, no stdout logging handler."
But `capture_habitat_logs` flushes Python's own buffers on both `__enter__` and `__exit__` (`audio_guard.py:245-252`), precisely so the caller's pending bytes do not land in the capture.
**Interleaved, in-thread `print()` is safe.**
What is forbidden is a *concurrent* fd-1/2 writer: a background thread, a timer-driven progress bar, an inherited subprocess descriptor, a logging handler flushed off-thread.

**2. Requirement 6 cannot be satisfied by omitting `__init__.py`, and this was verified rather than argued.**
PEP 420 namespace packages import `earshot.reference.memory.ltm` cleanly from a regular parent package — tested directly, and the behaviour is 3.3+ so it holds on the box's 3.9.
The only thing currently stopping it is `faiss` not being installed in `ss2`, which is luck, and which flips the day someone installs faiss to work on the memory follow-on.
So `reference/__init__.py` and `reference/memory/__init__.py` each **raise `ImportError`** with a pointer to the README.

**3. Ticket 17 contains a contradiction, and the layout dissolves it.**
`assert_env()` is specified to run "at the entry point, **before** `import habitat_sim`" (17, line 119), while one of its three checks is "`habitat_sim` audio via the enum **member** probe" — which cannot be done without importing habitat-sim.
With `pin_habitat_logging()` in `earshot/__init__.py`, importing `earshot.env_check` has already run the pin, so `env_check` is free to import habitat-sim.
`assert_env()` stays an explicit entry-point call because it is expensive and half box-only, not because of ordering.

### The leak requirement 10 wants closed is the current code, not a hypothetical

`build_report` (`anomaly_controller.py:302-316`) emits `"source_xyz": ev.get("source_xyz")` straight out of `ControllerState.investigation_event`, returns an untyped `Dict[str, Any]`, and mutates the state it was handed.
Ticket 10's "ports near-verbatim" would have carried all three in.

"The controller cannot see ground truth" is **not available as the rule**, because the oracle arm's controller legitimately holds `source_xyz` as its waypoint while task spec §5.1 requires an identical schema in both arms.
The boundary is drawn at the *type* instead: `AgentReport` is frozen with exactly §5.1's nine fields, so nothing privileged can appear in it whatever the controller holds.
**Deviation from ticket 10, taken deliberately: the anomaly controller does not port near-verbatim.** `build_report` moves to `earshot/report/`, and the state mutation goes with it.

### What the shared observation call forced

`sim.get_sensor_observations()` returns RGB, depth and the audio IR in one dict (`oneenv_probe.py:629`).
There is no separate audio render, so the frontier's depth frame and the onset detector's IR come from the same call, and criterion 1's "render count equals step count" is measured on it.

That is why `sim/` owns the lifecycle and is audio-blind, while `audio/spec.py` is the only `AudioSensorSpec()` call site in the tree — which makes requirement 2's key validator structural rather than remembered.

### The guard splits in two

`arm_audio_context()` stays once-per-episode (the 0.814 s / 32.2 MB OBJ write, the vertex floor, key validation).
A new light **`guarded_observe()`** wraps every step: fd capture, canary, fatal-line scan, no OBJ.

Ticket 16 measured `[Audio]` on **every** render specifically so this is possible, and it needs to be: the same ticket found the closed engine writing un-prefixed error blocks to fd 2 and `RLRA_SetListenerHRTF` returning `Success` over a failed load.
Those can happen at step 300.
Cost is two tempfiles per step, landing inside the per-step wall-clock the task spec already requires reporting — audited, not assumed.

### Three structural invariants, all Mac-runnable

`test_layering.py`, `test_report_boundary.py`, `test_no_env_flags.py`.
The layering test exists because both backwards dependencies rejected during the grilling — `agent/` reaching into `audio/` for a depth frame, `audio/` reaching into `agent/` for a room label — are one convenient import away.
Documentation-as-enforcement was rejected on this repo's own record: ticket 14's self-update gotcha, ticket 17's inert pin and ticket 13's version-blind skip were all written down and then quietly stopped being true.

### Smaller placements

- **Probes carry only where a live consumer exists.** `oneenv_gate.sh` → `tools/bootstrap_ss2.sh`; `audio_guard.py` → `audio/guard.py`; `test_audio_guard.py` → `tests/mac/`; `audioguard_probe.py` + `audioguard_gate.sh` → `tests/box/`. The other seven stay in `.scratch/`, which is **tracked** (`.gitignore:129` is `scratch/`, not `.scratch/`), so nothing is lost — it just stops pretending to be part of the build.
- **`tools/` is operator-facing and not part of the agent**, so the notify trio lands at `tools/notify/`. The dataset builder is *not* a tool — it enforces ADR-0010's source placement, which is task policy — so it sits at `task/dataset.py` beside the loader it shares a schema with.
- **`onset_rms` is not configuration.** §2.3 derives it at run start from the calibration sweep; `AudioConfig` holds the bed level and the audible band.
- **Python 3.9.19** means `int | None` needs `from __future__ import annotations`, and `test_report_boundary.py` must read `__dataclass_fields__` rather than `typing.get_type_hints()`, which raises on those annotations.

### What this unblocks

**Ticket 19 is unblocked, and inherits two constraints rather than deciding them freely**: the `tests/{mac,box}/` split pre-commits it to a where-it-runs taxonomy, and the injection rule makes its Mac surface most of the tree — `audio/`, `agent/`, `report/`, `metrics`, `types` — rather than the four layers it listed.

**The map's last fog patch graduates.** "The build itself, and the smoke run" was waiting on exactly this ticket, because the slices *are* modules and the modules now have names.
It becomes tickets **20–27**, in layer order: scaffold, then the four layers in parallel, then wiring, then the smoke, then the deletion.
None of them is blocked on a question.
