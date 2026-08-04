# 21 — `sim/world.py` and the ObjectNav episode loader

Type: task
Status: claimed
Blocked by: 20 (resolved 2026-08-04)

## Question

Build the one module in the tree that imports `habitat_sim`, and the episode loader that replaces habitat-lab.

## What to build

**`earshot/sim/world.py`** — `World(scene, sensor_specs)` taking a list of specs it does not interpret (ADR-0013: audio-blind), exposing `observe()` (the one shared call returning RGB, depth and the IR together), `step()`, the navmesh, and `sim.make_greedy_follower()` steering.

It asserts `HABITAT_SIM_LOG` is pinned immediately before `import habitat_sim`.
It is the only file in the tree that may import it — `tests/mac/test_layering.py` enforces that.

**`earshot/task/episodes.py`** — the ObjectNav `.json.gz` loader, extracted out of `embodied_memory/habitat_env.py` (623 LOC, habitat-lab-coupled, does not carry) across the dataset-path search, scene-label resolution, and the `content/<scene>.json.gz` lazy load.

## The box fact this settles

Ticket 08's outstanding question, deferred here deliberately because it is one line inside the loader: **does `objectnav_hm3d` v1 load against `hm3d_basis.scene_dataset_config.json`, or does it require `hm3d_annotated_basis.scene_dataset_config.json`?**

The old `habitat_env.py` reaches for the annotated one, which is suggestive, not proof.
Until it is answered, ticket 10 keeps the 9.3 GB of semantic annotations on the box.
**Record the answer on this ticket** — it is the only thing standing between that 9.3 GB and a deletion decision.

## Done when

A real HM3D scene loads, `observe()` returns all three modalities, the follower routes to a navmesh point, and the `scene_dataset_config` question has a measured answer written down.

## Notes

Requirement 3: the IR is trimmed to actual decay, not to `maxIRLength` — confirmed at three poses (1.64 s, 1.506 s, 1.26 s against a 4.0 s cap). No fixed-width buffer.
The observation is **not** a numpy array; `getattr(obs, "shape")` reads `None`, so anything wanting a shape must `np.asarray` it or walk the nesting (ticket 16).

---

## Built, 2026-08-04 — box run 1 RED on 2 of 21, fixed, awaiting run 2

Both modules exist, the Mac suite is **114 green** (was 84), `ruff check earshot/` is clean over 20 files.
Box run 1 (`4ok3usBNeis`, ObjectNav `val`, 74.8 s): **19 of 21 pass**, two errors on one cause, fixed below.
Re-run `bash earshot/tools/box_gate.sh --branch wayfinder/ss2-clean-room-21` and paste `runs/ss2-box-gate/box-suite.log`.

### What box run 1 established

| claim | measured |
| --- | --- |
| `SimulatorConfiguration().scene_dataset_config_file` | `'default'` — habitat-sim's own constructor default |
| a real HM3D scene loads with no scene-dataset config | **yes**, 1.19 s, navmesh loaded |
| the audio spec reaches the sensor suite via `AgentConfiguration.sensor_specifications` | **yes** — `AudioSensor`. The cross-version inference this ticket disclosed is **closed GREEN**; the `add_sensor` fallback is not needed |
| `observe()` returns all three modalities in one call | `rgb (256,256,4) uint8`, `depth (256,256) float32`, `audio_sensor (2, 57568) float64` |
| depth is metric, not normalised | **0.400 – 2.551 m** |
| the raw IR has no `.shape` | attribute **absent entirely** — ticket 16 recorded `None`; either way, coerce or walk it |
| `step()` never renders | `n_steps=10, n_renders=5` |
| the guard arms on `World.observe` | 335,370 verts held vs **335,362 submitted — the +8 gap reproduces** on a second scene |
| `snap_point` / `geodesic_distance` off the navmesh | `None` / `None`, never NaN, never `inf` |
| episode 0 parses from the published bytes | `bed`, 2 goals, **1187 view points**, authored `episode_id` preserved |

### The two errors, and the fix

`MultiGoalShortestPath.closest_end_point_index` **does not exist on the box's branch**. It is present on habitat-sim 0.3.3 — where the Mac pre-flight verified it — and absent on the 2022-era `RLRAudioPropagationUpdate`. Cross-version skew, on the one field the pre-flight had checked and therefore trusted.

It was read by `nearest_of`, a convenience returning *which* end was nearest. **Nothing consumed it, so it is removed** rather than papered over: a `getattr` fallback would have returned `-1` on the box forever, which is the silent-degradation shape this map keeps deleting. `geodesic_distance` now reads only `geodesic_distance`, exactly what habitat-lab reads. If ticket 25 ever needs the index, the branch-safe route is matching `path.points` (present on both versions) against `ends`.

### Ticket 08's decisive form was not what run 1 looked for

Run 1 hunted for a scene with **no** `.semantic.glb`, on the theory that the annotated config could then have had nothing to point at. All 20 ObjectNav `val` scenes on this box are annotated, so it reported the weaker "the plain path suffices".

The stronger claim was available all along and on every scene: **the annotation file is on disk and the loaded scene provably does not contain it.** habitat-sim said so three times in run 1's own log — `ResourceManager.cpp:473` "Not loading semantic mesh", `Simulator.cpp:471` "The active scene does not contain semantic annotations : activeSemanticSceneID_ = 0", and `AudioSensor.cpp:143` "Semantic scene does not exist or materials are disabled, will use default material". Run 2 asserts it through the API (`World.semantic_object_count() == 0`) rather than by scraping log lines.

So the 9.3 GB is not merely unneeded by the loader; it is untouched by the scene the simulator builds. The `setUpModule` scan also stops at the first usable scene now instead of parsing twenty tens-of-megabytes content files to answer a question that no longer needs asking.

### One observation for ticket 22, not a defect here

Per-step `guarded_observe` measured **~1230 ms mean (912–1510 ms)**. That is ticket 06's *unaffordable* default regime (1723 ms/step at stock acoustics), not its 27.2 ms `cheap_preset` — this test builds a bare `AudioSensorSpec` with only `sampleRate` set, because `audio/spec.py` and the preset are ticket 22's. Nothing to fix in `sim/`; recorded so the number is not later read as a `World` cost.

### The `scene_dataset_config` question had a false premise

It asked whether ObjectNav HM3D v1 loads against `hm3d_basis.scene_dataset_config.json` or requires `hm3d_annotated_basis.scene_dataset_config.json`.
**Stock habitat-lab uses neither.** The chain, read at `habitat_lab-0.3.320250127` rather than inferred:

| where | what it says |
| --- | --- |
| `config/benchmark/nav/objectnav/objectnav_hm3d.yaml` | never sets `scene_dataset` — the whole file is sensors, turn angle and agent geometry |
| `config/default_structured_configs.py:1744` | so it keeps the default, `scene_dataset: str = "default"` |
| `sims/habitat_simulator/habitat_simulator.py:326` | which is assigned straight to `SimulatorConfiguration.scene_dataset_config_file` |
| `habitat_sim.SimulatorConfiguration()` | whose own constructor default is already `'default'` (checked against the binding) |
| `datasets/object_nav/object_nav_dataset.py:163-168` | and the episode's `scene_id` is resolved as a **plain filesystem path** against `scenes_dir` |

That is the same form ticket 04 and ticket 16 already rendered against on the box — `backend_cfg.scene_id = <path to .basis.glb>`, no dataset config anywhere.

**Why the old tree reached for the annotated one.** `habitat_env.py:132` set it deliberately, and it bought exactly one thing: a working semantic sensor.
That sensor is gone three times over — ADR-0007 turns materials off permanently, ticket 03 found HM3D's v0.2 texture-based semantics appear to hand the audio context an empty mesh, and `CLAUDE.md` records the sensor returning all-zeros under every earlier result.
So **the clean room needs no scene-dataset config of any kind, and the semantic annotations ticket 10 kept only against this question are no longer load-bearing.**

`tests/mac/test_episodes.py::TestNoSceneDatasetConfig` pins it as an invariant rather than a comment: no module may set `scene_dataset_config_file` or carry a config path as a live string.
**AST-shaped, and that was learned here** — the first version scanned raw lines and went red on `episodes.py`'s own docstring, the citation chain establishing the config is unnecessary reading as a use of it. Ticket 19's "a grep verifies presence not truth", arriving in the one place the map had not applied it.

`tests/box/test_world_box.py::TestSceneDatasetConfigIsUnnecessary` is the measurement, and it is built to be **decisive rather than merely consistent**: it prefers a scene with **no `.semantic.glb` on disk**, which the annotated config would have had nothing to point at. Ticket 05 measured 100 basis meshes against 36 semantic ones in `val`, so one should exist; if every candidate is annotated the test says so and reports the weaker claim.

### Three defects the Mac caught before the box trip

Ticket 16's pattern: read the binding first, spend the box trip on what only the box can settle.

1. **`PathFinder` has no `geodesic_distance`** (BLOCKING). That name is habitat-**lab**'s wrapper method (`habitat_simulator.py:528-553`), which builds a `MultiGoalShortestPath`, calls `find_path` and reads the field off the result. The plausible one-liner `self._sim.pathfinder.geodesic_distance(...)` is an `AttributeError` no Mac test could have reached, because nothing here can construct a navmesh. Implemented the real way.
2. **`min_depth` / `max_depth` / `normalize_depth` do not exist on `CameraSensorSpec`** — all three are habitat-lab config fields applied by its own depth wrapper (`hasattr` is False for each). Setting them would have been three silently swallowed assignments, exactly what ticket 12's key validator exists to catch, one layer up. Removed; they are named in the docstring instead. The consequence is in our favour: raw habitat-sim depth is already **metric**, so the normalised-depth trap that collapsed Run 5's occupancy splat is one the clean room structurally does not have.
3. **A Simulator built with an empty spec list can never have a camera added afterwards.** `Configuration._sanitize_config` derives `create_renderer`, `requires_textures` and `load_semantic_mesh` from `sensor_specifications` *before* construction (`simulator.py:92-112`), and `add_sensor` then refuses any modality absent at init (`:265-284`). Measured on this Mac: `ValueError: Data for SensorType.COLOR sensor was not loaded during Simulator init`. So "every spec goes in at construction" is a constraint, not a style choice — and it is also the audio-blind shape, which is a happy coincidence rather than a design win.

### What `sim/world.py` is

`World(scene, sensor_specs)`, handed a list it does not interpret, exposing `observe()`, `step()`, `pose()`/`set_pose()`, the navmesh (`is_navigable` / `snap_point` / `geodesic_distance` / `random_navigable_point` / `seed_navmesh`), `follower()`, `sensor_handle()` and `semantic_object_count()`.

- **`step()` deliberately does not render.** `habitat_sim.Simulator.step` acts *and* renders; using it would make render count exactly twice step count and turn smoke criterion 1 into a tautology. The runner steps, then observes, and `n_renders` counts observations exactly.
- **`observe()` returns the raw dict.** Ticket 16's `getattr(obs, "shape") is None` finding survives only if nothing coerces on the way out, and coercing would force this module to know which key is audio.
- **`NoRouteError` is its own type.** The alternative reading of `None` is *arrived*, and that confusion is what made the old navigation unfalsifiable: the grid-A\* found no path on ~92% of steps and silently fell back to straight-line steering, so "a waypoint was chosen" and "the agent got there" came apart with nothing in the code marking where.
- **`snap_point` returns `None`, never NaNs**, and `geodesic_distance` returns `None`, never `inf` — an `inf` that reaches an SPL denominator produces a number instead of an error.
- **The backend flags raise on absence rather than `hasattr`-skipping.** The probes hedged because they did not yet know the branch; `load_semantic_mesh = False` is what keeps ticket 03's empty-mesh path shut, so a renamed field must fail loudly rather than turn semantics back on in silence.
- **Embodiment is habitat-lab's published ObjectNav HM3D configuration** (0.88 m, 0.18 m radius, 0.25 m steps, 30 degree turns, hfov 79, sliding off), carried so this run stays comparable with the prior record. habitat-sim's own `AgentConfiguration` defaults differ on all four, so accepting them would have silently changed the embodiment.

### What `task/episodes.py` is

Pure and stdlib-only — no numpy, no habitat-sim, I/O only through `gzip` + `json` — so the whole module is Mac-tested against real gzipped bytes rather than against a fake. The schema is habitat-lab's, read from source; the citations are in the module docstring.

**One deliberate divergence, and it is a fix.** habitat-lab overwrites every authored `episode_id` with the load index (`object_nav_dataset.py:141`). That renumbering is why the analysis pipeline had to re-key onto `(scene_id, target_category, visit_order)` after silently dropping pairs, and why `seed_only` had to ride in `episode.info` because the id could not carry it. The loader keeps the authored id and exposes the load index separately, so both are available and neither is a lie.

Every field it reads steers the agent, so it raises rather than defaults: a missing `start_position` would silently become the origin, and a missing goal set a permanently unreachable episode that still produces a number. `start_rotation` stays in the dataset's `[x, y, z, w]` order with the reorder to `(w, x, y, z)` happening once, in `world.set_pose`, at the only boundary where the two conventions meet.

### The invariant this ticket was told to tighten

`test_layering.py`'s `habitat_sim`-importer check is now an **equality**, not a subset. Both directions fail: a second file reaching for the simulator, and `sim/world.py` ceasing to be the file that owns it. `test_walker_scope.py` also pins `sim/world.py` in the walked set, because a walker that stopped reaching it would make the one-importer test pass by finding nothing to check.

### One correction to a neighbouring invariant

`test_no_env_flags.py` asserted `guard.py` had **exactly one** `os.environ` reach. Ticket 21 needed a second — `assert_habitat_logging_pinned()`, the check `sim/world.py` makes immediately before `import habitat_sim`, which lives in `guard.py` precisely so that module never touches the environment itself. The count-of-one assertion failed, which was the right alarm and the wrong question. It now names the **enclosing function** of every reach, so a third entry has to be added deliberately and a top-level `os.environ` anywhere in the module fails as `<module>`.

### Disclosed, and the box settles it

`World` puts the `AudioSensorSpec` through `AgentConfiguration.sensor_specifications` rather than `sim.add_sensor` after construction. `Agent.__init__` routes both through the identical `SensorFactory.create_sensors` (`agent/agent.py:158-171`) and tickets 04 and 16 proved the `add_sensor` form — but "same code path on habitat-sim 0.3.3" is a **cross-version inference** until it runs on the 2022-era branch the box builds. `test_the_audio_spec_arrives_through_the_agent_config` is the check; if it comes back red the fallback is one line and costs the audio-blindness. Worth noting the risk is *guarded* rather than silent either way: if the agent-config path yielded an empty mesh, `arm_audio_context` raises with a diagnosis.
