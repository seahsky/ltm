# 21 — `sim/world.py` and the ObjectNav episode loader

Type: task
Status: open
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
