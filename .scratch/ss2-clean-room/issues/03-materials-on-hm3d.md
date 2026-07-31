# 03 — Do acoustic materials resolve on HM3D?

Type: research
Status: open
Blocked by: none

## Question

How does SoundSpaces 2.0 assign acoustic materials to scene geometry, does that path work for HM3D, and what happens when it does not?

## Why it matters

The habitat-sim audio tutorial runs on **MP3D**, loads `mp3d.scene_dataset_config.json`, enables the semantic mesh, and points at a material config at `src/deps/rlr-audio-propagation/RLRAudioPropagationPkg/data/mp3d_material_config.json`.
Every part of that is MP3D-shaped.

Our scenes are HM3D `.basis.glb`, and this project has already established that the HM3D semantic sensor returns all-zeros (it is the bug that produced degenerate captions through the whole Phase-1/Phase-2 arc).
If material assignment rides on semantic categories, `enableMaterials=true` on HM3D may silently fall back to a single default material, which means every surface has identical absorption and the acoustics carry no room character at all.

That is not necessarily fatal for us. Our signal is onset detection plus an energy gradient, not realism. But it must be a known, stated property rather than a silent default, because "the room sounds like a box" would undercut any acoustic-realism claim in the paper.

It also feeds ticket 08: if HM3D cannot carry materials and MP3D can, that is a real argument for changing scene dataset while the tree is being rebuilt anyway.

## What would resolve it

- Read `mp3d_material_config.json` and find what it is keyed on (semantic category, `.mtl` material name, or something else).
- Determine the fallback when a lookup misses.
- Determine whether HM3D `.basis.glb` exposes anything usable, and whether HM3D-semantics v0.2 would change that.
- Compare against what MP3D would give us.

Deliverable: the assignment mechanism stated concretely, a yes/no/degraded verdict for HM3D, and the fidelity cost of the degraded case in terms our experiment cares about (does the energy gradient still point at the source).

## Note added by ticket 11

Two branch facts sharpen this before the research starts.

**Materials are OFF by default on our branch.** Ticket 01 read `enableMaterials` as defaulting `true` from `main`'s docs, but on `RLRAudioPropagationUpdate` it moved to `AudioSensorSpec.enableMaterials` and the constructor sets it to **`false`**. So the question is not only "do materials resolve on HM3D" but "we must opt in first", and any prior run that assumed materials were on was wrong.

**The material database is only ever loaded on the semantic path.** From `runSimulation`:

```cpp
if (audioSensorSpec_->enableMaterials_ && sim.semanticSceneExists()) {
    loadSemanticMesh(sim);     // calls RLRA_SetMaterialDatabaseJSON, then AddMeshVertices/Indices/AddObject/FinalizeObjectMesh
} else {
    loadMesh(sim);             // same mesh calls, but NO RLRA_SetMaterialDatabaseJSON at all
}
```

`loadMesh` never calls `RLRA_SetMaterialDatabaseJSON`. So the degraded case is not "materials resolve to a default entry" — it is **no material database in the context whatsoever**, and whatever the engine's built-in default absorption is applies uniformly. That is a cleaner, more testable claim than the "silent fallback to one material" the ticket originally hypothesised, and it means the question splits in two:

1. Does `sim.semanticSceneExists()` even return true for HM3D `.basis.glb`? If not, the material path is unreachable regardless of what the JSON is keyed on, and `enableMaterials` is inert.
2. If it is reachable, what is `mp3d_material_config.json` keyed on, and does HM3D produce those keys?

Note the material JSON ships in the propagation package (`RLRAudioPropagationPkg/data/mp3d_material_config.json`) — the only data file in that repo — so it is readable without the box.

The A/B that settles the fidelity question is cheap and belongs with ticket 06's timing work: render the same source/listener pair with `enableMaterials` off and on, and compare both the IR and the render time. If they are identical, materials are provably inert on HM3D.
