# 12 — Does the HM3D semantic path hand the audio sensor an empty mesh?

Type: task
Status: open
Blocked by: 04

## Question

On HM3D-Semantics v0.2 with `enableMaterials=True`, does `joinSemanticHierarchy` fail its cast and give the audio context zero scene geometry, and does forcing the vertex-colour path fix it?

## Why it matters

Ticket 03 found a failure mode nobody had named, and it is worse than the one everyone was worried about.

The annotated HM3D config sets `has_semantic_textures: true`.
That routes the semantic asset through `loadRenderAssetGeneral` into a `GenericMeshData` (`ResourceManager.cpp:1406-1413`, `:1722`, `:1744`, `:2407`).
But `joinSemanticHierarchy`, which is where the audio sensor gets its geometry, hard-requires a `GenericSemanticMeshData` (`ResourceManager.cpp:2937-2943`).

The cast fails.
The failure is a **bare `return`**, not an exception:

```cpp
if (!meshData) {
  ESP_ERROR() << "Could not get the GenericSemanticMeshData";
  return;
}
```

So the node is silently skipped, the joined mesh comes back empty, and `loadSemanticMesh` uploads **zero vertices** to the audio context.
The simulation then runs happily and produces a direct-path-only IR with no scene in it.

This is a code-level inference read at the pinned commit, not a measurement, which is exactly why it needs the box.

The reason this is worth its own ticket rather than a footnote: **a silent zero-geometry audio context still returns plausible-looking audio.**
That is the same class of bug that invalidated the `anommxv` headline, where the interrupt fired on the background bed and nobody noticed for a whole matrix.
The clean room should not be able to enter this state without failing loudly.

Note ticket 03's other relevant finding: `semanticSceneExists()` is not a trustworthy signal either, because `ResourceManager.cpp:292-296` assigns a fresh `SemanticScene::create()` *before* attempting the load and the failure branches return false without re-nulling it.
So the flag can be true over an empty scene.

## What would resolve it

On the box, in the ticket-04 env.

1. **The reachability probe.** Load `hm3d_annotated_basis.scene_dataset_config.json` with the v0.2 annots present, construct the audio sensor with `enableMaterials=True`, run one simulation, and grep stderr for `Could not get the GenericSemanticMeshData`.
   Check `len(sim.semantic_scene.objects)`, never the `semanticSceneExists()` flag.
   Then rerun with `SimulatorConfiguration.use_semantic_textures_if_found = False` to force the vertex-colour path, and compare.
2. **Does v0.2's `.semantic.glb` still carry per-vertex colours?** Only matters if probe 1 shows the texture path is the blocker. Dump `audio_sensor.writeSceneMeshOBJ(...)` and confirm the vertex count is scene-scale rather than zero.
3. **Characterise the degraded default material.** `RLRA_WriteSceneMeshOBJ` assigns "a random color corresponding to the material" (`RLRAudioPropagation.h:452-455`), so counting distinct vertex colours in the OBJ distinguishes "one uniform default" from "a database was silently applied". Do this for HM3D materials-off and, if 08 keeps MP3D in play, MP3D materials-on.
   `RLRA_WriteIRMetrics` would give RT60/EDT/DRR/C80 directly but is **not bound to Python** on this branch, so it needs the same class of wrapper patch ticket 02 proposed. Only take that if probe 3's colour count is ambiguous.

Deliverable: a yes/no on whether the semantic audio path is reachable on HM3D at all, plus **the invariant the new tree's audio wrapper must assert at context creation** so this state is unreachable in the clean room.
That invariant is the durable output; the probe result is the input to it.

## Note on scope

This does **not** block ticket 08.
Ticket 03 already supplies 08's fidelity evidence, and the MP3D-versus-HM3D vocabulary delta (85.7 % against 51.5 %) holds either way.
If 08 chooses to keep HM3D and accept the degraded path, this ticket stops being a decision input and becomes purely a guard, which is still worth doing.
