# 12 — The audio context must never silently accept an empty mesh

Type: task
Status: open
Blocked by: none (04 discharged; rescoped by 08 — no longer box-gated)

**Rescoped 2026-08-01 by ticket 08.** Originally "does the HM3D semantic path hand the audio sensor an empty mesh?" — a probe of the `enableMaterials=True` path. Materials are now permanently off (ADR-0007), so the probe is dropped and only the guard survives. The original question and its evidence are kept below for provenance; **read the ticket-08 note at the bottom first — it is the live scope.**

## Question

What must the new tree's audio wrapper assert at context creation so that a zero-geometry audio context, a swallowed config key, or an unchecked `RLRA_Error` is impossible rather than merely unlikely?

## Original question (superseded, kept for provenance)

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

## Note added by ticket 04 (now resolved — this ticket is unblocked)

**The gate did not test this trap, but it handed the experiment its control arm.**

Ticket 04's live render took the non-semantic path end to end, and logged the mesh it fed the audio context:

```
[Audio] Semantic scene does not exist or materials are disabled, will use default material
[Audio] Loading non-semantic mesh
Vertex count : 392356 , Index count : 1185054
```

So the control is **392,356 verts / 1,185,054 indices** on `minival/00800-TEEsavR23oF`, and it renders a non-silent IR (`ir_peak_abs` 0.163, `ray_efficiency` 0.548).
That converts this ticket's question from a source-reading argument into a **direct comparison**: run the same scene with `enableMaterials=True` on HM3D-Semantics v0.2 and read the vertex count off the same log line. Zero verts confirms the cast failure and the bare `return`; a non-zero count refutes it.
That is a much cheaper first move than the OBJ-colour proxy, and it needs no new instrumentation — the count is already printed at `AudioSensor.cpp(499)`.

**One complication, and it is the reason to check the data before the code.** The scene the gate ran has no semantics on disk at all:

```
SSD File Naming Issue! Neither ... TEEsavR23oF.basis.scn nor ... info_semantic.json exist on disk
The active scene does not contain semantic annotations : activeSemanticSceneID_ = 0
```

So on this box, that minival scene cannot exercise the trap — with no semantic asset there is nothing to mis-cast, and `enableMaterials=True` would fall through to the same default-material path. **This ticket needs a scene that actually has HM3D-Semantics v0.2 annotations present**, which is exactly what ticket 05's inventory counts (`semantic_glb` / `semantic_txt` per split). Confirm the data exists before concluding anything about the code path.

## Note added by ticket 08 (resolved 2026-08-01) — THIS TICKET IS RESCOPED

**HM3D stays and acoustic materials are permanently off** (`docs/adr/0007-hm3d-stays-mp3d-out-of-scope.md`). MP3D is out of scope entirely.

That guts the probe half of this ticket and sharpens the guard half.

**DROPPED — do not run:**

- Probe 1, the `enableMaterials=True` reachability check on HM3D-Semantics v0.2. It measures a path the clean room has now decided never to take. Whether the cast fails is no longer a decision input for anyone.
- Probe 2, the per-vertex-colour check, which was conditional on probe 1.
- Probe 3's MP3D materials-on arm — MP3D is out of scope.
- The `RLRA_WriteIRMetrics` wrapper patch. It was only ever a tie-break for probe 3.

**KEPT, and now the whole ticket:** the invariant this ticket already named as its durable output.

The audio wrapper must fail loudly at context creation on all three of:

1. **Non-empty audio mesh.** Ticket 04's control is `Vertex count : 392356 , Index count : 1185054` on the non-semantic path for `minival/00800-TEEsavR23oF`, printed at `AudioSensor.cpp(499)`. Nothing currently checks it.
2. **Every `RLRA_Error` checked.** The engine's failures are bare returns, not exceptions.
3. **Unknown spec keys rejected.** Per ticket 04, measured: `AudioSensorSpec` silently swallows unknown keys (`py::dynamic_attr`) while `acousticsConfig` raises — so the validator goes on the **spec** and nowhere else.

**Why this matters more under the materials-off decision, not less.** The path the clean room actually runs is the non-semantic one, and a zero-geometry context on *that* path still returns plausible-looking audio with no error. That is the same failure class that invalidated the `anommxv` headline, where the interrupt fired on the background bed for a whole matrix before anyone noticed. Materials being off removes the *cause* this ticket originally investigated; it does not remove the *state*.

**Type changes from `task` to a build item.** There is no longer a measurement to take, so this is no longer box-gated: it is the assertion suite the new tree's audio wrapper ships with. It overlaps the map's *Not yet specified* requirement 1 by design — that entry is the specification, this ticket is where it gets built and tested.

`Blocked by: 04` is discharged either way; nothing here needs the box before the package layout exists.
