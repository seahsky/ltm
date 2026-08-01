# 03 — Do acoustic materials resolve on HM3D?

Type: research
Status: resolved
Assignee: Sky
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

## Answer

**Materials are assigned by grouping scene triangles by their Habitat semantic category-name string and passing that string to `RLRA_AddMeshIndices`, where the engine picks the material with the most `labels` substring-matching the lowercased category.
For HM3D the verdict is NO by default and DEGRADED at best, and the degraded path is what the SoundSpaces authors themselves run on HM3D.**

Three independent gates each shut the material path off on their own, so this is not one fix away from working.

1. `AudioSensorSpec::enableMaterials_` is set to **`false`** in the constructor under `#ifdef ESP_BUILD_WITH_AUDIO`, so an audio build always starts materials-off regardless of the header's `= true` initialiser.
2. Plain HM3D ships only `*.basis.glb` and `*.basis.navmesh` with no `semantic_descriptor_filename` in its scene-dataset config, so `semanticSceneExists()` is false and `runSimulation` takes the `loadMesh` branch.
3. Even with HM3D-Semantics v0.2 present, the annotated config sets `has_semantic_textures: true`, which routes the semantic asset through `loadRenderAssetGeneral` into a `GenericMeshData`, and `joinSemanticHierarchy` hard-requires a `GenericSemanticMeshData`.

Gate 3 is the finding that was not on anyone's map, and it is worse than the failure it replaces. See "The trap" below.

Sources: `habitat-sim@4f61e321` (branch `RLRAudioPropagationUpdate`, the build branch), `rlr-audio-propagation@4fd446b4` (the exact submodule commit that branch pins), `sound-spaces@main`, `matterport/habitat-matterport-3dresearch` example configs, `niessner/Matterport metadata/mpcat40.tsv`.
Read from source, not run.
The structural claims below were re-verified verbatim against the pinned commits after the research landed; the coverage measurements were not independently re-run and are marked where they matter.

### 1. The mechanism

`mp3d_material_config.json` (59 KB, the only data file in the propagation package, readable without the box) holds **30 materials**, each with `name`, `absorption`, `scattering`, `transmission`, `labels`, `damping`, `density`, `speed`.

The lookup key is **`labels`**, and labels are **semantic category names**, not `.mtl` material names.
`RLRAudioPropagation.h:429-430` says `name` is decorative and ignored for matching.

```
Carpet  -> ["floor", "floor", "mat"]
Glass   -> ["blinds", "mirror", "tv_monitor", "lighting", "window"]
Curtain -> ["backpack", "clothes", "beanbag", "bed", "blanket", "cloth", ... "towel"]
Steel   -> ["bathtub", "beam", "handrail", "railing", "appliances", ... "sink"]
```

**Correction to the research: 17 of the 30 materials carry an empty `labels` list, not 8.
Only 13 materials are reachable through this lookup at all** (verified by parsing the file: 30 materials, 65 label entries, 64 unique).
The rest exist only for direct API use.

Matching is **substring, not exact** (`RLRAudioPropagation.h:426-428`): a match counts if the lowercased category name *contains* a label.
That direction matters. `"bathroom cabinet"` matches `cabinet`, but `"tv"` does **not** match the label `tv_monitor`.
On a miss the **default material** is used (`:169`, `:379`, `:390`).

The call chain in `AudioSensor::loadSemanticMesh`, verified:

1. `RLRA_SetMaterialDatabaseJSON`, but only if Python previously called `setAudioMaterialsJSON`. The flag latches and later files are refused, so the database is per-context and immutable after first mesh load.
2. `sim.getJoinedSemanticMesh(objectIds)`, giving a per-vertex `uint16` object id.
3. A per-triangle majority vote over its three vertices' object ids, resolving to `objects[id]->category()->name()` with a literal `"default"` when the object pointer is null.
4. One `RLRA_AddMeshIndices(..., catToUse.c_str())` **per distinct category name**. This is where material identity is attached.

One latent hazard worth carrying into the new tree: `objects[objectIds[...]]` indexes by raw semantic id with no bounds check, so an out-of-range id is undefined behaviour rather than an error.

### 2. `semanticSceneExists()` on HM3D

`Simulator::semanticSceneExists()` is `resourceManager_->semanticSceneExists()`, which is just `semanticScene_ != nullptr`.

**Plain HM3D: false.**
The example config's stage `default_attributes` carries only `shader_type`, `up`, `front`, `origin`.
No `semantic_descriptor_filename`, no `semantic_asset`.
The auto-construction fallback probes for `.house`, `.scn`, `_semantic.txt`, finds none, and the load fails.

**HM3D-Semantics v0.2: true.**
`hm3d_annotated_basis.scene_dataset_config.json` adds `semantic_descriptor_filename`, `semantic_asset`, and `has_semantic_textures: true`.
The build branch already has `HM3DSemanticScene.cpp`, dispatched on the file's first line reading `HM3D Semantic Annotations`, parsing `<id>,<hexRGB>,"<free-text category>",<regionID>`.
Caveat: that branch's `datasets_download.py` wires only v0.1 uids, so v0.2 must be fetched by hand. The C++ parser is version-agnostic.

**`semanticSceneExists()` is not a trustworthy test either way.**
`ResourceManager.cpp:292` nulls the pointer, then `:296` assigns a fresh `SemanticScene::create()` **before** attempting the load, and the failure branches return false **without** re-nulling it.
So a failed parse leaves the flag true over an empty scene.
The new tree should check `len(sim.semantic_scene.objects) > 1`, never the flag.

This also resolves the ticket's framing question about the all-zeros semantic sensor.
It and a false `semanticSceneExists()` are **the same event seen from two sides** when the plain config is in use, which is almost certainly what the earlier Phase-1/Phase-2 arc was running.

### 3. The trap: enabling materials on HM3D is worse than leaving them off

Verified verbatim at the pinned commit.

`ResourceManager.cpp:1406-1413`:

```cpp
bool ResourceManager::loadSemanticRenderAsset(const AssetInfo& info) {
  if (info.hasSemanticTextures) {
    return loadRenderAssetGeneral(info);   // -> loadMeshes -> GenericMeshData
  }
  return loadRenderAssetSemantic(info);    // -> GenericSemanticMeshData
}
```

`ResourceManager.cpp:2937-2943`, inside `joinSemanticHierarchy`, which is where the audio sensor gets its geometry:

```cpp
std::shared_ptr<GenericSemanticMeshData> meshData =
    std::dynamic_pointer_cast<GenericSemanticMeshData>(baseMeshData);
if (!meshData) {
  ESP_ERROR() << "Could not get the GenericSemanticMeshData";
  return;
}
```

`loadRenderAssetGeneral:1722` calls `loadMeshes:1744`, and `loadMeshes:2407` stores `GenericMeshData`.
So on HM3D v0.2 with defaults (`SimulatorConfiguration::useSemanticTexturesIfFound` defaults true, ANDed with the stage's `has_semantic_textures`), the cast fails.

Note the failure is a **bare `return`**, not an exception.
The node is silently skipped, the joined mesh comes back empty, and `loadSemanticMesh` then uploads **zero vertices** to the audio context.
The result is an audio simulation with no scene geometry at all, producing a direct-path-only IR.
The only visible signature is `Could not get the GenericSemanticMeshData` on stderr.

**Code-level inference, not measured.** It needs the box to confirm, and it is the highest-value probe on the list.
The escape hatch is `SimulatorConfiguration.use_semantic_textures_if_found = False`, forcing the vertex-colour path.
Whether v0.2's `.semantic.glb` still carries usable per-vertex colours after the move to textured annotation is **unverified**.

**Whatever the box says, the new tree must assert the audio sensor's mesh is non-empty after context creation.**
A silent empty-geometry mode that still renders plausible-looking audio is exactly the class of bug that invalidated the `anommxv` headline.

### 4. The degraded case, confirmed

`AudioSensor::loadMesh` contains no `RLRA_SetMaterialDatabaseJSON` call anywhere, verified.
It submits the entire scene in one call with a **`nullptr`** material category:

```cpp
error = RLRA_AddMeshIndices(context, sceneMesh->ibo.data(), sceneMesh->ibo.size(), 3, nullptr);
```

So the ticket's hypothesis is right and the original framing was wrong: the degraded case is **no material database in the context at all**, not a fallback to one entry in a loaded database.

What absorption the engine then applies is **not discoverable from source**.
The header exposes 40 `RLRA_*` functions and none read material properties back.
It is inside the closed `.so`.

Two incidental findings from the same read, both load-bearing elsewhere:

- `loadMesh` uses `getJoinedMesh(true)`, the **render/collision** mesh, not the semantic mesh, and it optionally folds in static collision objects. The degraded path therefore has different and arguably better geometry coverage than the semantic path.
- The whole mesh upload is wrapped in `if (newInitialization_)`, so geometry is uploaded **once per context**, not per step. Only `RLRA_Simulate` re-runs. **This is good news for ticket 06** and should be measured as such.

### 5. This is the reference configuration, not a corner case

In `sound-spaces`, every entry point sets `enableMaterials = False`: `soundspaces/simulator.py:155`, `soundspaces/continuous_simulator.py:118`, `PanoIR/render_panoIR.py:67`.
`INSTALLATION.md:71` advises setting it false if audio rendering crashes on semantic annotations.

Decisively, `render_panoIR.py` has an explicit `elif args.dataset == 'hm3d'` branch globbing `data/scene_datasets/hm3d/**/*.basis.glb`, and gates the material JSON away from it (verified):

```python
if args.dataset in ['mp3d', 'gibson']:
    audio_sensor.setAudioMaterialsJSON('data/mp3d_material_config.json')
```

**The SoundSpaces authors render HM3D with no material database.**
That is the strongest available evidence that the degraded path is the intended and supported HM3D configuration, which substantially de-risks accepting it.

### 6. If the path were reachable, the vocabulary only half matches

Measured by the research against the real `GLAQ4DNUx5U.semantic.txt` from the official v0.2 example tarball, reimplementing the header's documented substring rule.
Not independently re-verified, so treat the exact percentages as indicative.

| | matched a real material | fell to default |
|---|---|---|
| HM3D unique categories | 64 / 123 (52.0 %) | 59 |
| HM3D object instances | 467 / 907 (51.5 %) | 440 |
| MP3D mpcat40 categories | 36 / 42 (85.7 %), or 36 / 38 (94.7 %) excluding `void`/`unlabeled`/`misc`/`objects` | 6 |

The misses are systematic, not random: `lamp` misses because mpcat40 says `lighting`, `pillow` because mpcat40 says `cushion`, `couch` because mpcat40 says `sofa`, `tv` because the label is `tv_monitor` and the substring test runs the wrong way.
`glass` geometry misses because the `Glass` material has **no `glass` label** (verified).
`refrigerator` can never match because the shipped file fuses two labels into `"piperefrigerator"` (verified).
Substring matching also produces 8 ambiguous ties on that one HM3D scene (`shower wall` at 14 instances ties `Tile, Ceramic` against `Gypsum Board`) versus 1 across all of mpcat40; the tiebreak is inside the `.so` and is unverified.

### 7. Fidelity cost, in the terms this experiment actually cares about

The experiment needs a detectable **onset** and a received-energy **gradient** monotone enough to climb.
It does not need realism.

A material supplies per-frequency absorption, scattering, transmission and damping.
It therefore changes only the coefficients at surface interactions.
It changes nothing about geometry, ray topology, occlusion structure, or the direct path's distance law.

| Property | Changed by uniform materials? | Experiment depends on it? |
|---|---|---|
| Direct-path level vs distance | No | Yes, and safe |
| Occlusion step at doorways and walls | No, geometric (`sourceIsVisible` uses `RLRA_TraceRayAnyHit`, never a material) | Yes, and safe |
| Diffraction around edges | Magnitude only, not structure | Yes, and safe |
| RT60 and room character | Yes, erased. A tiled bathroom and a carpeted bedroom of equal volume become identical | No |
| Frequency colouration across bands | Yes | Only if a classifier keys on received spectral tilt, which would then need recalibrating on the uniform world |
| Absolute received level | Yes, an uncalibrated offset | No, provided every threshold is calibrated on the same uniform world the agent runs in |
| Gradient **contrast** | Compressed if the engine default is reflective | **Yes. This is the one thing worth measuring rather than arguing** |

Under Sabine, RT60 still varies room to room with `V/S` even at uniform absorption, so a uniform world keeps room-scale variation and loses only the furnishing-dependent part.
The genuine risk is quantitative: if the built-in default is highly reflective, the late diffuse field flattens within each room, energy encodes *which room* more than *where in the room*, and the final-metres gradient compresses.
If it is absorptive, direct and early energy dominate and the gradient is actually cleaner.
Both are climbable. They differ in contrast, which is signal-to-noise for the onset threshold.

One caveat that undercuts the occlusion argument: `transmission`'s default is contradictory across three primary sources on the same commit (header comment says `true`, pybind docstring and `docs/AUDIO.md` say `false`).
If transmission is on with a uniform default it leaks energy through walls and reduces exactly the doorway contrast this argument leans on.
Settle it with the ticket-04 defaults dump.

**Verdict for the experiment: a uniform-absorption world still yields a climbable gradient, because every load-bearing term is geometric.
The loss is realism this experiment does not consume.
Accepting the degraded path is defensible, and it is what SoundSpaces itself does on HM3D.**

### 8. Contradictions found, called out

1. **`docs/AUDIO.md` and the pybind docstring both say `enableMaterials` defaults `true`. Both are wrong** for any `--audio` build, because the constructor overwrites the header initialiser under `#ifdef ESP_BUILD_WITH_AUDIO`. Anyone reading the docs instead of the constructor gets this backwards.
2. **The shipped tutorial `examples/tutorials/audio_agent.py:39` is broken on this branch.** It does `acoustics_config.enableMaterials = True`, but on this branch the field lives on `AudioSensorSpec`, and `RLRAudioPropagationConfiguration` is bound **without** `py::dynamic_attr()`. Code-level inference: that line raises `AttributeError`. Do not use the tutorial as a working reference.
3. **Ticket 11's "unknown config keys are silently swallowed" is half right, and the half matters.** `AudioSensorSpec` does carry `py::dynamic_attr()`, so bad keys there are swallowed exactly as warned. `RLRAudioPropagationConfiguration` does not, so bad keys there **raise**. The new tree's key validation belongs on the **spec** object specifically.
4. **`docs/AUDIO.md` on this branch is stale**, documenting `irTime`, `updateDt`, `dumpWaveFiles`, `writeIrToFile`, `outputDirectory` and the Stereo/Quad/Surround layouts, none of which exist in the branch's own header. This independently confirms ticket 11 against a second file. Treat that doc as describing an older API.
5. **This ticket's own framing was incomplete.** It assumed the HM3D question is about vocabulary overlap. Overlap is real and measured at ~51.5 %, but it is the *second* problem. The first is the empty-mesh trap in section 3, which fails before any material name is ever looked up.

### 9. Probes for the box, ordered by value

1. **Does the joined semantic mesh come back empty on HM3D v0.2?** Load the annotated config, enable materials, grep stderr for `Could not get the GenericSemanticMeshData`. Run twice, once at the `use_semantic_textures_if_found` default and once forced `False`. Check `len(sim.semantic_scene.objects)`, never the `semanticSceneExists()` flag.
2. **Does v0.2's `.semantic.glb` still carry per-vertex colours?** Only if probe 1 shows the texture path is the blocker. `audio_sensor.writeSceneMeshOBJ(...)` and confirm the vertex count is scene-scale, not zero.
3. **What is the degraded path's default material?** `RLRA_WriteSceneMeshOBJ` assigns "a random color corresponding to the material" (`:452-455`), so counting distinct vertex colours in the OBJ distinguishes "one uniform default" from "a database was silently applied". `RLRA_WriteIRMetrics` would give RT60/EDT/DRR/C80 directly but is **not bound to Python** on this branch and would need the same class of wrapper patch ticket 02 already proposed.
4. **The full defaults dump** (ticket 04 already owns this), with `transmission` called out as contradictory.
5. **The gradient-contrast measurement** that section 7 argues for but does not prove: fix a source, walk a navmesh path toward it, record broadband IR energy per step under HM3D materials-off, MP3D materials-off, MP3D materials-on. Report Spearman correlation against negative geodesic distance and the far-to-near dynamic range in dB. This belongs with ticket 06.

Full findings with all citations: [`research/03-materials-on-hm3d-findings.md`](../research/03-materials-on-hm3d-findings.md).
