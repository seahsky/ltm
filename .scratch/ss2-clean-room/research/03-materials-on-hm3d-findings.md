# 03: Acoustic materials on HM3D in SoundSpaces 2.0

Label: `wayfinder:research`
Effort: `ss2-clean-room`
Resolved: 2026-08-01

## Headline verdict

SoundSpaces 2.0 assigns acoustic materials by **grouping scene triangles by their Habitat semantic-object category-name string and passing that string as the material-category argument to `RLRA_AddMeshIndices`, where the engine picks the material whose `labels` have the greatest number of substring matches against the lowercased category name**.
For HM3D the call is **DEGRADED, and in the default clean-room configuration it is a hard NO**.
Degraded because HM3D-Semantics emits raw free-text labels while `mp3d_material_config.json` is authored against MP3D's mpcat40 vocabulary, so on a real annotated HM3D scene only **51.5 % of annotated object instances** match any non-default material and the rest silently fall back to the engine default.
A hard no because three independent gates each shut the path off on their own: `AudioSensorSpec::enableMaterials_` is set to `false` in the constructor on the build branch, plain HM3D ships no semantic scene at all so `semanticSceneExists()` returns false, and even with the annotated dataset config the v0.2 **texture-based** semantic representation routes the semantic asset to `GenericMeshData`, which makes the `dynamic_pointer_cast<GenericSemanticMeshData>` inside `joinSemanticHierarchy` fail and hand the audio sensor an **empty** mesh.
The degraded case is confirmed exactly as the ticket suspected: `loadMesh` never calls `RLRA_SetMaterialDatabaseJSON`, so there is **no material database in the context at all**, not a fallback to one entry.
For this experiment that is acceptable, because the signal it depends on is onset plus a received-energy gradient, and both are dominated by geometry-driven direct and early-reflection energy rather than by per-surface absorption.

## Provenance

Every claim below was read from one of these, at these exact commits.

| Source | Branch | Commit | Date |
|---|---|---|---|
| `facebookresearch/habitat-sim` | `RLRAudioPropagationUpdate` | `4f61e321477708fa606fbd8f42b4bef41d67c672` | 2022-11-04 |
| `facebookresearch/habitat-sim` | `main` (for v0.2 dataset entries only) | `57ee4941dc4765240f0f91f70b2c97a919bf9038` | 2026-05-07 |
| `facebookresearch/rlr-audio-propagation` | default | `4fd446b4abb5c71fb7a232a083bbddd65f25fc6f` | 2022-11-01 |
| `facebookresearch/sound-spaces` | default | `287184fd7067a0385558492716355c54875500ee` | 2023-09-28 |
| `matterport/habitat-matterport-3dresearch` example configs + v0.2 semantic annots | `main` | fetched 2026-08-01 | n/a |
| `niessner/Matterport` `metadata/mpcat40.tsv` | `master` | fetched 2026-08-01 | n/a |

The habitat-sim build branch pins the propagation submodule at exactly the commit read here.

```
$ git ls-tree HEAD src/deps/
160000 commit 4fd446b4abb5c71fb7a232a083bbddd65f25fc6f  src/deps/rlr-audio-propagation
```

So the header and the material JSON quoted below are the ones this project actually builds against.

The `mp3d_material_config.json` and `libRLRAudioPropagation.so` are the only shipped data and binary in that repo, and the `.so` is Linux-x64 only per its `README.md`.

## 1. The assignment mechanism, concretely

### The material database file

`rlr-audio-propagation@4fd446b:RLRAudioPropagationPkg/data/mp3d_material_config.json` is a single-line JSON with exactly one top-level key.

```json
{"materials": [{"name": "Default", "absorption": [20.0, 0.1, 20000.0, 0.1],
  "scattering": [20.0, 0.5, 20000.0, 0.5], "transmission": [20.0, 0.0, 20000.0, 0.0],
  "labels": ["default"], "damping": [...], "density": 998.65, "speed": 1483.96}, ...]}
```

It holds **30 materials**, each with keys `name`, `absorption`, `scattering`, `transmission`, `labels`, `damping`, `density`, `speed`.

The lookup key is **`labels`**, and `labels` are **semantic category names**, not `.mtl` material names.
The header is explicit that `name` is decorative.

> NOTE: the material "name" attribute is not used for matching material categories to materials, is intended as human-readable name, and will be ignored.
>
> `RLRAudioPropagation.h:429-430`

Representative entries, showing the vocabulary is MP3D's.

| `name` | `labels` |
|---|---|
| `Default` | `["default"]` |
| `Gypsum Board` | `["wall"]` |
| `Carpet` | `["floor", "floor", "mat"]` |
| `Acoustic Tile` | `["ceiling"]` |
| `Tile, Ceramic` | `["shower-stall", "shower", "toilet"]` |
| `Glass` | `["blinds", "mirror", "tv_monitor", "lighting", "window"]` |
| `Foliage` | `["indoor-plant", "plant"]` |
| `Curtain` | `["backpack", "clothes", "beanbag", "bed", "blanket", "cloth", "clothing", "comforter", "cushion", "curtain", "handbag", "scarf", "sofa", "bag", "set-of-clothing", "towel"]` |
| `wood, Thick` | `["chair", "furniture", "chopping-board", "countertop", "counter", "shelving", "desk", "door", "seating", "chest_of_drawers", "stairs", "nightstand", "board_panel", "shelf", "stool", "table", "table-runner", "wardrobe"]` |
| `Steel` | `["bathtub", "beam", "handrail", "railing", "appliances", "major-appliance", "microwave", "piperefrigerator", "gym_equipment", "sink"]` |

`tv_monitor`, `chest_of_drawers`, `board_panel`, `gym_equipment` are mpcat40 names verbatim.
`beanbag`, `rug`, `mat` are the WordNet sub-keys listed in `mpcat40.tsv`'s `wnsynsetkey` column for the `chair` and `floor` rows.
There is one obvious data typo in the shipped file: `Steel` carries the label `"piperefrigerator"`, which is `pipe` and `refrigerator` fused.
That label can never match anything, so `refrigerator` geometry in any dataset falls to default.
Eight of the thirty materials carry an **empty** `labels` list (`Brick, Painted`, `Concrete`, `Grass`, `Gravel`, `Snow`, `Water`, `Sound Proof`, and others), so they are unreachable through this lookup and exist only for direct API use.

### The matching rule

The header states the rule precisely, and it is **substring** matching, not exact-key lookup.

> A material is determined from a material category string by inspecting all materials in the database, and finding the material which has the greatest number of label substring matches.
> A match is counted if the lowercase category name contains a label as a substring.
>
> `RLRAudioPropagation.h:426-428`

And on a miss:

> If a NULL or invalid material name is provided, the default material is used for those faces.
>
> `RLRAudioPropagation.h:169` (repeated at `:379` and `:390`)

This matters for HM3D more than for MP3D.
Free-text HM3D labels like `"bathroom cabinet"` do match, because they *contain* `cabinet`.
It also produces ties the header does not say how to break.

### The call chain in `loadSemanticMesh`

`habitat-sim@4f61e32:src/esp/sensor/AudioSensor.cpp:360-492`, in order.

1. `RLRA_SetMaterialDatabaseJSON(context, audioMaterialsJSON_.c_str())` at `:367`, but **only if** `audioMaterialsJSON_.size() > 0`, that is only if Python previously called `setAudioMaterialsJSON`.
   The flag `audioMaterialsJsonSet_` latches true, and `setAudioMaterialsJSON` at `:159-171` then refuses any later file with a warning.
   The database is therefore per-context and immutable after first mesh load.
2. `sim.getJoinedSemanticMesh(objectIds)` at `:380`, giving a world-space vertex buffer, an index buffer, and a **per-vertex `uint16` object id**.
3. `RLRA_AddMeshVertices(context, sceneMesh->vbo.data(), sceneMesh->vbo.size())` at `:390`.
4. A per-triangle vote over the three vertices' object ids at `:401-444`.
   Each vertex id indexes `semanticScene->objects()`, and the string is `objects[objectIds[ibo[i]]]->category()->name()`, with the literal fallback `"default"` when the object pointer is null.
   The vote: all three equal, use it; `cat1` differs from both others, use `cat1`; otherwise use the odd one out.
   Triangles accumulate into `std::unordered_map<std::string, std::vector<uint32_t>> categoryNameToIndices`.
5. One `RLRA_AddMeshIndices(context, indices, count, 3, catToUse.c_str())` **per distinct category name** at `:454-456`.
   This is where the material name is attached.
   The fifth argument is the category string straight from the semantic annotation.
6. `RLRA_AddObject(context)` at `:476`, then `RLRA_FinalizeObjectMesh(context, objectIndex)` at `:485`.

So the key is the **Habitat `SemanticCategory::name()` string with the default (empty) mapping argument**, and nothing else.
Note that `semanticScene->categories()` is fetched at `:383-384` and never used, which is dead code.

There is a latent robustness hazard worth recording: `objects[objectIds[...]]` indexes the `objects_` vector by raw semantic id with no bounds check, so any id larger than the object count is out-of-bounds and undefined behaviour rather than a clean error.

## 2. Does `sim.semanticSceneExists()` return true for HM3D?

### What it tests

```cpp
bool semanticSceneExists() const { return resourceManager_->semanticSceneExists(); }
```
`src/esp/sim/Simulator.h:100-102`

```cpp
bool semanticSceneExists() const { return (semanticScene_ != nullptr); }
```
`src/esp/assets/ResourceManager.h:220`

`semanticScene_` is set only in `ResourceManager::loadSemanticSceneDescriptor` (`ResourceManager.cpp:288-355`).
That function returns false immediately if `ssdFilename` is empty, and otherwise requires the file to exist on disk and to parse.
`ssdFilename` comes from the stage attribute `semantic_descriptor_filename` (`StageAttributesManager.cpp:438-448`, and the auto-construction fallback at `:249-269` which probes for `.house`, `.scn`, and `_semantic.txt`).

Note the subtlety: `semanticScene_` is assigned a fresh `SemanticScene::create()` at `:296` *before* the load is attempted, and is only reset to `nullptr` at `:292` on the next call.
So a **failed parse can still leave a non-null pointer** and make `semanticSceneExists()` return true over an empty scene.
That is a distinct third failure mode from the two the ticket named.

### What plain HM3D provides

Nothing.
The plain example config, `hm3d_example_basis.scene_dataset_config.json` from `matterport/habitat-matterport-3dresearch@main:example/hm3d-example-configs.tar`, has stage `default_attributes`:

```json
{"shader_type": "flat", "up": [0,0,1], "front": [0,1,0], "origin": [0,0,0]}
```

No `semantic_descriptor_filename`, no `semantic_asset`.
Combined with the fact that plain HM3D scene folders ship only `*.basis.glb` and `*.basis.navmesh`, the auto-construction fallback finds no `.house`, `.scn`, or `_semantic.txt`, sets the hacky `.scn` default, that file does not exist, and `loadSemanticSceneDescriptor` returns false with `semanticScene_` left non-null but empty.

**So plain HM3D gives you a scene with no semantic content.**

### What HM3D-Semantics changes

It changes it completely.
`hm3d_annotated_basis.scene_dataset_config.json`, from `example/hm3d-example-semantic-configs-v0.2.tar`, sets three extra stage `default_attributes`:

```json
"semantic_descriptor_filename": "%%CONFIG_NAME_AS_ASSET_FILENAME%%.semantic.txt",
"semantic_asset": "%%CONFIG_NAME_AS_ASSET_FILENAME%%.semantic.glb",
"has_semantic_textures": true
```

The full (non-example) annotated config lists **221** stage glob paths across `example/`, `minival/`, `test/`, and `train/`, with the identical `default_attributes` block.

Per-scene, HM3D-Semantics v0.2 ships exactly two files, confirmed by extracting the official example tarball:

```
00861-GLAQ4DNUx5U/GLAQ4DNUx5U.semantic.txt
00861-GLAQ4DNUx5U/GLAQ4DNUx5U.semantic.glb
```

No `.house` and no `.scn`.

The build branch already has the HM3D loader, `src/esp/scene/HM3DSemanticScene.cpp`, dispatched from `SemanticScene.cpp:43-45` on the file's first line reading `HM3D Semantic Annotations`.
It parses one instance per line as `<id>,<hexRGB>,"<free-text category>",<regionID>` (`HM3DSemanticScene.cpp:110-142`), and `HM3DObjectCategory::name()` returns that raw string unmodified (`HM3DSemanticScene.h:19-26`).

**So yes: with `hm3d_annotated_basis.scene_dataset_config.json` and the v0.2 annots present, `semanticSceneExists()` returns true for HM3D.**

Caveat on the build branch specifically: its `datasets_download.py` wires only `v0.1` uids (`hm3d_{split}_semantic_{annots,configs}_v0.1`, lines 210-257 and the `data_groups` at `:280-297`), whereas `main` has v0.2 (`main:src_python/habitat_sim/utils/datasets_download.py:289-487`).
The C++ parser is version-agnostic, so v0.2 files load on the build branch, but you must fetch them yourself rather than via that branch's downloader.

### The two failure modes are different, and there is a third

- **`semanticSceneExists() == false`** means no semantic scene descriptor was loaded at all.
  This is what plain HM3D gives you, and it is what routes `runSimulation` to `loadMesh`.
- **Semantic sensor returns all-zeros** means a semantic scene may exist but the rendered per-pixel ids are 0, the `Unknown` object.
  `ResourceManager.cpp:2462-2470` explains the mechanism directly: a 256^3 colour lookup table is built and "Unknown entries have semantic id 0x0 (corresponding to Unknown object in semantic scene)".
  An all-zeros sensor is what you get when the semantic mesh instance is absent or when no annotation colour matches.
- **The third mode, most likely for this project's prior work**: they were on the plain config, so both symptoms had the *same* root cause.
  The all-zeros sensor and a false `semanticSceneExists()` are the same event seen from two sides.
  That is consistent with the CLAUDE.md note that the semantic sensor returned all-zeros and every caption defaulted to `"room interior"`.

### The structural blocker nobody has named yet

Making `semanticSceneExists()` true is **not sufficient** to make `loadSemanticMesh` work on HM3D v0.2, and this is the most important finding in this document.

The annotated config sets `has_semantic_textures: true`, and `SimulatorConfiguration::useSemanticTexturesIfFound` defaults to `true` (`src/esp/sim/SimulatorConfiguration.h:92`).
The stage attribute is their AND (`ObjectAttributes.h:501-504`):

```cpp
void setUseSemanticTextures(bool useSemanticTextures) {
  set("use_textures_for_semantic_rendering",
      (useSemanticTextures && getHasSemanticTextures()));
}
```

That flag becomes `AssetInfo::hasSemanticTextures` (`ResourceManager.cpp:677`), which forks the loader (`ResourceManager.cpp:1406-1413`):

```cpp
bool ResourceManager::loadSemanticRenderAsset(const AssetInfo& info) {
  if (info.hasSemanticTextures) {
    // use loadRenderAssetGeneral for texture-based semantics
    return loadRenderAssetGeneral(info);
  }
  // special handling for vertex-based semantics
  return loadRenderAssetSemantic(info);
}
```

`loadRenderAssetSemantic` stores `GenericSemanticMeshData` into `meshes_` (`:1528`, `:1557`).
`loadRenderAssetGeneral` goes through `loadMeshes`, which stores `GenericMeshData` (`:2407`, `:2416`).

And the audio sensor's mesh source, `joinSemanticHierarchy`, hard-requires the former (`ResourceManager.cpp:2937-2943`):

```cpp
std::shared_ptr<GenericSemanticMeshData> meshData =
    std::dynamic_pointer_cast<GenericSemanticMeshData>(baseMeshData);
if (!meshData) {
  ESP_ERROR() << "Could not get the GenericSemanticMeshData";
  return;
}
```

**Code-level inference, needs the box to confirm.**
On HM3D v0.2 with the annotated config and default settings, the cast fails, the joined semantic mesh comes back **empty**, and `loadSemanticMesh` then calls `RLRA_AddMeshVertices` with zero vertices.
The result is an audio context with no scene geometry at all, which is worse than uniform materials.
The observable signature is `Could not get the GenericSemanticMeshData` in the log, followed by an IR that is direct-path only.

The escape hatch exists: set `SimulatorConfiguration.use_semantic_textures_if_found = False` to force the vertex-colour path.
Whether HM3D v0.2's `.semantic.glb` still carries usable per-vertex colours after the move to textured annotation is **UNVERIFIED** and is the single highest-value probe on the box.

## 3. If reachable, do HM3D's keys match?

Measured, not asserted, using the engine's documented rule reimplemented from `RLRAudioPropagation.h:426-428`, against the real `GLAQ4DNUx5U.semantic.txt` from the official v0.2 example tarball.

That scene has **907 annotated instances across 123 unique free-text categories**.

| | matched a non-default material | fell to default |
|---|---|---|
| unique categories | 64 / 123 (52.0 %) | 59 / 123 (48.0 %) |
| object instances | 467 / 907 (51.5 %) | 440 / 907 (48.5 %) |

Categories that work, because HM3D happens to use the mpcat40 word: `wall` to `Gypsum Board`, `floor` to `Carpet` (two label hits), `ceiling` to `Acoustic Tile`, `door` and `chair` to `wood, Thick`, `bed` to `Curtain`, `toilet` to `Tile, Ceramic`, `window` and `mirror` to `Glass`, `rug` to `Carpet, Heavy`, `sink` to `Steel`, `plant` to `Foliage`.

The largest instance-weighted misses, all silently defaulting:

| count | category | why it matters acoustically |
|---|---|---|
| 71 | `unknown` | unavoidable |
| 45 | `cardboard box` | no `box` label exists |
| 40 | `lamp` | `lighting` is the mpcat40 word, `lamp` is not a label |
| 31 | `clutter` | HM3D-only word |
| 23 | `pillow` | soft, mpcat40 calls it `cushion` |
| 21 | `book` | no label |
| 17 | `bathroom accessory` | no label |
| 14 | `glass` | **the `Glass` material has no `glass` label**, only `blinds`/`mirror`/`tv_monitor`/`lighting`/`window` |
| 13 | `bar` | no label |
| 12 | `toy` | no label |

Also `tv` misses, because the label is `tv_monitor` and the rule tests whether the label is a substring of the category, not the reverse.
`couch` misses, because mpcat40 says `sofa`.

Substring matching also produces **8 ambiguous categories** where two or more materials tie on hit count, and the header does not define the tiebreak:

- `shower wall` (14 instances) ties `Tile, Ceramic` against `Gypsum Board`
- `door/window frame` (3) ties `Glass` against `wood, Thick`
- `door/window` (2), `stairs railing` (1), `stairs` (1), `sofa chair` (1), `shower door frame` (1), `shower ceiling` (1)

Tie resolution is inside the closed `.so` and is **UNVERIFIED**.

### Behaviour on a miss

Documented, three times, and consistently: the **default material** is used.
`RLRAudioPropagation.h:169`, `:379`, `:390`.
The database's own `Default` entry has flat absorption 0.1 and flat scattering 0.5 from 20 Hz to 20 kHz, with `density` 998.65 and `speed` 1483.96, which are water's values and look like placeholder data.
Whether "the default material" means that JSON entry or a hardcoded engine constant is **UNVERIFIED**, and matters only when a database is loaded at all.

## 4. The degraded case, stated precisely

**CONFIRMED, exactly as the ticket suspected.**

`AudioSensor::loadMesh` (`AudioSensor.cpp:494-540`) contains no call to `RLRA_SetMaterialDatabaseJSON`.
It does four things: `getJoinedMesh(true)`, `RLRA_AddMeshVertices`, then

```cpp
error = RLRA_AddMeshIndices( context,
    sceneMesh->ibo.data(), sceneMesh->ibo.size(), 3,
    nullptr );
```
`AudioSensor.cpp:514-516`

and finally `RLRA_AddObject` plus `RLRA_FinalizeObjectMesh`.
One index submission for the entire scene, with a **`nullptr`** material category.

Since `RLRA_SetMaterialDatabaseJSON` is called *only* from `loadSemanticMesh:367`, the degraded case is **no material database in the context at all**.
It is not "falls back to one entry in a loaded database", it is "there is no database".
The `nullptr` category then hits the documented NULL path and every face in the scene gets the engine's built-in default material.

Note also that `getJoinedMesh(true)` returns the **render/collision** mesh (`Simulator.cpp:839-846`, via `createJoinedCollisionMesh`), not the semantic mesh, and it optionally folds in static collision objects.
So the degraded path also has different, and arguably better, geometry coverage than the semantic path.

**What absorption the engine then uses is not discoverable from the header or the docs.**
`RLRAudioPropagation.h` exposes 40 `RLRA_*` functions and none of them read material properties back.
There is no `RLRA_GetMaterial`, no documented built-in constant, and no default-material section.
It is inside `libRLRAudioPropagation.so`, which is a closed prebuilt Linux-x64 binary.
Two indirect probes exist and are listed in the last section.

### The official SoundSpaces code runs exactly this degraded path

This is not a corner case, it is the reference configuration.
In `sound-spaces@287184f`:

- `soundspaces/simulator.py:155` sets `audio_sensor_spec.enableMaterials = False`
- `soundspaces/continuous_simulator.py:118` sets `audio_sensor_spec.enableMaterials = False`
- `PanoIR/render_panoIR.py:67` sets `audio_sensor_spec.enableMaterials = False`
- `INSTALLATION.md:71` advises "If the audio rendering crashes due to errors in loading semantic annotations, try to set `audio_sensor_spec.enableMaterials = False`."

And decisively, `PanoIR/render_panoIR.py` is the official script that renders IRs on HM3D (`:161-162` globs `data/scene_datasets/hm3d/**/*.basis.glb`), and it gates the material JSON to non-HM3D datasets:

```python
if args.dataset in ['mp3d', 'gibson']:
    audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
    audio_sensor.setAudioMaterialsJSON('data/mp3d_material_config.json')
```
`PanoIR/render_panoIR.py:75-77`

**The SoundSpaces authors' own HM3D rendering path does not load a material database.**
That is the strongest available evidence that the degraded path is the intended and supported HM3D configuration.

## 5. The fidelity cost, in terms this experiment cares about

The experiment needs two things: a detectable **onset**, and a received-energy **gradient** that is monotone enough along a navmesh path to be climbed.
It does not need acoustic realism.

### What the engine computes

A bidirectional path tracer producing an impulse response per (listener, source) pair (`RLRA_Simulate`, header `:461-462`).
The energy decomposes into components each gated by a flag in `RLRA_ContextConfiguration`:

- `direct`, the source-to-listener path, "Whether or not direct sound is simulated" (`:147-148`)
- `indirect`, "everything but direct sound: reflections, reverb, diffraction" (`:149-150`)
- `diffraction`, "makes occlusion smoother" (`:151-152`)
- `transmission`, "sound transmission through geometry" (`:153-154`)

`frequencyBands` is "The number of log-spaced frequency bands that the simulation uses. 4 or 8 is recommended" (`:106-107`).

### What a material actually changes

A material supplies per-frequency `absorption`, `scattering`, `transmission`, and `damping` curves.
It therefore changes **only the coefficients applied at surface interactions and during propagation through matter**.
It changes nothing about geometry, ray topology, occlusion structure, or the direct path's distance law.

### The argument, component by component

**Direct sound is material-independent.**
When the source is in line of sight, the direct component's level is set by distance and source radius, not by any surface property.
Uniform materials do not perturb it at all.
Since this experiment's agent is navigating *toward* the source, the terminal part of every successful approach is line-of-sight-dominated, and that part of the gradient is exactly as good with uniform materials as with correct ones.

**Occlusion structure survives, because it is geometric.**
Whether a wall is between agent and source is a ray-intersection fact.
`AudioSensor::sourceIsVisible()` (`AudioSensor.cpp:226-254`) computes it with `RLRA_TraceRayAnyHit` and never consults a material.
The step-change in received energy at a doorway threshold, which is the most navigationally useful feature in the field, is produced by geometry plus diffraction, not by absorption coefficients.

**Diffraction survives.**
`maxDiffractionOrder` and the edge-diffraction solver are geometric.
Diffracted energy around a door frame is attenuated by path length and edge geometry.
Materials modulate the magnitude, not the existence or the spatial shape.

**Reverberation is where the loss is, and it is a contrast loss, not a structure loss.**
Under Sabine, `RT60 = 0.161 V / (S alpha)`.
With uniform `alpha`, RT60 still varies room to room because `V/S` varies.
So a uniform-absorption world still has a room-dependent reverberant character; it loses the *furnishing*-dependent part.
A carpeted, curtained bedroom and a tiled bathroom of the same volume become acoustically identical, when in reality they differ by an order of magnitude in tail energy.

**The genuine risk is diffuse-field flattening, and it is quantitative.**
If the engine's built-in default is highly *reflective* (low absorption), the late diffuse field becomes near-uniform within each room.
Received energy then encodes *which room* you are in far more than *where in the room* you are, which compresses the within-room gradient the agent has to climb in the final metres.
If the default is highly *absorptive*, the tail is short, direct and early energy dominate, and the gradient is actually cleaner and more distance-driven than in a correctly-materialed world.
Both are climbable; they differ in contrast, which is signal-to-noise for the onset threshold and the gradient step.
The `Default` entry in the shipped JSON has absorption 0.1, which is on the reflective side, but that entry is *not* what the degraded path uses since no database is loaded.

### What actually changes, and whether this experiment depends on it

| Property | Changed by uniform materials? | Does the experiment depend on it? |
|---|---|---|
| Reverberation tail length and RT60 per room | Yes, strongly. Furnishing-driven variation is gone. | **No.** |
| Room character, tiled bathroom vs carpeted bedroom | Yes, erased. | **No.** |
| Frequency colouration across the 4 bands | Yes. Per-surface spectral tilt collapses toward the source spectrum shaped only by distance. | **No**, if the onset detector is broadband. **Yes** if any classifier keys on received spectral tilt, which would then need recalibrating on the uniform world. |
| Absolute received level | Yes, uncalibrated offset. | **No**, provided every threshold is calibrated on the same uniform world the agent runs in. |
| Direct-path level versus distance | **No.** | Yes, and it is safe. |
| Occlusion step at doorways and walls | **No**, geometric. | Yes, and it is safe. |
| Diffraction around edges | Magnitude only, not structure. | Yes, and it is safe. |
| Monotonicity of received energy along a navmesh path | Contrast is compressed, ordering is preserved for the direct-dominated regime. | Yes. This is the one to measure. |

**Verdict for the experiment.**
A uniform-absorption world still yields a climbable gradient, because the gradient's load-bearing terms (direct-path spreading, occlusion, diffraction) are geometric and untouched.
What is lost is realism the experiment does not consume: tail, room character, colouration, and absolute level.
The one thing worth measuring rather than arguing is gradient **contrast**, since a reflective default compresses it.

Two caveats on this section.
First, it is a physics argument from the header's documented feature set, not a measurement; it should be validated with the cheap probe below.
Second, the `transmission` default is genuinely contradictory across sources, see the contradictions section, and if transmission is on with a uniform default it will leak energy through walls and *reduce* the doorway contrast the argument leans on.

## 6. The MP3D comparison, for the downstream decision

Evidence only, no dataset recommendation.

### MP3D emits the exact vocabulary the database was authored for

`Mp3dObjectCategory::name()` with the default empty mapping returns the **mpcat40** name (`src/esp/scene/Mp3dSemanticScene.cpp:63-72`):

```cpp
std::string Mp3dObjectCategory::name(const std::string& mapping) const {
  if (mapping == "" || mapping == "mpcat40") {
    return mpcat40Name_;
  } else if (mapping == "raw") {
    return categoryMappingName_;
  }
  ...
}
```

`AudioSensor::loadSemanticMesh` calls `category()->name()` with no argument (`AudioSensor.cpp:408`), so MP3D feeds mpcat40 names into the matcher.
`HM3DObjectCategory::name()` under the same default returns the raw free-text label (`HM3DSemanticScene.h:19-26`).
That single line is the whole fidelity delta.

### Quantified, against the real mpcat40.tsv

`niessner/Matterport@master:metadata/mpcat40.tsv` has 42 data rows, `mpcat40index` 0 to 41.

| | matched a non-default material | fell to default |
|---|---|---|
| **MP3D mpcat40 categories** | **36 / 42 (85.7 %)** | 6 |
| **HM3D free-text categories** (GLAQ4DNUx5U) | 64 / 123 (52.0 %) | 59 |
| **HM3D object instances** | 467 / 907 (51.5 %) | 440 |

The six MP3D misses are `void`, `unlabeled`, `misc`, `objects`, `picture`, `column`.
Four of those (`void`, `unlabeled`, `misc`, `objects`) are non-categories by construction, so against the 38 usable mpcat40 classes the real coverage is **36 / 38 (94.7 %)**, with only `picture` and `column` genuinely unmapped.

MP3D has **one** ambiguous tie across the whole vocabulary (`stairs`, between `wood, Thick` and `Wood Floor`).
HM3D has **eight** on a single scene, including `shower wall` at 14 instances.

Independently, 36 of the 64 database labels are **exact** mpcat40 names.
The remainder are either mpcat40 WordNet sub-keys (`rug`, `mat`, `beanbag` from the `floor` and `chair` rows of `mpcat40.tsv`) or hyphenated variants (`base-cabinet`, `shower-stall`, `indoor-plant`, `major-appliance`, `set-of-clothing`).
The file is an mpcat40 lookup table.

### The representation delta, which is larger than the vocabulary delta

| | MP3D | HM3D v0.2 |
|---|---|---|
| per-scene semantic files | `<scene>.house`, `<scene>_semantic.ply` | `<scene>.semantic.txt`, `<scene>.semantic.glb` |
| annotation carrier | per-vertex colours in the PLY | **textures** (`has_semantic_textures: true`) |
| habitat-sim loader | `loadRenderAssetSemantic`, stores `GenericSemanticMeshData` | `loadRenderAssetGeneral`, stores `GenericMeshData` |
| `joinSemanticHierarchy` cast succeeds | **yes** | **no** (see section 2) |
| category string fed to the engine | mpcat40, 40 fixed classes | raw free text, 123 classes on one scene alone |
| SoundSpaces reference config loads the material JSON | yes (`render_panoIR.py:75-77`) | **no** (same lines, HM3D excluded) |

Also relevant: `StageAttributesManager.cpp:271-280` auto-discovers a semantic asset by probing for `_semantic.ply`, described in the comment as "for back-compat with Mp3d".
The MP3D path is the one habitat-sim's defaults were built around; the HM3D texture path is newer and, on the audio seam specifically, untested.

So the concrete fidelity delta on materials is: MP3D gets **85.7 % vocabulary coverage with near-zero ambiguity, through a mesh representation the audio sensor can actually consume**, while HM3D gets **51.5 % instance coverage with eight tie cases per scene, through a mesh representation that appears to break the audio sensor's mesh extraction entirely unless the vertex-colour path is forced**.

## Unverified, needs the RACE V100

Ordered by value.

1. **Does the joined semantic mesh come back empty on HM3D v0.2?**
   This is the blocker, and it is a code-level inference, not a measurement.
   ```python
   import habitat_sim
   cfg = habitat_sim.SimulatorConfiguration()
   cfg.scene_dataset_config_file = ".../hm3d_annotated_basis.scene_dataset_config.json"
   cfg.scene_id = ".../00802-wcojb4TFT35/wcojb4TFT35.basis.glb"
   cfg.load_semantic_mesh = True
   # cfg.use_semantic_textures_if_found = False   # flip this and rerun
   sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [habitat_sim.AgentConfiguration()]))
   print("semanticSceneExists:", sim.semantic_scene is not None)
   ss = sim.semantic_scene
   print("n_objects:", len(ss.objects), "n_categories:", len(ss.categories))
   print("sample cats:", [o.category.name() for o in ss.objects[:20] if o])
   ```
   Then run the audio sensor with `enableMaterials = True` and grep stderr for `Could not get the GenericSemanticMeshData`.
   Run it twice, once with `use_semantic_textures_if_found` left at its `True` default and once forced to `False`, and compare.

2. **Does the v0.2 `.semantic.glb` still carry per-vertex colours?**
   Only matters if probe 1 shows the texture path is the blocker.
   With `use_semantic_textures_if_found = False`, check that `sim.semantic_scene.objects` is populated *and* that the audio sensor's semantic mesh is non-empty:
   ```python
   audio_sensor.writeSceneMeshOBJ("/tmp/scene_audio.obj")
   ```
   then check the OBJ vertex count is on the order of the scene mesh, not zero.

3. **What does the degraded path's default material actually sound like?**
   Two indirect probes, both from the header.
   `RLRA_WriteSceneMeshOBJ` "will be assigned a random color corresponding to the material" (`:452-455`), so counting distinct vertex colours in the OBJ tells you **how many distinct materials the engine assigned**, which distinguishes "one uniform default" from "the database was silently applied".
   `RLRA_WriteIRMetrics` computes "RT60, EDT, DRR, C80, C50, D50, TS ... for the frequency bands that are used in the simulation" (`:490-494`), which gives the absorption's effect directly.
   Neither is bound to Python on this branch except `writeSceneMeshOBJ` (`SensorBindings.cpp:426`); `RLRA_WriteIRMetrics` is **not** bound and would need a small wrapper patch, the same class of patch ticket 02 already proposed.
   ```python
   audio_sensor.writeSceneMeshOBJ("/tmp/scene_nomat.obj")   # enableMaterials=False
   audio_sensor.writeSceneMeshOBJ("/tmp/scene_mat.obj")     # enableMaterials=True, MP3D
   # count unique 'v x y z r g b' colour triples in each
   ```

4. **Every numeric default in `RLRA_ContextConfiguration`.**
   Still unverified from ticket 11, and now also contradictory for `transmission` specifically.
   ```python
   c = habitat_sim.sensor.RLRAudioPropagationConfiguration()
   for k in ("frequencyBands","directSHOrder","indirectSHOrder","directRayCount",
             "indirectRayCount","indirectRayDepth","sourceRayCount","sourceRayDepth",
             "maxDiffractionOrder","threadCount","sampleRate","maxIRLength","unitScale",
             "globalVolume","direct","indirect","diffraction","transmission",
             "meshSimplification","temporalCoherence"):
       print(k, getattr(c, k))
   spec = habitat_sim.AudioSensorSpec()
   print("enableMaterials default:", spec.enableMaterials)   # expect False
   ```

5. **The gradient-contrast measurement that section 5 argues for but does not prove.**
   Fix a source, walk a navmesh path toward it, and record broadband IR energy per step under three conditions: HM3D materials-off, MP3D materials-off, MP3D materials-on.
   Report the Spearman correlation between energy and negative geodesic distance, and the dynamic range in dB between the far and near ends.
   That converts "is the gradient climbable" from an argument into a number, and it is the only way to settle whether the uniform default is reflective enough to matter.

6. **Tie-breaking in the substring matcher.**
   Only matters if materials are ever enabled on HM3D.
   Probe via `writeSceneMeshOBJ` colour counts on a scene containing `shower wall`.

## Contradictions with the ticket's stated context

Called out loudly, as asked.

**1. `enableMaterials` default: the ticket is RIGHT, but the header says the opposite and the docs say the opposite.**
The prior ticket's claim of `false` is correct for any `--audio` build, but the evidence is subtler than a single line.
The header declares the member with `= true`:
```cpp
bool enableMaterials_ = true;
```
`src/esp/sensor/AudioSensor.h:38`

and the constructor immediately overwrites it:
```cpp
#ifdef ESP_BUILD_WITH_AUDIO
  ...
  enableMaterials_ = false;
#endif
```
`src/esp/sensor/AudioSensor.cpp:29-37`

The member only exists under `ESP_BUILD_WITH_AUDIO`, and the constructor body always runs, so the **effective default is `false`**.
But two shipped docs on this same branch state `true`: the pybind docstring `R"(bool | true | Enable audio materials)"` at `SensorBindings.cpp:405`, and `docs/AUDIO.md`'s config table row `| enableMaterials | bool | true | Enable audio materials |`.
Anyone reading the docs rather than the constructor will get this wrong.

**2. The control flow the prior ticket quoted is EXACT.**
`AudioSensor.cpp:138-147` matches the ticket's paraphrase verbatim, including the `else` comment "Semantic scene does not exist or materials are disabled, will use default material".
It is also guarded by `if (newInitialization_)` at `:132`, meaning geometry is uploaded **once per context**, on the first `runSimulation` after `createAudioSimulator`.
That guard was not in the ticket's excerpt and matters for a live-every-step renderer: geometry is not re-uploaded per step, only `RLRA_Simulate` re-runs.

**3. The shipped habitat-sim audio tutorial is BROKEN on this branch.**
`examples/tutorials/audio_agent.py:39` does:
```python
acoustics_config.enableMaterials = True
```
On this branch `enableMaterials` lives on `AudioSensorSpec`, not on `RLRAudioPropagationConfiguration`.
`py::class_<RLRA_ContextConfiguration>` is declared **without** `py::dynamic_attr()` (`SensorBindings.cpp:293-295`), and its `def_readwrite` list ends at `temporalCoherence` (`:353-355`) with no `enableMaterials`.
**Code-level inference:** that line raises `AttributeError` at runtime.
The tutorial cannot be used as a working reference for this branch.

**4. Ticket 11's "unknown config keys are silently swallowed" is HALF right, and the half matters.**
`AudioSensorSpec` **does** carry `py::dynamic_attr()` (`SensorBindings.cpp:395`), so `audio_sensor_spec.outputDirectory = "/tmp/AudioSimulation"` at tutorial line 54 is silently accepted and never read, exactly as ticket 11 warned.
`RLRAudioPropagationConfiguration` **does not** carry it (`:293-295`), so unknown keys there **raise** rather than swallow.
The new tree's wrapper still needs key validation, but on the spec object specifically, not on the config.

**5. `transmission`'s default is contradictory across three primary sources on the same commit.**
The header comment says `RLR_Bool transmission;// = true;` (`RLRAudioPropagation.h:154`).
The pybind docstring says `bool | false` (`SensorBindings.cpp:349`).
`docs/AUDIO.md` says `false`.
The header comment is the engine's own, and the two habitat-sim docs are second-hand.
The real value comes from `RLRA_ContextConfigurationDefault` inside the `.so`.
This is the sharpest reason to run probe 4, and it directly affects section 5's occlusion-contrast argument.

**6. `docs/AUDIO.md` on this branch is stale, confirming ticket 11 against a second file.**
It documents `irTime`, `updateDt`, `dumpWaveFiles`, `writeIrToFile`, `outputDirectory`, and the `Stereo`/`Quad`/`Surround_5_1`/`Surround_7_1` channel layouts.
None of these exist in the branch's own header or bindings: the header has `maxIRLength` not `irTime` (`:132`), no `updateDt`, and the enum has only `Unknown`/`Mono`/`Binaural`/`Ambisonics` (`:60-75`).
It also documents `enableMaterials` under **Acoustics configuration** when the branch puts it on the **spec**.
Treat `docs/AUDIO.md` as describing an older API and do not use it as a reference for the clean room.

**7. The ticket's framing that the HM3D question is about vocabulary overlap is INCOMPLETE.**
Vocabulary overlap is real and measured above at 51.5 %, but it is the *second* problem.
The first is that HM3D v0.2's texture-based semantics appear to make `joinSemanticHierarchy` return an empty mesh, which would mean the semantic audio path on HM3D fails before any material name is ever looked up.
If that inference holds on the box, the vocabulary number is moot until `use_semantic_textures_if_found=False` is set.

**8. `semanticSceneExists()` can return TRUE over an empty semantic scene.**
`ResourceManager.cpp:296` creates the object before attempting the load, and the failure branches at `:313` and `:336` return false **without** resetting `semanticScene_` to null.
So `semanticSceneExists()` is not a reliable test that annotations actually loaded, and the audio sensor's gate at `AudioSensor.cpp:138` inherits that weakness.
The clean room should check `len(sim.semantic_scene.objects) > 1` rather than trusting the flag.
