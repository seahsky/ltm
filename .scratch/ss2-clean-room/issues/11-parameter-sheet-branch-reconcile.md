# 11 — Reconcile the parameter sheet against the branch we actually build

Type: research
Status: resolved
Assignee: Sky
Blocked by: none

## Question

Ticket 01 documented 23 acoustics knobs from habitat-sim `main`.
Ticket 02 established that the branch we build, `RLRAudioPropagationUpdate`, is a **different API generation**.
Which of those 23 knobs actually exist on our branch, under what names, with what defaults?

## Why it matters

Ticket 06 sweeps four knobs (`irTime`, `indirectRayCount`, `threadCount`, `temporalCoherence`) to decide whether live-every-step is affordable, and ticket 09 hangs the retirement of ADR-0003's floor constraint on two more (`transmission`, `diffraction`).
Both are reasoning off a table read from the wrong branch.

Known divergences already found in ticket 02:

| | `main` | `RLRAudioPropagationUpdate` (ours) |
|---|---|---|
| API style | C++ `RLRAudioPropagation::Simulator` | C `RLRA_*` on an `RLRA_Context` |
| Config struct | `RLRAudioPropagationConfiguration` | `RLRA_ContextConfiguration`, caller sets a `thisSize` ABI field |
| `enableMaterials` | inside `acousticsConfig` | promoted to `AudioSensorSpec.enableMaterials` |

A `thisSize` field is a versioned-struct ABI marker, which is a strong hint the config struct changed shape between generations rather than merely being renamed.
If `temporalCoherence` or `transmission` is absent, renamed, or newly defaulted on our branch, ticket 06's sweep and ticket 09's floor-constraint argument both need rewriting.

This is cheap to settle and expensive to get wrong, so it lands before 06 spends box time.

## What would resolve it

Read `RLRAudioPropagationPkg/headers/RLRAudioPropagation.h` on the `rlr-audio-propagation` submodule pinned by the `RLRAudioPropagationUpdate` branch — the header is public and the struct is declared there.
Cross-check against `src/esp/bindings/SensorBindings.cpp` on that branch for what is actually reachable from Python, and against the branch's own `docs/AUDIO.md` if it has one.

Also settle, while in the header:
- What `RLRA_ContextConfiguration` fields exist, their types and their defaults.
- Whether `thisSize` has to be set from Python or only from C++ (the wrapper sets it in `createAudioSimulator`, so probably the latter — confirm, because it decides whether the config can be built Python-side at all).
- The `RLRA_Error` enum, since it is the only failure signal the C API gives and the new tree's wrapper has to raise on it explicitly rather than ignore it.

Deliverable: a corrected parameter table for **our** branch, marked against ticket 01's so the diff is visible, and an explicit list of any ticket-01 recommendation that no longer applies.

## Answer

**Four of ticket 01's knobs do not exist on our branch, one is renamed, two are new, and the defaults column cannot be verified from source at all.**
The good news: `transmission`, `diffraction`, `temporalCoherence` and the ray counts all survive, so ticket 09's floor-constraint argument and ticket 06's cost sweep both still stand — they just need different field names.

Source: `rlr-audio-propagation` `RLRAudioPropagationPkg/headers/RLRAudioPropagation.h` (repo archived 2023-10-31, so `main` is frozen) and `habitat-sim` branch `RLRAudioPropagationUpdate`, read 2026-07-31.
**Read from source, not run.** One caveat: habitat-sim pins the propagation repo as a submodule, and the pinned commit was not checked against archived `main`. Ticket 04 should print the submodule SHA during the build.

### The config struct on our branch

```c
#pragma pack(push,1)
typedef struct {
	size_t thisSize;
	size_t frequencyBands;      size_t directSHOrder;    size_t indirectSHOrder;
	size_t directRayCount;      size_t indirectRayCount; size_t indirectRayDepth;
	size_t sourceRayCount;      size_t sourceRayDepth;   size_t maxDiffractionOrder;
	size_t threadCount;
	float sampleRate;           float maxIRLength;       float unitScale;   float globalVolume;
	float hrtfRight[3];         float hrtfUp[3];         float hrtfBack[3];
	RLR_Bool direct;       RLR_Bool indirect;           RLR_Bool diffraction;
	RLR_Bool transmission; RLR_Bool meshSimplification; RLR_Bool temporalCoherence;
} RLRA_ContextConfiguration;
#pragma pack(pop)
```

### Diff against ticket 01

| ticket 01 (`main`) | our branch | status |
|---|---|---|
| `irTime` | **`maxIRLength`** | **RENAMED.** This is ticket 01's single biggest cheap win. |
| `enableMaterials` (in acousticsConfig, default **true**) | `AudioSensorSpec.enableMaterials`, default **`false`** | **MOVED AND FLIPPED.** |
| `updateDt` | — | **GONE** |
| `dumpWaveFiles` | — | **GONE** |
| `writeIrToFile` | — | **GONE** |
| `outputDirectory` (on the spec) | — | **GONE.** `RLRA_Simulate(context)` takes no folder. |
| `sampleRate` (int) | `sampleRate` (**float**) | type changed |
| — | **`directRayCount`** | **NEW**, distinct from `sourceRayCount` |
| — | **`hrtfRight` / `hrtfUp` / `hrtfBack`** | **NEW**, see the trap below |
| — | `thisSize` | ABI marker, stamped in C++, not yours to set |
| the other 14 | unchanged names | ✓ carried over |

Channel layout also narrowed. Ticket 01 listed Mono, Stereo, Binaural, Quad, Surround_5_1, Surround_7_1, Ambisonics. Our branch has **only** `Unknown`, `Mono`, `Binaural`, `Ambisonics`:

```c
typedef enum { RLRA_ChannelLayoutType_Unknown = 0, RLRA_ChannelLayoutType_Mono = 1,
               RLRA_ChannelLayoutType_Binaural = 3, RLRA_ChannelLayoutType_Ambisonics = 7,
               RLRA_ChannelLayoutType_COUNT } RLRA_ChannelLayoutType;
```

`Binaural` with `channelCount = 2` is already the constructor default, so ticket 01's "keep Binaural, it keeps the realizable-localization arm alive" needs no action — but the Stereo and Surround fallbacks it mentioned do not exist.

### The defaults are not readable from source

The struct carries no initialisers. Defaults come from inside the closed `.so`:

```c
RLRA_EXPORT RLRA_Error RLRA_ContextConfigurationDefault( RLRA_ContextConfiguration* config );
```

and habitat-sim calls it in the spec constructor:

```cpp
AudioSensorSpec::AudioSensorSpec() : SensorSpec() {
  uuid = "audio";
  sensorType = SensorType::Audio;
  sensorSubType = SensorSubType::ImpulseResponse;
  acousticsConfig_.thisSize = sizeof(RLRA_ContextConfiguration);
  RLRA_ContextConfigurationDefault( &acousticsConfig_ );
  channelLayout_.type = RLRA_ChannelLayoutType_Binaural;
  channelLayout_.channelCount = 2;
  enableMaterials_ = false;
}
```

So **every numeric default in ticket 01's table is unverified for our branch** (`indirectRayCount=5000`, `maxIRLength`/`irTime=4.0`, `threadCount=1`, `globalVolume=0.25`, …). They may well be identical; nothing in source says so.

**Cheapest fix, and it belongs in ticket 04's GREEN check:** construct an `AudioSensorSpec` in Python and print every `acousticsConfig` field. That is three lines, it runs the moment the build imports, and it converts the whole defaults column from hearsay into measurement. Do it before ticket 06 sweeps anything.

### The HRTF basis is a trap worth naming now

`hrtfRight`, `hrtfUp`, `hrtfBack` define the listener's orientation basis, and `RLRA_SetListenerHRTF` / `setListenerHRTF()` let an HRTF be loaded.
Get this basis wrong and left/right is silently swapped — which is precisely the failure this project has already hit once, when `render_rir_grid` rendered at identity listener yaw and the agent-frame comparison of `lateral_sign` was wrong until both frames were tested and it flipped to world-frame.
Binaural cues are the entire basis of ADR-0001's realizable localization, so **the basis convention must be pinned by an experiment, not by reading the header** — put a left/right sign check into the first smoke, with the source placed at a known bearing.

### Ticket-01 recommendations that no longer apply

1. **"`irTime` 0.5 to 1.0 s"** — the recommendation stands, the field is `maxIRLength`. Setting `irTime` on our branch is not an error, it is a silently ignored attribute (`AudioSensorSpec` is bound `py::dynamic_attr()`, so a typo'd field name just attaches a new Python attribute and is never read). Worth a guard in the new tree's wrapper: validate config keys against the real field list and raise, rather than let a typo cost a run.
2. **"`writeIrToFile` and `dumpWaveFiles` must be OFF in the loop"** — moot, neither exists. Disk writes are now explicit calls (`RLRA_WriteIRWave`, `RLRA_WriteIRMetrics`, `writeSceneMeshOBJ`) that you simply do not make per step. Strictly safer than a flag that could be left on.
3. **"`outputDirectory` still has to be set even when not writing"** — moot, gone.
4. **"`enableMaterials` (true). Depends on a material config"** — it is `AudioSensorSpec.enableMaterials` and defaults to **`false`**, so materials are OFF unless explicitly enabled. This is ticket 03's problem and it changes its shape: the question is no longer only "do materials resolve on HM3D" but "we must opt in first". Note `loadMesh` (the non-semantic path) never calls `RLRA_SetMaterialDatabaseJSON` at all — only `loadSemanticMesh` does, and that path is taken only when `enableMaterials_ && sim.semanticSceneExists()`.
5. **Add `directRayCount` to ticket 06's sweep.** It is new, it is a ray count, and ticket 01 never saw it.

### Other useful API surface found while in the header

- `RLRA_ResetContext` exists and is presumably cheaper than the destroy/recreate that `AudioSensor::reset()` does. Relevant if the runner needs a per-episode reset without re-uploading geometry.
- `RLRA_GetIRCount`, `RLRA_ClearListeners`, `RLRA_GetListenerCount`, `RLRA_SetListenerRadius` — the engine is multi-**listener** as well as multi-source (ticket 02).
- `RLRA_TraceRayFirstHit` alongside the `AnyHit` that backs `sourceIsVisible()` — gives distance to first occluder, not just a boolean, if occlusion needs to be graded.
- `RLRA_WriteIRMetrics` — writes IR metrics directly; may cover part of ticket 06's quality readout without hand-rolled analysis.
- `RLRA_Error` is the only failure signal (`RLRA_Success = 0`, plus `InvalidParam`, `BadSampleRate`, `MissingDLL`, `BadAlignment`, `Uninitialized`, `BadAlloc`, `UnsupportedFeature`). habitat-sim checks it in only some paths. **The new tree's wrapper must check every return and raise**, per the repo's "always raise errors explicitly" rule — a silently-ignored `RLRA_Error_UnsupportedFeature` is exactly how a knob like `temporalCoherence` would appear to be set while doing nothing.
