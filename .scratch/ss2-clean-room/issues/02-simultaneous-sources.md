# 02 — Can one audio sensor render simultaneous sources?

Type: research
Status: resolved
Assignee: Sky
Blocked by: none

## Question

The anomaly-response task needs up to three concurrent sounds: the continuous background bed, the anomaly, and a room-normal distractor.
Does the SoundSpaces 2.0 audio sensor support more than one sound source at a time, and if not, what is the workaround?

## Why it matters

`setAudioSourceTransform(np.array([x, y, z]))` reads as setting *the* source, singular.
habitat-sim issue #2532 is titled "Support for Simultaneous RIR Rendering for Multiple Sound Sources" and was open with no maintainer answer when checked on 2026-07-31, which is a strong prior that the answer is no.

If it is single-source, the task design changes:
- The bed cannot be a second point source. It becomes either a diotic signal added post-render (which is what ADR-0004 already treats it as, a noise floor) or a per-step re-render with the source moved, which multiplies the cost measured in ticket 06.
- The room-normal distractor (the only thing scene-conditioned discrimination is claimed on, ADR-0004) needs an explicit mechanism.
- Cost scales linearly with source count if the workaround is sequential re-rendering, which interacts directly with the live-every-step budget.

## What would resolve it

Read the actual bindings on the `RLRAudioPropagationUpdate` branch rather than the docs: `src/esp/sensor/AudioSensor.h` / `AudioSensor.cpp` and the pybind layer.
Look for whether the source is a single member or a container, and whether `reset()` is required between source changes.
Check whether issue #2532 has since been answered.

Deliverable: a yes/no with the API evidence quoted, plus, if no, a ranked list of workarounds with their cost implication stated in units of "extra renders per step".

## Answer

**No at the habitat-sim API level. Yes at the engine level, natively.**
The single-source limit is a hardcode in habitat-sim's wrapper, not a limitation of the audio engine underneath.
Since ticket 04 builds habitat-sim from source anyway, that hardcode is ours to remove.

Sources: `facebookresearch/habitat-sim` branch **`RLRAudioPropagationUpdate`** (`src/esp/sensor/AudioSensor.{h,cpp}`, `src/esp/bindings/SensorBindings.cpp`) and `facebookresearch/rlr-audio-propagation` `RLRAudioPropagationPkg/headers/RLRAudioPropagation.h`, read 2026-07-31.
All of this is **read from source, not run on the box.** Nothing here is verified against a built binary.

### The wrapper allows exactly one source

The whole audio sensor state is two positions, both scalars:

```cpp
  RLRA_Context context;
  vec3f sourcePosition_;
  vec3f listenerPosition_;
```

Sources are added once, at context creation, and never again:

```cpp
void AudioSensor::createAudioSimulator() {
  if (context) return;
  newInitialization_ = true;
  audioSensorSpec_->acousticsConfig_.thisSize = sizeof(RLRA_ContextConfiguration);
  RLRA_CreateContext(&context, &audioSensorSpec_->acousticsConfig_);
  RLRA_AddListener(context, &audioSensorSpec_->channelLayout_);
  RLRA_AddSource(context);          // <- once, unconditionally, exactly one
}
```

Every later call pins index `0`:

```cpp
void AudioSensor::setAudioSourceTransform(const vec3f& sourcePos) {
  createAudioSimulator();
  sourcePosition_ = sourcePos;
  const float pos[3] = {sourcePos[0], sourcePos[1], sourcePos[2]};
  RLRA_SetSourcePosition(context, 0, pos);      // source index 0
}
```
```cpp
  const float* ir = RLRA_GetIRChannel(context, 0, 0, channelIndex);   // listener 0, source 0
  return RLRA_WriteIRWave(context, 0, 0, wavePath.c_str());
```

The pybind block exposes no index and no list — `setAudioSourceTransform` is bound bare:

```cpp
    .def("setAudioSourceTransform", &AudioSensor::setAudioSourceTransform)
    .def("setAudioListenerTransform", &AudioSensor::setAudioListenerTransform)
    .def("runSimulation", &AudioSensor::runSimulation)
    .def("getIR", &AudioSensor::getIR)
    .def("sourceIsVisible", &AudioSensor::sourceIsVisible)
    .def("getRayEfficiency", &AudioSensor::getRayEfficiency)
    .def("reset", &AudioSensor::reset);
```

habitat-sim `main` (a different, older API — see the branch-divergence note below) makes the intent explicit in a comment: `// [NOTE] Currently, only one source is supported`.
Issue **#2532** is still **open with no maintainer reply** as of 2026-07-31, so upstream is not about to fix this.

### The engine underneath is multi-source and gives per-source IRs

`RLRAudioPropagation.h` is unambiguous:

```c
RLRA_EXPORT RLRA_Error RLRA_AddSource( RLRA_Context context );
RLRA_EXPORT RLRA_Error RLRA_ClearSources( RLRA_Context context );
RLRA_EXPORT size_t     RLRA_GetSourceCount( const RLRA_Context context );
RLRA_EXPORT RLRA_Error RLRA_SetSourcePosition( RLRA_Context context, size_t sourceIndex, const float position[3] );
RLRA_EXPORT RLRA_Error RLRA_SetSourceRadius( RLRA_Context context, size_t sourceIndex, float radius );

RLRA_EXPORT RLRA_Error RLRA_Simulate( RLRA_Context context );
RLRA_EXPORT size_t       RLRA_GetIRChannelCount( const RLRA_Context context, size_t listenerIndex, size_t sourceIndex );
RLRA_EXPORT size_t       RLRA_GetIRSampleCount ( const RLRA_Context context, size_t listenerIndex, size_t sourceIndex );
RLRA_EXPORT const float* RLRA_GetIRChannel     ( const RLRA_Context context, size_t listenerIndex, size_t sourceIndex, size_t channelIndex );
```

`RLRA_AddSource` appends, `RLRA_GetSourceCount` implies a container, and the IR accessors are keyed `(listenerIndex, sourceIndex)` — so the engine keeps a **separate IR per source** rather than a pre-mixed one.
One `RLRA_Simulate(context)` covers all sources; there is no per-source simulate call.

### Ranked workarounds

**1. Patch the wrapper. 0 extra renders per step. Recommended.**
Expose `RLRA_AddSource` / `RLRA_ClearSources` / `RLRA_GetSourceCount` / `RLRA_SetSourcePosition(idx, …)` and an IR getter that takes a source index.
Roughly a 40-line diff across `AudioSensor.h`, `AudioSensor.cpp` and `SensorBindings.cpp` — all three of which ticket 04 already compiles from source, so the marginal cost is a patch file in the build script, not a new build.
Two properties make this more than a convenience:
- **Per-source IRs make onset provenance structural.** The `anommxv` headline was invalidated partly because the interrupt fired on the background bed at step 0 and was attributed to the alarm. With one IR per source you know which source the energy came from by construction, so that class of break cannot recur. Any mixing workaround throws this away and forces provenance to be re-inferred from a summed signal.
- **`RLRA_SetSourceRadius` becomes available**, which is the honest way to render a background bed that is a region rather than a point.

Unquantified risk: `libRLRAudioPropagation.so` is closed-source, so how the simulate cost scales with source count cannot be read off — each source shoots its own `sourceRayCount` direct rays, and the shared indirect field may or may not amortise. **Ticket 06 must sweep 1 vs 2 vs 3 sources, not just the four knobs in ticket 01.**

**2. Sequential re-render. N sources = N renders per step.**
Move the single source and re-run per sound, then sum the IRs — valid, since propagation is linear.
No patch required, and geometry is *not* re-uploaded (`createAudioSimulator` early-returns on an existing context, and `newInitialization_` is only set at context creation), so each extra render is trace-only.
But it multiplies the per-step budget directly, which is the one budget the destination hangs on, and it loses per-source provenance unless kept unsummed — at which point it is strictly worse than option 1 at 3x the cost.

**3. Post-render diotic bed. 0 extra renders.**
Add the bed as an unspatialised L/R signal after convolution. This is what ADR-0004 already treats the bed as — a noise floor.
Cheapest and defensible for a bed with no location, but it cannot produce the **room-normal distractor**, which needs a position by definition. So it is at best half a mechanism, and it weakens ADR-0002's scene-conditioned normality claim.

**Rejected: pre-render the stationary sources.** The bed and distractor do not move, but the *listener* does, every step, so their IRs change every step too. That is the precomputed grid again, which the map has already ruled out.

### Operational details this settles

- **`reset()` must not be called between source moves.** On this branch it destroys the context outright (`RLRA_DestroyContext`) and clears the IR, which forces a full geometry re-upload on the next call. Moving a source needs only `setAudioSourceTransform` then `runSimulation`.
- **`sourceIsVisible()` exists and is cheap** — a single `RLRA_TraceRayAnyHit` between source and listener. This is a free, exact line-of-sight test, which is directly the primitive the deferred "non-LOS but audible" seed design needs, and a useful provenance annotation per step.
- **`getRayEfficiency()` exists.** Ticket 06's ray-count sweep has a built-in quality readout and does not have to rely solely on eyeballing gradient monotonicity.

### Branch divergence — this invalidates part of ticket 01

Ticket 01's parameter sheet was researched against `docs/AUDIO.md` and `AudioSensor.cpp` on **`main`**. The `RLRAudioPropagationUpdate` branch we are actually building is a **different API generation**, not a minor drift:

| | `main` | `RLRAudioPropagationUpdate` (ours) |
|---|---|---|
| API style | C++ `RLRAudioPropagation::Simulator` | C `RLRA_*` on an `RLRA_Context` |
| Config struct | `RLRAudioPropagationConfiguration` | `RLRA_ContextConfiguration`, with a `thisSize` ABI field set by the caller |
| `enableMaterials` | inside `acousticsConfig` | promoted to `AudioSensorSpec.enableMaterials` |
| `reset()` | nulls a `unique_ptr` | `RLRA_DestroyContext` |
| Extra methods | — | `setListenerHRTF`, `writeIRWave`, `getRayEfficiency`, `writeSceneMeshOBJ`, `sourceIsVisible` |

So the 23-knob table in ticket 01 is **not confirmed for our branch**, and at least one entry (`enableMaterials`) is known to have moved.
Raised as ticket 11, which now blocks 06.

(Aside, not ours to fix: `main`'s `runSimulation` reads `lastSourcePos_[1], lastSourcePos_[2], lastSourcePos_[3]` — indices 1..3 of a 3-component vector. Our branch does not have this. Noted only so nobody ports a fix from `main` and inherits it.)
