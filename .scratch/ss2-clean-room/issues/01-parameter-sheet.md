# 01 — SoundSpaces 2.0 parameter sheet

Type: research
Status: resolved
Blocked by: none

## Question

What parameters does SoundSpaces 2.0 expose, and which of them can be adjusted to suit the anomaly-response experiment?

## Answer

Source: habitat-sim `docs/AUDIO.md` and `SoundSpaces2.md` on `main`, plus `examples/tutorials/audio_agent.py`, fetched 2026-07-31.
Defaults below are quoted from that documentation.
Recommendations are marked as such and are **not** verified on our stack.

### The whole surface

SoundSpaces 2.0 is three objects hanging off habitat-sim:

```python
spec = habitat_sim.AudioSensorSpec()
spec.uuid = "audio_sensor"
spec.outputDirectory = "/tmp/AudioSimulation"
spec.acousticsConfig = habitat_sim.sensor.RLRAudioPropagationConfiguration()   # 23 knobs
spec.channelLayout   = habitat_sim.sensor.RLRAudioPropagationChannelLayout()   # 2 knobs
sim.add_sensor(spec)

audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
audio_sensor.setAudioSourceTransform(np.array([x, y, z]))
ir = sim.get_sensor_observations()["audio_sensor"]     # (channels, samples) impulse response
```

The listener is the agent, implicitly, wherever it currently is.
That is the entire difference from what we have today: no grid, no nearest-cell resolution, no floor ambiguity.

### RLRAudioPropagationConfiguration — 23 parameters

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `sampleRate` | int | 44100 | Sample rate of the simulated audio |
| `frequencyBands` | int | 4 | Number of frequency bands simulated |
| `directSHOrder` | int | 3 | Spherical-harmonic order for direct sound (max 9) |
| `indirectSHOrder` | int | 1 | SH order for reflections (max 5) |
| `threadCount` | int | 1 | CPU threads |
| `updateDt` | float | 0.02 | Simulation time step |
| `irTime` | float | 4.0 | Maximum render time budget / IR length |
| `unitScale` | float | 1.0 | Scene scale multiplier |
| `globalVolume` | float | 0.25 | Total initial pressure |
| `indirectRayCount` | int | 5000 | Rays for indirect paths |
| `indirectRayDepth` | int | 200 | Max indirect ray depth |
| `sourceRayCount` | int | 200 | Direct rays from the source |
| `sourceRayDepth` | int | 10 | Max direct ray depth |
| `maxDiffractionOrder` | int | 10 | Edge-diffraction events (max 10) |
| `direct` | bool | true | Enable direct-ray contribution |
| `indirect` | bool | true | Enable indirect-ray contribution |
| `diffraction` | bool | true | Enable diffraction |
| `transmission` | bool | **false** | Enable transmission of rays through geometry |
| `meshSimplification` | bool | false | Edge-collapse the mesh before tracing |
| `temporalCoherence` | bool | false | Temporal IR smoothing; ~10x fewer rays; requires continuous motion |
| `dumpWaveFiles` | bool | false | Write per-band wave files |
| `enableMaterials` | bool | true | Enable acoustic materials |
| `writeIrToFile` | bool | false | Write the IR to disk |

`RLRAudioPropagationChannelLayout` adds `channelType` (default `Binaural`) and `channelCount` (default 2).
Channel types: `Mono`, `Stereo`, `Binaural` (HRTF-spatialised), `Quad`, `Surround_5_1`, `Surround_7_1`, `Ambisonics`.

### The four knobs that decide whether live-every-step is affordable

This is the binding constraint on the whole reset.
Rendering is Monte-Carlo path tracing, so results are stochastic and cost scales with rays.
At an episode budget of 500 steps, anything above ~50 ms/step turns one episode into 25+ seconds of pure audio.

1. **`irTime` (4.0 s default).**
   The single biggest cheap win. We detect an onset and climb an energy gradient; we do not need 4 seconds of reverb tail.
   *Recommendation:* 0.5 to 1.0 s. Shorter IR means fewer ray bounces to resolve and a far cheaper convolution downstream.
2. **`indirectRayCount` (5000).**
   Dominant per-step cost. *Recommendation:* sweep 5000 → 1000 → 500 and watch when the energy gradient stops being monotonic toward the source. Accuracy only has to survive the gradient climb, not sound good.
3. **`threadCount` (1).**
   Free speed and currently left on the floor. *Recommendation:* set to the box's physical core count.
4. **`temporalCoherence` (false).**
   ~10x fewer rays, but the docs warn it is inappropriate for non-continuous motion.
   Our ObjectNav actions are discrete (0.25 m steps, 30 degree turns), which is exactly the "non-continuous" case, so this is a **risk knob**, not a free win. A/B it against the gradient-climb behaviour before trusting it.

Secondary: `indirectRayDepth` (200 is very deep for a domestic room), `meshSimplification` (HM3D meshes are dense), `frequencyBands`, `indirectSHOrder`.

### The two knobs that could retire ADR-0003

ADR-0003 constrains the anomaly source to the primary goal's floor, because the offline grid renders one floor and `nearest` resolves by xz, so an off-floor source fabricates audio.
Live rendering removes the grid, and then:

- **`transmission` (default OFF).**
  Turning it ON is the actual physics of sound passing through a floor or wall. A source one floor away then produces a *real* attenuated signal instead of a fabricated one. This is the knob that makes the floor constraint a modelling choice rather than a bug workaround.
- **`diffraction` (default ON, `maxDiffractionOrder` 10).**
  What makes a source audible around a corner without line of sight. Our deferred "non-LOS but audible seed" idea depends entirely on this being on and well-ordered.

Both need a decision in ticket 09, not just a flag flip: turning transmission on changes what the task is.

### The knobs that shape what the agent hears

- **`channelType = Binaural`.** Keep. The HRTF-spatialised L/R pair is what keeps the realizable-localization arm (ADR-0001, level-sign plus energy climb) alive. Switching to `Mono` kills it; `Ambisonics` would be a richer but unbuilt path.
- **`globalVolume` (0.25).** This is the new calibration point that replaces `bg_gain`. It sets absolute pressure, and therefore where `onset_rms` has to sit relative to the background bed (ADR-0004). Expect to recalibrate the onset threshold from scratch; the old 0.065 is meaningless under a different renderer path.
- **`sampleRate` (44100).** ESC-50 and FSD50K clips are 44.1 kHz, CLAP wants 48 kHz. Pick one convention now and resample in exactly one place.
- **`enableMaterials` (true).** Depends on a material config; see ticket 03, this is not free on HM3D.

### Plumbing

`unitScale` stays 1.0 (HM3D is metres).
`writeIrToFile` and `dumpWaveFiles` are debug-only and must be OFF in the loop, since they write per-step to disk.
`outputDirectory` still has to be set even when not writing.

### What this does not answer

- Whether more than one source can play at once (bed + anomaly + distractor). See ticket 02.
- Whether materials resolve at all on HM3D `.basis.glb`, given the tutorial's material config is MP3D-keyed and our semantic sensor returns zeros. See ticket 03.
- What any of this actually costs per step on the V100 box. See ticket 06.
