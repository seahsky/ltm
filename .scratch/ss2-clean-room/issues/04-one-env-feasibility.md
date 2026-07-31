# 04 — One-env feasibility: can the audio build hold the rest of the stack?

Type: task
Status: open
Blocked by: none

## Question

Can a single conda environment on RACE hold an audio-capable habitat-sim build **and** everything else the rebuilt agent needs to import, or does "purely SoundSpaces 2.0" force something to be dropped or served out of process?

## Why it matters

This is the gate the entire map hangs off.
The two-env split we are trying to kill was not an accident: `race-soundspaces-spike.sh` says explicitly that the working `ltm-embodied` env is never touched, because the SoundSpaces habitat-sim branch predates our pins.

The known constraints:
- habitat-sim `RLRAudioPropagationUpdate` branch, built with `--audio`
- Python 3.9
- numpy < 1.24 (the tree is 2022-era; numpy 2.x breaks it)
- `import quaternion` must precede `import habitat_sim` (habitat-sim issue #1813)
- `libRLRAudioPropagation.so` is prebuilt, closed-source, Linux x64 only, needs GLIBC >= 2.29
- cmake 3.14, gcc-10 toolchain

The thing that makes this newly tractable: memory is out of scope for this build, so the 7B planner and the VLM captioner are no longer required imports.
The question shrinks to habitat-sim-with-audio plus torch plus CLAP plus habitat-lab.
That is a much smaller ask than the one that forced the split originally, but it is still unproven.

Note the spike script records that habitat-lab fails to import in the audio env, and that this was acceptable *because rendering was offline*.
Under live rendering it is no longer acceptable if the runner needs habitat-lab. Establish whether it does.

## What would resolve it

On the RACE box, in a fresh env:
1. Build habitat-sim `RLRAudioPropagationUpdate` with `--audio` and prove the audio probe passes (`RLRAudioPropagationChannelLayoutType.Binaural` exists and is not just a stub binding; issue #2340 notes the spec is bound even in non-audio builds, so probe the enum member, not the class).
2. Install torch (V100, so CUDA 11.x era) and CLAP on top.
3. Prove co-import in one process, in the required order, and prove torch sees the GPU.
4. Establish whether habitat-lab v0.2.2 imports, and whether the rebuilt runner actually needs it or can drive `habitat_sim` directly.

GREEN = one process, all imports, GPU visible, audio sensor constructible.
RED = the printed blocker list is the deliverable, and ticket 07 has to be re-scoped around whatever cannot coexist.

Do not reuse the existing `soundspaces-spike` env. It is a spike artifact with unknown drift; build clean so the result is trustworthy.

## Note added by ticket 02

The build is likely to carry a **local patch**, so set it up to apply one from the start rather than retrofitting later.
Ticket 02 found habitat-sim hardcodes a single audio source (`RLRA_AddSource` called once, every accessor pinned to index 0) while the engine underneath is natively multi-source with per-source IRs.
Exposing that is ~40 lines across `src/esp/sensor/AudioSensor.{h,cpp}` and `src/esp/bindings/SensorBindings.cpp` — all already compiled here, so the marginal cost is the patch file, not a second build.

Whether we actually want it is ticket 09's call, gated on ticket 06's source-count cost sweep. What this ticket owes is a build that can take a patch and record which patches were applied, so the box state is reproducible.

Also worth recording during the audio probe: the branch exposes `sourceIsVisible()`, `getRayEfficiency()`, `setListenerHRTF()`, `writeIRWave()` and `writeSceneMeshOBJ()`, none of which exist on `main`. Confirming these are present is a sharper GREEN check than the enum probe alone, because it proves the build is on the expected branch generation and not a stale checkout.

## Note added by ticket 11

Add two things to the GREEN check, both nearly free:

- **Print the `rlr-audio-propagation` submodule SHA.** The parameter research was done against that repo's archived `main`; habitat-sim pins a submodule commit that was never checked against it. Recording the SHA makes the whole parameter sheet falsifiable.
- **Dump every `acousticsConfig` field from a freshly constructed `AudioSensorSpec`.** The defaults live inside the closed `.so` via `RLRA_ContextConfigurationDefault`, so no default value in tickets 01 or 11 is verified. Three lines here converts the entire defaults column from hearsay into measurement, and ticket 06 needs it before it can sweep anything.

One gotcha this makes concrete: `AudioSensorSpec` is bound `py::dynamic_attr()`, so assigning a field name that does not exist on this branch (say `irTime`, which was renamed to `maxIRLength`) **silently attaches a new Python attribute and is never read**. There is no error. Whatever wrapper the new tree puts around this should validate config keys against the real field list and raise.
