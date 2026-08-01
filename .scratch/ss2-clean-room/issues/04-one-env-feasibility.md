# 04 — One-env feasibility: can the audio build hold the rest of the stack?

Type: task
Status: resolved
Assignee: Sky
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

## Answer

**GREEN. One env holds it.** `nrun bash .scratch/ss2-clean-room/probes/oneenv_gate.sh` on `riftvm`, 2026-08-01, exit 0 in 24m50s. Env `ss2` at `/home/riftuser/miniconda3/envs/ss2`.

The two-env split is dead. habitat-sim-with-audio, torch-on-GPU and the CLAP stack coexist in one interpreter, the audio sensor renders a non-silent IR in a real HM3D scene, and **the numpy pin held through every layer** (1.23.5 after habitat-sim, after torch, and after transformers+scipy) — which was the actual question, and the one `race-soundspaces-spike.sh` never asked.

| | |
| --- | --- |
| python / numpy | 3.9.19 / 1.23.5, pin held after all three layers |
| habitat_sim | 0.2.2, audio-capable, all five branch-generation methods present |
| torch | 2.0.1+cu117, CUDA available, **Tesla V100-SXM3-32GB**, cap 7.0, 31.73 GB, alloc smoke OK |
| transformers / scipy | 4.57.6 / 1.13.1 |
| habitat-lab | not importable — **as designed**, it was excluded from the gate and merely measured |
| GLIBC | 2.39 (>= 2.29, so the prebuilt `.so` loads) |
| cores | **4** |

**Provenance** (this is what makes tickets 01/11's parameter sheet falsifiable):
`habitat-sim RLRAudioPropagationUpdate @ 4f61e321477708fa606fbd8f42b4bef41d67c672`, `rlr-audio-propagation @ 4fd446b4abb5c71fb7a232a083bbddd65f25fc6f`. No patches applied — stock branch.

### The defaults, measured

Every number below came out of a freshly constructed `AudioSensorSpec` on the real build. Tickets 01 and 11 quoted these from docs; the docs were reading a different branch generation and no value was verified. **Ticket 06 blocks on this table.**

```
diffraction 1        directRayCount 500     directSHOrder 3      direct 1
frequencyBands 4     globalVolume 1.0       indirect 1           indirectRayCount 5000
indirectRayDepth 200 indirectSHOrder 1      maxDiffractionOrder 10
maxIRLength 4.0      meshSimplification 0   sampleRate 44100.0
sourceRayCount 200   sourceRayDepth 10      temporalCoherence 0  threadCount 1
transmission 1       unitScale 1.0
channelLayout: Binaural, channelCount 2   |   layouts available: Mono, Binaural, Ambisonics
```

Four things this settles that were open:

1. **`transmission` defaults to `1` (ON).** The three primary sources contradicted each other at the same commit; the header comment was right, the pybind docstring and `AUDIO.md` were wrong. **This is not cosmetic** — ticket 03's argument that the energy gradient survives uniform materials leans on doorway occlusion contrast, and transmission-on with a uniform default leaks energy through walls and reduces exactly that contrast. Ticket 09 inherits this.
2. **`enableMaterials` prints `False`**, on the spec, exactly as ticket 03 predicted from the constructor. The build is what we think it is.
3. **`maxIRLength` present, `irTime` absent**, `directRayCount` present — ticket 11's rename confirmed against the binary rather than the header.
4. **The `dynamic_attr` trap, measured not restated**: the **spec** swallows unknown keys (`True`), `acousticsConfig` does **not** (`False`). Ticket 03 was right to split ticket 11's warning. **The new tree's key validator belongs on `AudioSensorSpec` specifically**, and nowhere else.

`multi-source surface: none` — ticket 02's source read confirmed against the built binary. The ~40-line patch is still the only route to concurrent sources, and this build is stock, so it is not taken yet (ticket 09's call).

### One render, and an early warning for ticket 06

`first_render_s = 0.6013`, `ir_shape = [2, 72300]`, non-silent (`ir_peak_abs` 0.163), `ray_efficiency` 0.548, on `minival/00800-TEEsavR23oF`. Mesh: 392,356 verts / 1,185,054 indices.

Two readings worth carrying into ticket 06, both flagged for what they are:

- **Inferred, not measured:** the geometry upload is a small fraction of that 0.60 s. The log timestamps put `createAudioSimulator` → `loadMesh` vertex count at ~17 ms, which would leave **~0.58 s in the Monte-Carlo simulate itself**. If that is the steady-state per-step cost, a 500-step episode is ~5 minutes of pure audio — which is ticket 06's own "forces the throttled variant" case, not its tolerable one. Ticket 06 must measure steady-state directly; do not take this inference as the number.
- **`threadCount` is a much weaker lever than assumed.** The map calls it "a free speed knob currently set to 1". This box has **4 cores**, so the ceiling is ~4x, not the order of magnitude that framing implies. The real levers are `indirectRayCount` (5000), `indirectRayDepth` (200), `maxIRLength` (4.0) and `temporalCoherence` (0, i.e. off).

Also a concrete design fact for the new tree: **the IR is trimmed to its actual decay, not to `maxIRLength`.** 72,300 samples at 44.1 kHz is 1.64 s against a 4.0 s `maxIRLength` (176,400). Anything downstream that assumes a fixed-width IR buffer is wrong.

### The one crack in the GREEN: CLAP cannot actually run

The gate reports `clap_symbols_importable: True`, but transformers printed its own verdict immediately above that line:

```
Disabling PyTorch because PyTorch >= 2.1 is required but found 2.0.1+cu117
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be
available and only tokenizers, configuration and file/data utilities can be used.
```

**transformers 4.57.6 has disabled its PyTorch backend**, so `ClapModel` is a dummy object. The symbol imports; the model cannot instantiate. This is precisely the false-positive class the gate was designed to avoid elsewhere (issue #2340: probe the member, not the class) — the CLAP check probed importability, which a dummy passes.

Precisely what is and is not established: the quoted message is transformers' own primary output, so the backend is definitively off. It was **not** directly exercised — `--load-clap` was not passed, so no `from_pretrained` call was made.

The `transformers>=4.30,<5` pin guarded the wrong end. The upper bound was set to stop a major bump changing the API; the failure came from the **lower** bound being loose enough to resolve 4.57, which requires `torch>=2.1` against a `torch==2.0.1` that this ticket's own comment flagged as an unverified V100-*era* guess.

**This does not retract the GREEN.** The coexistence question — the hard part, and the thing the whole map hangs off — is answered yes. This is a pin mismatch between two layers that both installed fine, not an architecture problem. It is ticket 13.

### What this unblocks

- **06** (blocked by 04 + 11, both now resolved) — joins the frontier with the real defaults table and the ~0.58 s warning.
- **07** and **12** — unblocked.
- **12 gets a free anchor**: this run took the *non-semantic* path (`enableMaterials=False`, and the scene has no semantics on disk — neither `.basis.scn` nor `info_semantic.json`, "The active scene does not contain semantic annotations"). So it does **not** test the empty-mesh trap, but it establishes the control: **392,356 verts on the non-semantic path**. If the semantic path yields zero, the trap is confirmed by direct comparison.
- **05** is partly answered incidentally — GLIBC 2.39, 4 cores, V100-32GB, driver 580.159.03. Its remaining value is the *existing-state* half: envs, disk, HM3D mesh coverage, weights, spike dir.

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

## Note added by ticket 03

Two things for the GREEN check, and one warning.

**Do not use `examples/tutorials/audio_agent.py` as the reference.** It is broken on this branch.
Line 39 does `acoustics_config.enableMaterials = True`, but on `RLRAudioPropagationUpdate` that field lives on `AudioSensorSpec`, and `RLRAudioPropagationConfiguration` is bound **without** `py::dynamic_attr()` (`SensorBindings.cpp:293-295`), so the line raises `AttributeError` rather than being swallowed.
`docs/AUDIO.md` on the same branch is stale too, documenting `irTime`, `updateDt`, `dumpWaveFiles`, `writeIrToFile`, `outputDirectory` and the Stereo/Quad/Surround layouts, none of which exist in the branch's own header.
Read the constructor and the bindings, not the docs.

**Ticket 11's swallowing warning needs splitting.** `AudioSensorSpec` **does** carry `py::dynamic_attr()` (`SensorBindings.cpp:395`), so bad keys there are silently attached and never read, exactly as warned.
`RLRAudioPropagationConfiguration` **does not**, so bad keys there raise.
The wrapper's key validation belongs on the **spec** object specifically.

**Add `transmission` to the defaults dump as a called-out item.** Its default is contradictory across three primary sources at the same commit: the header comment says `true` (`RLRAudioPropagation.h:154`), the pybind docstring says `false` (`SensorBindings.cpp:349`), and `docs/AUDIO.md` says `false`.
This is not cosmetic. Ticket 03's argument that the energy gradient survives uniform materials leans on doorway occlusion contrast, and transmission-on with a uniform default leaks energy through walls and reduces exactly that contrast.

Also worth confirming while probing: `enableMaterials` should print **`False`**, because the constructor overwrites the header's `= true` initialiser under `#ifdef ESP_BUILD_WITH_AUDIO`.
If it prints `True`, the build is not what we think it is.

## Note added by ticket 05

**Run ticket 05's inventory before this gate.** It is ~2 minutes and read-only:

```
nrun python3 .scratch/ss2-clean-room/probes/box_inventory.py
```

Not a blocking edge — this gate runs fine without it — but the order is worth ~1 hour:

- `SS2_TORCH_SPEC=torch==2.0.1` / cu117 here is a V100-*era* guess, and this ticket's own comment says it is overridable pending ticket 05. The inventory reads the driver's max supported CUDA, which is what actually settles the wheel.
- If an audio-capable build already exists on the box and is sound, the build step below is minutes rather than the full hour.
- GLIBC < 2.29 or short disk would fail this gate an hour in, on facts readable in two minutes. (GLIBC has never actually been measured on the box; the map treats >= 2.29 as an assumption.)

The inventory also opportunistically dumps `AudioSensorSpec` defaults if an audio-capable env already exists.
That does **not** substitute for the dump below — an existing env is exactly the "unknown drift" this ticket refuses to trust — but agreement is a strong prior and disagreement is a finding in itself.

## Comments

### 2026-08-01 — item 4 answered from source; items 1–3 handed to the box

Ticket 04 has two halves.
One is answerable by reading source and needs no box; the other needs the V100.
This session closed the first and built the driver for the second.

#### Item 4 — does the rebuilt runner need habitat-lab? No. (verified)

The old tree's entire habitat-lab surface is **five symbols across two files**:

| symbol | file | what it does |
| --- | --- | --- |
| `habitat.Env` + `habitat.config.default.get_config` + `read_write` | `habitat_env.py:165-168` | builds the ObjectNav env from `benchmark/nav/objectnav/objectnav_hm3d.yaml` |
| `habitat.config.default_structured_configs.HabitatSimSemanticSensorConfig` | `habitat_env.py:168` | adds the semantic sensor |
| `habitat.tasks.nav.nav.SPL`, `SoftSPL` | `habitat_env.py:181` | metrics (monkeypatched by `spl_guard`) |
| `habitat.tasks.nav.shortest_path_follower.ShortestPathFollower` | `episode_runner.py:2839, 2893, 2909` | the navmesh point-goal controller |

Three findings, each from a primary source rather than recollection:

1. **`ShortestPathFollower` is a thin wrapper.**
   Read at `habitat-lab@v0.2.2`: `_build_follower()` calls **`sim.make_greedy_follower()`**, stores a habitat-sim `GreedyGeodesicFollower`, and catches `habitat_sim.errors.GreedyFollowerError`.
   Its only habitat-lab-specific content is the `HabitatSimActions` action-id enum and a `HabitatSim` type annotation.
   So the clean room gets the controller that fixed navigation in Phase 2 by calling `sim.make_greedy_follower()` directly — no habitat-lab.
2. **The audio branch's own reference does not use habitat-lab at all.**
   `examples/tutorials/audio_agent.py` on `RLRAudioPropagationUpdate` constructs `habitat_sim.Simulator(cfg)` directly, attaches the sensor with `sim.add_sensor(audio_sensor_spec)`, reaches it via `sim.get_agent(0)._sensors["audio_sensor"]`, and per step calls `setAudioSourceTransform()` then `get_sensor_observations()`.
   (Ticket 03 is right that this file is *broken* on this branch — line 39 sets `enableMaterials` on the wrong object and raises. That kills it as a config reference; it survives as the structural reference for how the sensor attaches.)
3. **habitat-lab v0.2.2 has no RLR audio sensor config anyway.**
   SoundSpaces ships that separately.
   So attaching the audio sensor under `habitat.Env` means reaching through `env.sim` and calling `add_sensor` regardless — habitat-lab buys nothing on the audio path and still charges its config stack.

What habitat-lab genuinely provides is the third item: hydra config, the ObjectNav `.json.gz` episode loader, sensor wiring, and the episode iterator.
That is real but bounded work to replace, and it is the only piece with any weight.
`SPL`/`SoftSPL` are arithmetic.

**Conclusion: habitat-lab is a convenience, not a constraint.**
It is therefore deliberately **excluded from the GREEN gate** below and merely *measured*.
This also retires the spike script's caveat — the note that habitat-lab fails to import in the audio env "was acceptable because rendering was offline" does not become unacceptable under live rendering, because the runner does not need it.

#### Items 1–3 — the box gate

Built and handed over, since this Mac is edit-and-push only:

- `.scratch/ss2-clean-room/probes/oneenv_gate.sh` — the driver.
- `.scratch/ss2-clean-room/probes/oneenv_probe.py` — the introspection probe.
- `.scratch/ss2-clean-room/probes/patches/` — where a local habitat-sim patch goes.

Run it with `nrun bash .scratch/ss2-clean-room/probes/oneenv_gate.sh` (~1 h, mostly the build).
Remember the self-update gotcha: the driver git-pulls itself, so a change to it lands on the **second** invocation.

Five things it does that `race-soundspaces-spike.sh` did not, each traceable to a note on this ticket:

1. **One env, layered, re-probed after every layer.**
   habitat-sim(audio) → torch → transformers+scipy, with the audio probe re-run after each.
   Layering is the actual question and the spike never tested it; the most likely failure is a later `pip install` quietly resolving numpy 2.x and breaking the 2022-era tree, so a `numpy<1.24` constraint file is applied to *every* install and the pin is asserted in the report.
2. **Patch-capable from the start** (ticket 02's note), with applied patches recorded to `applied-patches.txt`.
3. **Provenance**: habitat-sim HEAD SHA **and** the `rlr-audio-propagation` submodule SHA (ticket 11's note) — this is what makes the parameter sheet falsifiable.
4. **The defaults dump** (tickets 01/03/11, and ticket 06 blocks on it): every field of a freshly constructed `AudioSensorSpec`, its `acousticsConfig`, and its `channelLayout`, with `transmission` and `enableMaterials` called out by name, plus whether `maxIRLength`/`irTime`/`directRayCount` exist on this branch.
   It also **demonstrates the `dynamic_attr` trap** rather than restating it: it assigns a bogus key to the spec and to `acousticsConfig` and records which one swallows it, which is exactly the evidence the new tree's key validator needs.
5. **habitat-lab installed last**, *after* the core verdict is already on disk, precisely because it is the layer most likely to rot the pin.
   Two reports are produced and compared, so "habitat-lab costs the audio build" becomes a measured statement instead of a worry.

GREEN is defined as: interpreter pins held, habitat_sim audio-capable (enum **member**, not the class — issue #2340), branch-generation methods present (`sourceIsVisible`, `getRayEfficiency`, `setListenerHRTF`, `writeIRWave`, `writeSceneMeshOBJ`), defaults dumped, torch sees the GPU **after** habitat_sim import, CLAP symbols import, and an audio sensor renders a non-silent IR in a real HM3D scene.
RED writes the full blocker list — every stage is independently guarded so one failure cannot hide the four behind it (verified locally: a bare Mac produces all seven blockers and still writes a valid report).

Two deliberate non-goals, so they are not read as omissions:

- The single render is **timed but not swept**. Path tracing is stochastic and needs repeats plus a knob sweep; that is ticket 06, which this only unblocks by printing the real defaults first.
- Materials are forced **off**. Ticket 03 settled that they do not resolve on HM3D, and ticket 12 owns the empty-mesh trap. This gate proves the sensor renders, not that materials work.

**Resolve this ticket by pasting `runs/ss2-oneenv-gate/report-core.json` back here**, then graduate the fog it clears.
