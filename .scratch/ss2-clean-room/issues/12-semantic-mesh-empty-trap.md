# 12 — The audio context must never silently accept an empty mesh

Type: task
Status: resolved
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

## Answer

**Built, unit-tested on the Mac, and the ticket's own framing corrected in three places.**
`probes/audio_guard.py` (the guard), `probes/test_audio_guard.py` (**27 tests, all green on this
Mac**, no habitat_sim), `probes/audioguard_probe.py` (the box verification, not yet run — ticket 16).
It ports verbatim into the new tree at reset phase 1; nothing about it depends on the layout ticket
09 has yet to decide.

### Three corrections, all read from source at the pinned commit `4f61e321`

**1. It cannot assert "at context creation" — the mesh does not exist yet.**
`createAudioSimulator()` sets `newInitialization_`; the **first `runSimulation()`** consumes it and
only then calls `loadMesh`/`loadSemanticMesh`. An assertion at construction would pass over nothing,
which is this ticket's own failure mode wearing a different hat. So `arm_audio_context` **performs
the first render itself** rather than asking the caller to sequence one correctly — the ordering
constraint is discharged by the interface, not documented in a comment.

**2. `RLRA_Error` is already checked — and then thrown away before Python.**
The ticket says "the engine's failures are bare returns, not exceptions", which is right about the
consequence and wrong about the cause. habitat-sim *does* compare every `RLRA_*` call against
`RLRA_Success` (`RLRA_CreateContext`, `RLRA_AddListener`, `RLRA_AddSource`, `RLRA_AddMeshVertices`,
`RLRA_AddMeshIndices`, `RLRA_AddObject`, `RLRA_FinalizeObjectMesh`, `RLRA_Simulate`). The handler is
`ESP_ERROR() << ...; return;` and `runSimulation` is `void`. So the error is detected in C++ and
**discarded before the binding** — no return code, no exception, no flag.

*"Check every `RLRA_Error`" is therefore not a thing a Python wrapper can do.* There are exactly two
routes: read the log, or patch the bindings. This ships the first and names the second.

And the log route has its own trap: habitat-sim logs from C++ to **file descriptor 2**, which
`contextlib.redirect_stderr` does not touch — it rebinds `sys.stderr`, a Python object the C++ logger
never sees. A guard built that way captures nothing and **passes vacuously**, which is precisely the
bug class this ticket exists to remove. Hence `capture_fd_stderr` (`os.dup2` on fd 2) plus a
**canary**: if the capture comes back without any recognisable habitat-sim audio line, the guard
raises saying invariant 2 is *unverified, not satisfied*.

**3. `vars(spec)` detects the `dynamic_attr` trap exactly, not heuristically.**
`def_readwrite` installs a data descriptor on the type, so a real field never reaches the instance
`__dict__`; `py::dynamic_attr` puts everything else there. So `vars(spec)` **is** the set of
swallowed keys — no allowlist needed for detection, and it stays correct across branch renames. The
allowlist is still used for *rejection* (`apply_audio_config` validates before writing, so a typo
cannot half-apply a config), and it is derived by introspecting `dir()` rather than hardcoded, so
ticket 11's `irTime` → `maxIRLength` rename surfaces as a rejected key instead of a silent no-op.

### The shape

```python
pin_habitat_logging()                      # before importing habitat_sim, or it raises
spec = apply_audio_config(AudioSensorSpec(), cfg)   # invariant 3
report = arm_audio_context(audio_sensor, render)    # invariants 1 + 2, owns the first render
```

`arm_audio_context` runs **every** check before raising and puts all failures in one
`AudioContextError` — a broken context usually trips several and the first is rarely the diagnosis.
That mirrors `oneenv_probe.py`'s "one failing stage must not hide the four behind it".

| invariant | mechanism | strength |
| --- | --- | --- |
| 1. non-empty mesh | `writeSceneMeshOBJ` → count `v ` lines → floor | **load-bearing** — reads the geometry the *engine* holds (`RLRA_WriteSceneMeshOBJ(...) == RLRA_Success`), not what habitat thinks it sent |
| 2. RLRA errors | fd-2 capture + fatal substrings + severity regex + canary | breadth; the only Python-visible channel |
| 3. unknown keys | `vars(spec)` sweep + introspected allowlist | **exact** |

**Invariant 1 is the backstop for invariant 2's whole class**: every way the upload can fail —
`RLRA_AddMeshVertices` failing, `joinSemanticHierarchy`'s cast returning bare — ends in a short
vertex count, whether or not the log scan sees the reason.

`getRayEfficiency()` and `sourceIsVisible()` are **recorded and never asserted**. Their values over a
zero-geometry context are unknown (the `.so` is closed), so there is no honest threshold; ticket 04's
0.548 is one sample on one healthy scene. With no geometry nothing occludes, so `sourceIsVisible()`
would read True everywhere — a good diagnostic against a known-occluded pair, not a gate.

### The floor is 10,000 verts, not `> 0`

`> 0` is the literal invariant and it would pass a degenerate three-vertex mesh, which produces the
same direct-path-only IR as an empty one. 10,000 is two orders of magnitude below ticket 04's
measured control (392,356) — far enough below any real HM3D scene never to fire spuriously, far
enough above zero to catch the degenerate case. The OBJ dump runs **every episode**: one ~25 MB
write against ~0.58 s × 500 steps of audio is well under 1% (an estimate; the probe prices it).

### Measured vs inferred, stated plainly

- **Verified** (source at `4f61e321`, and corroborated — the 20 `acousticsConfig` field names read
  off `SensorBindings.cpp` match ticket 04's measured dump exactly): lazy mesh upload, the
  `ESP_ERROR`+bare-return handler, `writeSceneMeshOBJ`'s bool contract, `py::dynamic_attr` on the
  spec and not on `acousticsConfig`, `HABITAT_SIM_LOG`'s grammar and levels.
- **Inferred, and the probe measures rather than assumes**: `HABITAT_SIM_LOG_PIN =
  "Sensor,Assets=Debug"` (the subsystem an `ESP_DEBUG` resolves to comes from its C++ namespace) and
  `DEFAULT_SEVERITY_RE = r"\[Error\]"` (habitat-sim's prefix format was never read verbatim). Both
  are defaults with a comment saying so; a probe mismatch is a finding about the constant.
- **Open, and the probe answers it**: whether a stock construct-and-configure leaves `vars(spec)`
  empty. If some legitimate dynamic attribute exists on this branch it must go on
  `assert_no_swallowed_keys(allowed=...)` permanently, or invariant 3 is a false positive forever.

### What this does not do

- It does not prove a genuinely empty audio mesh is *detectable on the box* — only that the assertion
  fires. Producing a real zero-geometry context would need the `enableMaterials=True` semantic path
  that ADR-0007 has permanently closed, so the negative control is a forced floor instead. This is
  the honest limit of a guard for a state we have deliberately made unreachable by other means.
- It does not close the `RLRA_Error` channel properly. That needs a bindings patch of the same class
  as ticket 02's multi-source one — **a second patch candidate, where the map currently says
  multi-source is the only one.** Cheaper than 02's (~returning the enum from `runSimulation`), and
  it converts invariant 2 from log-scraping into a return code. Routed to ticket 09's fork call.

### Follow-on

**Ticket 16** — run `audioguard_probe.py` on the box. Three negative controls, the two calibration
constants, the `vars(spec)` question, and the real OBJ-write cost.
