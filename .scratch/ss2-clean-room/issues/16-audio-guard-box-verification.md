# 16 — Verify the audio guard against the real binary

Type: task
Status: claimed
Assignee: Sky
Blocked by: none (12 resolved; needs the box)

## Question

Does ticket 12's audio context guard hold on the real `ss2` build — does it pass the healthy path,
fire under forced failure, and are its two calibrated constants right?

## Why it needs its own ticket

`audio_guard.py` is fully unit-tested against fakes (27 tests green on the Mac), and the fakes
reproduce the two pybind11 behaviours it depends on exactly. What a fake cannot settle is whether the
*real* objects behave as the source says they do — and two of the guard's constants are inferences
with a comment saying so, not measurements.

A guard that has only ever passed is indistinguishable from a guard that cannot fail. Until this
runs, the clean room's loudest safety property is unexercised.

## What would resolve it

```
git fetch && git checkout wayfinder/ss2-clean-room-16     # once, to get the driver
bash .scratch/ss2-clean-room/probes/audioguard_gate.sh
```

Four stages, ~2 minutes, read-only. Paste `report.json` **and** `freeze.txt` back here,
or just the `[5/5] verdict` block, which pulls out every field this ticket asks for.

`audioguard_gate.sh` is the driver: it self-updates (and **re-execs** if the pull changed
it, so the 10-hour-run gotcha cannot bite), activates `ss2` with `set +u` around conda,
asserts habitat-sim is audio-capable by probing the enum *member* rather than the class,
runs stage 0 and the probe, and prints the verdict. It installs nothing and builds
nothing. `--branch <name>` makes it do its own checkout; `SS2_SCENE` overrides the scene.

**Stage 0 now runs first**, so a guard failure still leaves ticket 17's freeze behind.
The driver also greps out the five versions ticket 17 could not find anywhere in the repo.

**Stage 0 — the `pip freeze`, for ticket 17.**
Not part of the guard at all; it rides along because this is a read-only box trip that was already going to happen.
Ticket 17 pinned the `ss2` env behind a nine-line constraints file, and **five of those nine versions are recorded nowhere in this repo**: `soundfile`, `numpy-quaternion`, `huggingface-hub`, `tokenizers`, `safetensors`.
The freeze supplies them, and it is also the forensic artifact ticket 17 asks the bootstrap to keep — the thing a future ticket-13 diagnosis diffs against.
Takes ~5 seconds and blocks nothing here; if the guard stages fail, the freeze is still the deliverable.

**Stage 1 — the key validator against the real `AudioSensorSpec`.**
Answers the one question ticket 12 left genuinely open: does a stock construct-and-configure leave
`vars(spec)` **empty**? If anything is in there it is a legitimate dynamic attribute and must go on
`assert_no_swallowed_keys(allowed=...)` permanently, or invariant 3 is a false positive forever.
Also confirms `irTime` is *rejected* (ticket 11's rename) rather than swallowed.

**Stage 2 — the healthy path.**
`arm_audio_context` must return a report, not raise, and `n_vertices` should be **392,356** against
ticket 04's control on `minival/00800-TEEsavR23oF`. Also prices the OBJ write, which is the one
number behind the "well under 1% overhead" claim that is currently an estimate rather than a
measurement.

**Stage 3 — negative controls.**
An impossible vertex floor must raise. Then three provocations (`setListenerHRTF`,
`setAudioMaterialsJSON`, `writeSceneMeshOBJ` against bad paths) capture a real `ESP_ERROR` so the
severity pattern can be validated against habitat-sim's actual log format.

## The two constants under test

| constant | current value | why it is an inference |
| --- | --- | --- |
| `HABITAT_SIM_LOG_PIN` | `"Sensor,Assets=Debug"` | the subsystem an `ESP_DEBUG` resolves to comes from its C++ namespace; `AudioSensor.cpp` is `esp::sensor` and `ResourceManager.cpp` is `esp::assets`, but that mapping was not read verbatim |
| `DEFAULT_SEVERITY_RE` | `r"\[Error\]"` | habitat-sim's log prefix is built by `buildMessagePrefix()` plus Corrade's severity output; the rendered format was never read |

Both matter for the same reason: if either is wrong the log scan comes back clean over a broken
context, which is the vacuous pass this whole guard exists to prevent. The probe dumps `raw_tail` on
every provocation precisely so a mismatch is fixable from the report rather than needing a second run.

## What it deliberately does not test

Whether a **genuinely** empty audio mesh is detectable. Producing one needs the `enableMaterials=True`
semantic path that ADR-0007 permanently closed, so stage 3 forces the floor instead. That is the
honest limit of a guard for a state we have made unreachable by other means — the assertion path is
proven live, the state itself is not reproduced.

## Note

Expect `canary_seen_on_second_render` to be **False**. The mesh uploads once per context
(`newInitialization_`), so the `Vertex count` line is a first-render artefact. It is recorded to make
explicit *why* `arm_audio_context` owns the first render rather than running later.

---

## Note from ticket 15 (2026-08-03) — stage 1 is already answered, and stage 0 is not

Ticket 15's budget run stood up a real sim + `AudioSensorSpec` on the box and hit this ticket's stage-1 question head-on.

**Stage 1 is resolved: `vars(spec)` is NOT empty on the real binary.** A stock construct-and-configure leaves `__noise_model_kwargs` in the instance `__dict__` — a genuine attribute set by the real constructor, distinct from the bound `noise_model_kwargs` field. Unfixed, `assert_no_swallowed_keys` raises on *every healthy spec*, which is exactly the "false positive forever" this ticket predicted. It now lives in `audio_guard.KNOWN_DYNAMIC_ATTRS`, subtracted on top of any caller-supplied `allowed` so a caller cannot re-open it. Three tests added (30 green), one asserting a real typo like `irTime` still fails alongside it, so the exclusion did not blunt the guard.

This vindicates the ticket's premise more than it shortens it: 27 tests were green against fakes that did not reproduce the one behaviour that mattered.

**Stages 2 and 3 still stand** — `arm_audio_context` itself, the 392,356-vertex floor, the OBJ write cost, and the negative controls are all still unexercised. What ticket 15 proved incidentally is only that a sim + audio sensor configured through `apply_audio_config` renders a real IR (`[2, 72300]`, binaural, 1.506 s at 48 kHz).

**Stage 0's `pip freeze` was NOT taken.** Ticket 17 still needs it, and it is still the cheapest thing on this trip.

**One bug removed from this ticket's path.** `audioguard_probe.py`'s `_find_scene` globbed `data/scene_datasets/hm3d/...`, which does not exist on the box — the canonical root is `data/hm3d/scene_datasets/hm3d/<split>`. It would have failed at stage 2 before touching the guard. Fixed; both roots are now tried.

---

## Pre-flight, 2026-08-03: the guard was reading the wrong file descriptor

Before spending a box trip, the two constants this ticket calls inferences were read at
the branch itself (`facebookresearch/habitat-sim @ RLRAudioPropagationUpdate`, plus
`mosra/corrade`). One is right, one could never have worked, and a third thing turned up
that would have failed stage 2 outright. All three are source citations, not predictions.

### 1. `ESP_DEBUG` writes to stdout, and the guard only captured stderr (BLOCKING, fixed)

The chain, verbatim:

| where | what it says |
| --- | --- |
| `Logging.h:326` | `ESP_DEBUG(...)` expands to `Corrade::Utility::Debug{__VA_ARGS__}` |
| Corrade `Debug.cpp:525` | `std::ostream* Debug::defaultOutput() { return &std::cout; }` |
| Corrade `Debug.cpp:526-527` | `Warning::defaultOutput()` and `Error::defaultOutput()` both return `&std::cerr` |
| `AudioSensor.cpp:499` | `ESP_DEBUG() << "Vertex count : " << sceneMesh->vbo.size()` |
| `AudioSensor.h:45` | `logHeader_ = "[Audio] "`, prefixed onto `ESP_DEBUG` lines |

So **both** canary substrings, `"Vertex count"` and `"[Audio]"`, arrive on **fd 1**.
`capture_fd_stderr` captured fd 2 only. On a healthy first render fd 2 is empty, so
`log_canary_seen` would have been `False`, `require_log_canary` defaults `True`, and
`arm_audio_context` would have raised. **Stage 2 would have come back RED on a perfectly
good audio context**, and stage 3's floor control would have "fired" partly for the wrong
reason, since the canary failure rides in the same exception.

The unit tests could not catch it: `make_render` wrote the whole fake log to fd 2 with
`os.write(2, ...)`, and the module docstring asserted "habitat-sim logs from C++ to file
descriptor 2" as settled fact. The fake encoded the assumption it was meant to test.

### 2. `DEFAULT_SEVERITY_RE` was structurally dead, not miscalibrated (fixed)

`buildMessagePrefix` (`Logging.cpp:149-152`) formats exactly:

```
[{HH}:{MM}:{SS}:{uuuuuu}]:[{Subsystem}] {file}({line})::{function} :
```

Corrade prepends no severity tag of its own. There is **no `[Error]` substring anywhere in
a rendered habitat-sim log line**, so `re.compile(r"\[Error\]")` matched nothing, ever, and
the generic arm of invariant 2 was silently blind. It read as breadth and delivered none.
Only `FATAL_LOG_SUBSTRINGS` was load-bearing.

The test that "proved" it worked, `test_severity_marker_fails`, fed the guard a fabricated
line containing `[Error]` in a format the binary does not produce.

**Severity is not recoverable from the text at all. It is carried entirely by the stream**,
because Corrade routes Warning and Error to `std::cerr` and Debug to `std::cout`, which is
the same fact as finding 1 seen from the other side.

### 3. `HABITAT_SIM_LOG_PIN = "Sensor,Assets=Debug"` is correct (confirmed, no change)

`ESP_DEBUG` resolves its subsystem through C++ namespace lookup on `espLoggingSubsystem()`
(`Logging.h:312`), and `ESP_ADD_SUBSYSTEM_FN(sensor)` installs that function in
`esp::sensor`, which is the namespace `AudioSensor.cpp` opens at `:12-13`. So its macros do
resolve to `Subsystem::sensor`, rendered `"Sensor"` by `subsystemNames`. `ResourceManager.cpp`
is `esp::assets`, rendered `"Assets"`. The inference held.

Also worth recording: `LoggingContext::DEFAULT_LEVEL` is already `Verbose` (`Logging.h:167`),
so on an untouched env the pin changes nothing. Its only job is surviving an operator who
turned logging down, which is exactly what its comment claimed.

### What changed in the code

`audio_guard.py`

- `capture_fd_stderr` becomes **`capture_habitat_logs`**, capturing fd 1 and fd 2 into
  separate buffers. `.stdout`, `.stderr` and a combined `.text` are exposed. Python's own
  buffers are flushed on both entry and exit, so the caller's `print()` output cannot leak
  into the capture.
- `DEFAULT_SEVERITY_RE` becomes **`HABITAT_LOG_PREFIX_RE`**, built from the `subsystemNames`
  array and the `buildMessagePrefix` format string. Its job changed from severity to
  **provenance**: on fd 2 the severity is already known, so the only open question is
  whether the line came from habitat-sim or from some third party that also writes there.
- Fatal now has two arms. `FATAL_LOG_SUBSTRINGS` on either stream, unchanged and still the
  load-bearing rule, plus **any habitat-prefixed line on fd 2**, which is new and general.
- The report gains `stdout_chars` and `stderr_chars` alongside `log_chars`.

`audioguard_probe.py`

- Stage 3 records both streams per provocation, both raw tails, and which stream the
  prefix matched on. `setListenerHRTF` is the reliable ESP_ERROR (`AudioSensor.cpp:181`);
  `setAudioMaterialsJSON` only `ESP_DEBUG`s the path it stored (`:169`), so it is *expected*
  on fd 1 and is the cleanest live demonstration that the split is real.
- Stage 3's canary check now reports `canary_on_stdout` / `canary_on_stderr` separately.
- **Placement is seeded and bounded.** `get_random_navigable_point()` was drawn twice
  independently and unseeded, so the source could land across the scene and the guard's
  non-silent-IR assertion would fail for a reason unrelated to the audio context, on a run
  nobody could reproduce. Now `pathfinder.seed(20260803)` plus a source redrawn until it is
  1.0 m to 8.0 m from the listener, with the placement recorded in the report.

`test_audio_guard.py`: **30 tests to 39, all green on the Mac.** The fakes now write
`ESP_DEBUG` to fd 1 and `ESP_ERROR` to fd 2 using the real rendered prefix. New cases cover
the stream split, a habitat-prefixed line on fd 2 being fatal, the same line on fd 1 not
being fatal, third-party stderr noise not being fatal, and a regression guard asserting no
`[Error]` tag exists to match.

### Two things checked and cleared, so they do not need the box

- **C++ buffering does not eat the capture.** `std::cout` is fully buffered when piped, so
  logs could have sat in the C++ buffer across the fd restore. Corrade's `Debug::newline`
  uses `std::endl` specifically "to force a flush" (`Debug.cpp:359-366`), per line. Nothing
  is left behind.
- **`writeSceneMeshOBJ` really returns `bool`** (`AudioSensor.h:118`), so
  `... is True` is safe rather than a `None`-versus-`True` trap.
- `_peak_abs` over a real `[2, 72300]` IR costs **0.062 s** in pure Python, so the
  numpy-free implementation is not a cost worth removing.

### What this does and does not settle

It closes the ticket's third bullet, "are the two calibrated constants right", from source:
one confirmed, one replaced. It does **not** close the ticket. Stages 0, 2 and 3 still need
the box, and the ticket's own premise is the reason: a guard that has only ever passed is
indistinguishable from one that cannot fail, and reading the source is not running it. What
the box now measures is a guard whose two known-wrong assumptions have already been removed,
so a RED comes back about the audio context rather than about the plumbing.

**Status stays claimed. The run command is unchanged** (`--out runs/ss2-audioguard/report.json`),
and `freeze.txt` for ticket 17 still rides along.

---

## Box run 1, 2026-08-03 (commit `6c49f2f`): GREEN, and the pre-flight call held

`VERDICT: GREEN`, all three stages ok, 8.25 s of probe time.

**The fd-1 finding is confirmed on the binary.** The healthy first render logged
**916 chars on stdout and 0 on stderr**. The pre-fix guard captured fd 2 only, so
`log_canary_seen` would have been False and `arm_audio_context` would have raised. Stage 2
would have come back RED on a good audio context, exactly as predicted.

**Stage 0 delivered.** All five versions ticket 17 could not find: `huggingface_hub==0.36.2`,
`numpy-quaternion==2023.0.4`, `safetensors==0.7.0`, `soundfile==0.13.1`, `tokenizers==0.22.2`.
71 packages in `runs/ss2-audioguard/freeze.txt`.

**Stage 1 reproduces ticket 15 exactly.** `vars(spec)` is `['__noise_model_kwargs']` after both
a bare construct and a configure, nothing else. `irTime` rejected, swallowed key detected.
`KNOWN_DYNAMIC_ATTRS` is right and complete.

**Stage 2 passes**, on a seeded placement 3.146 m apart: 392,364 verts, IR peak 0.130,
`ray_efficiency` 0.669, `source_is_visible` False. Guard total **2.24 s**, of which the
**OBJ write is 0.893 s for 32.2 MB** — the "well under 1%" claim is now measured, and it
holds against a 500-step episode.

**Stage 3's forced floor fires.** And because the canary now passes on fd 1, the floor is
the *only* failure in that exception, so the negative control is clean rather than tripping
two invariants at once.

### Four things run 1 corrected

1. **`ir_shape` was `None` over a real IR.** The audio observation is not a numpy array, so
   `getattr(ir, "shape")` finds nothing. `_shape_of` walks the nesting instead, still
   numpy-free. `_peak_abs` was unaffected and read 0.130.
2. **392,364, not ticket 04's 392,356.** Not a discrepancy: `AudioSensor.cpp:499` logs what
   habitat **submitted**, `writeSceneMeshOBJ` reports what the engine **holds**. Two numbers
   either side of the upload. Both are now recorded with the delta, so invariant 1's
   "reads the geometry the engine holds, not what habitat thinks it sent" is visible in the
   artifact rather than being a claim in a docstring.
3. **`canary_seen_on_second_render` is True**, and this ticket's "expect False" was wrong.
   `Vertex count` is a first-render artefact, but the *other* canary substring is
   `logHeader_`, and `runSimulation` logs `[Audio] Running the audio simulator`
   (`AudioSensor.cpp:130`) on **every** render. The canary stays armed for the whole episode,
   which is strictly better than predicted.
4. **No habitat-prefixed line reached fd 2 on any provocation**, yet stderr carried 406 and
   238 chars. Diagnosed from the stored tails via `--tails`, no re-run. See below.

### The engine has its own stderr, and it reports success anyway

The un-prefixed fd-2 text is the closed RLR engine, Meta's AudioSDK (`ovra`), writing
around Corrade entirely:

```
File: arvr/libraries/audio/AudioSDK/Research/Source/Wrapper/PropagationWrapper.cpp
Function: ovrResult PropagationWrapper::WriteSceneMeshOBJ(const std::string &), Line 1025
Error writing scene OBJ mesh at location:
/nonexistent-dir/x.obj
```

A block, not a line: two stable header lines, then free text and often the offending value.

**The sharper half is what is absent.** `setListenerHRTF`'s stdout carries the `ESP_DEBUG` at
`AudioSensor.cpp:177` and **no `ESP_ERROR` at `:181`**, so `RLRA_SetListenerHRTF` printed
`Error reading HRTF file` and returned `RLRA_Success`. Ticket 12 stated invariant 2 as "the
error is detected in C++ and discarded before Python". For this class **it is never detected
at all** — the engine does not report it through the return code. A failure with no return
code and no habitat log entry is visible **only** as that block.

Two consequences beyond this ticket:

- **Invariant 2 is load-bearing, not a stopgap.** The log scan is not standing in for a
  return code that exists behind a missing binding; for this class there is no return code.
- **It weakens the map's second fork candidate.** Ticket 12 proposed binding `RLRA_Error` to
  Python to convert invariant 2 from log-scraping into a return code. That would **not** have
  caught this, because the call returned Success. The patch is worth less than it looked.

`setAudioMaterialsJSON` produced **0 chars on stderr**, predicted from source (`:169` is
`ESP_DEBUG`-only, it just stores the path). That is the control: the stream split is real,
not an artefact of one call.

### Run 2 is required, and why

The fd-2 rule changed, so run 1 no longer validates the current code. `RLR_ENGINE_RE` matches
the verbatim box text off-box and third-party stderr noise does not trip it, but that is a
fixture, not the binary. Run 2 also settles two open items: whether `writeSceneMeshOBJ`
**returns** false on a bad path (if it returns success like `setListenerHRTF`, invariant 1's
`is True` check is decorative and the vertex floor is the whole arm), and the submitted-vs-engine
vertex delta from the healthy-path tails.

```
bash .scratch/ss2-clean-room/probes/audioguard_gate.sh
```
