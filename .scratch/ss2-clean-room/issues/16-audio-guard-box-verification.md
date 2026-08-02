# 16 — Verify the audio guard against the real binary

Type: task
Status: open
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
conda activate ss2
nrun python3 .scratch/ss2-clean-room/probes/audioguard_probe.py \
    --out runs/ss2-audioguard/report.json
```

Three stages, ~2 minutes, read-only. Paste `report.json` back here.

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
