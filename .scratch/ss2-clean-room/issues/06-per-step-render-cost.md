# 06 — What does live-every-step audio actually cost?

Type: task
Status: claimed
Assignee: Sky
Blocked by: 04, 11 (both resolved — unblocked)

## Question

How many milliseconds does one in-sim audio render take on the RACE box, at default parameters and at a cheap preset, and does that make live-every-step affordable?

## Why it matters

"Live, every step" was chosen as a design decision before anyone measured it.
Rendering is Monte-Carlo path tracing at `indirectRayCount=5000`, `indirectRayDepth=200`, `irTime=4.0 s`, `threadCount=1`.
At a 500-step episode budget, 50 ms/step is 25 seconds of pure audio per episode and tolerable; 500 ms/step is four minutes per episode and forces the throttled variant instead.

The answer decides whether the destination as named is reachable, so it is worth measuring before any agent code is written.

## What would resolve it

On the box, in the ticket-04 env, with one HM3D scene loaded:

0. **Print the real defaults first.** Ticket 11 found they live inside the closed `.so` (`RLRA_ContextConfigurationDefault`) and that every number in ticket 01's table is unverified for our branch. Construct an `AudioSensorSpec` and dump every `acousticsConfig` field before timing anything, so the sweep has a real baseline.
1. Time `setAudioSourceTransform` + `get_sensor_observations()` at those defaults, over enough repeats to see variance (path tracing is stochastic).
2. Sweep the cost knobs one at a time and record the time/quality curve. **Names and defaults per ticket 11, not ticket 01** — `irTime` does not exist on our branch, and no default below is verified until printed on the box:
   - `maxIRLength` (was `irTime`) 4.0 → 1.0 → 0.5
   - `indirectRayCount` 5000 → 1000 → 500
   - `directRayCount` — new on our branch, ticket 01 never saw it
   - `threadCount` 1 → physical core count
   - `temporalCoherence` off → on
3. For each setting, also record whether the received energy still decreases monotonically as the listener moves away from the source along a navmesh path. Speed is worthless if the gradient stops being climbable.
4. Check whether cost depends on listener-source distance or scene size, since that determines whether the budget is per-scene stable.
5. **Sweep source count 1 → 2 → 3.** Ticket 02 found the engine is natively multi-source and that a wrapper patch exposes it at zero extra *renders* per step — but the `.so` is closed-source, so whether one `RLRA_Simulate` over 3 sources costs 1x or 3x cannot be read and must be measured. This decides whether the bed and distractor are real sources or post-render additions, so it is not optional.

Note two readouts ticket 02 found on the branch that this sweep should use:
- **`getRayEfficiency()`** gives a built-in quality number, so the ray-count sweep does not rest on eyeballing gradient monotonicity alone.
- **`sourceIsVisible()`** is a single-ray line-of-sight test, useful for labelling each timing sample as LOS or non-LOS, since diffraction-dominated paths are the expensive case.

Take the knob names and defaults from ticket 11, not ticket 01 — ticket 01 was researched against a different branch.

Deliverable: a table of ms/step against setting, a recommended preset, and an explicit verdict on whether live-every-step holds or whether the map's destination needs amending to the throttled variant.

## Note added by ticket 03

**Good news for the budget: scene geometry is uploaded once, not per step.**
The whole mesh upload in `AudioSensor::runSimulation` is wrapped in `if (newInitialization_)`, so `loadMesh` / `loadSemanticMesh` run on the first simulation after `createAudioSimulator` and never again.
Only `RLRA_Simulate` re-runs per step.
Measure the two separately: first-call cost (geometry upload plus simulate) against steady-state cost (simulate only).
The ticket's affordability verdict hangs on the steady-state number, and the first-call number is a per-episode constant.

**The materials A/B this ticket was asked to run is now mostly answered, and its scope changes.**
Ticket 03 established that materials are off by default on this branch, that plain HM3D has no semantic scene at all, and that HM3D-Semantics v0.2 appears to break the semantic mesh path entirely (ticket 12).
So "render with `enableMaterials` off and on and compare" is not a meaningful A/B on HM3D.
What replaces it:

- **Add a gradient-contrast measurement to the sweep.** Fix a source, walk a navmesh path toward it, record broadband IR energy per step. Report the Spearman correlation between energy and negative geodesic distance, plus the far-to-near dynamic range in dB.
  Ticket 03 argues from the engine's physics that a uniform-absorption world still yields a climbable gradient, because every load-bearing term (direct-path spreading, occlusion, diffraction) is geometric and material-independent. What it cannot argue is **contrast**: a reflective built-in default flattens the within-room field and compresses the final-metres gradient.
  That number is the one thing section 7 of ticket 03 could not settle from source, and it belongs here because it is the same walk the timing sweep already does.
- Run it under HM3D materials-off, and under MP3D materials-on/off if ticket 08 keeps MP3D in play.

## Note added by ticket 04 (now resolved — this ticket is unblocked)

**Step 0 is done. The defaults are measured, so the sweep has a real baseline:**

```
diffraction 1        directRayCount 500     directSHOrder 3      direct 1
frequencyBands 4     globalVolume 1.0       indirect 1           indirectRayCount 5000
indirectRayDepth 200 indirectSHOrder 1      maxDiffractionOrder 10
maxIRLength 4.0      meshSimplification 0   sampleRate 44100.0
sourceRayCount 200   sourceRayDepth 10      temporalCoherence 0  threadCount 1
transmission 1       unitScale 1.0
channelLayout Binaural / channelCount 2
```

Provenance for the sweep to quote: habitat-sim `RLRAudioPropagationUpdate @ 4f61e321`, `rlr-audio-propagation @ 4fd446b4`, stock (no patches).

Four things that change how this ticket should be run:

1. **The single timing is worse than this ticket's tolerable case.** `first_render_s = 0.6013` on `minival/00800-TEEsavR23oF` (392,356 verts). Ticket 03's note above is right that geometry uploads once — and the log timestamps put that upload at only **~17 ms**, which would leave **~0.58 s in `RLRA_Simulate` itself**. *That is an inference off log timestamps, not a measurement* — settling it is this ticket's first job. But if it holds, live-every-step at defaults is ~5 min/episode of pure audio, which is this ticket's "forces the throttled variant" case. **Measure steady-state before assuming the destination is affordable.**
2. **`threadCount` is a weaker lever than the map assumes.** The map calls it "a free speed knob currently set to 1"; the box has **4 cores**, so the ceiling is ~4x, not an order of magnitude. The real levers are `indirectRayCount` (5000), `indirectRayDepth` (200), `maxIRLength` (4.0) and `temporalCoherence` (currently **0**, i.e. off — so it is a pure win to test, nothing is being given up by enabling it).
3. **`transmission` defaults to ON (`1`).** It is a cost knob *and* a contrast knob: it leaks energy through walls, which works directly against the gradient-contrast measurement ticket 03 added above. Sweep it on both axes, not just for speed.
4. **The IR is trimmed to actual decay, not to `maxIRLength`.** The gate returned `ir_shape [2, 72300]` = 1.64 s at 44.1 kHz against a 4.0 s `maxIRLength` (176,400 samples). So IR width is *scene- and pose-dependent*: any fixed-width buffer downstream is wrong, and `maxIRLength` is a cap rather than a size. Worth recording IR width alongside ms/step, since a shorter IR is part of how a cheaper preset pays for itself.

Two readouts are confirmed present on the built binary and free to use: `getRayEfficiency()` (the gate measured 0.548 at defaults) and `sourceIsVisible()` (measured `False` for that pose — a non-LOS, diffraction-dominated sample, which is the expensive case).

## Comments

### 2026-08-01 — probe built and handed to the box; two decisions taken off-box first

The measurement half of this ticket has no off-box component — it is a wall-clock
number on a V100 and nothing else. So this session built the probe and settled the
two questions that had to be answered *before* the box time was worth spending.

```
nrun bash .scratch/ss2-clean-room/probes/rendercost_sweep.sh
```

~10–20 min. Reuses the `ss2` env ticket 04 built; installs nothing, builds nothing,
applies no patches. Writes `runs/ss2-render-cost/report.json`.
**Resolve this ticket by pasting the verdict block back here.**

- `.scratch/ss2-clean-room/probes/rendercost_probe.py` — the measurement
- `.scratch/ss2-clean-room/probes/rendercost_sweep.sh` — the driver

#### Decision 1 — the multi-source sweep is deferred, and that is not a dodge

Item 5 asks to sweep source count 1 → 2 → 3. On the **stock** build that cannot be
done: ticket 04 confirmed `multi-source surface: none` against the built binary, so
concurrent sources need ticket 02's ~40-line patch, and ticket 02's own note routes
the decision to take that patch through *this* ticket's cost sweep. That is a loop.

It is broken by ordering rather than by patching:

1. **The sequential upper bound is measurable now, unpatched.** N sources as N
   renders (ticket 02's workaround 2) needs no patch and no rebuild, and it brackets
   the answer — the patched single-`RLRA_Simulate` cost cannot be worse than N
   sequential renders, and cannot be better than one. The probe measures it.
2. **If one render at the cheapest admissible preset is already over budget, the
   patch is moot** — the task is throttled regardless of how well multi-source
   amortises, and writing ~40 lines of C++ plus a rebuild to learn that would be
   backwards.
3. **If the sequential 3x is affordable, the patch is also moot for the budget** and
   becomes purely an onset-provenance argument, which is ticket 09's call on its
   merits rather than on cost.

So the patch only earns a rebuild in the narrow band where 1x fits and 3x does not.
The probe reports exactly the numbers needed to see whether we are in that band.
**The fork-cost question in the map's Not yet specified is unchanged by this** — it
just gets a cheaper input than a rebuild.

#### Decision 2 — the verdict thresholds are pre-registered

Written into the probe *before* any number came back, so pasting the report resolves
this ticket instead of opening a fresh argument about what the number means:

| verdict | steady-state ms/step | per 500-step episode |
| --- | --- | --- |
| `LIVE_EVERY_STEP_HOLDS` | ≤ 50 | ≤ 25 s |
| `LIVE_EVERY_STEP_TOLERABLE` | ≤ 150 | ≤ 75 s |
| `THROTTLE_REQUIRED` | > 150 | destination gets amended |

Every one of those is **gated on the gradient still being climbable** — Spearman(energy,
geodesic distance) ≤ −0.70 and a far-to-near dynamic range ≥ 6 dB. A preset that is
fast and flat does not count, per this ticket's own item 3. Cost is quoted at the
**worst** scene measured, and a config only counts as admissible if it is admissible
in **every** scene, so the recommendation is not read off the friendliest geometry.

#### A correction to this ticket's own framing: the 0.6013 s is not an audio number

Ticket 04 timed `get_sensor_observations()` on a **default agent config, which carries
an RGB camera**, and that call renders *every* attached sensor. So 0.6013 s is audio
**plus** a visual render, and the "~0.58 s left in `RLRA_Simulate`" inference is
subtracting the geometry upload from a total that also contains something else
entirely. The scary number may be materially smaller than it looks.

The probe configures the agent with **no camera sensors**, so its steady-state figure
is audio alone, and `--with-camera-delta` re-measures one config with the camera
attached so the confound becomes a number rather than an argument. This does not
overturn ticket 04's warning — it may well still be slow — but the warning should not
be quoted as an audio cost until this run reports one.

#### What the probe measures, and why it is one walk

Everything is derived from a single **walk**: fix a source, drop the listener several
metres away, step it along the navmesh path toward the source, and time every render.
One walk yields all five of this ticket's items at once — first-call vs steady-state,
ms/step per knob, gradient monotonicity, cost-vs-distance, and LOS/non-LOS split via
`sourceIsVisible()`. `getRayEfficiency()` and IR sample width are recorded per step,
so the ray-count sweep has a quality readout that does not rest on eyeballing.

The cheap preset is **derived from the measurements, not guessed** — every knob that
was both faster and gradient-admissible is combined, and the combination is then
measured too, because these effects are not additive.

One addition beyond the ticket text: **the non-LOS gradient is scored separately**
(`rho_nlos`). A config can post a healthy overall gradient purely from its
line-of-sight samples while being flat wherever a wall is in the way — and climbing
toward a source it cannot see is the entire premise of the anomaly response. Ticket 09
needs that number specifically to decide `transmission` and `diffraction`.

#### Four bugs found in local verification, all of which would have produced plausible wrong numbers

Verified against a stub `habitat_sim` (the Mac cannot run the real one), to the bar
tickets 04 and 05 set. The stub run was not ceremony — it caught four defects, and
every one of them would have returned a *confident, well-formatted, wrong* answer:

1. **Sensor accumulation.** `walk_config` attached a new audio sensor per config to
   one simulator, and `get_sensor_observations()` renders every attached sensor — so
   config *k* paid for *k* renders. The stub's cost curve rose monotonically with
   sweep position regardless of the knob, with `diffraction=0` measured as the
   **slowest** config. Fixed: one fresh simulator per config.
2. **Walk truncation.** Sampling at fixed spacing and slicing `[:walk_steps]` covered
   only the first 4 m of a 13.6 m path — the far end, where the gradient is shallowest.
   Every config scored FLAT, and the final metres that ticket 03 explicitly asked about
   were never sampled. Fixed: samples are spread evenly across the whole path.
3. **The verdict mixed scenes.** It took the cheapest admissible row across all scenes,
   so it reported a config as a clean win while that same config was flat in the other
   scene. Fixed: admissible-everywhere, costed at the worst scene.
4. **The cheap preset auto-adopted physics knobs.** `transmission` and `diffraction`
   are cheaper when off and still scored "climbable" on a mostly-LOS walk, so they were
   being folded into a preset labelled *cheap* — smuggling a task-design decision in as
   a performance tweak. They are the non-line-of-sight audio path, and ticket 09 owns
   them. Fixed: swept and reported, never auto-adopted.

Also verified: the walk poses are computed once and replayed into every config, so
knob comparisons are over identical journeys; unknown `acousticsConfig` keys raise
rather than being silently swallowed (ticket 04 measured that the **spec** swallows
them, so a typo'd knob name would otherwise time the *default* value and look fine);
and a machine with no `habitat_sim` still writes a valid report whose blocker list is
the deliverable, rather than crashing.

## Note added by ticket 08 (resolved 2026-08-01) — run the HM3D arm only

**HM3D stays; MP3D is out of scope; acoustic materials are permanently off** (`docs/adr/0007-hm3d-stays-mp3d-out-of-scope.md`).

Two changes to this ticket, both narrowing it:

1. **Drop the MP3D arms.** Ticket 03's note asked for gradient contrast under "HM3D materials-off, and under MP3D materials-on/off if ticket 08 keeps MP3D in play". It does not. Run **HM3D materials-off only** — which is the configuration the clean room ships, so the sweep now measures exactly the deployed path and nothing else. No MP3D download is authorised.
2. **The gradient-contrast number is no longer a dataset gate.** It was framed as the measurement that could reopen the HM3D-vs-MP3D question. It cannot: 08 ruled MP3D out unconditionally, on the reasoning that materials are off in the MP3D reference configuration too, so a flat gradient would not be fixed by switching datasets. **If the gradient comes back flat, the lever is source placement or source gain, not the dataset.** Report the number the same way; only its consequence changed.

Everything else in this ticket stands unchanged — the timing sweep, the `transmission` on/off axis, the sequential multi-source upper bound, and the steady-state-versus-first-call split are all untouched by 08.
