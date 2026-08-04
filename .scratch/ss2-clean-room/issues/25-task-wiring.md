# 25 — `task/` wiring: the runner, the dataset builder, and the CLI

Type: task
Status: resolved
Assignee: Sky
Blocked by: 21 (resolved), 22 (resolved), 23 (resolved), 24 (resolved)

## Question

Wire the four layers together. `task/` is the only layer that may import all of them (ADR-0013), so this is where the tree stops being modules and starts being a run.

## What to build

**`task/dataset.py`** — the episode builder. Places exactly **one** positioned source per episode (the anomaly), on the primary goal's floor at `|Δy| < ~1.0 m` (ADR-0010, checked here in the builder because there is no grid and no `nearest` to defend against — there is **no runtime guard and none is needed**), and carries the xz separation rule that decouples the source from the primary goal.

Audibility is **not** screened at build time (§2.5): pre-screening would reintroduce offline rendering by the back door, and the attrition is carried in the funnel instead.

**`task/runner.py`** — the episode loop. Constructs the World from `sim/`, the sensor specs from `audio/spec.py`, the handle from `audio/sensor.py` (which arms the guard), the proposer pool and detector from `agent/`, and every model **eagerly at startup** (requirement 9: 5.547 GiB against 31.73 GiB usable, so there is no lazy-loading seam and the layout must not grow one).

Per step: `guarded_observe()` → onset → `step_controller` → apply the `ControllerDecision` → record §3.2's per-step row (measured RMS, lateral sign, source playing, `sourceIsVisible()`, **the action taken** — the action is there so a rotation-driven RMS rise is distinguishable from a translation-driven one after the fact).

Tracks §6's funnel: episodes run → `t_anom` reached → onset fired → investigate entered → source reached → primary resumed, denominator at stage 2.

**`__main__.py` and `config.py`** — `argparse` → `RunConfig`, composing the per-module frozen sub-configs. The two arms are enums: `Localization.{REALIZABLE,ORACLE}` and `Detector.{ORACLE,CAPTION}`. `assert_env()` is called here.

## Done when

An episode runs end to end on the box under `Localization.REALIZABLE` + `Detector.ORACLE`, emitting both artefacts. Green is ticket 26's job; this ticket ends when it runs.

## Watch for

**Requirement 1(d), as ADR-0013 narrowed it.** The guard flushes Python's buffers, so interleaved in-thread `print()` between steps is safe. What is forbidden is a *concurrent* fd-1/2 writer — no background thread, no timer-driven progress bar, no logging handler flushed off-thread, no subprocess inheriting the descriptor.

**Metrics, per §6.** Find-SR at 1.0 m primary and 0.1 m diagnostic. soft-SPL computed but not headlined. Benchmark SPL computed and **never cross-quoted from this map**. Two new ones: distance-at-STOP as a distribution, and per-step audio render wall-clock reported every run.

Resume is unchanged: restore primary state, force a re-query, return to SEARCH. `is_diverting()` must suppress the primary STOP during INVESTIGATE / CHECK / RESUME.

---

## Built, 2026-08-05 — and the error type had to move before the loop could catch it

Five modules, **76 new Mac tests (484 → 560 green)**, ruff clean, and four planted
violations that each fired and were each restored. The tree is now a runnable program:
`python -m earshot --run-dir runs/<tag>` composes the config, asserts the environment,
builds the episodes, runs the loop and writes both artefacts.

**The box run is NOT done, and that is the one part of "Done when" this session could
not deliver** — see "Handed to the box" below.

### The finding: an exception type cannot be caught by a layer that may not name it

`sim/world.py` raises `NoRouteError` when the follower cannot route, and ticket 21 gave
it its own type for a stated reason: `None` already means *arrived*, and conflating the
two is what made the old tree's navigation unfalsifiable.

`task/runner._steer` is the caller that re-proposes on a no-route. To catch that type it
has to name it, and naming it means `from earshot.sim.world import NoRouteError` — which
imports habitat-sim into the one module the entire Mac suite needs to be able to import.
Three ways out, and two of them are the shapes this map keeps removing:

- catch `RuntimeError` and sniff `type(exc).__name__` — quietly wrong the first time
  another `RuntimeError` crosses that line;
- `try: from earshot.sim.world import NoRouteError / except ImportError: class
  NoRouteError(RuntimeError)` — the Mac then catches a type nothing raises, so the
  no-route branch is **vacuously green** on every Mac run. That is ticket 13's
  version-blind skip wearing an import guard.

So `NoRouteError` **moves to `earshot/types.py`** and `sim/world.py` re-exports it (it
stays in `__all__`, so `sim.world.NoRouteError` still resolves and the follower's own
docstring stays true). It is a leaf type about routing, not about habitat, and it now
sits where both the raiser and the catcher can name it. The Mac tests raise the **real**
type from a fake follower, so the branch is exercised rather than merely present.

### `earshot/vlm.py` does not exist, and two seams wait on it

ADR-0013's tree names `vlm.py` — the Qwen2-VL-2B connector — as a leaf. No ticket built
it. Ticket 22 declared the `Captioner` protocol at its consumer and ticket 23 declared
`Grounder` at its, both deliberately, so the *seams* are complete and the *connector* is
absent. Two things therefore cannot run:

- `Detector.CAPTION` (`agent/detector.CaptionDetector` needs a `Grounder`);
- `CaptionerRoomLabeler`, so `AgentReport.room` is always `None`.

`make_detector` **raises** for the caption arm with a message naming `vlm.py` rather
than substituting the oracle, because an arm that silently ran the other arm would
produce numbers labelled `caption` in every audit record. Both are R2, which is out of
scope for this map (§4.3: the smoke does not exercise the room gate; §8: the smoke runs
`Detector.ORACLE`), so this is a disclosed gap rather than a new ticket.

### Three homeless things found homes, each disclosed

**CLAP's encoder → `task/models.py`.** `audio/clap.py` is pure and takes an injected
encoder; ticket 22 dissolved the constructor ticket 17's CLAP assertion used to live in,
and ticket 24 recorded that it had nowhere to go. ADR-0013's tree names no module for
the concrete model, so it lands in the wiring layer — a **disclosed addition to the
tree, not a correction to the ADR**: the layer graph is unchanged, and what changes is
that the seam ships with both sides rather than one, which is the property
`audio/clap.py`'s own docstring argues for. torch and transformers are imported inside
the constructor (the discipline `audio/clips.py` applies to scipy), and there is no
fallback encoder — `load_clap_encoder` raises, naming the torch/transformers **pair**
because that is ticket 13's actual failure.

**soft-SPL → `metrics.compute_soft_spl`.** §6 requires it computed and not headlined.
The old tree read it off habitat-lab's own `SoftSPL` measure (`episode_runner.py:2426`
reads `step.info["softspl"]`), and habitat-lab is deliberately not a dependency — so it
is re-derived here from `habitat/tasks/nav/nav.py` with the citation, exactly as
`task/episodes.py` did for the dataset loader. One divergence, at the case habitat-lab
divides by zero on (`start_end_distance == 0`, the cold-start-on-goal this project has
produced before): 1.0 if the agent is still there, 0.0 if it left. 7 new cases.

**The calibration's other half → `runner.calibration_poses`.** `band_poses` returns
geodesic *distances* and says why; this draws navigable points, measures each one's
geodesic distance to the source, and assigns the closest match to each target without
reuse. Deliberately greedy and inexact: a scene may simply have no navmesh at 8 m from
the source, and what matters is that the samples span the band rather than clustering.

### Five corrections to how this would otherwise have been built

**(a) The calibration is per EPISODE, and that is a reading of §2.3 rather than a
departure.** The spec says `onset_rms` is derived "at run start". But ADR-0009 puts one
positioned source in each episode, so the anomaly's RMS distribution is a property of
*that* episode's placement and *that* scene's geometry. One sweep per source is the
faithful form; it is measurement either way (the 6 dB gate still fails on overlap and
the correction is still `globalVolume`), and each threshold lands on its own audit
record with its own separation margin. A run of several episodes therefore carries
several thresholds, which is a fact about the record rather than a hidden one.

**(b) Criterion 1 needs two counters, and ticket 26 must use them.** "Render count
equals step count exactly" is not checkable on `World.n_renders`: arming the guard
performs one render before the first step (it owns the first render — the mesh upload is
lazy), the §2.5 start-pose audibility check performs another, and the calibration sweep
performs `sweep_poses` more. All are legitimately outside the loop. The audit therefore
records **`n_renders_in_loop`** and **`n_loop_steps`**, and criterion 1 is their
equality. Measured on the lifetime counter it would fail for a reason that is not a
defect.

**(c) The dataset builder's separation bar is against EVERY primary view point.** The
old builder measured against the one goal view point it had chosen as the cold start.
These episodes come from the published dataset and the agent succeeds at **any**
instance of the category, so separation from one instance is not separation from the
goal — a source 4 m from instance A can be 0.5 m from instance B, and the detour is a
second route to the goal again. The cost is that a scene can fail to place a source;
that attrition is counted per episode in `DatasetBuild.skipped` and is **distinct from
§2.5's audibility attrition**, which is not screened at all and shows up as the funnel's
stage 3. A Mac test holds the second half structurally: `task/dataset.py` may not import
`audio`, because the only way to pre-screen audibility is to render.

**(d) There is no invented fallback action.** When the follower reports arrival or no
route, the runner re-proposes and tries once more; after two attempts it records
`action: null` and takes **no** simulator step. The old tree's answer here was a
straight-line fallback, which is precisely how "a waypoint was chosen" and "the agent
got there" came apart with nothing in the code marking where. A recorded null is a
visible symptom in the per-step record; an invented action is a trajectory that lies.
`n_no_action_steps` counts them.

**(e) `NaN` and `Infinity` never reach an artefact.** `json.dump` writes both as bare
literals that are not valid JSON, and "unreachable" is a different fact from "the number
is missing". Three metrics (`min_d2g_m`, `primary_dist_at_stop_m`, `min_d2source_m`) are
**omitted** when they do not exist rather than written as sentinels.

### A prediction for the box, from a test that would not converge

The Mac fake's IR started omnidirectional — amplitude falling with distance, ears split
by the lateral offset — and `TestTheStallTurnsTowardTheSource` **could not reach the
source**. The cause is a real property of `realizable_investigate_step` and not of the
fake: its only route back to `move_forward` is `rising`, and if turning in place cannot
raise the measured level, a stalled climb can never re-arm. It turns for ever.

What makes the rule terminate is **front-back gain** — a real head shadows a source
behind it, so turning toward the louder half-plane raises the level and the next tick
walks. The fake now carries a `1 + 0.5·cos(bearing)` directivity term for that reason,
and the ear split is power-normalised so that *only* distance and facing move the level
(an unnormalised split makes a source abeam read louder than the same source dead ahead,
and the climb would be following an artefact).

**So the box has a specific thing to watch.** If SoundSpaces' HRTF gives weak front-back
discrimination at this preset, a stalled realizable climb will rotate until the detour
sub-budget aborts, and the symptom in the record is `investigate_aborted` with a long
run of `turn_*` actions and a flat `measured_rms`. That is a finding, not a bug in the
controller — §4.1 says the rotation-versus-translation conflation is **instrumented, not
fixed**, and the per-step `action` is what makes it decidable from the artefact.

### The one test of mine that caught a real drift

`TestTheRunnerAndTheRealCollaboratorsAgreeOnNames` pins the `World` and
`AudioSensorHandle` APIs the runner reaches for. No Mac can import `sim/world.py`, so
the real subject is read out of its `ast` — ticket 23's move for `PlannerConfig`'s FOV.
It went red immediately: `FakeWorld` had no `sensor_handle`, because `run_episode` never
calls it and only `run()` does. A fake that publishes only what one function happens to
call drifts from the real class in the direction nothing checks. `run()` is the part of
this module that has never executed anywhere, and this is the cheapest assertion that
covers its riskiest failure mode.

### Planted and fired

Four violations, four reds, tree restored and re-run green:

| plant | fires |
|---|---|
| `task/dataset.py` imports `audio.clips` | `test_the_builder_imports_nothing_from_the_audio_layer` |
| `task/dataset.py` imports `sim` | the module cannot even be collected on a Mac |
| a `RunConfig` field with no CLI flag | `test_every_run_level_field_is_settable_or_deliberately_composed` **and** `test_as_dict_carries_every_top_level_field` |
| `World.snap_point` renamed | `test_the_real_world_publishes_everything_the_runner_calls` |

### Handed to the box — this ticket's "Done when", undelivered

An episode has **not** run on the box. This session runs on the Mac, which cannot load
habitat-sim at all, and every box result on this map has come from the operator running
a driver and pasting the log back. The run is the first action of ticket 26's trip, and
it needs one prerequisite the tree cannot supply itself:

```bash
# on RACE, in the ss2 env, from the repo root
conda activate ss2
export PYTHONPATH="$PWD"
python -m earshot.audio.clips --out-dir data/anomaly_audio     # once: stage ESC-50
python -m earshot --run-dir runs/ss2-first-episode --n-episodes 1 --max-steps 250
```

There is deliberately no synthetic clip fallback (ticket 22), so a missing staging
raises with the fetch command in the message rather than calibrating CLAP against a
noise burst. `earshot/tools/box_gate.sh` still runs the box unittest suite and is
unchanged by this ticket.

**What a green Mac suite licenses and does not** (ADR-0014): it licenses the loop's
control flow, the funnel ladder, the artefact split, the criterion-1 accounting, and the
fact that the greedy climb closes on a source whose loudness and lateral cue are real
functions of the pose. It licenses nothing about the follower, the navmesh, the
renderer, or the frame the live cue arrives in.
