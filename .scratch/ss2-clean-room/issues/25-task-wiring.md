# 25 — `task/` wiring: the runner, the dataset builder, and the CLI

Type: task
Status: open
Blocked by: 21, 22, 23, 24

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
