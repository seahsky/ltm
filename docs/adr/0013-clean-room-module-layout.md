# The clean room is `earshot/`, a single package whose only simulator-touching module is `sim/world.py`

**Status:** accepted (2026-08-04, grilling session on ticket 18 of the `ss2-clean-room` map).

The agent is ADR-0008.
The task is `docs/anomaly_response_task_spec.md`.
The reset order and carry list are ticket 10.
This ADR fixes the third side: what the modules are, where the seams fall, and what the root is called.

It is what ticket 10's phase 1 has been waiting on, so it is on the critical path to the deletion commit.

## The root is `earshot/`, and the root is the package

One name, not two.
`tools/`, `reference/` and `tests/` are plain subdirectories inside it.

Ticket 18's `<newroot>/<pkg>/env_check.py` notation implied a container root holding a package, and that was inherited rather than decided.
It does not survive contact with this repo: there is **no `pyproject.toml`, no `setup.py`, no lint config and no CI**.
Everything runs as `python -m <pkg>.<mod>` from the repo root, so a nested package needs either a root `__init__.py` — putting a two-level prefix on every import for no benefit — or a `sys.path` insertion, which is the class of silent footgun this map keeps finding.
Nothing here is ever pip-installed (ticket 17 pins an *environment*; it does not build a distribution), so the usual reason to separate root from package does not apply.

`earshot` names the premise that distinguishes this rebuild: the agent knows only what it can hear from where it stands.
It survives the follow-on memory effort plugging into ADR-0008's proposer seam, it is not a dependency name so it cannot go stale if the renderer changes, and it shares no prefix with `embodied_memory` — so a stale import fails loudly instead of reading as a typo during the window when both trees exist.

## The tree

```
earshot/
├── __init__.py          pin_habitat_logging() — the whole file
├── __main__.py          argparse → RunConfig; assert_env(); dispatch
├── config.py            RunConfig; Localization / Detector enums
├── env_check.py         assert_env() — metadata half Mac, capability half box
├── types.py             Pose, Xyz                                    ← leaf
├── metrics.py           verbatim port of the SPL arithmetic          ← leaf
├── vlm.py               Captioner (Qwen2-VL-2B connector)            ← leaf
├── sim/
│   └── world.py         THE only `import habitat_sim` in the tree.
│                        World(scene, sensor_specs), observe(), step(),
│                        navmesh, greedy follower
├── audio/
│   ├── guard.py         ticket 12 + 16, verbatim   ← leaf, stdlib-only
│   │                    arm_audio_context() once · guarded_observe() every step
│   ├── spec.py          audio_sensor_spec() — THE only AudioSensorSpec() call site
│   ├── sensor.py        AudioSensorHandle — arms the guard, set_source,
│   │                    source_is_visible (analyst-only, spec §3.3)
│   ├── bed.py           synthesized diotic bed, mixed after rendering
│   ├── onset.py         one-shot RMS threshold + §3.1 provenance assertions
│   ├── calibration.py   §2.3 sweep → onset_rms, and the separation gate
│   ├── clips.py         resolve_anomaly_clip, ESC-50 fetch, normalize_clip
│   ├── lateral.py       lateral_sign — frame convention pinned by tests/box/
│   ├── clap.py          open-set normal-vs-anomaly gate + prompt banks
│   ├── normality.py     RoomLabeler protocol, ROOM_PRIOR, Null + Captioner impls
│   └── config.py        AudioConfig
├── agent/
│   ├── occupancy.py     depth → occupancy grid
│   ├── proposers.py     FrontierProposer (~300 LOC, rewritten from 1129)
│   ├── scorer.py        picks one waypoint from the pool
│   ├── reachability.py  navmesh filter + snap — injected callables (the invariant)
│   ├── detector.py      GoalDetector protocol · OracleDetector · CaptionDetector
│   │                    (L3 snap-gate kept, OWLv2 dropped)
│   ├── controller.py    NavMode, ControllerState, step_controller,
│   │                    realizable_investigate_step — build_report REMOVED
│   └── config.py        PlannerConfig, ControllerConfig
├── report/
│   ├── agent.py         AgentReport — frozen, exactly §5.1's nine fields
│   ├── audit.py         EpisodeAudit, with AudioContextReport nested
│   └── artifacts.py     THE only writer: write_env_report / write_episode / read_episode
├── task/
│   ├── episodes.py      ObjectNav .json.gz loader — settles ticket 08's box fact
│   ├── dataset.py       the builder: source placement, ADR-0010 |Δy| < 1.0 m,
│   │                    the xz separation rule
│   ├── runner.py        the episode loop; applies ControllerDecision; funnel stages
│   └── smoke.py         the nine acceptance assertions
├── tools/
│   ├── bootstrap_ss2.sh MOVED from .scratch/…/probes/oneenv_gate.sh
│   ├── ss2-constraints.txt
│   └── notify/          notify_email.py, notify-run.sh — as-is
├── reference/           __init__.py raises ImportError, at both levels
│   └── memory/          6 vendored files, inert, + README
└── tests/
    ├── mac/             runs on the laptop
    └── box/             needs the V100 and the `ss2` env
```

## The layer graph, and why it is enforced

```
types, metrics, audio.guard, vlm   → nothing                    (leaves)
sim                                → audio.guard, types
audio                              → audio.guard, vlm, types     (NOT sim)
agent                              → vlm, types                  (NOT sim)
report                             → audio.guard, types
task                               → everything                  (the only wiring layer)
```

The load-bearing edge is the one that is **absent**.

Ticket 12's guard is Mac-testable with 42 tests and no `habitat_sim` because every simulator object it touches is injected.
That pattern generalises: `audio/sensor.py` takes an injected `observe` callable and the sensor handle, `agent/reachability.py` takes injected `snap_point` and `geodesic`, `agent/detector.py`'s oracle takes an injected distance function.
So neither `audio/` nor `agent/` imports `sim` at all, `import habitat_sim` appears in exactly one file, and `task/` is the only layer that wires the two together.

This is not tidiness.
It is what makes ticket 19's Mac surface most of the tree instead of a corner of it, and it turns requirement 1(a) — "whichever module touches the simulator first is a designed fixed point" — into a one-line check rather than a property to be argued.

**Three structural invariants, all Mac-runnable**, in the spirit of ADR-0008's "assert the invariant instead of carrying the flag":

- `tests/mac/test_layering.py` walks each module's `ast` imports and asserts only the edges above exist.
  Both backwards dependencies rejected during the grilling — `agent/` reaching into `audio/` for a depth frame, `audio/` reaching into `agent/` for a room label — are one convenient import away, and this is what stops them.
- `tests/mac/test_report_boundary.py` — see below.
- `tests/mac/test_no_env_flags.py` asserts `os.environ` appears nowhere outside `guard.py`'s pin and `env_check.py`.
  That is ADR-0008's "no flag surface" made checkable rather than intended.

## The fixed points the requirements demanded

### `earshot/__init__.py` pins the logging, and Python enforces the ordering

The whole file is one call to `pin_habitat_logging()`.

`HABITAT_SIM_LOG` is read at habitat-sim import time, so a late pin is a silent no-op — which is why ticket 12's implementation **raises** if `habitat_sim` is already in `sys.modules` (`audio_guard.py:660`) rather than doing nothing.
Putting the call in `__init__.py` means every path into the package runs it first *by construction*: there is no convention for an entry point to forget, and no REPL or ad-hoc box script that can bypass it.
It stays cheap and Mac-safe because `guard.py` is stdlib-only (`os`, `re`, `sys`, `tempfile`, `dataclasses`, `typing`) — no habitat-sim, no torch, no models, no measurable import cost.

`earshot/sim/world.py` additionally asserts the pin is in force immediately before its `import habitat_sim`.

**This dissolves a contradiction inside ticket 17.** That ticket specifies `assert_env()` as running "at the entry point, **before** `import habitat_sim`", while one of its three checks is "`habitat_sim` audio via the enum **member** probe" — which cannot be done without importing habitat-sim.
With the pin in `__init__.py`, importing `earshot.env_check` has already run it, so `env_check` is free to import habitat-sim for its enum probe.
`assert_env()` stays an explicit call from entry points because it is expensive and half box-only, not because of ordering.

### `sim/` owns the simulator lifecycle; `audio/` owns the sensor

The audio side has to participate at two moments of the simulator's life: the `AudioSensorSpec` must exist *before* `Simulator(cfg)`, and the `AudioSensor` handle only exists *after* it.

`earshot/sim/world.py` is audio-blind — it is handed a list of sensor specs and returns a dict of observations.
`earshot/audio/spec.py` is the only place in the tree that constructs an `AudioSensorSpec`, so requirement 2's key validator (`apply_audio_config` + `assert_no_swallowed_keys`) is structural rather than remembered — a bare `setattr` elsewhere has nowhere to happen.
`earshot/audio/sensor.py` wraps the handle and arms the guard in the same constructor, satisfying requirement 1(b)'s "whichever module constructs the sensor also arms it" literally.

**The per-step observation is one shared call.** `sim.get_sensor_observations()` returns RGB, depth and the audio IR together (`oneenv_probe.py:629`); there is no separate audio render.
So the depth frame the frontier proposer consumes and the IR the onset detector consumes come out of the same call, and smoke criterion 1's "render count equals step count" is measured on it.

### The guard runs on every render, split into two entry points

`arm_audio_context()` stays the heavy once-per-episode call: the scene-OBJ write (0.814 s / 32.2 MB, ticket 16), the 10,000-vertex floor, the key validation.
A new light `guarded_observe()` wraps every step: fd 1 + 2 capture, the canary, the fatal-line scan, and no OBJ write.

Ticket 16 measured that `[Audio]` is logged on **every** render (`AudioSensor.cpp:130`), so "the canary stays armed for the whole episode, not just at arm time".
That measurement only has a consumer if the guard actually scans every render — and it needs one, because the same ticket found that the closed engine writes un-prefixed error blocks to fd 2 and that `RLRA_SetListenerHRTF` returns `Success` over a failed load.
Those failures can happen at step 300 as easily as at step 0.
The cost is two tempfiles per step, and it lands inside the per-step audio wall-clock the task spec already requires reporting every run — so it is audited, not assumed.

**This ADR narrows ticket 18's requirement 1(d).** That requirement reads "nothing else in the process may write to stdout during that window — no progress print, no `tqdm` bar, no stdout logging handler."
`capture_habitat_logs` flushes Python's own buffers on both `__enter__` and `__exit__` (`audio_guard.py:245-252`), precisely so the caller's pending bytes do not land in the capture.
So interleaved, in-thread `print()` between steps is **safe**.
What is actually forbidden is a *concurrent* writer to fd 1 or 2: a background thread, a timer-driven progress bar, a subprocess that inherited the descriptor, a logging handler flushed off-thread.

### The agent's testimony cannot reach ground truth, by type

`build_report` leaves `agent/controller.py`.

The leak requirement 10 wants closed is not hypothetical — it is the current code.
`build_report` (`anomaly_controller.py:302-316`) emits `"source_xyz": ev.get("source_xyz")` out of `ControllerState.investigation_event`, returns an untyped `Dict[str, Any]`, and mutates the state it was handed.
Ticket 10's "ports near-verbatim" would have carried all three in.

"The controller cannot see ground truth" is not available as the rule, because the oracle arm's controller legitimately holds `source_xyz` as its waypoint while the task spec requires an **identical schema in both arms**.
So the boundary is drawn at the report type instead: `AgentReport` is a frozen dataclass with exactly §5.1's nine fields and no others, so nothing privileged can appear in it whatever the controller holds.
`EpisodeAudit` holds the privileged set.
`tests/mac/test_report_boundary.py` asserts the field-name sets are disjoint and that `report/agent.py` imports nothing that can supply ground truth.

This is a **documented deviation from ticket 10's "near-verbatim"** for the controller, and it also fixes the state mutation.

### One writer, one run directory, testimony in its own file

Requirements 1(c), 8 and 10 each add a claimant and say the three "should be one location, not three" — but they are not the same scope.
`env_report.json` is per-run; `AudioContextReport` and the audit are per-episode; and `AgentReport` is a fourth artefact that must stay separable.

`earshot/report/artifacts.py` is the only module in the tree that writes anything.

```
runs/<tag>/
├── env_report.json          per-run  (requirement 8)
└── episodes/
    ├── ep0000.agent.json    §5.1 only
    ├── ep0000.audit.json    §5.2, with audio_context nested
    └── …
```

Splitting the testimony into its own file buys a property the paper can use: a reviewer can be handed `ep0000.agent.json` and physically cannot see the answer key.
The realizability claim stops being a source-level argument and becomes a demonstrable artefact.
`runs/` is already gitignored and the notify/RACE tooling already knows it.

### `reference/` is kept out by a raising `__init__.py`, not by an absent one

Ticket 10 vendors ~3,400 LOC deliberately **broken** — it imports `faiss` and `sentence-transformers`, and `memory_bridge.py` is built against the deleted `episode_runner` and the env-flag surface ADR-0008 removed — and states that if it can be imported by accident, vendoring it was a mistake.

Omitting `__init__.py` does not achieve that, and this was verified rather than assumed: PEP 420 namespace packages make `earshot.reference.memory.ltm` import cleanly from a regular parent package (Python 3.3+, so it holds on the box's 3.9).
The only thing that currently stops it is `faiss` not being installed in the `ss2` env, which is luck, and which flips the day someone installs faiss to work on the memory follow-on.

So `reference/__init__.py` and `reference/memory/__init__.py` each raise `ImportError` with a pointer to the README.
Active rather than absent: it survives faiss being installed, and it explains itself at exactly the point a confused session lands.

### Configuration is one `RunConfig` of per-module frozen sub-configs, from the CLI

Each module defines its own frozen config beside itself — `AudioConfig`, `PlannerConfig`, `ControllerConfig` — and `earshot/config.py` composes them into one `RunConfig` built from `argparse` in `__main__.py`.
The two surviving experimental arms (ADR-0008: oracle vs realizable localization, oracle vs caption-grounded detector) are **enums**, not booleans, so a third option is addable without a flag explosion.
The old tree read `LTM_REALIZABLE_LOCALIZATION` from the environment at the runner; `test_no_env_flags.py` makes sure nothing does that again.

**`onset_rms` is not configuration.** Task spec §2.3 derives it at run start from the calibration sweep; config holds the bed level and the audible band to sweep.

### Tests split by where an assertion can run

`earshot/tests/mac/` and `earshot/tests/box/`.

Ticket 19 owns the *strategy*; this ADR owns the paths, and it makes the strategy's central rule a directory rather than a convention — "the Mac suite" is `python -m unittest discover earshot/tests/mac`, not something to remember.
It also keeps `earshot/`'s shipping modules free of test files, which is what makes requirement 6's "outside the test surface" a path property.

Ticket 17's `env_check` deliberately splits **across** these: its metadata half in `mac/`, its capability half in `box/`.
That is ticket 19's "per-assertion, not per-module" rule showing up in the tree on day one.

### Probes carry only where a live consumer exists

`.scratch/ss2-clean-room/` is tracked (`.gitignore:129` is `scratch/`, not `.scratch/`), so leaving a probe there is durable, not ephemeral.

- **Move in:** `oneenv_gate.sh` → `tools/bootstrap_ss2.sh` (ticket 17), `audio_guard.py` → `audio/guard.py`, `test_audio_guard.py` → `tests/mac/`, and `audioguard_probe.py` + `audioguard_gate.sh` → `tests/box/` — because ticket 16's box verification is exactly the box-side check the guard must keep having once the tree depends on it.
- **Stay:** `vram_probe.py` (ticket 15 retired VRAM as a constraint), `box_inventory.py` (one-off), `rendercost_probe.py` and its sweep and tests (criterion 7 measures wall-clock *in the runner* now), `oneenv_probe.py` (the gate's payload), `kill_nrun.sh` (operational, cited by the runbook), `patches/` (empty — ticket 09 ruled no fork).

`tools/` is operator-facing and not part of the agent, which is also why the notify trio lands at `tools/notify/`.
The dataset builder is **not** a tool: it enforces ADR-0010's source placement, which is task policy, and it shares the episode schema with the loader, so it sits at `task/dataset.py`.

## Python 3.9 constrains two of these

The box is **Python 3.9.19** (ticket 13's known-good set).

- `int | None` in an annotation needs `from __future__ import annotations`, and even then `typing.get_type_hints()` on those dataclasses raises.
  `test_report_boundary.py` must read `__dataclass_fields__` names, not resolved hints.
- `Protocol` is available from `typing` in 3.8+, so the `GoalDetector` and `RoomLabeler` seams need no backport.

## Considered and rejected

- **A container root holding a package** (ticket 18's literal notation). Rejected: with no `pyproject.toml` it costs either a two-level import prefix everywhere or a `sys.path` insertion, and buys nothing that is not hypothetical for a tree that is never installed.
- **`embodied_audio` as the name.** Closest to repo habit. Rejected: that is exactly the near-miss risk — tab-completion is ambiguous while both trees exist, and a stale `embodied_memory` import reads as a typo rather than as deleted code.
- **One `world` module owning sim, audio and the guard together.** Honest about the shared observation call and needs no two-phase protocol. Rejected: it makes the one module that can never be Mac-tested the largest in the tree, and onset / CLAP / bed / calibration have to live outside it anyway, so the split reappears one layer up and less visibly.
- **Arming the guard on the first render only.** Cheapest. Rejected: it makes ticket 16's every-render canary measurement a fact with no consumer, and leaves a context that degrades mid-episode invisible — the exact silent-fabrication class tickets 12, 13 and 16 each caught.
- **Keeping `build_report` in the controller with `source_xyz` deleted by hand.** Ticket 10's literal instruction, one fewer module. Rejected: enforcement drops to a convention, and the code being ported is itself the proof that the convention fails.
- **A hyphenated `reference-memory/` directory.** Airtight and zero code. Rejected: it explains nothing at the failure point, and it silently contradicts ticket 10's written path.
- **A YAML/JSON run config file.** Diffable and commitable. Rejected: a parse-and-validate layer for roughly eight numbers and two enums, when the audit record already captures what was run.
- **Documenting the layer graph without enforcing it.** Rejected on this repo's own record: ticket 14's self-update gotcha, ticket 17's inert pin and ticket 13's version-blind skip are all things that were written down and then quietly stopped being true.

## Consequences

**Ticket 10's phase 1 is unblocked.** The root exists as a name and a shape, so vendoring and porting can start, which puts the deletion commit back on a critical path with no open questions in front of it.

**Ticket 19 inherits two constraints from this ADR rather than deciding them freely.** The mac/box directory split pre-commits it to a where-it-runs taxonomy, and the injection rule above means its Mac surface is most of the tree — `audio/`, `agent/`, `report/`, `metrics`, `types`, `vlm`'s interface — rather than the four layers it listed.

**Three things ticket 18 stated are corrected here**, all in the direction of less work rather than more: requirement 1(d) forbids concurrent fd writers, not interleaved prints; requirement 6 needs an active guard because an absent `__init__.py` does not work; and ticket 17's `assert_env()` ordering contradiction is dissolved rather than worked around.

**One deviation from ticket 10 is taken deliberately and disclosed**: the anomaly controller does not port near-verbatim. `build_report` moves out, and the state mutation goes with it.

**`import habitat_sim` in a second file is now a test failure.** That is the intended cost: adding a simulator dependency to `audio/` or `agent/` is meant to be a decision someone makes on purpose, not one that happens because a depth frame was convenient to reach.
