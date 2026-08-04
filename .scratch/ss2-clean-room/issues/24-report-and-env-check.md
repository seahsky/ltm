# 24 — `report/` and `env_check.py`

Type: task
Status: resolved
Blocked by: 20 (resolved 2026-08-04)

## Question

Build the two output types, the one writer, and ticket 17's runtime assertion.

Grouped because they share a destination: `env_report.json`, `AudioContextReport` and the episode audit were three separate claimants on "somewhere to land", and ADR-0013 made them one location.

## What to build

**`report/agent.py`** — `AgentReport`, a frozen dataclass with **exactly** task spec §5.1's nine fields and no others: `primary_completed`, `heard_at_step`, `room`, `anomaly_class`, `stopped_at_pose`, `visual_confirm_object`, `investigate_aborted`, `resumed`, `n_benign_ignored`.

`source_xyz` is gone. The oracle arm's privilege shows in its trajectory and its audit record, never in its testimony — and the schema is **identical in both arms**, which is what makes "the sound is just a stopwatch, the coordinate is handed to the agent" answerable by reading the type.

**`report/audit.py`** — `EpisodeAudit`: ground-truth source position, distance-at-STOP, the `sourceIsVisible()` history, §3's provenance assertions and the measured pre-onset bed RMS, the calibration separation margin and threshold in force, the funnel stage reached, and per-step audio render wall-clock. `AudioContextReport` nests inside it.

**`report/artifacts.py`** — the **only** module in the tree that writes anything.

```
runs/<tag>/
├── env_report.json          per-run
└── episodes/
    ├── ep0000.agent.json    §5.1 only
    ├── ep0000.audit.json    §5.2, audio_context nested
    └── …
```

Testimony gets its own file so a reviewer can be handed `ep0000.agent.json` and physically cannot see the answer key — the realizability claim becomes a demonstrable artefact rather than a source-level argument. `read_episode()` exists because ticket 26's smoke asserts against these.

**`env_check.py`** — ticket 17's `assert_env()`, importable so `bootstrap_ss2.sh` (`python -m earshot.env_check --strict`) and the runtime share one implementation. Returns a report that lands as `env_report.json`.

It splits **across** `tests/{mac,box}/`, which is the clearest case that ticket 19's rule is per-assertion:
- **Mac:** numpy `< 1.24` metadata, the constraints-versus-resolved comparison.
- **Box:** a real sm_70 allocation, the `habitat_sim` audio enum **member** probe, a live `ClapModel` instantiation.

Enforcement is **capability-shaped, not provenance-shaped** — ticket 13's failure would have passed every version check.

## Done when

`tests/mac/test_report_boundary.py` is real rather than a stub: the field-name sets are disjoint, and `report/agent.py` imports nothing that can supply ground truth. It must read `__dataclass_fields__`, not `typing.get_type_hints()`, which raises on `from __future__ import annotations` under Python 3.9.

## Watch for

Ticket 17's knock-on, still outstanding: `docs/race-box-runbook.md` §3 says its version table is "a record of what worked once, not a lockfile". That goes stale the moment `ss2-constraints.txt` exists. Fix the sentence here or in ticket 27's doc pass.

---

## Built, 2026-08-05 — and ADR-0013 contradicted its own enforcement

Five modules, **62 new Mac tests (422 → 484 green)**, ruff clean, one new box file, and the
bootstrap's verdict is now the assertion the runtime shares. Every structural invariant on
the new code was verified by **planting a violation** and watching it fire — seven plants,
seven reds, tree restored and re-run green.

### The finding: the one-importer rule and ADR-0013's own dissolution of ticket 17 collide

`test_layering.py` went red on the first run, and it was right to.

ADR-0013 states the one-importer rule — `import habitat_sim` appears in exactly
`sim/world.py` — and then, dissolving ticket 17's ordering contradiction, states that
"importing `earshot.env_check` has already run [the pin], so **`env_check` is free to
import habitat-sim for its enum probe**". Both halves are in the same ADR. Ticket 24 is
the first ticket to need them at the same time.

**The exemption is real and cannot be met any other way.** The audio enum **member** probe's
subject *is* habitat-sim's build: `AudioSensorSpec` is bound even in non-audio builds
(habitat-sim #2340), so only resolving `SensorType.AUDIO` distinguishes them. It cannot
arrive injected, because `env_check` runs **before** a `World` exists — on an environment
that may not be able to build one, which is the case it exists to catch — and `env_check`
sits at layer `()` so it cannot import `sim` to ask.

`importlib.import_module("habitat_sim")`, which the AST walker does not see, was considered
and **rejected outright**: dodging a structural test with a dynamic import is the
written-down-then-quietly-untrue pattern these invariants exist to stop.

So the allowlist is two, in `_tree.SIMULATOR_IMPORT_ALLOWED`, each entry carrying its
reason — the shape `ENV_ACCESS_ALLOWED` already had. And it comes with the same discipline:
a test that the exemption is **still spent on what earned it**, asserting the reach sits
inside `probe_habitat_sim_audio_enum_member` and nowhere else. A *top-level*
`import habitat_sim` there would be the real regression — it would fire on every import of
the module including from a Mac, turning a probe that reports `NOT_RUN` into an
`ImportError` at the entry point of a tree that is otherwise fine.

### `env_check` is shaped around ticket 13's *other* bug

The pin/capability half is ticket 17 verbatim: every probe does the thing — allocates on the
GPU and reads the result back, resolves the enum member, instantiates CLAP and reads a
finite logit — because `transformers` reported **4.57.6 both before and after** the fix and
`ClapModel` imported cleanly the whole time it was a `DummyObject`.

The module's *shape*, though, answers the bug underneath that one: the gate's torch layer
**skipped on mere importability and reported success**. A layer that computed the right
answer and then did not use it. So:

- **`ProbeStatus.NOT_RUN` exists and is never green.** An import that raised, a probe that
  could not complete, a check that was skipped: one verdict.
- **`judge()` takes the set of probes it expects**, and a missing name is red. A probe that
  silently stopped being emitted cannot pass by absence — the inert-pin class, one level up.
- **`judge()` is pure**, which is ticket 19's third row (*given a failing probe result, does
  `assert_env()` raise*) made testable with no box at all. That row is now 11 Mac tests.
- **`assert_green()` is split out of `assert_env()`** so the *raise* is Mac-testable too,
  plus a static test that `assert_env` still calls it. Without that pair, `assert_green`
  could be perfect and unreferenced — `assert_env` would compute the right answer and hand
  it back green, which is the original bug with a new coat on.

### Three things ticket 17 specified that did not survive contact, all disclosed

**(a) `assert_clap_instantiable()` has no `AudioClassifier.__init__` to live in.** Ticket 17
put it there, "paid only by runs that use it". Ticket 22 made `audio/clap.py` **pure** — the
encoder is injected — so there is no such class in the clean room and the construction
happens in `task/`. The probe therefore lives here, **requested rather than required**
(`--clap` / `assert_env(clap=True)`), which preserves ticket 17's cost argument exactly
while putting the code where the layer graph allows it.

**(b) `assert_env()` does not write `env_report.json`.** ADR-0013 makes `report/artifacts.py`
the only writer in the tree, and `env_check` sits at layer `()` so it cannot reach it. So
`--strict` **prints** and exits non-zero; `task/` does
`write_env_report(run_dir, assert_env().as_dict())`. The report still "lands as
`env_report.json`" — through the one writer, which is the stronger reading of both tickets.

**(c) Stage 7's provenance comparison was one rule in two languages.** It lived as an inline
bash heredoc, and ticket 19 assigns exactly that comparison to `env_check.py`'s
Mac-testable half. That is the drift trap ticket 17 refused to accept for the build recipe,
sitting inside the script ticket 17 wrote. There is one implementation now, unit-tested
against `ss2-constraints.txt` itself — including the two cases only it can see (an **inert
pin**, and a resolver that ignored a pin) and the one it must not report (`torch==2.2.2`
matching a resolved `2.2.2+cu118`; PEP 440 makes `+cu118` a local version, and a raw string
compare would cry skew on the package this env is most careful about).

### The bootstrap's verdict is now the runtime's assertion

`bootstrap_ss2.sh` grew from 8 stages to 9. Stage 7 calls
`python -m earshot.env_check --provenance` (WARN, no `--strict` — the recipe already
succeeded and killing a 40-minute build over an unjudged skew destroys the evidence); the
new **stage 8 is `--strict`, and it gates**.

**The ticket-04 probe is kept, demoted to stage 9, not replaced.** The map's hand-off said
ticket 24 replaces it — but `env_check` never opens a scene, and this script's headline is
"audio sensor renders **in a scene**". Replacing it would have traded a stronger verdict for
a weaker one. So the assertion and the render are two stages, both gating.

### `report/`: the boundary is drawn twice, once on the type and once on the bytes

`test_report_boundary.py` was **armed** — its `skipTest` scaffolding is gone, so a subject
that disappears is a red rather than a green skip — and its privileged-name pin gained the
check it was missing: every name pinned as privileged must genuinely be **on** the audit,
or the exclusion is enforcing nothing for it. (It reads properties as well as fields,
because `source_is_visible_history`'s correct home *is* a property derived from the
per-step rows; storing it twice would be the drift trap, and a field-only check would have
called the right design a violation.)

`test_report_artifacts.py` then asserts the same property one layer down, **on the bytes** a
reviewer actually meets: the testimony file contains no privileged key, and it reads back
with the audit file **deleted** — asserted by deleting it, because a reader that quietly
reached for its sibling would pass every other test in the file.

Four decisions inside `report/` worth naming:

1. **The two audio projections are not the audio types.** `report` may reach only `report`,
   `audio.guard` and `types`, so `CalibrationRecord` and `OnsetRecord` mirror
   `CalibrationResult` and `OnsetState` — and §5.2 asks for *the separation margin and the
   threshold in force*, not the whole sweep, so the narrowing is the spec rather than a
   workaround. The drift is closed the way ticket 23 closed the frame convention: a **test**
   sits outside the layer graph, imports both sides, and asserts every projected name still
   exists on its source. Planted a rename; it fired.
2. **`OnsetRecord.provenance_asserted` is the audit's own field.** §3.1's assertions raise,
   so a record that exists *looks* like proof they passed — unless they were never called.
   That is ticket 16's log canary in a third costume, and it is what makes smoke criterion 4
   checkable from the artefact rather than from the absence of a traceback.
3. **`FunnelStage` is an `IntEnum`.** §6's denominator is stage 2, so per-stage counts are
   `stage >= T_ANOM_REACHED`. With a plain `Enum` that ordering would live in whichever
   module happened to aggregate — inside the one metric §6 singles out because an aggregate
   has hidden the mechanism on this project before.
4. **`audio_render_summary()` returns `{}` rather than zeros when nothing rendered.** A
   ceiling check against a fabricated `0.0` would pass criterion 7 on an episode whose audio
   never rendered at all. `n_render_steps` is criterion 1's numerator for the same reason.

**Writes are atomic and refuse to overwrite.** Both answer one incident: this project's own
audit found committed run directories holding a different run's data, quoted against numbers
they did not come from. A half-written JSON and a silently re-used `--tag` are the two ways
that happens without leaving a trace. `AgentReport.from_dict` refuses unknown keys for the
third form of it — a tolerant reader is how a privileged field gets into a run directory and
out again while the type stays clean and the disjointness test stays green.

`types.Pose.from_dict` was added (the inverse of the existing `as_dict`) so the pose's flat
layout is known in one module rather than re-derived in the reader.

### One test of mine would have passed if the code were wrong

Caught on a self-audit before the ticket closed, in the shape ticket 23's adversarial pass
kept finding. `test_a_red_report_raises_with_the_summary_attached` staged its **own**
`raise EnvCheckError(...)` inside the test and asserted the message — so it passed whether
or not `assert_env` raised at all. That is what forced the `assert_green` split above: the
raise is now a real function the test calls, plus the static check that `assert_env` still
calls it.

### Everything the ticket asked for

`AgentReport` is frozen with exactly §5.1's nine fields and no others; `source_xyz` is gone.
`EpisodeAudit` carries §5.2's list with `AudioContextReport` nested. `report/artifacts.py`
is the only writer and lays out `runs/<tag>/` as ADR-0013 draws it; `read_episode` exists
for ticket 26. `env_check.py` is importable, shared by both callers, splits three ways
across `tests/{mac,box}/`, and is capability-shaped throughout. The "done when" is met:
`test_report_boundary.py` is real rather than a stub, reads `__dataclass_fields__` rather
than `typing.get_type_hints()`, and its two fixture tests still pin *why*.

The box file ships both arms where they exist (a bogus `SensorType` member, an unresolvable
model id) and prints its measurements. The GPU allocation probe has **no forced-failure
arm** — you cannot uninstall CUDA for one test — which is ticket 19's disclosed permanent
gap, named in the file rather than papered over.

### The "watch for" was already discharged

`docs/race-box-runbook.md` §3's "a record of what worked once, not a lockfile" was fixed by
ticket 20 when `ss2-constraints.txt` landed. §4 gained the new material instead: the verdict
is `env_check --strict`, it is the same assertion the runtime makes, and here is how to run
it standalone on the box in seconds.
