# 24 — `report/` and `env_check.py`

Type: task
Status: open
Blocked by: 20

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
