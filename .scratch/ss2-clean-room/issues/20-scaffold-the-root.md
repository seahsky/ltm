# 20 — Scaffold `earshot/` and its three structural invariants

Type: task
Status: open
Blocked by: 19

## Question

Create the root, the leaves, the inert `reference/`, the moved `tools/`, and the three invariant tests — everything in ADR-0013's tree that needs no simulator.

This is ticket 10's phase 1 for the parts that are pure movement plus the parts that assert the tree's own shape.
Nothing here imports `habitat_sim`, so all of it runs on the Mac.

## What to build

**The root.** `earshot/` with `__init__.py` containing exactly one call to `pin_habitat_logging()` and nothing else (ADR-0013: Python's import machinery is the enforcement, so the file must stay cheap — `guard.py` is stdlib-only, keep it that way).

**The leaves.** `types.py` (`Pose`, `Xyz`), `metrics.py` (verbatim from `embodied_memory/metrics.py`, 55 LOC, one pure function), `audio/guard.py` (verbatim from `.scratch/…/probes/audio_guard.py`, plus the new `guarded_observe()` split out of `arm_audio_context()` per ADR-0013).

**`reference/`.** Vendor ticket 10's six files inert, with `reference/__init__.py` and `reference/memory/__init__.py` each raising `ImportError` pointing at the README, and the README's five required sections from ticket 10.

**`tools/`.** `bootstrap_ss2.sh` **moved** (not copied) from `probes/oneenv_gate.sh` per ticket 17, `ss2-constraints.txt` with its 9 pins plus the habitat-sim SHA `4f61e321`, and `tools/notify/` carrying the three notify files as-is.

**`tests/{mac,box}/`** with the three invariants:
- `test_layering.py` — walks each module's `ast` imports, asserts only ADR-0013's edges exist, and asserts `import habitat_sim` appears in exactly one file.
- `test_report_boundary.py` — stub until 24 lands; assert the harness reads `__dataclass_fields__`, not `get_type_hints()` (Python 3.9).
- `test_no_env_flags.py` — `os.environ` appears nowhere outside `guard.py`'s pin and `env_check.py`.

Plus `test_audio_guard.py` ported into `tests/mac/` (42 tests, must stay green), and `audioguard_probe.py` + `audioguard_gate.sh` into `tests/box/`.

## Done when

`python -m unittest discover earshot/tests/mac` is green on this Mac, `import earshot` works and pins `HABITAT_SIM_LOG`, and `import earshot.reference.memory.ltm` raises `ImportError` with the intended message rather than a `ModuleNotFoundError` about faiss.

## Watch for

Ticket 12's warning shot applies to `guarded_observe()`: it is new code, so its fakes have never met the binary. Its box confirmation is ticket 26's, not this ticket's — do not let a green Mac suite read as verified.
