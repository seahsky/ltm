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

## Added by ticket 19 (resolved 2026-08-04) — the test strategy's scaffolding

`docs/adr/0014-test-strategy-across-the-mac-box-split.md` decided the strategy, and five of its pieces are scaffolding rather than test content, so they land here rather than in a new ticket.

**The shared structural walker.** The three invariants above do **not** each roll their own tree walk. One helper in `tests/mac/` enumerates the live tree, **denylist-shaped** so new top-level code is checked by default, with `reference/` as the sole exemption — and a test asserting the exemption set equals exactly `{reference}`, so widening it fails before it lands.

**The `earshot-mac` env and its one dependency declaration.** A conda env at **Python 3.9** with numpy under the box's `< 1.24` pin, installing from `earshot/tools/mac-requirements.txt`. The CI job installs from the same file: two declarations of one set would drift silently. Do **not** reuse `ltm-embodied` (deleted tree's env, numpy 1.26.4 sits above the pin).

**The interpreter refusal.** The Mac suite asserts its own interpreter is 3.9.x at start and refuses otherwise. This Mac's default `python3` is 3.14.3, so without it a green run proves nothing about the box — and ADR-0013's two 3.9 constraints (`int | None`, `get_type_hints()`) are invisible under 3.14.

**`ruff.toml`** — rules `F` + `E9` **only**, `reference/` excluded. Not style, not import ordering. It exists for one file: `sim/world.py` is the sole `import habitat_sim` module, so the Mac can never import it, so a static AST check is its only Mac-side verification.

**The CI workflow** — one job, `ubuntu-latest`, Python 3.9, `unittest discover earshot/tests/mac` plus the ruff check, on push and PR, **named for its scope** so green cannot read as "the stack works".

**Two ports change shape from what is written above.** `audioguard_probe.py` becomes a `unittest` suite in `tests/box/` (its four negative controls as test methods, printing their measurements); `audioguard_gate.sh`'s driver concerns move to `earshot/tools/box_gate.sh` — **carried, not rewritten**, keeping the SIGPIPE-safe conda directory check, the enum-member preflight, and the pip-freeze-first ordering.

**One fact to verify here rather than assume**: whether `actions/setup-python` still provides Python 3.9 in 2026 (EOL late 2025). If it does not, pin the nearest available and widen the interpreter refusal to match, and record that the refusal is weakened. ADR-0014 discloses this as a decision for this ticket, not a discovery in a red build.

## Done when

`python -m unittest discover earshot/tests/mac` is green **under the `earshot-mac` 3.9 env** (and refuses to run under this Mac's default 3.14), `ruff check earshot/` is clean, the CI job is green, `import earshot` works and pins `HABITAT_SIM_LOG`, and `import earshot.reference.memory.ltm` raises `ImportError` with the intended message rather than a `ModuleNotFoundError` about faiss.

## Watch for

Ticket 12's warning shot applies to `guarded_observe()`: it is new code, so its fakes have never met the binary. Its box confirmation is ticket 26's, not this ticket's — do not let a green Mac suite read as verified.

ADR-0014 states the licence to quote here: **a green Mac suite is evidence about our own logic and nothing else.** Ticket 12's guard passed 27 fake-based tests and then raised on the first real spec.

Every fake added here that reproduces a third-party behaviour carries a `file:line` citation and a note on what breaks if it changes (ADR-0014's source layer, review-enforced). Every tuned constant carries its provenance: `box` / `source` / `fake` / `runtime`.
