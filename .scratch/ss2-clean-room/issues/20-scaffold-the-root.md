# 20 — Scaffold `earshot/` and its three structural invariants

Type: task
Status: resolved
Blocked by: 19 (resolved 2026-08-04)

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

---

## Answer

**Built, and green on every gate the ticket named** (commit `2c705dc`, branch
`wayfinder/ss2-clean-room-20`).
`earshot/` exists, three structural invariants assert its own shape over one shared
walker, and the tree is now something tickets 21–24 add modules *to* rather than
argue about.

Measured, not claimed:

| gate | result |
|---|---|
| `python -m unittest discover earshot/tests/mac` under `earshot-mac` 3.9.25 | **84 tests OK** (5 skipped — the report-boundary assertions wait on ticket 24) |
| the same command under this Mac's default 3.14.3 | **exit 1, all 7 modules error** — the refusal holds |
| `ruff check earshot/` | **clean over 16 files**; `reference/` genuinely excluded (11 `F` errors when forced with `--no-force-exclude`) |
| CI, `ubuntu-latest` | **green in 31 s** — CPython **3.9.25**, `numpy-1.23.5 ruff-0.14.4` from the shared requirements file, 84 tests OK |
| `import earshot` | pins `HABITAT_SIM_LOG=Sensor,Assets=Debug` |
| `import earshot.reference.memory.ltm` | raises the intended `ImportError`, **not** a `ModuleNotFoundError` about faiss |

Test population: 52 in `test_audio_guard.py` (**42 carried, all still green**, plus 10
new for `guarded_observe`), 8 in `test_metrics.py`, 24 structural, and 10 box tests
awaiting the box.

### The one fact this ticket was told to verify rather than assume

**`actions/setup-python` still provides Python 3.9, so the interpreter refusal did not
have to widen.**
Checked against `actions/python-versions`' `versions-manifest.json` on 2026-08-04:
**32 `3.9.x` entries, up to 3.9.25, `linux-x64` present**.
CI then resolved 3.9.25 in practice.
ADR-0014 disclosed this as its one known way for the refusal to weaken; it did not
happen, and the disclosure can be closed rather than carried.

### Four corrections, all found by building

**1. The walker's excluded set is `{reference, tools, tests}`, not `{reference}`.**
ADR-0014 asks for "a test asserting the exemption set equals exactly `{reference}`".
That set would have produced a **red suite on day one**, and the two it misses are
load-bearing rather than cosmetic — both invariants fire on the test tree itself:

- `tests/box/` **must** `import habitat_sim`. It drives the real artefact, which is
  ADR-0014's own definition of a box test, so the one-importer rule cannot cover it.
- `tests/mac/test_audio_guard.py` **must** touch `os.environ`. `TestLoggingPin` saves
  and restores `HABITAT_SIM_LOG` around the pin it is testing.

`tools/` is the third, and ADR-0013 already supplies the reason in its own words —
"`tools/` is operator-facing and not part of the agent" — which is why
`notify_email.py` reading `RESEND_API_KEY` is credentials rather than the flag surface
ADR-0008 removed.
So there are now **two pins, not one**: `test_walker_scope.py` asserts the excluded set
exactly, and `test_no_env_flags.py` asserts its own allowlist is the two modules the ADR
names. Each entry in `_tree.NON_AGENT_ROOTS` carries its reason inline, and a further
test asserts every excluded root **exists on disk** — an exclusion for a directory that
is not there is an inert pin, the same class as ticket 17's constraint on a package that
is never installed.

The failure direction is unchanged and is the whole design: a new `earshot/experimental/`
is checked **by default** and fails `test_every_agent_module_is_in_the_graph` until
someone puts it in ADR-0013's graph or in the excluded set with a reason.

**2. The interpreter refusal cannot live in `tests/mac/__init__.py`.**
`unittest discover earshot/tests/mac` sets `top_level_dir` to the start directory, puts
it on `sys.path`, and imports each `test_*.py` as a **top-level** module — so the
package `__init__.py` is never executed and a refusal placed there would never run.
Read out of CPython's `TestLoader._find_tests`, not assumed.
It lives in `_interpreter.py`, which raises at import, and every test module imports it,
so the wrong Python turns the whole suite into collection errors rather than letting a
subset quietly pass. The obvious gap — a new test file that forgets the import —
is closed structurally by `test_suite_hygiene.py`, in the same shape as the other
invariants.

**3. The constraints file was being *generated*, not read.**
`oneenv_gate.sh:151` was `echo "numpy<1.24" > "$NP_CONSTRAINT"` into `$BUILD_ROOT`, so
the pin lived **on the box and nowhere in the repo**. Ticket 17 called widening it "a
content change, not a plumbing change" because `-c` was already threaded through every
`pip install` — correct about the threading, but one plumbing change was still needed:
stop writing the file. It is now git-tracked and read-only, with a startup check that
it exists at all.

**4. `reference/memory/__init__.py`'s message is unreachable on the normal path.**
Importing a subpackage always imports its parent first, so all three of
`earshot.reference`, `earshot.reference.memory` and `earshot.reference.memory.ltm`
surface the **parent's** message. ADR-0013 asks for both levels and both are there, but
as defence in depth against someone editing the parent — not as a second message anyone
will read. Worth stating rather than implying two messages exist.

### One thing the reset would have dropped silently

`embodied_memory/scripts/test_metrics.py` — 8 cases on `compute_benchmark_spl`, the
cross-quotable headline number of ADR-0005 — sits inside a tree ticket 10 phase 3
deletes **wholesale**, and neither this ticket's carry list nor ticket 10's named it.
Carried to `tests/mac/test_metrics.py` as `unittest`; the eight cases are unchanged and
only the harness moved. `metrics.py`'s own docstring pointer was stale in two ways (it
named `scripts/test_metrics.py`, which is not even where the file lived) and now points
at the new location — the "near-verbatim" allowance ticket 10 grants for this file,
spent on exactly one line.

### The guard's split

`arm_audio_context` keeps the heavy once-per-episode work (the OBJ write, the
10,000-vertex floor, the key validation) and a new `guarded_observe` wraps every step
with fd 1+2 capture, the canary and the fatal-line scan, and no OBJ write. Both call one
extracted `_scan_logs`, and a test asserts they **agree on what is fatal**: a per-step
scan recognising a different set from the arming scan would be worse than no per-step
scan at all, because a context could arm clean and then degrade into a state only one of
them could see.
`guarded_observe` takes no sensor argument — the mesh uploads once, so re-reading it
every step would pay ticket 16's measured 0.814 s / 32.2 MB to re-answer a question that
cannot have changed — and a test checks no `.obj` appears.

Ticket 12's warning shot applies in full: **this is new code, its fakes have never met
the binary, and 42 green fake-based tests preceded the last raise on a real spec.** Its
box confirmation is ticket 26's, and `tests/box/` already carries the ten assertions
that will do it.

### Two deliberate omissions and one deliberate non-change

- **`types.py` has no bearing helper.** A plausible `atan2` would be a frame convention
  asserted by nobody, and ticket 09 found the lateral sign silently inverts from world
  frame to agent frame under live rendering with no code change. `audio/lateral.py` owns
  it and `tests/box/` pins it. Written into the module docstring so the omission reads
  as a decision.
- **`bootstrap_ss2.sh` does not call `python -m earshot.env_check --strict`.** Ticket 24
  builds that module and owns the wiring; adding a guarded call now would be the
  version-blind-skip pattern this map keeps killing. The bootstrap keeps the ticket-04
  probe as its verdict meanwhile, which ADR-0013 supports — `oneenv_probe.py` **stays**
  in `.scratch` as "the gate's payload".
- **The bootstrap's self-update stays a comment, not the re-exec.** `box_gate.sh` carries
  the re-exec because `audioguard_gate.sh` had it; `oneenv_gate.sh` never did, and ticket
  17 named three specific changes for this file. Left alone to keep the move a move.
  **Loose end for whoever next touches it** — the gotcha is one of the 33 ticket 14
  counted and has already cost a 10-hour run.

### Knock-on, taken here rather than deferred

`docs/race-box-runbook.md` §3's "**a record of what worked once, not a lockfile**" went
false the moment `ss2-constraints.txt` landed. Ticket 24's "watch for" offered it to
24 or 27; taking it here means the repo never carries the false sentence. Two stale
pointers to `probes/oneenv_gate.sh` in §4 and §6 were repointed at
`earshot/tools/bootstrap_ss2.sh` in the same pass. The ADRs and the resolved tickets keep
their original paths — they are historical records of decisions, not instructions.

### What the graph now enforces

```
types, metrics, audio.guard, vlm   -> nothing                 (leaves)
sim                                -> audio.guard, types
audio                              -> audio.guard, vlm, types  (NOT sim)
agent                              -> vlm, types               (NOT sim)
report                             -> audio.guard, types
task, __main__                     -> everything               (the wiring layers)
__init__                           -> audio.guard
config                             -> audio.config, agent.config, types
env_check                          -> nothing
```

Encoded once and completely, so tickets 21–25 add modules without touching the test.
The `import habitat_sim` rule is a **subset** assertion today (`sim/world.py` lands in
ticket 21) rather than an equality — not vacuous, because it fires the moment a *second*
file reaches for the simulator, which is the failure it exists to catch. **Ticket 21's
arrival is what makes it an equality, and tightening it is part of that ticket.**
`test_layering.py` also asserts nothing imports `habitat` (habitat-lab is deliberately
not installed, and a stray import would fail only on the box, only when it ran).
