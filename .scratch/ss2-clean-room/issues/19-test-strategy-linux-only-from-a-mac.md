# 19 — Test strategy for a Linux-only stack from a Mac

Type: grilling
Status: resolved
Blocked by: none (18 resolved 2026-08-04)
Blocks: 20 (and through it, the whole build)

## Question

Which layers of the clean room stay pure enough to unit-test on this Mac, what has to be a box-only integration test, and what is the rule that decides which is which?

The map's execution environment is split by construction: the RACE V100 is the only place the stack runs (`libRLRAudioPropagation.so` is a prebuilt Linux-x64 binary needing GLIBC ≥ 2.29, measured at 2.39 by ticket 05), and this Mac is edit-and-push only.
A test strategy that ignores that split either tests nothing locally or pretends a green local suite means something it does not.

Blocked by ticket 18 because the split is per-module and per-assertion, so it needs the module tree to attach to.

## What the resolved tickets already establish

**Dropping habitat-lab helps rather than hurts** (ticket 04).
Episode loading becomes a gzipped-JSON parse the Mac can unit-test, where `habitat.Env` was box-only by construction.

**Ticket 07 adds two Mac-testable layers by construction**: the anomaly controller (a pure function over `(energy_history, lateral_sign, visual_confirm)`, no simulator) and the navmesh reachability filter (its `snap_point` / `geodesic` callables are injected).
The SPL arithmetic is a third, and ticket 10 pinned it: `metrics.py` is 55 LOC, one pure function, ported verbatim.

**Ticket 12 makes the fourth a demonstration rather than a plan.**
The audio guard unit-tests on this Mac with 27 tests and no `habitat_sim`, because every simulator object it touches is injected and the pybind11 behaviours it depends on (`py::dynamic_attr` on the spec, data descriptors for real fields) are reproducible in ~40 lines of plain Python.
That is the pattern to generalise: **fake the binding semantics, inject the object.**
It leaves box-only exactly what a fake cannot settle, namely whether the real objects behave as the source says.

**Ticket 15 supplies the cleanest worked example, and the warning shot.**
Its attribution and summary stages are stdlib-only and testable off-box (the `nvidia-smi` process-table parser and the teardown's process-tree walk were both unit-tested here against fabricated tables, including a bash-3.2-compatible awk fixpoint so the walk itself is testable); the budget is box-only by construction, because the measurement *is* a driver query.
But: **ticket 12's guard passed 27 tests against fakes and then raised on the first real spec**, because the fakes did not reproduce `__noise_model_kwargs`.
The strategy must state plainly that a green fake-based suite licenses nothing about binding behaviour, and that every constant calibrated against a fake needs a box confirmation before anything depends on it.

**Ticket 16 is the proof of that warning at full strength.**
Two of the guard's assumptions were false and were caught **from source before the box trip, not by it**: `ESP_DEBUG` writes to stdout (`Logging.h:326` → Corrade `Debug.cpp:525`), so capturing fd 2 alone would have raised on a *good* context; and `DEFAULT_SEVERITY_RE = r"\[Error\]"` could never match, because `buildMessagePrefix` emits no severity tag — severity is the **stream**, not the text.
Then the box found a channel neither ticket knew about: the closed engine writes un-prefixed blocks to fd 2, and `RLRA_SetListenerHRTF` returns `Success` over a failed load while `RLRA_WriteSceneMeshOBJ` correctly returns failure.
So **reading the source is a distinct and productive verification layer between "fake" and "box"**, and the strategy should name it as one.

**Ticket 17 adds a layer that deliberately splits across the line.**
`env_check.py` is half metadata (numpy's version, the constraints-versus-resolved comparison — pure, Mac-testable) and half capability (a real sm_70 allocation, the audio enum member, a live `ClapModel` instantiation — box-only, because the capability *is* the assertion).
It is the clearest case that the split is **per-assertion, not per-module**.

**Ticket 09 adds three more Mac-testable layers**, all pure by construction: the onset detector (a one-shot threshold on an RMS series), the level-calibration gate (a separation statistic over two distributions), and the report/audit split (a schema assertion — the agent's testimony must not be constructible from ground truth, which is a static check).
It also adds one that is box-only and load-bearing: the **lateral-sign frame convention**, which inverted from world-frame to agent-frame under live rendering with no code change. A fake cannot settle which frame the real renderer produces.

## Note added by ticket 18 (resolved 2026-08-04) — two constraints inherited, and the Mac surface is much larger than this ticket assumed

`docs/adr/0013-clean-room-module-layout.md` decided the layout, and it pre-decides two things this ticket listed as open.

**1. The paths are fixed: `earshot/tests/{mac,box}/`.** Where an assertion can run *is* the directory boundary, so this ticket's central rule is already a path rather than a convention. Ticket 17's `env_check` splits deliberately across the two — metadata half in `mac/`, capability half in `box/` — which is "per-assertion, not per-module" showing up in the tree on day one. This ticket inherits the taxonomy; what it still owns is the naming and the discipline for deciding which side an assertion lands on.

**2. The Mac surface is most of the tree, not the four layers listed above.** ADR-0013 generalised ticket 12's proven pattern — *fake the binding semantics, inject the object* — to the whole layout. `audio/sensor.py` takes an injected `observe` callable and sensor handle; `agent/reachability.py` takes injected `snap_point` and `geodesic`; the oracle detector takes an injected distance function. The result is that **neither `audio/` nor `agent/` imports `sim`, and `import habitat_sim` appears in exactly one file** (`sim/world.py`), enforced by `tests/mac/test_layering.py`.

So the Mac-testable set is `audio/`, `agent/`, `report/`, `metrics`, `types`, the `vlm` interface, and half of `env_check` — rather than the anomaly controller, the reachability filter, the SPL arithmetic and the guard. That is a materially different strategy problem: the question shifts from "which four things can we test here" to "what does a green Mac suite over most of the tree actually license", which is this ticket's warning shot at full strength.

**Two facts that constrain the answer:**

- The repo has **no `pyproject.toml`, no `setup.py`, no lint config and no CI**, and tests are stdlib `unittest` executed directly. So the "CI story for a repo whose primary target cannot run in CI" is a greenfield question, not a migration.
- Three structural invariants already exist as Mac tests and are load-bearing rather than incidental: `test_layering.py`, `test_report_boundary.py`, `test_no_env_flags.py`. They assert the tree's own shape, which is a fourth verification layer alongside fake / source-read / box, and this ticket should say whether it names it as one.

**This ticket now blocks ticket 20**, and through it the entire build — because building before the strategy lands means writing the tests twice.

## What would resolve it

A grilling session producing the rule (per-assertion, not per-module), the named verification layers (fake / source-read / box, and whether the structural invariants are a fourth), the CI story for a repo whose primary target cannot run in CI, and where `reference/memory/` sits relative to all of it — noting that ADR-0013 already settled the *import* half with a raising `__init__.py`, leaving the lint and test halves to this ticket.

## Answer (resolved 2026-08-04)

**`docs/adr/0014-test-strategy-across-the-mac-box-split.md`.**
Eleven decisions, and the one that reframes the ticket is a divergence axis it did not list.

**The rule is ownership, not dependency.**
An assertion is box-only when its *subject* is behaviour we did not write (the closed `.so`, pybind11, CUDA, the weights); everything about our own logic is Mac with whatever it touches injected.
The corollary does the work: an our-logic assertion that cannot be stated without the real artefact is a **seam defect to fix, not a box test to relocate**.
Dependency ("box when it imports `habitat_sim`/`torch`") was rejected as per-module in a per-assertion costume, and it splits `env_check` the wrong way; falsifiability ("box when a fake could pass vacuously") was rejected as near-universal, which would destroy ADR-0013's Mac surface.

**Names stay, meaning pinned**: `mac/` means *no box required*, `box/` means *the subject lives on the box*.
`mac/` never meant macOS, which is what makes the Linux CI runner consistent rather than a name that quietly stopped being true.

**Four layers, and the structural one earns its place on a specific ground**: it is the only Mac layer that reads the **real** subject (`ast` over actual `earshot/` source, not a fake), so it cannot go vacuous.
The **source** layer is named because ticket 16 proved it productive, with citations at `file:line` in the fake and **review enforcement, deliberately not a test** — a grep confirms a citation is present, not that it is true, which is the vacuous pass the layer exists to prevent.
The **box** layer takes two mandatory rules, each earned by a named incident: a detector ships **both arms** (healthy passes, forced failure fires, on `audioguard_probe.py`'s existing stage shape), and a capability is **exercised, never proxied**.
Both directions are covered because the record has both: under-firing (13's importability skip, `clap_symbols_importable`'s `DummyObject`, 17's version-blind skip at `oneenv_gate.sh:216`) and over-firing (15's false positive that would have fired on every healthy run).

**What green licenses, stated with its proof cited rather than asserted**: a green Mac suite is evidence about our own logic and nothing else, and ticket 12's *27 fake-based tests then a raise on the first real spec* is the proof.
Three gates, each cheaper than what it protects: Mac-green gates a box trip, `tests/box`-green gates the smoke, smoke plus hermeticity gates the deletion commit.

**The axis this ticket did not list: version skew.**
Measured today — this Mac's default `python3` is **3.14.3 with no numpy**; the box is **3.9.19**.
A Mac suite green on 3.14 licenses nothing about whether the code *imports* on the box, and ADR-0013's two 3.9 constraints (`int | None`, `get_type_hints()`) are invisible to a 3.14 run.
So: a dedicated **`earshot-mac`** conda env at 3.9 with numpy under the box's `< 1.24` pin, and **the suite refuses to run under any other interpreter** — the capability-shaped discipline turned on the suite itself, so it cannot silently pass on the wrong Python the way ticket 13's gate silently passed on the wrong torch.
`ltm-embodied` was rejected despite being free: it is the deleted tree's env and its numpy sits above the pin.

**CI exists, one job, named for its scope**: `ubuntu-latest`, 3.9, `unittest discover earshot/tests/mac` plus the lint check, both it and the conda env installing from **one** `tools/mac-requirements.txt` so the dependency set is declared once.
**One unverified fact, disclosed rather than assumed**: whether `actions/setup-python` still provides 3.9 in 2026 (EOL late 2025). If not, the job pins the nearest available and the refusal widens — a decision at ticket 20, not a discovery in a red build.

**The box suite becomes `unittest`** with the four negative controls as test methods, behind a thin `tools/box_gate.sh` **carrying** (not rewriting) `audioguard_gate.sh`'s footgun hardening — the SIGPIPE-safe conda check, the enum-member preflight, the pip-freeze-first ordering.
**Box tests print their measurements**, because ticket 16's trip left the numbers that made 15, 17 and 09 decidable, and a pass/fail run discards exactly that.

**Corrects ticket 17's own split: `env_check` is three-way, not two.**
Metadata comparison → mac; capability probes → box; and *"given a failing probe result, does `assert_env()` raise"* → **mac with injected results**.
That third one is ticket 13's exact bug (a layer that computed the right answer and then skipped), it is the highest-value assertion in the module, and it needs no box at all.
It also exposes a permanent limit: where a forced-failure arm is **unavailable** (you cannot uninstall CUDA), the no-proxy rule is the whole discipline — weaker than what everything else gets, and disclosed rather than papered over.

**Constants carry provenance** (`box` / `source` / `fake` / `runtime`) at their definition; a `box` tag names the measuring test; a `fake` threshold is **generous until a box measurement tightens it** (ticket 26's ceiling already models this, and gives the reason: 06 measured 2.3x pose variance, so a tight bound fails for a reason that is not a regression).

**`reference/` is excluded three times by three mechanisms** — import (ADR-0013's raising `__init__.py`), test (a shared **denylist** walker whose exemption set is asserted to equal exactly `{reference}`, so new top-level code is checked by default and widening fails a test first), and lint.
**Lint exists for exactly one file**: `ruff` with `F` + `E9` only, because `sim/world.py` is the sole `import habitat_sim` module, so the Mac can never import it, so a static AST check is the **only** Mac-side verification it can ever have.
Before this, ticket 10's "excluded from lint" had no subject.

**Honest cost, stated rather than hidden**: this adds three surfaces the repo did not have (CI, a linter, a pinned Mac env), and one duplication accepted rather than engineered away (`reference/` in two exclusion lists — the walker's assertion fails loudly, ruff's does not, and reading `ruff.toml` from 3.9 would need a `tomli` dependency to check a one-line string).

**Ticket 20 grows rather than a new ticket being created**, since all of it is scaffolding for tests that ticket already builds.
