# 19 — Test strategy for a Linux-only stack from a Mac

Type: grilling
Status: open
Blocked by: 18

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

## What would resolve it

A grilling session producing the rule (per-assertion, not per-module), the named verification layers (fake / source-read / box), the CI story for a repo whose primary target cannot run in CI, and where `reference/memory/` sits relative to all of it (ticket 10 requires it outside the lint, test and import surface).
