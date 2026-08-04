# Tests split by whose behaviour they assert, and a green Mac suite licenses only our own logic

**Status:** accepted (2026-08-04, grilling session on ticket 19 of the `ss2-clean-room` map).

The module layout is ADR-0013.
The agent is ADR-0008.
The reset order is ticket 10, and the smoke's nine assertions are ticket 26.

This ADR fixes the fourth side: which assertions run where, what each layer of verification is worth, and what has to be true before anything depends on a check.

It blocks ticket 20 and through it the entire build, because building first means writing the tests twice.

## The rule: ownership, not dependency

An assertion is **box-only when its subject is the behaviour of something we did not write**: the closed `libRLRAudioPropagation.so`, the pybind11 binding layer, the CUDA driver, the model weights.
Everything whose subject is our own logic is Mac, with whatever simulator object it touches injected.

The two rules this rejects both look reasonable and both fail on a real case in this tree.

**Dependency** ("box when the test imports `habitat_sim` / `torch` / `transformers`") is mechanical and unarguable, and it is a per-module rule wearing a per-assertion costume.
It splits `env_check.py` the wrong way: its constraints-versus-resolved comparison is arithmetic over two dicts, and it would land on the box because the module it lives in also probes CUDA.

**Falsifiability** ("box when a fake could pass while the real thing is broken") targets this map's three vacuous-green incidents precisely, and it is close to universal.
Almost any fake could pass vacuously, so the rule drags most of the tree boxward and destroys ADR-0013's Mac surface.

### The corollary is what does the work

An assertion that is about **our** logic but cannot be stated without the real artefact is a **seam defect, not a box test**.
Fix the injection point.

This is the rule ADR-0013 already applied without naming: `audio/sensor.py` takes an injected `observe` callable, `agent/reachability.py` takes injected `snap_point` and `geodesic`, the oracle detector takes an injected distance function.
That is why `import habitat_sim` appears in exactly one file and why the Mac surface is most of the tree rather than a corner of it.

Stating the corollary gives a reviewer something to push back with when a test drifts boxward because injection was inconvenient.

## The names stay, and the meaning is pinned

`earshot/tests/mac/` and `earshot/tests/box/`, as ADR-0013 wrote them.

- `mac/` means **no box required**.
- `box/` means **the assertion's subject lives on the box**.

`mac/` never meant macOS.
That definition matters because the CI job below runs `tests/mac` on a Linux runner, which under any machine-shaped reading of the name would be exactly the sort of thing this map keeps catching: a name that reads as true and quietly is not.

`box` is load-bearing repo vocabulary already (`docs/race-box-runbook.md`, "box inventory", "box-only" across eight resolved tickets), and "Mac-testable" is the phrase tickets 12, 15, 17, 18 and ADR-0013 all use.
Renaming would have cost that continuity to buy precision the definition supplies for free.

## Four verification layers

### fake

Our logic against injected doubles.
`tests/mac/`, 42 tests already green in `test_audio_guard.py`.

**It licenses nothing about binding behaviour.**
The proof is in this map's own record and is cited rather than asserted: ticket 12's guard passed **27 tests against fakes and then raised on the first real spec**, because the fakes did not reproduce `__noise_model_kwargs`.
Under ADR-0013 the Mac surface is now most of the tree, so that warning applies to far more code than it did when it was written.

### source

Reading the real source before booking box time, and a productive layer in its own right rather than a way of writing better fakes.

Ticket 16 is the evidence at full strength.
Two of the guard's assumptions were false and were caught from source **before** the box trip: `ESP_DEBUG` writes to stdout (`Logging.h:326` into Corrade `Debug.cpp:525`), so capturing fd 2 alone would have raised on a *good* context; and `DEFAULT_SEVERITY_RE = r"\[Error\]"` could never match, because `buildMessagePrefix` (`Logging.cpp:149-152`) emits no severity tag.
Severity is the stream, not the text.
Either would have burned a box trip on a repo whose runbook records a footgun that already cost a 10-hour run.

**Discipline.** Every fake that reproduces a third-party behaviour cites it at `file:line` and states what would break if that behaviour changed.
`test_audio_guard.py`'s module docstring already does this for four citations, so this codifies an existing habit.

**Enforced by review, deliberately not by a test.**
A test that greps for a citation confirms that a string is present, not that it is true.
A stale or wrong citation would pass exactly as green as a correct one, which is the vacuous pass this layer exists to prevent.
Automating it would be the failure mode wearing the costume of the fix.

### box

The real artefact, on the RACE V100 in the `ss2` env.

Two mandatory rules, each earned by a named incident rather than by convention.

**1. A detector ships with both arms.**
A healthy arm that passes and a forced-failure arm that fires, on `audioguard_probe.py`'s existing stage shape (stage 2 healthy, stage 3 four negative controls, raising `the guard did not fire under a forced failure`).

The failures run in both directions, which is why one arm is not enough.
Under-firing: ticket 13's torch layer skipped on mere importability, so the fix would have reported green as a no-op; `clap_symbols_importable` could not fail, because transformers substitutes a `DummyObject` that only raises on instantiation; ticket 17 found the version-blind build-skip still alive at `oneenv_gate.sh:216`, making a SHA pin inert on every re-run.
Over-firing: ticket 15 found a false positive in ticket 12's guard that would have fired on **every healthy run**.

**2. A capability is exercised, never proxied.**
Version numbers, import success and symbol presence are never the assertion.
Ticket 17 named this "capability-shaped, not provenance-shaped"; ticket 16 applied it by probing the audio enum **member** rather than the class; ticket 13 applied it by instantiating `ClapModel` rather than importing it.

**Where rule 1 is unavailable, rule 2 substitutes for it.**
You cannot uninstall CUDA to prove the sm_70 probe fires.
For that class of assertion the no-proxy rule is the whole of the discipline, and this is stated so the gap is disclosed rather than discovered.

### structural

`test_layering.py`, `test_report_boundary.py`, `test_no_env_flags.py`.

These are a fourth layer and not ordinary Mac tests, on one ground: they are the **only Mac layer that reads the real subject**.
`test_layering.py` `ast`-parses actual `earshot/` source, not a fake, so it cannot go vacuous the way a fake-based suite can.
They are also the enforcement ADR-0008 and ADR-0013 chose *instead of* flags and documentation, on this repo's record of things written down that quietly stopped being true.

**Scope, and the exemption.**
`reference/memory/` is ~3,400 LOC vendored deliberately broken.
`memory_bridge.py` reads `LTM_*` from the environment throughout, so `test_no_env_flags.py` fires on it; its modules import `faiss`, `sentence-transformers` and each other, so `test_layering.py` fails on it.

One shared walker in `tests/mac/` enumerates the live tree, **denylist-shaped** so new top-level code is checked by default.
`reference/` is the sole exemption, and a test asserts the exemption set equals exactly `{reference}`, so widening it fails a test before it lands.

An allowlist was rejected for the direction of its default: a new `earshot/experimental/` would be silently unchecked until someone remembered to add it, which is the wrong failure mode for this repo.

## The gate chain, and what green licenses

**A green Mac suite is evidence about our own logic and nothing else.**
It licenses no claim about binding behaviour, about the stack, or about the value of any constant.
The one thing it supports is: *nothing we wrote regressed since the last box confirmation.*

Three gates, each strictly cheaper than what it protects.

| Gate | Protects |
| --- | --- |
| Mac suite green | a box trip |
| `tests/box` green | the smoke run |
| smoke green plus the hermeticity re-run | the deletion commit (ticket 10 phase 3) |

The third is ticket 10's, restated here for the chain rather than added.
The first two are new, and they exist because a box trip on this repo has cost 10 hours and the smoke is the most expensive thing on the map.

## Where things run

### The Mac suite is pinned to Python 3.9 and refuses to run elsewhere

Version skew is a **third divergence axis**, alongside fake-versus-real and Mac-versus-box, and neither ticket 19 nor ADR-0013 listed it.

Measured 2026-08-04: this Mac's default `python3` is **3.14.3 with no numpy**, and the box is **Python 3.9.19**.
A Mac suite green on 3.14 licenses nothing about whether the code even imports on the box.
ADR-0013 already names two 3.9 constraints (`int | None` in an annotation needs `from __future__ import annotations`, and `typing.get_type_hints()` raises on those dataclasses, which is why `test_report_boundary.py` reads `__dataclass_fields__`), and both are invisible to a 3.14 run.
`match`, runtime PEP 604 unions and resolved `list[str]` annotations would all sail through here and break there.

So: a dedicated **`earshot-mac`** conda env at Python 3.9, numpy under the box's `< 1.24` pin, and the suite **asserts its own interpreter version at start and refuses to run otherwise**.

The refusal matters more than the pin.
It is the capability-shaped discipline applied to the suite itself, so the suite cannot silently pass on the wrong Python the way ticket 13's gate silently passed on the wrong torch.

The existing `ltm-embodied` env (3.9.23, numpy 1.26.4) was rejected despite being free: it is the deleted tree's env, ticket 27 retires what it serves, and its numpy sits above the box's pin.

### One CI job, named for its scope

`ubuntu-latest`, Python 3.9, a small pinned requirements set, running `python -m unittest discover earshot/tests/mac` plus the lint check below, on push and pull request.

This is a greenfield addition: the repo has **no `.github/`, no `pyproject.toml`, no `setup.py`, no lint config and no CI**.

What it buys that nothing else does: the suite runs without anyone remembering, and it runs on a machine that is not the author's, rebuilt from pinned inputs every time.
That is ticket 17's *pin the inputs, record the outputs* applied to the Mac side.

What it risks is a green badge on a repo whose primary target cannot run in CI.
The licence sentence above bounds what green means, and the job is **named for its scope** so it cannot be read as verifying the stack.

**One dependency declaration, not two.**
The `earshot-mac` env and the CI job both install from `earshot/tools/mac-requirements.txt`.
Two declarations of the same set would drift, and the drift would be silent.

**One unverified fact, to check at ticket 20 rather than assume.**
Python 3.9 reached end of life in late 2025, and whether `actions/setup-python` still provides it in 2026 has not been confirmed.
If it does not, the job pins the nearest available version and the interpreter refusal has to widen, which weakens it.
That is a known and disclosed weakening, not a surprise to discover in CI.

### The box suite is `unittest`, behind a thin driver

`python -m unittest discover earshot/tests/box`, run in the `ss2` env on the V100.
`audioguard_probe.py`'s four negative controls become test methods.

`audioguard_gate.sh` is **not** a test suite, and the split is drawn there.
It git-pulls with self-update, activates conda as a directory check (because a matching `grep` under `pipefail` SIGPIPEs conda and turns found-it into a failure, runbook section 7), preflights the audio enum member so it fails in seconds rather than 90, drops ticket 17's `pip freeze` deliverable **first** so a guard failure still leaves it behind, and writes an artefact directory.
Those are driver concerns and they survive as `earshot/tools/box_gate.sh`.
The footgun hardening is not rewritten, it is carried.

**Box tests print their measurements.**
Ticket 16's box trip left numbers, not just green: 916 chars stdout versus 0 stderr on a healthy render, the +8 vertex gap between what habitat submits (392,356) and what the engine holds (392,364), 0.814 s and 32.2 MB for the OBJ write.
Those measurements made tickets 15, 17 and 09 decidable.
A bare pass/fail run discards exactly the evidence the next decision needs, so the captured driver log is the evidence record.

### `env_check` splits three ways, not two

Ticket 17 described two halves.
There are three, and the missing one is pure logic.

| Assertion | Layer |
| --- | --- |
| constraints-versus-resolved comparison, numpy's version | `tests/mac` |
| given a failing probe result, does `assert_env()` **raise** | `tests/mac`, injected results |
| a real sm_70 allocation, the audio enum member, a live `ClapModel` instantiation | `tests/box` |

The middle row is ticket 13's exact bug: a layer that computed the right answer and then skipped.
It is the highest-value assertion in `env_check` and it needs no box at all.

This is "per-assertion, not per-module" at its sharpest, and it is why the rule is stated in terms of subjects rather than imports.

## Constants carry their provenance

Every tuned constant records where its value came from, at the point it is defined: **`box`**, **`source`**, **`fake`**, or **`runtime`**.

A `box` provenance **names the box test that measures it**, which the previous section made a real artefact rather than a promise.

A `fake`-provenance threshold is **set generously and tightened only by a box measurement**.
Ticket 26 already models this for one case and gives the reason: criterion 7's wall-clock ceiling is set generously rather than at ticket 06's 27.2 ms, because ticket 06 measured 2.3x pose variance against ticket 04 on the same scene, so a tight bound fails for a reason that is not a regression.

The population is real and its provenance genuinely differs.
Ticket 12 shipped two constants it flagged as inferences the box must confirm, the 10,000-vertex mesh floor among them.
Ticket 26 defers five to the smoke: `investigate_max_steps`, the bed level, the audible band, criterion 7's ceiling, and the tolerance on the pre-onset RMS assertion.
Ticket 06 measured `cheap_preset` on the box.
`DEFAULT_SEVERITY_RE` came from a source read that corrected a fake-era guess.
Task spec section 2.3 derives `onset_rms` at run start and fixes it nowhere.

Review-enforced, for the source layer's reason: a grep confirms a tag exists but not that it is true.

## `reference/` is excluded three times, by three different mechanisms

Ticket 10 says `reference/memory/` is excluded from lint, test and import.
Each half has a different owner and only one was settled before this ADR.

- **Import**: ADR-0013, `reference/__init__.py` and `reference/memory/__init__.py` each raise `ImportError`. Active rather than absent, because PEP 420 makes an absent `__init__.py` no barrier at all.
- **Test**: the structural walker's pinned denylist above, asserted to be exactly `{reference}`.
- **Lint**: `ruff`'s exclude, below.

The `reference/` name therefore appears in two exclusion lists.
That duplication is accepted rather than engineered away: the walker's assertion fails loudly if it drifts, ruff's does not, and reading a `ruff.toml` from a 3.9 test would need a `tomli` dependency to check a one-line string.

## Lint exists for exactly one file

`ruff` with a deliberately narrow rule set, **`F` plus `E9`**, excluding `reference/`, run locally and in the CI job.
Not style, not formatting, not import ordering.

The justification is specific rather than conventional.
`sim/world.py` is the only module that imports `habitat_sim`, so the Mac can never import it, so it has **zero Mac-side verification** under everything else in this ADR.
A static checker is AST-based and executes no imports, which makes it the only thing on this side of the split that can look inside that file at all.
Without it, an undefined name or a typo'd attribute in the one module that touches the simulator survives until a box trip.

Before this ADR, ticket 10's "excluded from lint" had no subject, because no linter existed.
It has one now.

## Considered and rejected

- **Renaming `mac/` to `offline/`.** States the property rather than a machine, and stays true on a Linux CI runner. Rejected: the pinned definition buys the same precision without discarding the vocabulary of eight resolved tickets.
- **Renaming both to `ours/` and `theirs/`.** Encodes the ownership rule in the path, so choosing a directory forces the question. Rejected for the same continuity reason, and because `box` is the runbook's word.
- **Enforcing source citations with a Mac test.** Rejected as self-defeating: it can only check that a string is present, and a wrong citation passes as green as a right one.
- **A ledger file of every third-party behaviour and its confirmation status.** A single place to audit before a box trip. Rejected: documentation-as-enforcement, which ADR-0013 already rejected on this repo's record.
- **An allowlist of live packages in the structural walker.** Unambiguous about what is covered. Rejected on the direction of its default: new code would be silently unchecked.
- **Both arms only for engine-facing detectors.** Less test surface on a one-episode smoke. Rejected: the version-blind skip that survived into ticket 17 was not engine-facing.
- **Reusing `ltm-embodied` as the Mac env.** Free today, already at 3.9.23. Rejected: it is the deleted tree's env and its numpy 1.26.4 sits above the box's `< 1.24` pin.
- **A pre-push git hook instead of CI.** Same "without being remembered" win, no new platform and no badge to misread. Rejected: it is per-clone, bypassable with `--no-verify`, adds a `hooksPath` wrinkle under worktrees, and never rebuilds the dependency set on a machine that is not the author's.
- **CI running only the structural invariants.** Stdlib-only, zero dependency install, and it protects exactly the tests that exist as enforcement. Rejected as under-coverage once the dependency set turned out to be numpy and ruff.
- **Keeping the box side as a bash gate script.** Nothing ticket 16 verified gets rewritten. Rejected: it leaves the box side with a permanently different mental model from the Mac side, when only the driver concerns actually need bash.
- **Box tests asserting without printing measurements.** Cleaner output. Rejected: a box trip would then leave green rather than the numbers that made tickets 15, 16, 17 and 09 decidable.
- **A central `constants.py` with provenance.** A single-file pre-trip audit. Rejected: it fights ADR-0013's per-module frozen configs and separates values from the code that gives them meaning.
- **`py_compile` in CI instead of ruff.** Zero new dependencies and it still reaches `sim/world.py`. Rejected: it catches syntax only, so a typo'd name still reaches the box.
- **Ruff's full default rule set.** Conventional, and uniform as the tree grows. Rejected: style and import-ordering churn on a repo that has never had a linter, for value unrelated to the one file that justifies having one.

## Consequences

**Ticket 20 grows.** The scaffold now also carries the `earshot-mac` env and `tools/mac-requirements.txt`, the interpreter refusal, the shared structural walker with its pinned exemption, `ruff.toml`, and the CI workflow. Ticket 20's own ticket is updated rather than a new one created, because all of it is scaffolding for the tests that ticket already builds.

**Three surfaces this repo did not have, it now has to keep alive**: CI, a linter, and a pinned Mac env. Each is justified by a named incident rather than by convention, and that is the honest cost of the strategy rather than a hidden one.

**The interpreter refusal has one known way to weaken.** If `actions/setup-python` no longer provides 3.9, the CI job pins the nearest available and the refusal widens to match. Disclosed here so it is a decision at ticket 20 and not a discovery in a red build.

**One thing this ADR cannot fix.** A capability probe that cannot be forced to fail is verified by the no-proxy rule alone, which is weaker than the two-arm rule everything else gets. The sm_70 allocation and the CUDA-side probes live in that gap permanently.
