# 17 — Does the clean room pin the `ss2` env, or is the recipe enough?

Type: grilling
Status: resolved
Assignee: Sky
Blocked by: none

## Question

The env the entire destination runs on is reproduced by re-running a resolver, not by restoring a pinned state.
Does the clean room ship an actual pin, and if so what kind — or is the written recipe plus a loud version-pair assertion the answer?

## Why it matters

`oneenv_gate.sh` installs from version *ranges* (`transformers>=4.40,<5`, `numpy>=1.16.1,<1.24`), so every rebuild re-resolves against whatever PyPI serves that day.

That is not hypothetical. It is exactly how the env broke: `transformers` drifted to 4.57.6 against a frozen `torch==2.0.1`, disabled its own PyTorch backend, and turned `ClapModel` into a dummy object that imports fine and cannot instantiate (ticket 13).

Ticket 13 fixed the **instance** (move the torch pin) and the **detection** (the gate now fails loudly when the resolved pair does not actually work).
Neither touches the **class**.

## What changed since the fog patch was written

Two things, both from tickets 13 and 14, and they cut in opposite directions:

- **The working set is now known** (ticket 13): python 3.9.19 / numpy 1.23.5 / torch 2.2.2+cu118 / transformers 4.57.6 / scipy 1.13.1, on habitat-sim `4f61e321` stock. So there is now something concrete to pin *to*.
- **It is written down and has a home** (ticket 14): `docs/race-box-runbook.md` section 3 carries that table and section 4 carries the build recipe, both stating outright that this is a record of what worked once, not a lockfile.

The fog patch asked whether this was "part of 14 rather than its own ticket".
Ticket 14 answered no: writing the recipe down does not make it a pin.

## What makes it a real question rather than an obvious yes

- A `pip freeze` of this env is **not** a reproducible artifact on its own. habitat-sim is compiled from source against a git branch and a submodule SHA, so a lockfile covers only part of the surface and the expensive part is the part it misses.
- Half the failure surface is already covered by ticket 13's loud gate. A pin that duplicates a working assertion buys less than it costs to maintain.
- Python 3.9 already bounds the worst drift axis: transformers 5.x requires >= 3.10, so resolution is capped at the 4.x line.
- Against all that: the box is a three-month reservation, and a pod that comes back without `ss2` is a rebuild with a resolver that has moved on.

## What would resolve it

A decision, plus wherever it lands in the clean room's layout:

- pin / do not pin, and if pinning, **which artifact** (a constraints file, a full `pip freeze`, a conda `environment.yml`, or just the version table plus the assertion) and **what asserts it at runtime**;
- whether the habitat-sim branch SHA + submodule SHA are pinned alongside, since they are the half a Python lockfile cannot reach;
- and whether the `ss2` rebuild path is a documented recipe (status quo) or an executable script the clean room owns.

Related: the clean room's bootstrap script is where a pin would naturally live, and that script does not exist yet.

## Answer — Pin, at the input side, in three artifacts. The clean room owns the rebuild path.

**Pin the inputs, record the outputs.**
A constraints file pins what the resolver is allowed to choose; a `pip freeze` is kept as forensic evidence and is never installed from.
Enforcement is **capability-shaped, not provenance-shaped**, because ticket 13's failure would have passed every version check.

### 1. `ss2-constraints.txt` — nine exact pins, replacing the one-line `np-constraint.txt`

The six packages the gate installs by name, plus the transitive trio of the one stack that has already broken:

```
# direct installs — the gate names these
numpy==1.23.5
numpy-quaternion==?          # unrecorded, needs the box
torch==2.2.2                 # PEP 440: matches the 2.2.2+cu118 local version
transformers==4.57.6
scipy==1.13.1
soundfile==?                 # unrecorded, needs the box

# transitive — pinned because this stack is what broke
huggingface-hub==?
tokenizers==?
safetensors==?
```

**This is a content change, not a plumbing change.** `NP_CONSTRAINT` is already passed as `-c` to every `pip install` in `oneenv_gate.sh` — the numpy layer, habitat-sim's own `requirements.txt`, torch, and the CLAP stack. Widening the file it points at reaches all four.

Rejected, with reasons:

- **Full `pip freeze` as the pin.** Over-constrained: one yanked transitive breaks the entire rebuild, and habitat-sim is a source install a freeze records as `habitat-sim @ file:///root/ss2-build/...`, which cannot be reinstalled from. It survives as a *recorded artifact*, below.
- **conda `environment.yml`.** It destroys the thing that makes the env work: the install **order**. numpy must be pinned before anything else can resolve it, and the toolchain before the build. The ordering *is* the recipe, and a declarative file has no way to express it. habitat-sim still builds from source outside the file regardless.
- **Named direct installs only.** Three lines cheaper and leaves `huggingface-hub` / `tokenizers` / `safetensors` floating against a frozen `transformers==4.57.6` — a live breakage class in that ecosystem, independent of the transformers version.

Constraints do not force installation, so nothing here is installed that would not otherwise be. **The corollary is a footgun worth naming: a constraint on a package that is never installed is a silent no-op**, so a misspelled name is an inert pin that reports success — the same class as ticket 13's version-blind skip. Section 4 handles it.

### 2. habitat-sim pinned to the SHA; conda left alone

`SIM_SHA=4f61e321`, with `RLRAudioPropagationUpdate` demoted to a comment.

**Measured, and it corrects this ticket's own framing** (see below): the branch HEAD **is** `4f61e321`, last committed **2022-11-04**, dormant for 3.7 years. So the source half is already frozen in practice and the pin costs one line.

Two mechanics matter:

- **Fetch the branch, reset to the SHA** — `git fetch origin RLRAudioPropagationUpdate` then `git reset --hard 4f61e321`. Not `git fetch origin <sha>`, which needs the server to allow SHA-in-want. The chosen form also fails **loudly** if a force-push ever makes that SHA unreachable from the branch, which is exactly the signal the pin exists to produce. Today the gate runs `git reset --hard "origin/$SIM_BRANCH"` (`oneenv_gate.sh:234`) — it does not merely *name* a branch, it re-syncs to whatever the remote serves, by design.
- **The build-skip must check the pin.** `oneenv_gate.sh:216` skips the whole habitat-sim stage when an audio-capable `habitat_sim` merely *imports*, never checking which tree it was built from. **That is the exact bug ticket 13 killed in the torch layer, still alive in the habitat-sim layer** — and it makes a SHA pin inert on every re-run against an existing env. The skip becomes `quick_audio_probe && [ "$(git -C "$SIM_DIR" rev-parse HEAD)" = "$SIM_SHA" ]`.

**The submodule is not pinned separately.** `git submodule update --init --recursive` forces to the gitlink recorded in the superproject commit, so pinning habitat-sim to `4f61e321` pins `rlr-audio-propagation` to `4fd446b4` deterministically. A second pin is redundant; the probe already records the SHA, and a force-push in the submodule repo makes the gitlink unresolvable, which fails loudly on its own.

**The conda side stays on ranges**: `python=3.9`, `cmake=3.14.0`, `gcc_linux-64=10.*`. The governing line is **pin where failure is silent, leave ranges where failure is loud.** Conda drift dies as a build error you cannot miss; PyPI drift dies as a dummy object that imports fine. Pinning `python=3.9.19` exactly also buys a new failure mode — `PackagesNotFoundError` when a channel retires a patch build — in exchange for guarding an axis where patch drift inside 3.9 is ABI-stable.

### 3. The clean room owns the rebuild path, and the assertion is extracted

```
<newroot>/tools/bootstrap_ss2.sh      # the recipe, executable
<newroot>/tools/ss2-constraints.txt   # the pin
<newroot>/<pkg>/env_check.py          # the assertion, importable
```

`bootstrap_ss2.sh` calls `python -m <pkg>.env_check --strict`; the runtime entry point calls `from <pkg>.env_check import assert_env`. **One implementation of the assertion, two callers** — which is the whole reason for splitting build from assert rather than promoting the gate verbatim and leaving the assertion stranded in bash.

**Moved, not copied.** Two copies of a build recipe is a drift trap, and ticket 14 already named the risk in the other direction: a probe inside a wayfinder ticket directory is not where an operator looks, and `.scratch` reads as disposable. The ticket-04 scaffolding does not come along — the habitat-lab arm (habitat-lab is deliberately not installed) and the report-JSON comparison machinery were feasibility-experiment shape, not production shape.

This closes the ticket's own "Related" note: the bootstrap script that did not exist is now specified, and the pin lives in it.

### 4. What enforces it, and where

**Capability probes, not provenance checks.** This is ticket 13's lesson stated as a rule: `transformers` reported **4.57.6 both before and after** the fix, and `ClapModel` imported cleanly the whole time it was a `DummyObject`. A version-set comparison would have printed green through the entire failure.

| where | what | cost |
| --- | --- | --- |
| entry point, **before** `import habitat_sim` (beside ticket 12's `pin_habitat_logging()`) | numpy `< 1.24` (metadata); torch `>= 2.1` **and** a real sm_70 allocation; `habitat_sim` audio via the enum **member** probe | ~free |
| `AudioClassifier.__init__` | `assert_clap_instantiable()` — a real `from_pretrained` and a finite logit | 153.5 M params, 0.713 GB VRAM, paid only by runs that use it |

Both fail before the episode rather than mid-run. The entry point is already a fixed point in the layout — ticket 12 forced `pin_habitat_logging()` to run before anything imports `habitat_sim`, so the module that touches the simulator first is a named place, not an accident.

**One provenance check does earn its place, at bootstrap time only**: after the installs, compare the resolved set against `ss2-constraints.txt`. It catches the inert-pin class from section 1 (a misspelled or never-installed constraint), which a capability probe cannot see. Two different jobs — bootstrap-time provenance catches a pin that does nothing, runtime capability catches a backend that turned itself off.

**Record the outputs.** `assert_env()` returns a report that lands beside ticket 12's `AudioContextReport` in the run's output, and the bootstrap writes a full `pip freeze` as forensic evidence. That freeze is the artifact a future ticket 13 diffs against — the diagnosis that cost a whole ticket last time existed nowhere on disk.

### This ticket's own framing needed correcting in three places

**"habitat-sim is compiled from source against a git branch and a submodule SHA, so a lockfile covers only part of the surface and the expensive part is the part it misses."**
Backwards, on measurement. The source half is the **cheapest** to pin — one line against a branch that has not moved since 2022-11-04 — and it is already frozen in practice. The genuinely unpinned surface is the PyPI half the ticket treated as the easy part, and it is **wider than the runbook's seven-row table shows**: five of the nine pins above (`soundfile`, `numpy-quaternion`, `huggingface-hub`, `tokenizers`, `safetensors`) have their resolved versions recorded nowhere in this repo, alongside all of habitat-sim's 2022-era `requirements.txt`.

**"A pin that duplicates a working assertion buys less than it costs to maintain."**
They are not duplicates. **The gate is the detector; the pin is the fix.** The gate tells you the env is broken at the *end* of a build; without a pin, that is where diagnosis starts — and the last time it happened, diagnosis was ticket 13 in its entirety. They compose: the pin prevents the common case, the gate catches the case a pin cannot (a pinned set that stops working for a reason outside its versions).

**"Python 3.9 already bounds the worst drift axis."**
True and narrower than it sounds. It caps resolution at the transformers 4.x line — and **the break happened inside that line**, `>=4.40` drifting to 4.57.6. The bound offered as reassurance did not prevent the actual failure, and would not prevent its recurrence.

### What this does not settle

- **The file cannot be authored yet.** Five of the nine versions are unknown, so this decision hands out one box command. Folded into **ticket 16** as a stage 0 (`pip freeze`, ~5 s, same read-only trip, already unclaimed) rather than spawning a ticket for one line.
- **Nothing is built.** `<newroot>` does not exist, so the three artifacts land with the layout work, not here. This blocks nothing in ticket 10 — the reset's delete list never touches `.scratch/`, so the move is an addition to phase 1, not a prerequisite of phase 3.
- **`docs/race-box-runbook.md` §3 goes stale on landing.** Its "this is a record of what worked once, not a lockfile" is accurate today and false the moment `ss2-constraints.txt` exists. Whoever builds the bootstrap updates §3 to point at it.
- **Pin maintenance has no policy.** When a pin must move, the answer is a new measured set and a green bootstrap run, not a widened range — but nothing enforces that beyond this sentence.
