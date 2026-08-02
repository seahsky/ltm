# 14 — Extract the RACE box runbook before `scripts/` is deleted

Type: task
Status: resolved
Blocked by: none

## Question

What are the facts about the RACE box that currently live only inside `scripts/`, and what do they look like written down once the scripts are gone?

## Why it matters

Ticket 10 ruled that `scripts/` is deleted wholesale — ~35 `race-*.sh` drivers plus the setup and verify scripts, ~8,200 LOC across 44 files.
The drivers all invoke `embodied_memory/run_hm3d_pol.py` with the flag surface ADR-0008 deleted, so they are dead on arrival in the clean room.

But the knowledge and the artifacts have come apart.
Several of those scripts encode box facts that outlive the code they were written for, and `race-setup.sh` is the clearest case: it is half-dead (it activates `ltm-embodied` and exports `REMEMBR_*` for a stack being deleted) and half-irreplaceable (it is the only record of how to get a working shell on the box after a pod restart).

**This is a hard prerequisite of ticket 10's phase 3.** Once `scripts/` is deleted the facts are recoverable only by reading 8,200 LOC out of a git tag, which is exactly the "one `git show` a future session will not spend" problem that pushed ticket 10 to vendor rather than point.

Sized as its own ticket rather than done inside ticket 10 because it is a read-44-files-and-write-a-document job, not a decision.

## What would resolve it

`docs/race-box-runbook.md`, written from the scripts themselves, covering at minimum:

- **Conda hook restore after a pod restart.** `race-setup.sh:4-11` — the `conda` CLI leaves `PATH` but `~/miniconda3` survives; `eval "$(~/miniconda3/bin/conda shell.bash hook)"`. And the reason it must be **sourced, not executed**: `conda activate` only sticks in the calling shell, and the script refuses to run when executed directly.
- **The conda `set -u` trap.**
- **The apt line that resolves across Ubuntu releases.**
- **The driver self-update gotcha** — the `race-*.sh` drivers `git pull` themselves, so a change to a driver needs a second invocation to take effect. Present in ~10 of them; the single most likely fact to be re-learned the hard way.
- **The absolute-symlink trap** on rsync'd HM3D — `race-setup.sh:62`, an rsync from the laptop copies an absolute symlink that does not resolve on the box.
- **The box inventory from ticket 05**, so it survives outside a `.scratch` ticket: GLIBC 2.39, 4 cores, V100-32GB, 680 G free of 773.9 G, HM3D val mesh coverage 20/20, no MP3D present.
- **How `ss2` and `~/ss2-build` were actually built** (ticket 04). `race-soundspaces-spike.sh` (288 LOC) is the *ancestor* recipe, not the real one, and both it and the spike env are removed by ticket 10's reset — so if the real recipe is not written down here, the environment the entire map depends on becomes unrebuildable.
- **The `nrun` / `notify-run.sh` wiring** and what it needs from `.env`. The three notify files are carried as live code by ticket 10, so this is usage, not archaeology.

Two guards on scope:

- This is a runbook, not a history. Facts a future session needs to operate the box — not a record of what each driver measured, which is `PHASE2_ABLATION_REPORT.md`'s job.
- Do not port `race-setup.sh` to `ss2` here. The clean room writes its own thin setup script once its root exists; this ticket captures what that script will need to know.

Deliverable: `docs/race-box-runbook.md`, committed, before ticket 10's phase 3 runs.

## Answer

**Written: `docs/race-box-runbook.md`, 8 sections, from the 44 files in `scripts/` plus the measured facts in tickets 04, 05 and 13.**
All eight items on the list above are covered.
Ticket 10's phase 3 is no longer gated on this.

Section map, so a reader knows what is where: 1 the box (measured inventory), 2 getting a shell after a pod restart, 3 the four conda envs and the known-good version set, 4 how `ss2` and `~/ss2-build` were actually built, 5 `nrun` / `notify-run.sh` / `.env`, 6 HM3D layout, symlink traps and the download recipe, 7 the footguns, 8 (folded into 7) the audio-spec traps.

### The ticket's own framing needed correcting in two places

**"Present in ~10 of them" understates the self-update gotcha by 3x.**
**33** of the `race-*.sh` drivers `git pull` the repo they are running from, and **exactly one** (`race-r1-objectnav.sh`) self-heals it.
It is also not merely "the most likely fact to be re-learned the hard way" — it has already cost a **10-hour run** (`r1spin` executed the pre-anti-spin body at commit `32b3493`, `n_unreachable_escape=0`).
The runbook carries the self-heal block verbatim, because a warning is weaker than the six lines that make the class impossible.

**"If the real recipe is not written down here, the environment the entire map depends on becomes unrebuildable" overstates the risk.**
The real `ss2` recipe was never in `scripts/` at all: it is `.scratch/ss2-clean-room/probes/oneenv_gate.sh`, which is **tracked in git** (27 files under `.scratch/` are) and is **not** on ticket 10's delete list, which never mentions `.scratch/`.
What `scripts/` uniquely held was the *ancestor* recipe (`race-soundspaces-spike.sh`, a different env in a different build dir that never layers torch) and the shell bootstrap.
The extraction still earns its place — a probe script inside a wayfinder ticket directory is not where a future operator looks, and a directory named `.scratch` is a plausible later cleanup target — but the runbook now states plainly which artifact is authoritative rather than implying the knowledge would be gone.

### Seven things worth recording that were not on the list

Found by reading rather than predicted, all now in section 7:

1. **SIGPIPE under `pipefail`** — one cause, two shipped bugs. `conda env list | grep -q` turned found-it into a pipeline failure; `ldd --version | head -1 | grep` flunked a healthy GLIBC 2.35 because the `|| echo 0.0` fallback appended a bogus line that `sort -V` then picked. Both fixed by using commands that consume their whole input.
2. **Three cmake/compiler knobs** the conda cross-toolchain makes mandatory (`CMAKE_LIBRARY_PATH`, `CMAKE_INCLUDE_PATH`, and a `CPATH` *shim* of GL-only header symlinks — deliberately not all of `/usr/include`, which would shadow the sysroot's glibc headers).
3. **The poisoned half-configure**: a `build/` with `CMakeCache.txt` but no `compile_commands.json` makes `setup.py`'s argument cache skip cmake and die. Wipe it.
4. **`--no-replace` is mandatory under `nrun`** — the HM3D downloader otherwise prompts, reads stdin, and a detached process has none (`OSError [Errno 9]` about 21s in).
5. **Probe the member, not the class** — the same false-positive class twice: `AudioSensorSpec` is bound even in non-audio builds (issue #2340), and `ClapModel` imports as a DummyObject that only raises on instantiation. Both are why "it imports" proved nothing.
6. **The sourced-vs-executed guard runs both ways.** `race-setup.sh` refuses execution (conda activate would not propagate); `setup-vm.sh`, `verify-setup.sh` and `setup-anomaly-vm.sh` refuse sourcing (their `exit`s would kill the session). `notify-run.sh` is the interesting case: it is safe to source *and* runnable, because it defines `nrun` and returns before any `set` or `exit`.
7. **`race-setup.sh` never exported `PYTHONPATH`** — every driver did it itself. A one-line fact that will bite the clean room's first script otherwise.

### Two scope guards held

No history: what each driver *measured* is not in the runbook.
No port: `race-setup.sh` is described, not translated to `ss2`. The runbook says outright that it is the list of things the clean room's own bootstrap will need to know.

### What this does NOT settle, and the fog it sharpens

The map's fog patch **"How the `ss2` env is pinned, and how it is recreated"** wondered whether it was "part of 14 rather than its own ticket".
**It is not.**
The runbook records the recipe and ticket 13's known-good version table, and says explicitly that this is *a record of what worked once, not a lockfile* — writing the recipe down does not make it a pin.
Whether the clean room ships an actual pin is a decision with a real cost on both sides, and it now has enough concrete content to be asked properly, so it graduates to **ticket 17**.
