# 14 — Extract the RACE box runbook before `scripts/` is deleted

Type: task
Status: open
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
