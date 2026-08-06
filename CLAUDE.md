# Earshot — LTM-grounded anomaly response on live SoundSpaces 2.0 audio

## Mission

An embodied agent runs a primary find-task in a Habitat/HM3D home. An anomaly sound
fires. The agent interrupts, investigates the source, resumes, and reports what it heard
and where. Audio is rendered **live in the simulator at every step** (SoundSpaces 2.0 /
RLR geometric acoustics), not looked up from a precomputed grid.

The method spec is `docs/anomaly_response_task_spec.md`; the vocabulary it uses —
primary find-task, anomaly response, Find-SR, benchmark SPL, onset provenance, background
bed, room-normal distractor — is defined in `CONTEXT.md` and is binding.

## The tree

`earshot/` is the package and the root. One rule shapes it (ADR-0013): **neither `audio/`
nor `agent/` imports `sim`, so `import habitat_sim` lives in exactly one file**
(`earshot/sim/world.py`). That absence is what makes most of the tree testable on a
machine with no simulator.

| directory | what it is |
|---|---|
| `sim/` | the only `habitat_sim` importer: `World`, the audio sensor, the navmesh follower |
| `audio/` | spec, guard, IR handling, the bed, onset, CLAP, the lateral cue |
| `agent/` | proposers → scorer → waypoint → follower; the anomaly controller; the detector seam |
| `task/` | the runner, the episode/dataset builders, the smoke gate, the CLI |
| `report/` | the agent's testimony and the audit record, written atomically, never overwritten |
| `tools/` | bootstrap, the box gate, `nrun`, and the reset machinery (below) |
| `tests/mac`, `tests/box` | see ADR-0014 — `mac/` means "no box required", not macOS |
| `reference/` | the memory stack, vendored inert and deliberately un-importable |

## Running it

**Linux + CUDA + the `ss2` conda env only.** A Mac cannot load habitat-sim: the audio
propagation library is a prebuilt Linux-x64 binary. Editing here, running on the box.

```bash
# on the box, once: build the env, then stage the ESC-50 recordings
nrun bash earshot/tools/bootstrap_ss2.sh
python -m earshot.audio.clips --out-dir data/anomaly_audio

# one episode, end to end
python -m earshot --run-dir runs/<tag> --n-episodes 1 --max-steps 250

# the nine acceptance criteria, tallied over EVERY episode in the run directory
python -m earshot.task.smoke --run-dir runs/<tag>          # --episode N judges just one

# how much of HM3D can pose the task at all — one directory is one run, so the tag
# must be fresh; nonzero exit if any scene failed or any scene's gate went red
nrun bash earshot/tools/yield_sweep.sh --tag <fresh-tag>

# why a detour ended: metres walked per metre of gap closed, abandoned vs reached,
# and what one forward step was worth against the threshold it had to clear
python -m earshot.tools.detour_report runs/<tag>/<scene>

# how many abandoned episodes stood inside the arrival ring, and how many episodes had
# no navmesh route to their source at all — the whole sweep, by scene. Counts only:
# scenes are different rooms, so bands and epsilons stay per-scene above
python -m earshot.tools.detour_report runs/<tag> --across-scenes

# both counts over several finished runs, after a pull. Read-only, no GPU, minutes
bash earshot/tools/arrival_audit.sh --tags "cast-1 eps-1 yield-2"

# did a change move the funnel — the arm WITHOUT it first. Refuses to subtract scenes
# the two sweeps built differently, and prints how big a delta the renderer alone makes
python -m earshot.tools.funnel_diff runs/<before-tag> runs/<after-tag>

# the same two sweeps PAIRED BY EPISODE — 365 comparisons rather than 20, which is what
# a delta of a dozen episodes needs. Verifies each pair is the same task before
# subtracting it, and needs no flip-rate estimate: the flips ARE the discordant pairs
python -m earshot.tools.episode_diff runs/<before-tag> runs/<after-tag>

# is the run-to-run variance a knob or a fact? One scene, several ray counts, N repeats
# each (~1.5h at the defaults). `flip_report` reads the arms back: the aggregate rate AND
# the fraction of episodes whose outcome is not unanimous, which is the one that decides
# how many repeats a matrix cell needs
nrun bash earshot/tools/ray_variance.sh --tag <fresh-tag>
python -m earshot.tools.flip_report runs/<tag>-r500-* runs/<tag>-r2500-*

# the box test suite (a few minutes, read-only, installs nothing)
bash earshot/tools/box_gate.sh
```

On a Mac:

```bash
conda activate earshot-mac      # 3.9; the suite refuses to run on anything else
PYTHONPATH=. python -m unittest discover earshot/tests/mac
ruff check earshot/
```

## Conventions

- **No environment flags.** ADR-0008 removed that surface; behaviour is typed on
  `RunConfig` and `tests/mac/test_no_env_flags.py` holds the line.
- **A criterion that could not be evaluated is never green.** `NOT_RUN` is red. Two
  incidents are behind this: a probe that skipped and reported success, and a canary that
  was never armed reading as a pass.
- **A capability is exercised, never proxied**, and a detector ships both arms — the
  healthy path passing *and* the forced failure firing (ADR-0014).
- **A claim that X broke because of a change needs the arm where the change is absent.**
  The hermeticity gate called a pre-existing failure a leak for want of a control run.
- **Box tests print their measurements.** The numbers are what make the next decision.
- `data/`, `runs/`, `models/` are gitignored.

## The reset

The old `embodied_memory/` and `dialogue_memory/` trees, `scripts/`, and the MSC path
were deleted in one commit once the smoke was green *and* green again with them moved out
of the repo. `earshot/tools/reset_manifest.py` records exactly what went and asserts it
stays gone; `earshot/tools/hermeticity_gate.sh` is the gate that licensed it, kept
working so a revert restores a tree it can still check.

Rollback is a single `git revert` of that commit, or
`git checkout archive/pre-reset-2026-08-06 -- embodied_memory dialogue_memory scripts`.

## Where the history is

- `PHASE2_ABLATION_REPORT.md` — every measured result, including the closed arcs. Read it
  before proposing a lever; most of them have been tried and have a number attached.
- `docs/adr/` — the decisions, with the arguments. 0013 (tree and layering) and 0014 (test
  strategy) govern the code above.
- `docs/race-box-runbook.md` — the box: envs, data, `nrun`, and the footguns that have
  cost real runs.
- `.scratch/ss2-clean-room/` — the map this rebuild was planned and executed from.
- `Research Proposal_Embodied Agent.md`, `ICRA2027_PAPER_DRAFT.md` — the research framing.

The long-term-memory stack that produced the earlier results is archived at
`earshot/reference/memory/` with its own README. It is not wired in, and reviving it is a
new effort rather than an import.
