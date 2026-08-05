# The `CLAUDE.md` the deletion commit lands

Ticket 27 phase 3 rewrites `CLAUDE.md` **in the same commit** as the deletion, so the file
never describes a tree that does not exist. This is the text, staged here so it can be
reviewed before the irreversible commit rather than written during it.

580 lines → the file below. The ~450 lines of outcome narrative are **not** summarised
into it: that history lives in `PHASE2_ABLATION_REPORT.md`, and duplicating it is how the
file reached 580 lines in the first place. What replaces it is a pointer.

Everything between the rulers is the new file, verbatim.

---

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
| `tools/` | bootstrap, the box gate, the reset manifest, the hermeticity gate, `nrun` |
| `tests/mac`, `tests/box` | see ADR-0014 — `mac/` means "no box required", not macOS |
| `reference/` | the memory stack, vendored inert and deliberately un-importable |

## Running it

**Linux + CUDA + the `ss2` conda env only.** This Mac cannot load habitat-sim: the audio
propagation library is a prebuilt Linux-x64 binary. Editing here, running on the box.

```bash
# on the box, once: build the env, then stage the ESC-50 recordings
nrun bash earshot/tools/bootstrap_ss2.sh
python -m earshot.audio.clips --out-dir data/anomaly_audio

# one episode, end to end
python -m earshot --run-dir runs/<tag> --n-episodes 1 --max-steps 250

# the nine acceptance criteria, judged off the run directory
python -m earshot.task.smoke --run-dir runs/<tag>

# the box test suite (a few minutes, read-only, installs nothing)
bash earshot/tools/box_gate.sh
```

On this Mac:

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
- **Box tests print their measurements.** The numbers are what make the next decision.
- `data/`, `runs/`, `models/` are gitignored.

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

---

## Notes for the session that lands this

- `docs/archive/README.md` also needs one edit in the same commit: its heading points at
  `scripts/race-setup.sh`, which the commit deletes.
- Do not carry the "Running the ablation" section forward in any form. It invokes
  `python -m embodied_memory.run_hm3d_pol`, which will not exist.
