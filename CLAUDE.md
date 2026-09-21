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
| `audio/` | spec, guard, IR handling, the bed, onset, CLAP, the lateral cue, the sounding window, the accumulation buffer |
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

# the first run of ADR-0017's windowed task. THREE arms, because CONTINUOUS is the
# control: a windowed run differenced against the historic sweeps crosses both the offset
# step and the ADR-0019 renderer. Not the ADR-0018 matrix and it says so in its header
nrun bash earshot/tools/window_pilot.sh --tag <fresh-tag>

# the pilot's readout, over any sweep laid out as <tag>/<arm>/<scene>/. Read-only, no
# GPU, seconds — so a finished sweep can be re-read without re-running it, which is what
# `pilot-1` needed after its readout looked for the wrong filename and reported three
# dead arms over 120 episodes that were on disk the whole time
python -m earshot.tools.window_report runs/<tag>

# THE OVERNIGHT SWEEP: the paper's HM3D baseline and the ablation table, in one run.
# NINE arms by default. `full` is the baseline of record (ADR-0021); `no-climb`,
# `no-cue`, `scan-only` and `anechoic` each remove one component; `dream` and
# `dream-nomem` are the memory pair (ADR-0025) and differ in `lambda_memory` alone.
# `oracle-loc` and `oracle-loc-matched` are CEILINGS and not ablations -- they ADD the
# source coordinate. Difference only the MATCHED one against `full`: `oracle-1` measured
# the other at 94.3% source-reached and 2 of 270 Find-SR@1m out of one arm, because it
# arrives at 1.5 m and the metric scores 1.0 m (ADR-0028).
# Every arm shares its episodes with the reference, so `episode_diff` pairs them. It
# prints the MDE it is buying BEFORE it spends the night earning it -- read that estimate
# rather than a figure quoted here, because it is computed from the arms you asked for.
# `--arms` runs a subset, and a subset without `full` is quoted against its FIRST arm
nrun bash earshot/tools/ablation_sweep.sh --tag <fresh-tag>
python -m earshot.tools.window_report runs/<tag> \
  --arms "full no-climb no-cue scan-only anechoic dream dream-nomem"

# THE LOCALIZATION CEILING, on the baseline's own criterion (ADR-0028). Two arms, ~3.5 h:
# `full` is the in-run control, because `repeat-1` measured 16.2% of outcomes flipping on
# byte-identical reruns and the decisive contrast does not get a control from last night
nrun bash earshot/tools/ablation_sweep.sh --tag <fresh-tag> \
  --arms "full oracle-loc-matched"
python -m earshot.tools.episode_diff runs/<tag>/full runs/<tag>/oracle-loc-matched

# ADR-0029's GATE, BEFORE ANY CONTROLLER CODE: does a finished cast leg know which way
# the source is? Grades the verdict `READ_LEGS` would act on, over `full`'s own legs,
# against the change in route to the source. The legs are the controller's, rebuilt
# and checked step by step against the recorded `realizable_action`; a leg the record
# disagrees with is never graded. Prints the pre-registered branch (BUILD / ONE BRANCH /
# STOP) and the grid of `T_LEG` it chose from. PRICES THE VERDICT, NOT THE SWEEP: those
# legs were walked under blind alternation. Refuses any run that is not `full`'s rule.
# It read STOP on 2026-09-21 (ADR-0029). BY SOUNDING STATE splits the legs by where the
# source was, off each record's window: before the offset, across it, or on the bed
# alone. That split is NOT a gate, because it was chosen after the STOP.
# Read-only, no GPU, minutes
python -m earshot.tools.leg_replay runs/abl-2/full runs/oracle-1/full runs/oracle-2/full

# WHAT DREAM's MEMORY ACTUALLY DID — the numbers `window_report` and `episode_diff` cannot
# see. `dream-1` wrote `dream_omega_e_spread` onto 282 episodes and no reader could print
# it, so the run's own central quantity reached nobody.
# Four sections, in the order that decides what to run next: (A) did `omega_t` MOVE, the
# spread and never the mean, because a mean of 0.5 is what a live omega and a dead one
# both report; (B) how degenerate `M^E`'s keys are, BANDED BY MEMORY SIZE, because
# "the keys are the same key" and "the memory had two rows" are different findings;
# (C) whether `eta` retained or refused anything; (D) how large a memory the sweep ever
# built — it names a per-scene reset, which is what the sweep's one-runner-per-scene loop
# produces. Read-only, no GPU, seconds. Exits 2 if the arm recorded omega nowhere:
# unreadable is not flat.
# (G) DID THE MEMORY TERM CHANGE THE PICK — eq. 26's own counterfactual, re-ranking each
# step's pool at lambda_memory 0. `dream_informed_steps` only ever said the memory
# ANSWERED. The denominator EXCLUDES steps where the answer was fixed before the memory
# was read (divert override in force, pool of one, nothing retrieved). An INERT verdict
# with a zero S_mem spread names `k_experience`/store diversity as the fix and rules out
# lambda_memory; a LIVE verdict makes an outcome difference attributable to steering
python -m earshot.tools.dream_report runs/<tag>          # --arm NAME for a non-`dream` arm

# PRICE `eta` BEFORE BOOKING A NIGHT (ADR-0024 step 1). `dream-2` set eta 0.5 against the
# box's EMPTY-memory `I_j`, a regime that occurs once per chain, and the gate never opened
# again: 275 of 282 episodes retained NOTHING, 38 of the arm's 45 rows from one walk. The
# cause was a units bug — `C_j`/`U_j` were shares carrying a hidden `1/J` that `N_j` does
# not — so eq. 10 degenerated to eq. 12. Fixed; `mean(C_j + U_j)` is now 2.0 for ANY `J`,
# and eta is MEANINGFUL AND UNPRICED. One scene, minutes, prints its own readout.
# READ SECTION C: it prints `dream_segments_over_eta` beside the cap and NAMES which of
# the two retained, because `dream_rows_added` cannot — eta passing twelve segments and a
# cap of twelve truncating forty write the same twelve rows. "THE CAP IS THE RETENTION
# RULE" means eta is decoration: ADR-0024's top-k deviation, which needs its own ADR and
# not a quietly raised cap. A reportable DREAM comparison costs 14h15m; this rules it out
# first. `dream_report` reads a bare run directory too, so `runs/<tag>` off this driver is
# readable without the sweep's `<arm>/<scene>/` layout
nrun bash earshot/tools/eta_pass.sh --tag <fresh-tag>   # --eta E --max-retained N --scene S

# THE CHAIN, ADDED AFTER `dream-1` CAME BACK A NULL. That sweep built nineteen memories
# and threw each away — the driver invokes the runner once per scene, so section D above
# found all 19 scenes starting from an empty `M^E`. `omega_t` reaches the agent ONLY
# through `memory_consistency`, which renormalises omega^E against omega^P, so it cannot
# act until `M^P` is non-empty — and `M^P` needs two successes sharing a concept triple,
# which 15 episodes in one room at 33% reach rarely give. The `dream` arm now chains
# `M^E` through one file, so `abstract` sees nineteen scenes' successes instead of one
# scene's. Nothing about `G` changes. SCENE ORDER IS PART OF THE RESULT for that arm now
nrun bash earshot/tools/ablation_sweep.sh --tag <fresh-tag> --arms "full dream"

# THE CONTROL THAT MAKES A DREAM ARM READABLE (ADR-0024). `dream-2` was a TWO-variable
# contrast: `pick_plan` does NOT reduce to `pick_waypoint` at `lambda_feasibility 0.5`,
# so `full` vs `dream` differenced the memory term AND the feasibility term at once.
# `dream-nomem` is `dream` with `lambda_memory 0.0` and every other knob identical, built
# by substitution so the two CANNOT drift apart. Three arms decompose it:
#   dream vs dream-nomem  = the memory term alone, which is eq. 26's whole claim
#   dream-nomem vs full   = the feasibility term alone, which dream-2 confounded
# `--dream-eta`/`--dream-max-retained` are flags so a sweep runs at the value
# `eta_pass.sh` priced, without editing this driver at 11pm.
#
# `--resume` PICKS IT UP AFTER A CRASH, at the scene grain: a scene that wrote a
# summary.json is skipped, an unfinished one is cleared and re-run. It refuses to change
# the knobs mid-sweep, and for the chained `dream` arm it refuses unless the memory
# file's own scene list matches the scenes it is about to skip -- a chain that silently
# lost a house cannot be detected afterwards. Pass the SAME flags on the resume
nrun bash earshot/tools/ablation_sweep.sh --tag <fresh-tag> \
  --arms "full dream dream-nomem" --dream-eta <priced>
nrun bash earshot/tools/ablation_sweep.sh --tag <same-tag> \
  --arms "full dream dream-nomem" --dream-eta <priced> --resume

# the same chain by hand, one scene at a time. `--dream-memory-in` on a MISSING file is
# an error and never a silent empty memory: the two are indistinguishable afterwards,
# which is exactly how `dream-1` looked like a working run
python -m earshot --run-dir runs/<tag>/dream/<scene> --clap --dream <knobs> \
  --dream-memory-out runs/<tag>/dream/memory.json        # first scene: write only
python -m earshot --run-dir runs/<tag>/dream/<scene2> --clap --dream <knobs> \
  --dream-memory-in runs/<tag>/dream/memory.json \
  --dream-memory-out runs/<tag>/dream/memory.json        # every later scene

# DID ADR-0022's PLACEMENT CHANGE ACTUALLY REACH THIS SWEEP — the question a moved SR
# cannot answer, because a re-run of the SAME task moves by 3.0 points on identical bytes.
# Splits every arm into anchored / geometric / MISSING and gives each branch its own
# reached-rate; MISSING is never folded into geometric, and an arm that recorded the field
# nowhere exits nonzero. Read-only, no GPU, seconds
python -m earshot.tools.placement_report runs/<tag>

# WOULD A WIDER CLASS SET RAISE THAT ANCHORED FRACTION — asked BEFORE spending a night on
# it. Runs the real `build_anomaly_episodes` over the published goals for every sounding
# class, so it is the build a sweep would do and not a model of one. It reproduces `abl-2`'s
# measured 134 of 282 as its own check. READ THE `ALWAYS-` ROWS, NOT THE TOTALS: it prints
# what a memory that learned nothing scores under each design, and the design with the most
# anchored episodes is the one that hands the null hypothesis the highest score. THE UNIT IS
# THE ROOM: `chair`, `sofa` and `tv_monitor` are all the living room, so a design balanced
# over four anchor OBJECTS is one room holding half the scenes. No GPU, no simulator, seconds
python -m earshot.tools.anchor_yield

# THE PRIOR PASS: walk the scripted tour over real scenes, render REAL audio of the given
# classes at every reached stop, and dump `<tag>/store.json` -- what `run()`'s new
# `--memory-store` flag reads. `--scenes`/`--classes` have no default on purpose: pass
# `anchor_yield`'s own room-balanced assignment, not a guess. One tour per scene serves
# every class in the bank that anchors at a room the scene has, so pass the whole bank in
# one invocation. Continue-on-failure at the scene grain; nonzero only if every scene failed
nrun bash earshot/tools/prior_pass.sh --tag prior-1 \
  --scenes "sceneA sceneB sceneC" --classes "toilet_flush snoring keyboard_typing"

# ONE EPISODE UNDER ONE MATRIX CELL -- reads the store the prior pass dumped and runs
# `run_episode` with a real `MemoryContext` built for that cell. `--memory-condition none`
# (the default) is byte-identical to no memory flags at all. Any other value needs
# `--memory-store` and `--clap`; a condition with no store is a usage error caught before
# the environment probe is paid for, not a silent run under `none`
python -m earshot --run-dir runs/<tag> --clap --anomaly-class toilet_flush \
  --memory-condition heard_seen --memory-store runs/prior-1/store.json

# THE MATRIX SWEEP: ADR-0018's four cells, carved on the assignment `anchor_yield`
# computes fresh and the ONE prior pass that assignment needs. `heard_seen`,
# `heard_unseen`, `not_heard_seen`, `not_heard_unseen` -- named after `MemoryCondition`'s
# own values, so `window_report.py`/`episode_diff.py` read them with no new tool. Prints
# both pre-registered contrasts (heard_seen vs not_heard_unseen, and the SEMANTIC-store-
# alone co-primary heard_unseen vs not_heard_unseen) at the end, always
nrun bash earshot/tools/matrix_sweep.sh --tag matrix-1
python -m earshot.tools.window_report runs/matrix-1 \
  --arms "heard_seen heard_unseen not_heard_seen not_heard_unseen"

# ADR-0026: DOES THE PRIOR HELP WHEN IT HAS TO EARN THE PICK? The recalled place used to
# REPLACE the acoustic estimate outright, unranked, which is what matrix-2 measured at
# -10.3 points against a RIGHT prior. It is a second investigate candidate now and eq. 26
# chooses. BOTH ARMS ARE DREAM ARMS and that is forced: both candidates are diverts, so
# without a memory term nothing can separate them and --memory-proposes is a no-op.
# FOUR arms, because `-a`/`-b` are the SAME command twice: repeat-1 measured a 16.2% flip
# on byte-identical reruns, dream-3's -5.3 pts became dream-4's -0.7, so the repeat is
# measured IN THE SAME RUN and the contrast is read against tonight's own noise.
# ~10.3-11.3 h at 4 arms x 19 val scenes x 15 episodes (32.4-35.8 s/ep, MEASURED on DREAM
# arms). Run --prior-only FIRST: it is the assignment, the tour and the coverage gate in
# about two minutes, and it answers "does this assignment tour cleanly" before the night
nrun bash earshot/tools/propose_sweep.sh --tag propose-1 --prior-only
nrun bash earshot/tools/propose_sweep.sh --tag propose-1 --resume

# THE PRIMARY OUTCOME IS STAGE 4 -> STAGE 5 CONVERSION, not Find-SR (ADR-0026): the
# mechanism acts during the detour, so the attrition in front of it is noise. The driver
# prints this for all four pairs; this is the same reader by hand. It drops a pair unless
# BOTH arms reached the gate, and prints each arm's drop count — conditioning is only
# sound while the gate is UPSTREAM of what the arms differ in, and a gap between those two
# counts is the evidence that it is not
python -m earshot.tools.episode_diff runs/<tag>/replace-a runs/<tag>/propose-a \
  --given-stage INVESTIGATE_ENTERED

# THE MATRIX-1 REVIEW'S FOUR READ-ONLY QUESTIONS, off a finished sweep's own artefacts:
# A. store coverage -- which assigned scenes the prior pass completed, and the SILENT
#    drops a pre-`pass_provenance` store hides (a scene in no provenance list at all ran
#    its seen cells byte-identical to its unseen cells with no error anywhere);
# B. seen-axis liveness -- `memory_prior_instances` differing at a SHARED voted category
#    is the episodic store actually narrowing something; a category flip is render noise;
# C. what the not_heard cells were told -- a confident WRONG category, almost never
#    `no_prediction`, because `without_class` strips one class and `_vote` has no abstain:
#    heard-vs-not-heard is right-prior vs wrong-prior until a NONE arm runs beside them;
# D. determinism for free -- zero-episodic-row scenes ran seen==unseen on identical
#    inputs, so every discordant pair there is the apparatus flipping on its own, the
#    number repeat-1 bought with a full re-run.
# E. CAN THE VOTE ABSTAIN -- the question C leaves open, decided off data already on
#    disk. Grades every recall against the sweep's own `assignment.tsv` (correct = the
#    voted category IS the anchor object of the class that scene ran) and reports the
#    largest confidence floor that loses NO correct recall, plus the max-J operating
#    point. Separable means `_vote` can decline and `not_heard` becomes an arm with no
#    prior; overlap means only a NONE arm controls for a wrong one (ADR-0023). Blind to
#    every miss: confidence reaches the audit only on a RESOLVED prior, and the section
#    prints how many episodes that is.
# The same module is the sweep's own coverage gate (`--gate-scenes`, exit 2 on a missing
# scene), wired between the prior pass and the cells. No GPU, seconds
python -m earshot.tools.matrix_audit runs/matrix-1

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
