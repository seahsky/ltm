#!/bin/bash
# earshot/tools/propose_sweep.sh — ADR-0026: does the prior help when it has to EARN the pick?
#
#   source earshot/tools/notify/notify-run.sh
#   nrun bash earshot/tools/propose_sweep.sh --tag propose-1
#
# THE CONTRAST. `runner.py` used to do `target = memory_prior.target` while diverting, so
# the recalled place REPLACED the acoustic estimate outright, and because the divert class
# held exactly one candidate the structural override in `_rank` made it the pick with
# nothing ranking it. `matrix-2` measured that at -10.3 points against a RIGHT prior:
# redundant when the memory agrees with the cue, unchecked when it does not. PR #133 made
# the prior emit a SECOND investigate candidate instead, so eq. 26 chooses. This sweep is
# the two arms, and they differ in `--memory-proposes` and in nothing else.
#
# **BOTH ARMS ARE DREAM ARMS, AND THAT IS FORCED.** Both candidates are diverts, so the
# structural override cannot separate them; without a memory term there is nothing to
# choose WITH and `--memory-proposes` correctly reduces to the acoustic behaviour. So the
# contrast needs `--clap`, a prior-pass store AND the DREAM knobs in both arms. Neither
# `ablation_sweep.sh` (DREAM, no prior pass) nor `matrix_sweep.sh` (prior pass, no DREAM)
# could carve it, which is why this file exists rather than a flag on one of them.
#
# FOUR ARMS, BECAUSE TWO OF THEM ARE THE CONTROL FOR THE OTHER TWO. `propose-a`/`propose-b`
# and `replace-a`/`replace-b` are the SAME command run twice, which is `repeat-1`'s design:
# the apparatus flips 16.2% of outcomes on byte-identical reruns, so a single pair of arms
# cannot tell an effect from that. `dream-3` measured -5.3 points at p = 0.0275 and
# `dream-4` measured -0.7 at p = 0.8555 on the same contrast, four weeks of box time to
# learn one thing: MEASURE THE REPEAT IN THE SAME RUN. The within-arm pairs below price the
# apparatus on tonight's box, tonight's code and tonight's scenes, so the contrast is read
# against its own noise rather than against a historical number from a different build.
#
# THE PRIMARY OUTCOME IS STAGE 4 -> STAGE 5 CONVERSION, NOT FIND-SR (ADR-0026). The
# mechanism acts during the detour, so `SOURCE_REACHED` given `INVESTIGATE_ENTERED` is the
# transition it can move; Find-SR carries the stage 2 and stage 3 attrition in front of it
# as noise. `episode_diff --given-stage INVESTIGATE_ENTERED` is the reader, and it prints
# each arm's drop count so the conditioning is checkable rather than assumed. Find-SR is
# printed beside it, always, and both tests are reported for both (ADR-0016).
#
# THE CONDITION IS HELD FIXED AT `heard_unseen`. It is the SEMANTIC store alone, which is
# what proposes a place; `heard_seen` adds the episodic narrowing that ADR-0023 records as
# measuring a selection rule rather than a memory (matrix-1 defect D1). Holding it fixed is
# what keeps this a one-knob contrast.
#
# THE ASSIGNMENT IS COMPUTED, NOT HARD-CODED, and one tour serves every arm -- both
# inherited from `matrix_sweep.sh` unchanged, along with the coverage gate, `--resume`,
# `--prior-only`, and the rule that a zero-yield scene is a measured fact and not a
# failure. See that file's header for the reasoning behind each.
#
# WHAT IT COSTS. 4 arms x 19 val scenes x 15 episodes = 1140 episodes. `dream-3` measured
# 32.4 s/episode and `dream-4` 35.8 s/episode all-in for DREAM arms, so this is 10.3 to
# 11.3 h plus about two minutes of prior pass. The estimate is printed BEFORE the night is
# spent, from the arms actually asked for.
#
# ONE DIRECTORY IS ONE RUN, enforced before any work starts.
#
# Flags: --tag T (required in practice), --classes "a b c" (default the room-balanced bank
#        `toilet_flush snoring keyboard_typing`), --n-episodes N (default 15, PER SCENE PER
#        ARM), --max-steps M (default 250), --sounding-steps N (default 60), --limit N
#        (scene cap, default 0 = no limit), --seed N, --arms "a b" (default all four),
#        --condition C (default heard_unseen), --propose-max-offset M (the ADR-0026 safety
#        rail, default 6.0; negative removes it), --leg-budget N, --goal-radius M,
#        --start-draws N, --max-tour-dy M, --split S (default val), --prior-only, --out-dir
#        DIR, --no-pull, --force, --resume.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then :; else
  echo "ERROR: execute this script, don't source it — its exit calls would kill your shell." >&2
  return 1
fi

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

ENV_NAME="${SS2_ENV_NAME:-ss2}"
TAG="propose-$(date +%Y%m%d-%H%M%S)"
CLASSES="toilet_flush snoring keyboard_typing"
N_EPISODES=15
MAX_STEPS=250
SOUNDING_STEPS=60
SEED=20260821
LIMIT=0
# `-a` and `-b` are the SAME command, twice. Not a knob: `repeat-1` measured a 16.2%
# outcome flip on byte-identical reruns, so the repeat IS the measurement of the noise the
# contrast has to clear. Directory names because `window_report` and `episode_diff` take
# arbitrary arm names and pair on `(scene, episode index)`, which the repeats share.
ARMS="propose-a propose-b replace-a replace-b"
# Held fixed across every arm. The SEMANTIC store alone is what proposes a place;
# `heard_seen` adds the episodic narrowing ADR-0023 records as a selection rule.
CONDITION="heard_unseen"
# ADR-0026's safety rail, in metres of navmesh route from the acoustic estimate. Negative
# removes it. A first choice this run prices.
PROPOSE_MAX_OFFSET="6.0"
# 4.5 AND NOT `ablation_sweep.sh`'s DEFAULT OF 2.0. `eta-1` ran at 2.0 and the cap bound
# on 15 of 15 episodes, so top-k was the retention rule and eta was decoration; `eta-2`
# priced 4.5 and `dream-3` and `dream-4` both ran at it explicitly. The other driver's
# default is stale against its own runs of record, so this one carries the measured value
# rather than inheriting a number no result was produced at.
DREAM_ETA=4.5
DREAM_MAX_RETAINED=12
# The DREAM knobs, VERBATIM from `ablation_sweep.sh`'s `DREAM_KNOBS` at the eta `eta-2`
# priced, so the memory term this arm ranks with is the one every DREAM result was
# measured under. `tests/mac/test_propose_sweep.py` holds the two lists against each other.
DREAM_KNOBS="--dream \
  --dream-stm-horizon 8 --dream-stm-decay 0.8 --dream-present-weight 0.7 \
  --dream-coherence 0.99 --dream-min-segment 3 --dream-max-segment 12 \
  --dream-alpha 1.0 --dream-beta 1.0 --dream-gamma 1.0 \
  --dream-eta $DREAM_ETA --dream-max-retained $DREAM_MAX_RETAINED \
  --dream-min-support 2 \
  --dream-k-experience 3 --dream-k-pattern 2 --dream-k-knowledge 1 \
  --dream-temperature 0.5 \
  --dream-lambda-plan 1.0 --dream-lambda-memory 0.5 --dream-lambda-feasibility 0.5"
LEG_BUDGET=200
GOAL_RADIUS=1.0
# THE TWO TOUR FIXES prior-5 and prior-7 MEASURED, defaulted ON, because the values that
# reproduce today's behaviour are the values that fail this script's own coverage gate.
#
# --start-draws 20: `plan_until_non_empty` redraws ONLY while the plan has no stop, so
# raising it cannot re-roll a scene that already plans one. `prior-5` measured exactly
# that -- qyAac8rV8Zk 0 of 0 -> 1 of 2, and the ziup5kvtCCR control unmoved.
#
# --max-tour-dy 1.0: not a new number. `build_anomaly_episodes` has screened placements at
# `max_dy_m=1.0` since ADR-0010 (dataset.py), so a tour without it stores stops NO EPISODE
# can put a source at -- qyAac8rV8Zk's bathroom was 2.24 m of height from its start. The
# default makes the two halves of one experiment agree about what is placeable.
#
# Both were unreachable from here until now: this script calls `prior_driver` directly and
# never grew the flags, so every matrix sweep would have hit prior-2's red gate.
START_DRAWS=20
MAX_TOUR_DY=1.0
# HOW MANY STOREYS ONE SCENE'S TOUR MAY VISIT, defaulted to 3 because 1 is measured
# broken. `plan_tour` filters candidates against ONE random start, so a scene's whole
# tour is decided by a single dice roll and can only ever see one floor. `prior-9`
# measured the cost on the real 73: 16 scenes BLIND on the seen axis, 14 of them
# `snoring` -- 58% of that arm -- with drop reasons reading `on another floor`.
#
# The episodes never had this problem: each has its own start and is screened against
# it, so those bedrooms are placeable and the tour simply never went upstairs.
TOUR_FLOORS=3
# WHICH SCENE POOL. Every other entry point took a split and this one hardcoded `val`,
# so a sweep could not be pointed at `train` at all.
#
# It is not a small pool difference. ObjectNav HM3D v1 has 20 val scenes and 80 train
# ones, and `anchor_yield --split train` measured the train half at 73 scenes usable,
# 1068 of 1200 episodes anchored (89.0%) against val's 210 of 282 (74.5%), with the
# null-hypothesis score LOWER (35.1% against 39.5%) -- more episodes AND a harder
# experiment. Balance is free there too: greedy and balanced both reach 1068.
SPLIT="val"
PRIOR_ONLY=0
# --resume implies --force: it reuses a non-empty directory ON PURPOSE, which is the
# one case "one directory is one run" is not what the operator wants.
RESUME=0
OUT_DIR=""
NO_PULL=0
FORCE=0
ORIGINAL_ARGS="$*"

need_value() { [ "$1" -ge 2 ] || { echo "FATAL: $2 needs a value"; exit 2; }; }
while [ $# -gt 0 ]; do
  case "$1" in
    --tag)            need_value $# "$1"; TAG="$2";            shift 2 ;;
    --classes)        need_value $# "$1"; CLASSES="$2";        shift 2 ;;
    --n-episodes)     need_value $# "$1"; N_EPISODES="$2";     shift 2 ;;
    --max-steps)      need_value $# "$1"; MAX_STEPS="$2";      shift 2 ;;
    --sounding-steps) need_value $# "$1"; SOUNDING_STEPS="$2"; shift 2 ;;
    --seed)           need_value $# "$1"; SEED="$2";           shift 2 ;;
    --limit)          need_value $# "$1"; LIMIT="$2";          shift 2 ;;
    --arms)           need_value $# "$1"; ARMS="$2";           shift 2 ;;
    --condition)      need_value $# "$1"; CONDITION="$2";      shift 2 ;;
    --propose-max-offset) need_value $# "$1"; PROPOSE_MAX_OFFSET="$2"; shift 2 ;;
    --dream-eta)      need_value $# "$1"; DREAM_ETA="$2";      shift 2 ;;
    --dream-max-retained) need_value $# "$1"; DREAM_MAX_RETAINED="$2"; shift 2 ;;
    --leg-budget)     need_value $# "$1"; LEG_BUDGET="$2";     shift 2 ;;
    --goal-radius)    need_value $# "$1"; GOAL_RADIUS="$2";    shift 2 ;;
    --start-draws)    need_value $# "$1"; START_DRAWS="$2";    shift 2 ;;
    --max-tour-dy)    need_value $# "$1"; MAX_TOUR_DY="$2";    shift 2 ;;
    --tour-floors)    need_value $# "$1"; TOUR_FLOORS="$2";    shift 2 ;;
    --split)          need_value $# "$1"; SPLIT="$2";          shift 2 ;;
    --prior-only)     PRIOR_ONLY=1;                             shift ;;
    --resume)         RESUME=1; FORCE=1;                        shift ;;
    --out-dir)        need_value $# "$1"; OUT_DIR="$2";        shift 2 ;;
    --no-pull)        NO_PULL=1;                                shift ;;
    --force)          FORCE=1;                                  shift ;;
    -h|--help) sed -n '2,80p' "$0"; exit 0 ;;
    *) echo "FATAL: unknown argument: $1"; exit 2 ;;
  esac
done
OUT_DIR="${OUT_DIR:-runs/$TAG}"

banner() { printf '\n========== %s ==========\n' "$1"; }

is_zero_yield() {
  [ -f "$1/summary.json" ] || return 1
  python -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['n_episodes']==0 else 1)" \
    "$1/summary.json" 2>/dev/null
}

# A cell that already finished. `report/` writes its artefacts ATOMICALLY and never
# overwrites them, so a `summary.json` that parses is a cell that ran to the end -- a
# crash mid-write leaves no such file rather than half of one.
#
# A ZERO-YIELD cell counts as finished, and must: it wrote a real summary saying the scene
# placed no episode, which is a measured fact about HM3D and not work left to redo.
is_finished_cell() {
  [ -f "$1/summary.json" ] || return 1
  python -c "import json,sys; json.load(open(sys.argv[1]))['n_episodes']; sys.exit(0)" \
    "$1/summary.json" 2>/dev/null
}

# A CELL ABOUT TO RUN OWNS ITS DIRECTORY, so whatever is in it belongs to an attempt that
# did not finish. `matrix-2`'s recovery found this the level below the store: with the
# tour correctly skipped, the one unfinished cell still died before its first episode --
#
#     ArtifactExistsError: runs/matrix-2/not_heard_seen/LcAd9dhvVwh/env_report.json
#     already exists. Re-using a run tag mixes two runs into one directory with nothing
#     on disk saying so
#
# -- because its first attempt got as far as writing `env_report.json` and then crashed.
# `report/` refuses to overwrite, and that refusal is right: it answers a real incident,
# committed run directories holding a different run's data.
#
# PASSING `--overwrite` WOULD BE THE WRONG FIX, and for the reason the refusal exists. A
# cell that died at episode 3 of 4 leaves episodes 0-2 on disk; an overwriting re-run
# replaces them, but any episode file the new build does NOT write survives and the
# readout counts it as this run's. That is two invocations in one directory with nothing
# saying so -- exactly what the error is guarding.
#
# Clearing is the honest version of the same intent. A cell reaches here only when it is
# about to be run, which under --resume means `is_finished_cell` said no. The directory is
# then a crashed attempt's debris, not a record of anything, and the invariant holds
# exactly afterwards: one cell directory, one invocation.
#
# The path is checked before anything is removed. `$OUT_DIR` is operator-supplied via
# --out-dir, and an `rm -rf` built from a variable deserves the belt and the braces.
clear_cell_dir() {
  [ -n "$1" ] || { echo "FATAL: clear_cell_dir got an empty path"; exit 1; }
  case "$1" in
    *..*) echo "FATAL: refusing to clear '$1' — it contains '..'"; exit 1 ;;
  esac
  case "$1" in
    "$OUT_DIR"/*/*) ;;
    *) echo "FATAL: refusing to clear '$1' — not a <arm>/<scene> under $OUT_DIR"
       exit 1 ;;
  esac
  [ -d "$1" ] || return 0
  echo "      clearing $(find "$1" -type f | wc -l | tr -d ' ') file(s) left by an attempt that did not finish"
  rm -rf "${1:?}"
}

# --- ONE DIRECTORY IS ONE RUN, before anything expensive ------------------
if [ -d "$OUT_DIR" ] && [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]; then
  if [ "$FORCE" = 0 ]; then
    echo "FATAL: $OUT_DIR already exists and is not empty."
    echo "       One directory is one run. Pass a fresh --tag, or --force to reuse it."
    exit 1
  fi
  echo "WARN: --force — reusing a non-empty $OUT_DIR."
fi

# --- 1. self-update by re-exec ---------------------------------------------
if [ "$NO_PULL" = 0 ]; then
  banner "[1/6] git pull --ff-only"
  _self_before="$(md5sum "$0" 2>/dev/null | awk '{print $1}')"
  git pull --ff-only || echo "WARN: git pull failed — running the checked-out copy"
  _self_after="$(md5sum "$0" 2>/dev/null | awk '{print $1}')"
  if [ -n "$_self_before" ] && [ "$_self_before" != "$_self_after" ] && [ -z "${_REEXEC:-}" ]; then
    echo "  this script changed in the pull — re-execing the new body"
    export _REEXEC=1
    _force_flag=""
    [ "$FORCE" = 1 ] && _force_flag="--force"
    _prior_only_flag=""
    [ "$PRIOR_ONLY" = 1 ] && _prior_only_flag="--prior-only"
    _resume_flag=""
    [ "$RESUME" = 1 ] && _resume_flag="--resume"
    exec bash "$0" --tag "$TAG" --classes "$CLASSES" --n-episodes "$N_EPISODES" \
         --max-steps "$MAX_STEPS" --sounding-steps "$SOUNDING_STEPS" --seed "$SEED" \
         --limit "$LIMIT" --arms "$ARMS" --leg-budget "$LEG_BUDGET" \
         --condition "$CONDITION" --propose-max-offset "$PROPOSE_MAX_OFFSET" \
         --dream-eta "$DREAM_ETA" --dream-max-retained "$DREAM_MAX_RETAINED" \
         --goal-radius "$GOAL_RADIUS" --start-draws "$START_DRAWS" \
         --max-tour-dy "$MAX_TOUR_DY" --tour-floors "$TOUR_FLOORS" \
         --split "$SPLIT" --out-dir "$OUT_DIR" \
         ${_prior_only_flag:+--prior-only} ${_resume_flag:+--resume} ${_force_flag:+--force}
  fi
else
  banner "[1/6] git pull SKIPPED (--no-pull)"
fi
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  commit: $COMMIT"

# --- 2. the env -------------------------------------------------------------
banner "[2/6] conda env: $ENV_NAME"
MINICONDA="${HOME}/miniconda3"
[ -x "$MINICONDA/bin/conda" ] || { echo "FATAL: $MINICONDA/bin/conda missing"; exit 1; }
[ -d "$MINICONDA/envs/$ENV_NAME" ] || { echo "FATAL: env '$ENV_NAME' missing — run bootstrap_ss2.sh"; exit 1; }
set +u
eval "$("$MINICONDA/bin/conda" shell.bash hook)"
conda activate "$ENV_NAME" || { echo "FATAL: conda activate $ENV_NAME failed"; exit 1; }
set -u
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
echo "  python: $(python -V 2>&1)"

[ -d "data/anomaly_audio" ] || {
  echo "FATAL: no data/anomaly_audio — stage the ESC-50 recordings once:"
  echo "       python -m earshot.audio.clips --out-dir data/anomaly_audio"
  exit 1
}

# Idempotent — a second run prints "already staged" and returns — so it is safe to leave
# in the path. `prior_pass.sh` and `clap_gate.sh` both do this; this script calls
# `task/prior_driver.py` directly rather than through `prior_pass.sh`, so it is not
# inherited for free and has to be its own step. Both this sweep's CLAP arm (`--clap`
# below) and the prior pass's rendering need the local safetensors copy — the box pins
# torch 2.2.2+cu118, and transformers refuses `torch.load` on the Hub's raw
# `pytorch_model.bin` below torch 2.6 (CVE-2025-32434).
python -m earshot.task.models \
  || { echo "FATAL: could not stage the CLAP checkpoint — nothing below can classify"; exit 1; }

mkdir -p "$OUT_DIR"

# --- 3. the assignment -------------------------------------------------------
banner "[3/6] the room-balanced assignment"
ASSIGNMENT="$OUT_DIR/assignment.tsv"
python -m earshot.tools.anchor_yield \
  --split "$SPLIT" \
  --classes "$CLASSES" \
  --n-episodes "$N_EPISODES" \
  --limit "$LIMIT" \
  --emit-assignment "$ASSIGNMENT" \
  | tee "$OUT_DIR/assignment.log"
ASSIGNMENT_STATUS=${PIPESTATUS[0]}
[ "$ASSIGNMENT_STATUS" -eq 0 ] || { echo "FATAL: could not compute an assignment"; exit 1; }
[ -s "$ASSIGNMENT" ] || { echo "FATAL: $ASSIGNMENT is empty"; exit 1; }

SCENES=""
declare -A CLASS_OF_SCENE
while IFS=$'\t' read -r scene class; do
  [ -n "$scene" ] || continue
  SCENES="$SCENES $scene"
  CLASS_OF_SCENE["$scene"]="$class"
done < "$ASSIGNMENT"
# shellcheck disable=SC2206
SCENE_LIST=($SCENES)
N_SCENES="${#SCENE_LIST[@]}"
[ "$N_SCENES" -gt 0 ] || { echo "FATAL: the assignment named no scenes"; exit 1; }
echo "  $N_SCENES scene(s) assigned"

# --- 4. the prior pass --------------------------------------------------------
# ONE tour serves every condition and every scene: the store this writes is the UNFILTERED
# pair `run()` filters four ways per cell via `stores_for_cell`, so this step runs once,
# not once per condition.
banner "[4/6] the prior pass"
STORE="$OUT_DIR/prior/store.json"
# A RESUME REUSES THE STORE, IT DOES NOT REBUILD IT. `matrix-2`'s recovery run spent
# 10m 34s re-touring all 73 scenes, reported `73 of 73 scene(s) complete`, and then died
# in `write_pass_store` on the store it had just been told not to overwrite -- so the one
# crashed cell it existed to re-run never started.
#
# The fix is to skip the tour, not to pass `--overwrite`. `--overwrite` would destroy the
# artefact the finished cells actually consumed and replace it with one nothing has
# checked. A same-seed rebuild is LIKELY identical -- `prior_driver` calls
# `world.seed_navmesh(seed)` before every draw -- but likely is not the standard here:
# step 1 git-pulls and re-execs, so the code rebuilding the store is by construction not
# guaranteed to be the code that built it, and the climb loop's early exits decide how
# many draws the seeded sequence even consumes. Reuse is provable, rebuild is hope.
#
# The gate below still runs on the reused store. A store is checked, never trusted.
if [ "$RESUME" = 1 ] && [ -s "$STORE" ]; then
  echo "  --resume: reusing the store already on disk, NOT re-touring."
  echo "    $STORE"
  echo "  It is the store the finished cells consumed. The coverage gate below still runs."
  PRIOR_STATUS=0
else
  python -m earshot.task.prior_driver \
    --run-dir "$OUT_DIR/prior" \
    --split "$SPLIT" \
    --scenes "$SCENES" \
    --classes "$CLASSES" \
    --seed "$SEED" \
    --leg-budget "$LEG_BUDGET" \
    --goal-radius "$GOAL_RADIUS" \
    --start-draws "$START_DRAWS" \
    ${MAX_TOUR_DY:+--max-tour-dy "$MAX_TOUR_DY"} \
    --tour-floors "$TOUR_FLOORS" \
    2>&1 | tee "$OUT_DIR/prior_pass.log"
  PRIOR_STATUS=${PIPESTATUS[0]}
fi
[ "$PRIOR_STATUS" -eq 0 ] && [ -f "$STORE" ] || {
  echo "FATAL: the prior pass did not produce $STORE"
  exit 1
}

# THE COVERAGE GATE (matrix-1 review, D3). The store EXISTING is not the store COVERING
# the assignment: a scene whose tour left one leg unreached is excluded from the merge,
# and before `pass_provenance` recorded incompletes it landed in NO provenance list — the
# sweep then ran that scene's seen cells byte-identical to its unseen cells with no error
# anywhere. NOT_RUN is never green: an incomplete assignment stops the sweep here, before
# any cell is paid for. Fix the tour (raise --leg-budget) or drop the scene from the
# assignment; --force does not bypass this, because the cells it would run are not the
# experiment the assignment names.
python -m earshot.tools.matrix_audit --store "$STORE" --gate-scenes "$SCENES" || {
  echo "FATAL: the prior pass does not cover the assignment — the gate's lines above name the scenes"
  exit 1
}

# --prior-only: everything above, nothing below. The first four steps are the sweep's own
# first half -- the same assignment, the same tour flags, the same gate -- and they cost
# under two minutes against the cells' several hours (`prior-2` measured the pass at
# 1m 40s over 19 scenes). So the question "does this assignment tour cleanly, and what did
# the tour drop" is answerable BEFORE a night is committed, by the code that will run it
# rather than by a hand-assembled approximation of it.
if [ "$PRIOR_ONLY" = 1 ]; then
  # Its own provenance: the full block below never runs on this path, and a store whose
  # tour parameters are not on disk beside it cannot be compared with the next one.
  {
    echo "tag:            $TAG (--prior-only: assignment, tour and gate; no cells)"
    echo "commit:         $COMMIT"
    echo "args:           $ORIGINAL_ARGS"
    echo "split:          $SPLIT"
    echo "classes:        $CLASSES"
    echo "scenes:         ${SCENE_LIST[*]}"
    echo "seed:           $SEED"
    echo "store:          $STORE"
    echo "tour:           leg_budget=$LEG_BUDGET goal_radius=$GOAL_RADIUS start_draws=$START_DRAWS max_tour_dy=${MAX_TOUR_DY:-<unset>} tour_floors=$TOUR_FLOORS"
    echo "finished:       $(date -Is)"
  } > "$OUT_DIR/provenance.txt"
  banner "--prior-only: stopping after the gate"
  echo "  store:      $STORE"
  echo "  assignment: $ASSIGNMENT ($N_SCENES scene(s))"
  echo "  read it:    python -m earshot.tools.matrix_audit --store $STORE"
  echo "  the COMPLETE, SHORT A ROOM lines are the rooms the tour dropped and why."
  exit 0
fi

# shellcheck disable=SC2206
ARM_LIST=($ARMS)
N_ARMS="${#ARM_LIST[@]}"
# Every arm name must say which side of the contrast it is, because the ONLY thing that
# turns proposing on below is a `propose-` prefix. A typo would otherwise run a control
# arm under a treatment name and the sweep would report a null it never tested.
for _arm in "${ARM_LIST[@]}"; do
  case "$_arm" in
    propose-*|replace-*) ;;
    *) echo "FATAL: arm '$_arm' is neither propose-* nor replace-*, so nothing decides"
       echo "       whether it passes --memory-proposes. Name it for its side."
       exit 2 ;;
  esac
done
TOTAL_EPISODES=$((N_SCENES * N_ARMS * N_EPISODES))
# 32.4 and 35.8 s/episode are MEASURED, on DREAM arms: `dream-3` ran 855 episodes in
# 7h41m and `dream-4` ran 570 in 5h40m. `matrix_sweep.sh`'s 24.2 is a non-DREAM figure and
# would under-price this run by a third. A range rather than a point, because the two
# measurements differ and pretending otherwise is how a night gets booked at the wrong size.
EST_LOW=$(awk "BEGIN{printf \"%.1f\", $TOTAL_EPISODES * 32.4 / 3600.0}")
EST_HIGH=$(awk "BEGIN{printf \"%.1f\", $TOTAL_EPISODES * 35.8 / 3600.0}")
echo "  $N_ARMS arm(s): ${ARM_LIST[*]}"
echo "  condition held fixed at $CONDITION; arms differ in --memory-proposes alone"
echo "  $N_EPISODES episodes per scene per arm -> $TOTAL_EPISODES total"
echo "  estimated wall clock: ${EST_LOW} to ${EST_HIGH} h, at dream-3's 32.4 and dream-4's"
echo "                        35.8 s/episode, both MEASURED on DREAM arms"

{
  echo "tag:            $TAG"
  echo "commit:         $COMMIT"
  echo "args:           $ORIGINAL_ARGS"
  echo "split:          $SPLIT"
  echo "classes:        $CLASSES"
  echo "scenes:         ${SCENE_LIST[*]}"
  echo "arms:           ${ARM_LIST[*]}"
  echo "condition:      $CONDITION (held fixed across every arm)"
  echo "propose_rail_m: $PROPOSE_MAX_OFFSET"
  echo "dream_knobs:    $DREAM_KNOBS"
  echo "n_episodes:     $N_EPISODES (per scene, per arm)"
  echo "max_steps:      $MAX_STEPS"
  echo "sounding_steps: $SOUNDING_STEPS (fixed_steps, ADR-0017)"
  echo "seed:           $SEED"
  echo "store:          $STORE"
  echo "tour:           leg_budget=$LEG_BUDGET goal_radius=$GOAL_RADIUS start_draws=$START_DRAWS max_tour_dy=${MAX_TOUR_DY:-<unset>} tour_floors=$TOUR_FLOORS"
  echo "started:        $(date -Is)"
} > "$OUT_DIR/provenance.txt"

HERM_BEFORE="$OUT_DIR/.hermeticity-before.json"
if ! python -m earshot.tools.reset_manifest --verify-absent --when before > "$HERM_BEFORE"; then
  echo "WARN: could not record the pre-run hermeticity check — criterion 9 will be NOT_RUN"
  rm -f "$HERM_BEFORE"
fi

# --- 5. the four cells --------------------------------------------------------
banner "[5/6] $N_ARMS arm(s) x $N_SCENES scene(s)"
FAILED_RUNS=0
ZERO_YIELD=""
RESUMED=0
for arm in "${ARM_LIST[@]}"; do
  # THE ONE KNOB. Everything else below is byte-identical between the two sides, which is
  # what makes this a one-variable contrast rather than dream-2's two-variable one.
  case "$arm" in
    propose-*) PROPOSE_FLAG="--memory-proposes" ;;
    *)         PROPOSE_FLAG="" ;;
  esac
  echo ""
  echo "  --- arm $arm (${PROPOSE_FLAG:-<replace: no flag>}) ---"
  for scene in "${SCENE_LIST[@]}"; do
    anomaly_class="${CLASS_OF_SCENE[$scene]}"
    run_dir="$OUT_DIR/$arm/$scene"
    if [ "$RESUME" = 1 ] && is_finished_cell "$run_dir"; then
      RESUMED=$((RESUMED + 1))
      echo "    $arm / $scene — already finished, skipped (--resume)"
      continue
    fi
    echo "    $arm / $scene ($anomaly_class)   ($(date +%H:%M:%S))"
    clear_cell_dir "$run_dir"
    python -m earshot \
      --run-dir "$run_dir" \
      --split "$SPLIT" \
      --scene "$scene" \
      --n-episodes "$N_EPISODES" \
      --max-steps "$MAX_STEPS" \
      --seed "$SEED" \
      --localization realizable \
      --detector oracle \
      --anomaly-class "$anomaly_class" \
      --sounding-policy fixed_steps \
      --sounding-steps "$SOUNDING_STEPS" \
      --clap \
      --memory-condition "$CONDITION" \
      --memory-store "$STORE" \
      --memory-propose-max-offset "$PROPOSE_MAX_OFFSET" \
      $PROPOSE_FLAG \
      $DREAM_KNOBS \
      > "$OUT_DIR/$arm-$scene.log" 2>&1
    status=$?
    if [ "$status" -ne 0 ]; then
      if is_zero_yield "$run_dir"; then
        echo "      ZERO YIELD — this scene placed no episode. Recorded, not a failure."
        ZERO_YIELD="$ZERO_YIELD $arm/$scene"
        continue
      fi
      echo "      FAILED (exit $status) — tail:"
      tail -n 12 "$OUT_DIR/$arm-$scene.log" | sed 's/^/        /'
      FAILED_RUNS=$((FAILED_RUNS + 1))
      continue
    fi
    if [ -f "$HERM_BEFORE" ]; then
      python -m earshot.tools.reset_manifest --verify-absent --when after \
          > "$run_dir/.hermeticity-after.json" \
        && python -m earshot.tools.reset_manifest --write-record \
             --run-dir "$run_dir" --before "$HERM_BEFORE" \
             --after "$run_dir/.hermeticity-after.json" --commit "$COMMIT" \
             >/dev/null \
        || echo "      WARN: hermeticity incomplete — criterion 9 will not be green"
    fi
  done
done

# --- 6. the readout ------------------------------------------------------------
# No new reader: `window_report.py` and `episode_diff.py` already take arbitrary arm
# directory names, and this sweep's layout (`<tag>/<condition>/<scene>/`) is exactly the
# shape both already assume.
banner "[6/6] the readout"
python -m earshot.tools.window_report "$OUT_DIR" --arms "$ARMS"
READ_STATUS=$?

# `pair` prints one contrast twice: the PRE-REGISTERED conditional rate first, because
# ADR-0026 names stage 4 -> stage 5 conversion as the primary outcome, and Find-SR under
# it because ADR-0016 says both tests are reported and a reader will want the headline
# number anyway. Continue-on-failure: a missing arm must not cost the other three rows.
pair() {
  local label="$1" a="$2" b="$3"
  echo ""
  echo "  === $label:  $a  vs  $b ==="
  if [ ! -d "$OUT_DIR/$a" ] || [ ! -d "$OUT_DIR/$b" ]; then
    echo "    SKIPPED: one of the two arms has no directory. Not a smaller result, no result."
    return 0
  fi
  echo "  -- PRIMARY: SOURCE_REACHED given INVESTIGATE_ENTERED (ADR-0026) --"
  python -m earshot.tools.episode_diff "$OUT_DIR/$a" "$OUT_DIR/$b" \
    --given-stage INVESTIGATE_ENTERED 2>&1 | sed 's/^/    /' \
    || echo "    (episode_diff did not run)"
  echo "  -- beside it: Find-SR@1m over every paired episode --"
  python -m earshot.tools.episode_diff "$OUT_DIR/$a" "$OUT_DIR/$b" 2>&1 | sed 's/^/    /' \
    || echo "    (episode_diff did not run)"
}

banner "THE CONTRAST"
pair "repeat 1" replace-a propose-a
pair "repeat 2" replace-b propose-b

banner "THE APPARATUS, MEASURED ON THIS RUN"
echo "  The same command, twice. Under a null these are two draws from one distribution,"
echo "  so whatever they disagree by is the floor the contrast above has to clear."
echo "  repeat-1 measured 16.2% on byte-identical reruns and dream-3's -5.3 points"
echo "  evaporated to dream-4's -0.7 on exactly this kind of unmeasured noise. These two"
echo "  rows are that number, on tonight's box and tonight's code."
pair "within REPLACE" replace-a replace-b
pair "within PROPOSE" propose-a propose-b

banner "DID THE MECHANISM RUN AT ALL"
# ADR-0026's fourth branch, and it is checked BEFORE the contrast is believed: if eq. 26
# ranked the proposal first on under 5% of eligible steps, the rail or the store is
# suppressing the mechanism and the run is not a result about memory. `dream-1` wrote its
# central quantity to disk and no reader could print it; this is that lesson applied.
python - "$OUT_DIR" <<'PYEOF' || echo "  (the proposal counters could not be read)"
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for arm in sorted(d.name for d in root.iterdir() if d.is_dir() and d.name.startswith("propose")):
    tot = {}
    for audit in root.joinpath(arm).glob("*/episodes/*/audit.json"):
        try:
            metrics = json.loads(audit.read_text()).get("metrics") or {}
        except (OSError, ValueError):
            continue
        for key, value in metrics.items():
            if key.startswith("memory_propose_") and key != "memory_propose_rail_m":
                tot[key] = tot.get(key, 0.0) + float(value)
    eligible = tot.get("memory_propose_eligible", 0.0)
    first = tot.get("memory_propose_ranked_first", 0.0)
    share = "n/a" if eligible <= 0 else "{:.2%}".format(first / eligible)
    print("  {:<12} eligible={:.0f} ranked_first={:.0f} ({})  emitted={:.0f} "
          "railed={:.0f} unrouted={:.0f}".format(
              arm, eligible, first, share, tot.get("memory_propose_emitted", 0.0),
              tot.get("memory_propose_railed", 0.0),
              tot.get("memory_propose_unrouted", 0.0)))
    if eligible > 0 and first / eligible < 0.05:
        print("       UNDER 5%: ADR-0026's fourth branch. The rail or the store is")
        print("       suppressing the mechanism, and this run is NOT a result about")
        print("       memory until that is fixed. Read it before reading the contrast.")
PYEOF

{
  echo "finished:       $(date -Is)"
  echo "failed_runs:    $FAILED_RUNS"
  echo "zero_yield:     ${ZERO_YIELD:-<none>}"
  echo "resumed_cells:  $RESUMED"
} >> "$OUT_DIR/provenance.txt"

banner "done"
if [ "$RESUMED" -gt 0 ]; then
  # Said out loud, because a resumed run's numbers come from two invocations and a reader
  # who does not know that would read one wall clock for all of them.
  echo "  RESUMED: $RESUMED cell(s) were already finished and were not re-run."
  echo "           This directory's episodes come from more than one invocation."
fi
echo "  artefacts: $OUT_DIR/{provenance.txt,assignment.tsv,prior/store.json,<arm>/<scene>/}"
echo "  finished=$(date -Is)"
if [ "$FAILED_RUNS" -gt 0 ]; then
  echo "  EXIT NONZERO: $FAILED_RUNS run(s) failed for a reason other than zero yield."
  exit 1
fi
if [ "$READ_STATUS" -ne 0 ]; then
  echo "  EXIT NONZERO: the readout found nothing to report."
  exit "$READ_STATUS"
fi
exit 0
