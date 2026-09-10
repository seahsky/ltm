#!/bin/bash
# earshot/tools/matrix_sweep.sh — ADR-0018's four cells, on the room-balanced assignment.
#
#   source earshot/tools/notify/notify-run.sh
#   nrun bash earshot/tools/matrix_sweep.sh --tag matrix-1
#
# THE LAST PIECE `window_pilot.sh`'s HEADER SAID DID NOT EXIST. "ADR-0018's four cells need
# two memory stores, a prior pass wired into a run, and the episode plan consumed by the
# runner. None of the three exists." All three now do: `task/prior_driver.py` walks the
# tour and dumps the store, `run()` takes `--memory-condition`/`--memory-store` and builds
# the cell's `MemoryContext` itself, and this script is what carves the four cells and runs
# them. Nothing here is a new mechanism -- it is the four pieces that already existed,
# called in the order ADR-0018 always said they would be.
#
# THE ASSIGNMENT IS COMPUTED, NOT HARD-CODED. `anchor_yield.py --emit-assignment` is the
# room-balanced design (PR #77's fix: balanced over ROOMS, because `chair`/`sofa`/
# `tv_monitor` are all the living room and an object-balanced design hands that room half
# the scenes). Re-derived fresh every run rather than pasted from a prior printout, so a
# scene that stops publishing a mesh, or a HM3D refresh, changes the assignment rather than
# silently going stale in a comment.
#
# ONE TOUR SERVES EVERY CELL. `class_at_category` reads the class bank per stop, so one
# prior pass over the assigned scenes yields rows for whichever class anchors at each room
# -- and `stores_for_cell` (inside `run()`) filters that ONE store four different ways.
# `episode_diff.py` and `window_report.py` already assume this: THE SAME EPISODE runs under
# every condition (same scene, same class, same seed), which is what makes them paired.
#
# THE FOUR CELLS ARE `MemoryCondition`'s OWN NAMES, used as directory names on purpose:
# `heard_seen`, `heard_unseen`, `not_heard_seen`, `not_heard_unseen`. `window_report.py`
# and `episode_diff.py` take arbitrary arm directory names, so no new reader was built --
# `window_report.py runs/<tag> --arms "heard_seen heard_unseen not_heard_seen
# not_heard_unseen"` is the readout, unchanged.
#
# THE PRIMARY CONTRASTS, PER THE 2026-09-01 AMENDMENT: `heard_seen` vs `not_heard_unseen`
# is the pre-registered primary (both memories against neither); `heard_unseen` vs
# `not_heard_unseen` is co-primary and isolates the SEMANTIC store alone (both cells are
# unseen, so only whether the class was heard before can move the delta). Both contrasts,
# always, via `episode_diff.py runs/<tag>/<arm-a> runs/<tag>/<arm-b>`.
#
# --CLAP IS NOT OPTIONAL HERE. Without it every cell's `is_anomaly` verdict is None and
# `memory_prior`'s query embedding never exists, which `run_episode` already refuses --
# "a memory arm was passed but no CLAP encoder" -- rather than silently running four
# identical un-conditioned arms. This script always passes it.
#
# CONTINUE-ON-FAILURE AT THE (CONDITION, SCENE) GRAIN, `ablation_sweep.sh`'s own rule: a
# zero-yield scene is a measured fact about HM3D, recorded and skipped, never a failure.
#
# ONE DIRECTORY IS ONE RUN, enforced before any work starts.
#
# Flags: --tag T (required in practice), --classes "a b c" (default: the room-balanced
#        bank `toilet_flush snoring keyboard_typing`), --n-episodes N (default 15, PER
#        SCENE PER CONDITION), --max-steps M (default 250), --sounding-steps N (default
#        60), --limit N (scene cap on the assignment, default 0 = no limit), --seed N,
#        --conditions "a b" (default all four), --leg-budget N (prior pass, default 200),
#        --goal-radius M (prior pass, default 1.0), --start-draws N (prior pass, default
#        20), --max-tour-dy M (prior pass, default 1.0), --split S (default val; `train`
#        is the 80-scene pool), --prior-only (stop after the coverage gate), --out-dir
#        DIR, --no-pull, --force.
#
# --split IS THE SCENE POOL, and val is small. ObjectNav HM3D v1 publishes 20 val scenes
# and 80 train ones; `anchor_yield --split train` measured 73 of the 80 usable, 1068 of
# 1200 episodes anchored (89.0%) against val's 210 of 282 (74.5%), and a LOWER null
# (35.1% against 39.5%). Every other entry point already took a split; this one did not,
# so no sweep could be pointed anywhere but val.
#
# --prior-only IS THE DRY RUN. Steps 1-4 are the assignment, the tour and the gate, and
# they cost under two minutes; the cells cost a night. Run it first on a fresh tag, read
# `matrix_audit --store` on what it wrote, and only then spend the night.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then :; else
  echo "ERROR: execute this script, don't source it — its exit calls would kill your shell." >&2
  return 1
fi

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

ENV_NAME="${SS2_ENV_NAME:-ss2}"
TAG="matrix-$(date +%Y%m%d-%H%M%S)"
CLASSES="toilet_flush snoring keyboard_typing"
N_EPISODES=15
MAX_STEPS=250
SOUNDING_STEPS=60
SEED=20260821
LIMIT=0
CONDITIONS="heard_seen heard_unseen not_heard_seen not_heard_unseen"
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
    --conditions)     need_value $# "$1"; CONDITIONS="$2";     shift 2 ;;
    --leg-budget)     need_value $# "$1"; LEG_BUDGET="$2";     shift 2 ;;
    --goal-radius)    need_value $# "$1"; GOAL_RADIUS="$2";    shift 2 ;;
    --start-draws)    need_value $# "$1"; START_DRAWS="$2";    shift 2 ;;
    --max-tour-dy)    need_value $# "$1"; MAX_TOUR_DY="$2";    shift 2 ;;
    --split)          need_value $# "$1"; SPLIT="$2";          shift 2 ;;
    --prior-only)     PRIOR_ONLY=1;                             shift ;;
    --out-dir)        need_value $# "$1"; OUT_DIR="$2";        shift 2 ;;
    --no-pull)        NO_PULL=1;                                shift ;;
    --force)          FORCE=1;                                  shift ;;
    -h|--help) sed -n '2,68p' "$0"; exit 0 ;;
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
    exec bash "$0" --tag "$TAG" --classes "$CLASSES" --n-episodes "$N_EPISODES" \
         --max-steps "$MAX_STEPS" --sounding-steps "$SOUNDING_STEPS" --seed "$SEED" \
         --limit "$LIMIT" --conditions "$CONDITIONS" --leg-budget "$LEG_BUDGET" \
         --goal-radius "$GOAL_RADIUS" --start-draws "$START_DRAWS" \
         --max-tour-dy "$MAX_TOUR_DY" --split "$SPLIT" --out-dir "$OUT_DIR" \
         ${_prior_only_flag:+--prior-only} ${_force_flag:+--force}
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
  2>&1 | tee "$OUT_DIR/prior_pass.log"
PRIOR_STATUS=${PIPESTATUS[0]}
STORE="$OUT_DIR/prior/store.json"
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
    echo "tour:           leg_budget=$LEG_BUDGET goal_radius=$GOAL_RADIUS start_draws=$START_DRAWS max_tour_dy=${MAX_TOUR_DY:-<unset>}"
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
CONDITION_LIST=($CONDITIONS)
N_CONDITIONS="${#CONDITION_LIST[@]}"
TOTAL_EPISODES=$((N_SCENES * N_CONDITIONS * N_EPISODES))
EST_SECONDS=$(awk "BEGIN{printf \"%d\", $TOTAL_EPISODES * 24.2}")
EST_HOURS=$(awk "BEGIN{printf \"%.1f\", $EST_SECONDS / 3600.0}")
echo "  $N_CONDITIONS condition(s): ${CONDITION_LIST[*]}"
echo "  $N_EPISODES episodes per scene per condition -> $TOTAL_EPISODES total"
echo "  estimated wall clock: ${EST_HOURS} h at ablation_sweep.sh's measured 24.2 s/episode"

{
  echo "tag:            $TAG"
  echo "commit:         $COMMIT"
  echo "args:           $ORIGINAL_ARGS"
  echo "split:          $SPLIT"
  echo "classes:        $CLASSES"
  echo "scenes:         ${SCENE_LIST[*]}"
  echo "conditions:     ${CONDITION_LIST[*]}"
  echo "n_episodes:     $N_EPISODES (per scene, per condition)"
  echo "max_steps:      $MAX_STEPS"
  echo "sounding_steps: $SOUNDING_STEPS (fixed_steps, ADR-0017)"
  echo "seed:           $SEED"
  echo "store:          $STORE"
  echo "tour:           leg_budget=$LEG_BUDGET goal_radius=$GOAL_RADIUS start_draws=$START_DRAWS max_tour_dy=${MAX_TOUR_DY:-<unset>}"
  echo "started:        $(date -Is)"
} > "$OUT_DIR/provenance.txt"

HERM_BEFORE="$OUT_DIR/.hermeticity-before.json"
if ! python -m earshot.tools.reset_manifest --verify-absent --when before > "$HERM_BEFORE"; then
  echo "WARN: could not record the pre-run hermeticity check — criterion 9 will be NOT_RUN"
  rm -f "$HERM_BEFORE"
fi

# --- 5. the four cells --------------------------------------------------------
banner "[5/6] $N_CONDITIONS condition(s) x $N_SCENES scene(s)"
FAILED_RUNS=0
ZERO_YIELD=""
for condition in "${CONDITION_LIST[@]}"; do
  echo ""
  echo "  --- condition $condition ---"
  for scene in "${SCENE_LIST[@]}"; do
    anomaly_class="${CLASS_OF_SCENE[$scene]}"
    run_dir="$OUT_DIR/$condition/$scene"
    echo "    $condition / $scene ($anomaly_class)   ($(date +%H:%M:%S))"
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
      --memory-condition "$condition" \
      --memory-store "$STORE" \
      > "$OUT_DIR/$condition-$scene.log" 2>&1
    status=$?
    if [ "$status" -ne 0 ]; then
      if is_zero_yield "$run_dir"; then
        echo "      ZERO YIELD — this scene placed no episode. Recorded, not a failure."
        ZERO_YIELD="$ZERO_YIELD $condition/$scene"
        continue
      fi
      echo "      FAILED (exit $status) — tail:"
      tail -n 12 "$OUT_DIR/$condition-$scene.log" | sed 's/^/        /'
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
python -m earshot.tools.window_report "$OUT_DIR" --arms "$CONDITIONS"
READ_STATUS=$?

echo ""
echo "  primary contrast (both memories vs neither), per the 2026-09-01 amendment:"
python -m earshot.tools.episode_diff \
  "$OUT_DIR/heard_seen" "$OUT_DIR/not_heard_unseen" 2>&1 | sed 's/^/    /' \
  || echo "    (episode_diff did not run for heard_seen vs not_heard_unseen)"

echo ""
echo "  co-primary contrast (the SEMANTIC store alone: both cells unseen), same amendment:"
python -m earshot.tools.episode_diff \
  "$OUT_DIR/heard_unseen" "$OUT_DIR/not_heard_unseen" 2>&1 | sed 's/^/    /' \
  || echo "    (episode_diff did not run for heard_unseen vs not_heard_unseen)"

{
  echo "finished:       $(date -Is)"
  echo "failed_runs:    $FAILED_RUNS"
  echo "zero_yield:     ${ZERO_YIELD:-<none>}"
} >> "$OUT_DIR/provenance.txt"

banner "done"
echo "  artefacts: $OUT_DIR/{provenance.txt,assignment.tsv,prior/store.json,<condition>/<scene>/}"
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
