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
#        --goal-radius M (prior pass, default 1.0), --out-dir DIR, --no-pull, --force.

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
    --out-dir)        need_value $# "$1"; OUT_DIR="$2";        shift 2 ;;
    --no-pull)        NO_PULL=1;                                shift ;;
    --force)          FORCE=1;                                  shift ;;
    -h|--help) sed -n '2,54p' "$0"; exit 0 ;;
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
    exec bash "$0" --tag "$TAG" --classes "$CLASSES" --n-episodes "$N_EPISODES" \
         --max-steps "$MAX_STEPS" --sounding-steps "$SOUNDING_STEPS" --seed "$SEED" \
         --limit "$LIMIT" --conditions "$CONDITIONS" --leg-budget "$LEG_BUDGET" \
         --goal-radius "$GOAL_RADIUS" --out-dir "$OUT_DIR" ${_force_flag:+--force}
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

mkdir -p "$OUT_DIR"

# --- 3. the assignment -------------------------------------------------------
banner "[3/6] the room-balanced assignment"
ASSIGNMENT="$OUT_DIR/assignment.tsv"
python -m earshot.tools.anchor_yield \
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
  --scenes "$SCENES" \
  --classes "$CLASSES" \
  --seed "$SEED" \
  --leg-budget "$LEG_BUDGET" \
  --goal-radius "$GOAL_RADIUS" \
  2>&1 | tee "$OUT_DIR/prior_pass.log"
PRIOR_STATUS=${PIPESTATUS[0]}
STORE="$OUT_DIR/prior/store.json"
[ "$PRIOR_STATUS" -eq 0 ] && [ -f "$STORE" ] || {
  echo "FATAL: the prior pass did not produce $STORE"
  exit 1
}

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
  echo "classes:        $CLASSES"
  echo "scenes:         ${SCENE_LIST[*]}"
  echo "conditions:     ${CONDITION_LIST[*]}"
  echo "n_episodes:     $N_EPISODES (per scene, per condition)"
  echo "max_steps:      $MAX_STEPS"
  echo "sounding_steps: $SOUNDING_STEPS (fixed_steps, ADR-0017)"
  echo "seed:           $SEED"
  echo "store:          $STORE"
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
