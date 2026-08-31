#!/bin/bash
# earshot/tools/window_pilot.sh — the first run of the task ADR-0017 defines.
#
#   source earshot/tools/notify/notify-run.sh
#   nrun bash earshot/tools/window_pilot.sh --tag pilot-1
#
# THIS IS NOT THE GENERALIZATION MATRIX AND CANNOT BE. ADR-0018's four cells need two
# memory stores, a prior pass wired into a run, and the episode plan consumed by the
# runner. None of the three exists: `grep -rn "semantic_store\|episodic_store" earshot/`
# is empty, no caller of `task/prior_pass.walk_tour` exists outside its own tests, and
# `task/runner.py` names neither `plan_episodes` nor the bank of record. Run the four
# cells today and they would be four identical arms, because nothing varies between them.
# That is worth saying out loud rather than discovering in a report.
#
# What this DOES measure is everything the matrix rests on and nobody has ever seen:
#
#   1. THE EPISODE'S COST. A windowed episode has never been timed. Every budget
#      downstream — `build_plan.py`'s 15 hours for 500 episodes, ADR-0018's 2000 runs —
#      extrapolates 27 s/episode from the ANOMALY-RESPONSE task, which had a different
#      renderer and a different stopping rule. This run replaces a guess with a number,
#      and it is the reason the defaults here are small.
#
#   2. SWS, EVER, AT ALL. The metric is implemented, gated and box-tested, and has never
#      returned a value from a real episode. `n_window_closed` being zero would mean
#      every episode ends before its own offset step, which would make the whole silent
#      phase unreachable at these step budgets — a design fact, discoverable in one run.
#
#   3. WHETHER THE CLIMB SURVIVES A BURSTY CLIP. The known risk, and the arm that is here
#      for it. A 0.6 s transient on a 5 s loop is audible on one fold in five: measured
#      cue SD/level 2.014 against a PERFECTLY DETERMINISTIC renderer, so it is the loop's
#      phase and not renderer noise. If `climb_eps` lands above the level the reading
#      actually reaches, `is_rising` never fires and the agent casts until the budget
#      ends. `glass_break` is the arm; `alarm` is its control at the same window.
#
# THE CONTROL ARM IS NOT OPTIONAL AND IT IS WHY THERE ARE THREE. ADR-0019 changed the
# renderer and ADR-0017 changed the stopping rule in the same tree. A windowed run
# differenced against `arrive-2` or `yield-2` crosses BOTH at once and can attribute
# neither. `WindowPolicy.CONTINUOUS` is the pre-ADR-0017 stopping rule THROUGH the new
# renderer, so `cont-alarm` vs `win-alarm` isolates the offset step, and only a
# comparison against the historic sweeps carries the renderer too. This repo's own rule:
# a claim that X broke because of a change needs the arm where the change is absent.
#
# THE ARMS PAIR BY EPISODE. Same scenes, same `--seed`, same `--n-episodes`, same
# `--max-steps`, so episode k is the same task in all three and `tools/episode_diff.py`
# can run an exact McNemar over the pairs. `repeat-1` measured a 16.2% flip rate on
# BYTE-IDENTICAL reruns and a net of +11 on nothing at all, so at this `n` the pairing is
# not a refinement, it is the only test with any power. The pilot is still far under the
# 15-episode MDE — it is sized to time the episode and expose a mechanism failure, NOT to
# resolve a difference between arms. Any delta it prints is a direction, never a result.
#
# ONE DIRECTORY IS ONE RUN, enforced before any work starts (`yield-1` pooled two
# invocations into one 41% and nothing on disk said so).
#
# CONTINUE-ON-FAILURE: one scene that cannot load must not cost the others. Continuing is
# not passing — the exit code is NONZERO if any arm lost a scene or any smoke gate failed.
#
# WHAT `pilot-1` COST, because the fix is in three places below and the reason is one.
# It ran all 120 episodes, wrote all of them, and reported NONE of the three numbers.
#   * the readout was a heredoc that walked for `audit.json`; the writer names them
#     `ep0000.audit.json`, so all three arms printed "NO EPISODES ON DISK". It is now
#     `tools/window_report.py`, held by `tests/mac/test_window_report.py` over
#     directories the real writer wrote. A reader inside a bash string is untestable and
#     this repo does not get to ship one.
#   * `episode_diff` was handed SCENE directories, which it refuses by design. It now
#     gets the ARM directories, which pairs 40 episodes rather than 10 four times.
#   * criterion 9 was never armed, so all twelve gates went red on a structural NOT_RUN
#     and the run exited 1 with its other eight criteria green. Armed here exactly as
#     `yield_sweep.sh` arms it.
# Everything the run itself measured was sound. What failed was the join, and only the
# `window_report` half of it could ever have been caught by a test — which is the half
# that now has one.
#
# Flags: --tag T (required in practice; one directory is one run), --n-episodes N
#        (default 10, PER SCENE PER ARM), --max-steps M (default 250), --scenes "a b c"
#        (default: the first --limit scenes with a mesh), --limit N (default 4),
#        --sounding-steps N (default 60, the provisional ADR-0017 value), --seed S,
#        --out-dir DIR, --no-pull, --force.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then :; else
  echo "ERROR: execute this script, don't source it — its exit calls would kill your shell." >&2
  return 1
fi

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

ENV_NAME="${SS2_ENV_NAME:-ss2}"
TAG="pilot-$(date +%Y%m%d-%H%M%S)"
N_EPISODES=10
MAX_STEPS=250
SOUNDING_STEPS=60
SEED=20260805
SCENES=""
LIMIT=4
OUT_DIR=""
NO_PULL=0
FORCE=0
ORIGINAL_ARGS="$*"

need_value() { [ "$1" -ge 2 ] || { echo "FATAL: $2 needs a value"; exit 2; }; }
while [ $# -gt 0 ]; do
  case "$1" in
    --tag)             need_value $# "$1"; TAG="$2";             shift 2 ;;
    --n-episodes)      need_value $# "$1"; N_EPISODES="$2";      shift 2 ;;
    --max-steps)       need_value $# "$1"; MAX_STEPS="$2";       shift 2 ;;
    --sounding-steps)  need_value $# "$1"; SOUNDING_STEPS="$2";  shift 2 ;;
    --seed)            need_value $# "$1"; SEED="$2";            shift 2 ;;
    --scenes)          need_value $# "$1"; SCENES="$2";          shift 2 ;;
    --limit)           need_value $# "$1"; LIMIT="$2";           shift 2 ;;
    --out-dir)         need_value $# "$1"; OUT_DIR="$2";         shift 2 ;;
    --no-pull)         NO_PULL=1;                                shift ;;
    --force)           FORCE=1;                                  shift ;;
    -h|--help) sed -n '2,77p' "$0"; exit 0 ;;
    *) echo "FATAL: unknown argument: $1"; exit 2 ;;
  esac
done
OUT_DIR="${OUT_DIR:-runs/$TAG}"

banner() { printf '\n========== %s ==========\n' "$1"; }

# --- ONE DIRECTORY IS ONE RUN, before anything expensive ------------------
if [ -d "$OUT_DIR" ] && [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]; then
  if [ "$FORCE" = 0 ]; then
    echo "FATAL: $OUT_DIR already exists and is not empty."
    echo "       One directory is one run. Re-using it pools two pilots into one"
    echo "       aggregate with nothing on disk saying so. Pass a fresh --tag."
    exit 1
  fi
  echo "WARN: --force — reusing a non-empty $OUT_DIR. Earlier records will be pooled"
  echo "      into the aggregates below and cannot be told apart."
fi

# --- 1. self-update by re-exec (bash runs the body it loaded, not the file) -
if [ "$NO_PULL" = 0 ]; then
  banner "[1/5] git pull --ff-only"
  _self_before="$(md5sum "$0" 2>/dev/null | awk '{print $1}')"
  git pull --ff-only || echo "WARN: git pull failed — running the checked-out copy"
  _self_after="$(md5sum "$0" 2>/dev/null | awk '{print $1}')"
  if [ -n "$_self_before" ] && [ "$_self_before" != "$_self_after" ] && [ -z "${_REEXEC:-}" ]; then
    echo "  this script changed in the pull — re-execing the new body"
    export _REEXEC=1
    _force_flag=""
    [ "$FORCE" = 1 ] && _force_flag="--force"
    exec bash "$0" --tag "$TAG" --n-episodes "$N_EPISODES" --max-steps "$MAX_STEPS" \
         --sounding-steps "$SOUNDING_STEPS" --seed "$SEED" --limit "$LIMIT" \
         ${SCENES:+--scenes "$SCENES"} --out-dir "$OUT_DIR" ${_force_flag:+--force}
  fi
else
  banner "[1/5] git pull SKIPPED (--no-pull)"
fi
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  commit: $COMMIT"

# --- 2. the env -----------------------------------------------------------
banner "[2/5] conda env: $ENV_NAME"
MINICONDA="${HOME}/miniconda3"
[ -x "$MINICONDA/bin/conda" ] || { echo "FATAL: $MINICONDA/bin/conda missing"; exit 1; }
# A DIRECTORY check, never `conda env list | grep -q`: under pipefail a matching grep
# exits early, SIGPIPEs conda, and turns found-it into a pipeline failure.
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

# --- 3. scenes ------------------------------------------------------------
banner "[3/5] scenes"
if [ -z "$SCENES" ]; then
  SCENES="$(python - <<'PY'
# The same check task/runner._pick_scene makes: load the content file, then test that its
# resolved mesh is on this box. A content file without its .glb fails at load, which is
# not the same fact as a scene that cannot pose the task.
from earshot.task.episodes import available_scenes, find_scenes_dir, find_split_dir, load_scene
split_dir = find_split_dir("val", root=".")
scenes_dir = find_scenes_dir(root=".")
found = []
for label in available_scenes(split_dir):
    try:
        load_scene(split_dir, label, scenes_dir=scenes_dir)
    except Exception:
        continue
    found.append(label)
print(" ".join(found))
PY
)" || { echo "FATAL: scene discovery failed"; exit 1; }
fi
# shellcheck disable=SC2206
SCENE_LIST=($SCENES)
if [ "$LIMIT" -gt 0 ] && [ "${#SCENE_LIST[@]}" -gt "$LIMIT" ]; then
  SCENE_LIST=("${SCENE_LIST[@]:0:$LIMIT}")
fi
[ "${#SCENE_LIST[@]}" -gt 0 ] || { echo "FATAL: no scene with a mesh on this box"; exit 1; }
echo "  ${#SCENE_LIST[@]} scene(s): ${SCENE_LIST[*]}"
echo "  $N_EPISODES episodes per scene per arm, 3 arms"

mkdir -p "$OUT_DIR"
{
  echo "tag:            $TAG"
  echo "commit:         $COMMIT"
  echo "args:           $ORIGINAL_ARGS"
  echo "scenes:         ${SCENE_LIST[*]}"
  echo "n_episodes:     $N_EPISODES (per scene, per arm)"
  echo "max_steps:      $MAX_STEPS"
  echo "sounding_steps: $SOUNDING_STEPS (provisional, ADR-0017 open question)"
  echo "seed:           $SEED"
  echo "started:        $(date -Is)"
} > "$OUT_DIR/provenance.txt"

# --- 4. the three arms ----------------------------------------------------
# ARM NAME -> policy, class. Held as parallel arrays rather than an associative array so
# the ORDER is fixed: the control runs first, so a crash in the windowed arms still
# leaves the arm every comparison needs on disk.
ARM_NAMES=(cont-alarm win-alarm win-burst)
ARM_POLICY=(continuous fixed_steps fixed_steps)
ARM_CLASS=(alarm alarm glass_break)
ARM_WHY=(
  "the CONTROL: pre-ADR-0017 stopping rule through the ADR-0019 renderer"
  "the TASK: the window closes, on a clip whose energy is spread over its length"
  "the RISK: a 0.6 s transient on a 5 s loop, audible one fold in five"
)

# --- criterion 9's evidence, armed once around the whole sweep ------------
# Copied in behaviour from `yield_sweep.sh`, and `pilot-1` is why it is not optional:
# criterion 9 was structurally NOT_RUN on all twelve gates, NOT_RUN is red, and the
# driver exited 1 over twelve smoke gates whose other eight criteria were green. A
# criterion that is red on every ordinary run is one the reader learns to skip — which
# is how a never-armed canary read as a pass in the first place. `--verify-absent` is a
# filesystem existence check over the delete set and costs milliseconds.
HERM_BEFORE="$OUT_DIR/.hermeticity-before.json"
if ! python -m earshot.tools.reset_manifest --verify-absent --when before > "$HERM_BEFORE"; then
  echo "WARN: could not record the pre-run hermeticity check — criterion 9 will be NOT_RUN"
  rm -f "$HERM_BEFORE"
fi

FAILED_RUNS=0
banner "[4/5] the three arms"
for i in "${!ARM_NAMES[@]}"; do
  arm="${ARM_NAMES[$i]}"
  echo ""
  echo "  --- arm $arm: ${ARM_WHY[$i]} ---"
  for scene in "${SCENE_LIST[@]}"; do
    run_dir="$OUT_DIR/$arm/$scene"
    echo "    $arm / $scene"
    # `--detector oracle` and `--localization realizable` are the defaults and are passed
    # explicitly: the ORACLE STOP deletes the stop_miss half of the failure mass, so
    # these find numbers are an upper bound and the command line should say so rather
    # than the reader having to know the default.
    python -m earshot \
      --run-dir "$run_dir" \
      --scene "$scene" \
      --n-episodes "$N_EPISODES" \
      --max-steps "$MAX_STEPS" \
      --seed "$SEED" \
      --localization realizable \
      --detector oracle \
      --anomaly-class "${ARM_CLASS[$i]}" \
      --sounding-policy "${ARM_POLICY[$i]}" \
      --sounding-steps "$SOUNDING_STEPS" \
      > "$OUT_DIR/$arm-$scene.log" 2>&1
    status=$?
    if [ "$status" -ne 0 ]; then
      echo "      FAILED (exit $status) — tail:"
      tail -n 12 "$OUT_DIR/$arm-$scene.log" | sed 's/^/        /'
      FAILED_RUNS=$((FAILED_RUNS + 1))
      continue
    fi
    # stderr is NOT suppressed: `--verify-absent` prints "STILL PRESENT: <paths>" there
    # and that list is the entire diagnostic. A criterion 9 that went red with its
    # reason discarded is the shape of failure this repo keeps paying for.
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

# --- 5. read it back ------------------------------------------------------
banner "[5/5] what the pilot measured"
GATE_FAILED=0
READ_FAILED=0
for arm in "${ARM_NAMES[@]}"; do
  for scene in "${SCENE_LIST[@]}"; do
    [ -d "$OUT_DIR/$arm/$scene" ] || continue
    echo ""
    echo "  --- smoke gate: $arm / $scene ---"
    python -m earshot.task.smoke --run-dir "$OUT_DIR/$arm/$scene" || GATE_FAILED=1
  done
done

echo ""
echo "  --- the three numbers this pilot exists for ---"
# The readout is `earshot/tools/window_report.py` and NOT a heredoc, because the heredoc
# is what cost `pilot-1`. It walked for a file named `audit.json`; the writer names them
# `ep0000.audit.json`; three arms and 120 episodes on disk printed "NO EPISODES ON DISK",
# and no test in the tree could have caught it, because the reader was a bash string.
# `tests/mac/test_window_report.py` now holds that seam over directories the REAL writer
# wrote, and asserts this line still calls it.
python -m earshot.tools.window_report "$OUT_DIR" || READ_FAILED=1

echo ""
echo "  --- the offset step, isolated: win-alarm against its own control ---"
echo "  Paired by episode. At this n the sign is a direction and not a result:"
echo "  repeat-1 measured a 16.2% flip rate on byte-identical reruns."
# ARM directories, not scene directories. `load_outcomes` wants the tag directory that
# holds one subdirectory per scene, and for this sweep that IS the arm directory — a
# scene directory it refuses by design, with the message `pilot-1` printed four times
# where its only between-arm comparison should have been. Pairing across all four scenes
# at once is also the correct unit: 40 pairs answer a question 10 cannot.
if [ -d "$OUT_DIR/cont-alarm" ] && [ -d "$OUT_DIR/win-alarm" ]; then
  python -m earshot.tools.episode_diff \
    "$OUT_DIR/cont-alarm" "$OUT_DIR/win-alarm" 2>&1 | tail -n 30
fi

echo ""
echo "  logs:       $OUT_DIR/<arm>-<scene>.log"
echo "  provenance: $OUT_DIR/provenance.txt"
echo "  audits:     $OUT_DIR/<arm>/<scene>/"

if [ "$FAILED_RUNS" -ne 0 ]; then
  echo ""
  echo "RED: $FAILED_RUNS run(s) failed. A sweep missing runs is NOT_RUN for those cells,"
  echo "     and NOT_RUN is red. The aggregates above are over what survived."
  exit 1
fi
if [ "$READ_FAILED" -ne 0 ]; then
  echo ""
  echo "RED: the readout found no episode under any arm. Runs that produced nothing and a"
  echo "     reader that cannot find what they produced look identical from here, and"
  echo "     pilot-1 was the second: check $OUT_DIR/<arm>/<scene>/episodes/ by hand."
  exit 1
fi
if [ "$GATE_FAILED" -ne 0 ]; then
  echo ""
  echo "RED: at least one smoke gate did not pass. The measurements above stand as"
  echo "     measurements; what fails is the claim that the loop behaved."
  exit 1
fi
echo ""
echo "GREEN — every arm ran and every gate passed. This is a PILOT: it times the episode"
echo "        and exposes mechanism failures. It does not resolve a difference between"
echo "        arms, and it is not the generalization matrix."
