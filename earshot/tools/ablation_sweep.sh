#!/bin/bash
# earshot/tools/ablation_sweep.sh — the ablation table, and the baseline row it needs.
#
#   source earshot/tools/notify/notify-run.sh
#   nrun bash earshot/tools/ablation_sweep.sh --tag abl-1
#
# TWO DELIVERABLES, ONE SWEEP, because they are the same runs read two ways.
#
#   1. THE BASELINE, ON HM3D. The `full` arm is the complete system on the ADR-0017
#      windowed task, and it is the number every other row in the paper is quoted
#      against. It is an HM3D number by decision (ADR-0021): SAVN-CE is MP3D-bound —
#      its config is `magnet/config/mp3d/rgbd_ddppo_clean.yaml`, its episodes are MP3D
#      episodes and its checkpoint was trained on MP3D over 4xA800 for 14 days — so it
#      cannot be re-pointed at HM3D by a flag, and ADR-0015 forbids subtracting it from
#      an earshot number in any case. The baseline this paper compares against is
#      therefore INTERNAL and lives in this sweep.
#
#   2. THE ABLATION. Four arms, one component removed each, every one of them landed as
#      a typed `RunConfig` enum in commit 511b52f and never yet run. Until now the
#      ablation table in the paper was four empty rows.
#
# THE ARMS PAIR BY EPISODE, which is the only reason this sweep has any power. Same
# scenes, same `--seed`, same `--n-episodes`, same `--max-steps`, same class, same
# window, so episode k is the SAME TASK in all five arms and `tools/episode_diff.py`
# runs an exact McNemar over the pairs. `repeat-1` measured a 16.2% outcome flip rate on
# BYTE-IDENTICAL reruns and a net of +11 episodes on nothing at all, so an unpaired
# reading of this sweep would be measuring the apparatus.
#
# WHAT IT CAN AND CANNOT RESOLVE, before it is started rather than after. At the default
# 15 episodes per scene per arm over the 19 scenes that yield anything, each arm holds
# 285 episodes and `tools/power.py` prices the paired MDE at 6.68 points (80% power,
# alpha 0.05, the measured flip rate). 20 episodes a scene buys 5.78 points and costs
# 12.8 hours instead of 9.6. Anything smaller than about 7 points this sweep cannot see,
# and saying so here is cheaper than discovering it in the readout.
#
# THE SCENE-LEVEL TEST IS THE ONE THAT WILL DISAGREE. `sign_test_threshold(19)` is 15:
# an arm has to move 15 of 19 scenes the same way to clear a scene-level sign test, and
# `yield-2`->`arrive-2` already produced a mechanism that was green and exact at the
# episode level and null at the scene level. Both tests are reported. Neither is the
# tie-break for the other.
#
# THE CONTROL ARM RUNS FIRST so a crash in the ablation arms still leaves on disk the one
# arm every comparison needs.
#
# WHY THESE FOUR COMPONENTS. Each is a claim the paper makes about how the agent finds a
# source, and each has an arm because ADR-0014 requires a capability to be exercised
# rather than proxied:
#
#   no-climb   `--climb-rule off`. The energy climb never steers INVESTIGATE, so the
#              agent runs the scan/cast cycle alone. This is the largest expected effect
#              and the one that says whether live audio is doing any work at all.
#   no-cue     `--lateral-cue off`. The interaural sign is treated as ambiguous, so the
#              turn decision falls to its zero/absent default. Isolates BINAURAL
#              localization from mere loudness.
#   scan-only  `--cast-policy scan_only`. Every dead step turns instead of walking a leg.
#              This is the pre-`eps-1` control, and `eps-1`->`cast-1` is the comparison
#              that was DEMOTED from confirmed to "direction consistent, magnitude
#              unresolved" (McNemar p=0.18) for want of exactly this sweep's n.
#   anechoic   `--ir-policy anechoic`. Every rendered IR is replaced by a flat,
#              reverberation-free stand-in at all three render sites. This is row R5 and
#              it exists because `pilot-2` found win-alarm's silent phase audible in
#              356 of 356 episodes and win-burst's in 0 of 356, at SWS 0.115 against
#              0.112 — a well-powered null whose comparison is CONFOUNDED BY SOUND CLASS.
#              This arm asks the same question with class held fixed.
#
# NO MEMORY ARM IS IN THIS SWEEP, deliberately. ADR-0018's four cells need the stores
# wired into the runner and a prior pass that has run; neither exists yet, and four
# identical arms named after four conditions is worse than no table. This sweep is the
# ablation and the baseline. The matrix is a separate run.
#
# CONTINUE-ON-FAILURE, with the two kinds of failure told apart. A scene that yields ZERO
# EPISODES is a measured fact about that scene and not a broken run — `mL8ThkuaVTM` has
# posed no episode in any sweep this repo has ever run, and it is why `pilot-2` exited 1
# with 1095 good episodes on disk. It is now counted separately and printed by name. A
# run that failed for any OTHER reason is still red.
#
# Flags: --tag T (required in practice; one directory is one run), --n-episodes N
#        (default 15, PER SCENE PER ARM), --max-steps M (default 250), --scenes "a b c"
#        (default: every scene with a mesh), --limit N (default 0 = no limit),
#        --sounding-steps N (default 60), --anomaly-class C (default alarm),
#        --seed S, --out-dir DIR, --arms "a b" (default: all five), --no-pull, --force.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then :; else
  echo "ERROR: execute this script, don't source it — its exit calls would kill your shell." >&2
  return 1
fi

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

ENV_NAME="${SS2_ENV_NAME:-ss2}"
TAG="abl-$(date +%Y%m%d-%H%M%S)"
N_EPISODES=15
MAX_STEPS=250
SOUNDING_STEPS=60
ANOMALY_CLASS=alarm
SEED=20260805
SCENES=""
LIMIT=0
OUT_DIR=""
WANTED_ARMS=""
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
    --anomaly-class)   need_value $# "$1"; ANOMALY_CLASS="$2";   shift 2 ;;
    --seed)            need_value $# "$1"; SEED="$2";            shift 2 ;;
    --scenes)          need_value $# "$1"; SCENES="$2";          shift 2 ;;
    --limit)           need_value $# "$1"; LIMIT="$2";           shift 2 ;;
    --arms)            need_value $# "$1"; WANTED_ARMS="$2";     shift 2 ;;
    --out-dir)         need_value $# "$1"; OUT_DIR="$2";         shift 2 ;;
    --no-pull)         NO_PULL=1;                                shift ;;
    --force)           FORCE=1;                                  shift ;;
    -h|--help) sed -n '2,79p' "$0"; exit 0 ;;
    *) echo "FATAL: unknown argument: $1"; exit 2 ;;
  esac
done
OUT_DIR="${OUT_DIR:-runs/$TAG}"

banner() { printf '\n========== %s ==========\n' "$1"; }

# A ZERO-YIELD cell is a scene that placed no episode: a measured fact about HM3D, not a
# broken run. `runner.run` writes `summary.json` and THEN re-raises on EmptyDatasetError
# precisely so the answer lives in the artefact rather than in a log line, which is what
# lets both the run loop and the readout loop below ask the same question of the same file.
# `abl-1` is why they must: the run loop skipped `mL8ThkuaVTM` correctly and the readout
# loop then judged it anyway, `smoke` returned 2 for NOT_RUN as it should, and a sweep with
# 1410 good episodes and five complete arms reported RED over five empty directories.
is_zero_yield() {
  [ -f "$1/summary.json" ] || return 1
  python -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['n_episodes']==0 else 1)" \
    "$1/summary.json" 2>/dev/null
}

# What one scene's smoke gate means for the sweep. `gate_verdict <arm> <rc> <output>`:
#   0  the gate passed
#   1  criterion 5 ALONE, in an ablation arm: 0 episodes in this scene closed the loop.
#      A measurement, not a failure -- see the readout loop for why.
#   2  red.
#
# This is a function so the decision can be exercised rather than described: a rule this
# permissive, asserted only by reading the script's text, is a rule nothing checks.
gate_verdict() {
  local arm="$1" rc="$2" out="$3"
  [ "$rc" -eq 0 ] && return 0
  # A glob on the captured string, never `| grep -q`: under pipefail a matching grep
  # exits early and SIGPIPEs its producer, which is the footgun line 182 already names.
  # The match is ANCHORED at the end, so "criteria 5, 7" is red -- an arm that also
  # failed the audio or hermeticity criteria must not ride through on this allowance.
  if [ "$arm" != "full" ] && [[ "$out" == *"criteria 5" ]]; then
    return 1
  fi
  return 2
}

# --- ONE DIRECTORY IS ONE RUN, before anything expensive ------------------
if [ -d "$OUT_DIR" ] && [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]; then
  if [ "$FORCE" = 0 ]; then
    echo "FATAL: $OUT_DIR already exists and is not empty."
    echo "       One directory is one run. Re-using it pools two sweeps into one"
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
         --sounding-steps "$SOUNDING_STEPS" --anomaly-class "$ANOMALY_CLASS" \
         --seed "$SEED" --limit "$LIMIT" ${SCENES:+--scenes "$SCENES"} \
         ${WANTED_ARMS:+--arms "$WANTED_ARMS"} --out-dir "$OUT_DIR" ${_force_flag:+--force}
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

# --- 4. the arms ----------------------------------------------------------
# Parallel arrays rather than an associative array so the ORDER is fixed and the control
# is first. `ARM_FLAGS` is a single string per arm, word-split at the call site: every
# entry is a literal flag and value with no spaces inside either, so the split is safe
# and `shellcheck` is told so once, there.
ARM_NAMES=(full no-climb no-cue scan-only anechoic)
ARM_FLAGS=(
  ""
  "--climb-rule off"
  "--lateral-cue off"
  "--cast-policy scan_only"
  "--ir-policy anechoic"
)
ARM_WHY=(
  "the BASELINE: the complete system, and the row every other one is quoted against"
  "R1 the energy climb never steers — does live audio do any work at all"
  "R2 the interaural sign is ambiguous — loudness without binaural localization"
  "R3 every dead step turns instead of walking a leg — the pre-eps-1 control"
  "R5 flat IRs at all three render sites — does the reverb tail buy any SWS"
)

if [ -n "$WANTED_ARMS" ]; then
  # shellcheck disable=SC2206
  _wanted=($WANTED_ARMS)
  _names=(); _flags=(); _why=()
  for want in "${_wanted[@]}"; do
    _hit=0
    for i in "${!ARM_NAMES[@]}"; do
      if [ "${ARM_NAMES[$i]}" = "$want" ]; then
        _names+=("${ARM_NAMES[$i]}"); _flags+=("${ARM_FLAGS[$i]}"); _why+=("${ARM_WHY[$i]}")
        _hit=1
      fi
    done
    [ "$_hit" = 1 ] || { echo "FATAL: unknown arm '$want'. Known: ${ARM_NAMES[*]}"; exit 2; }
  done
  ARM_NAMES=("${_names[@]}"); ARM_FLAGS=("${_flags[@]}"); ARM_WHY=("${_why[@]}")
fi

N_SCENES="${#SCENE_LIST[@]}"
N_ARMS="${#ARM_NAMES[@]}"
TOTAL_EPISODES=$((N_SCENES * N_ARMS * N_EPISODES))
# 24.2 s/episode all-in is the MEASURED figure from `pilot-2`: 26516 s of wall clock over
# 1095 episodes, including scene loads, calibration and the smoke gates. It is not the
# per-step render cost and it is not an extrapolation from the anomaly-response task.
EST_SECONDS=$(awk "BEGIN{printf \"%d\", $TOTAL_EPISODES * 24.2}")
EST_HOURS=$(awk "BEGIN{printf \"%.1f\", $EST_SECONDS / 3600.0}")
PER_ARM=$((N_SCENES * N_EPISODES))
echo "  $N_SCENES scene(s): ${SCENE_LIST[*]}"
echo "  $N_ARMS arm(s): ${ARM_NAMES[*]}"
echo "  $N_EPISODES episodes per scene per arm -> $PER_ARM per arm, $TOTAL_EPISODES total"
echo "  estimated wall clock: ${EST_HOURS} h at the measured 24.2 s/episode"
echo ""
echo "  what this n can resolve, at 80% power and alpha 0.05. Read the PAIRED block:"
echo "  the arms share episodes, so the unpaired table is the wrong column for them."
python -m earshot.tools.power \
    --n-per-cell "$PER_ARM" --paired-n "$PER_ARM" --n-scenes "$N_SCENES" \
    2>/dev/null | sed 's/^/    /' \
  || echo "    (power.py did not run — the MDE is UNKNOWN for this sweep)"

mkdir -p "$OUT_DIR"
{
  echo "tag:            $TAG"
  echo "commit:         $COMMIT"
  echo "args:           $ORIGINAL_ARGS"
  echo "scenes:         ${SCENE_LIST[*]}"
  echo "arms:           ${ARM_NAMES[*]}"
  echo "n_episodes:     $N_EPISODES (per scene, per arm) -> $PER_ARM per arm"
  echo "max_steps:      $MAX_STEPS"
  echo "sounding_steps: $SOUNDING_STEPS (fixed_steps, ADR-0017)"
  echo "anomaly_class:  $ANOMALY_CLASS"
  echo "seed:           $SEED"
  echo "started:        $(date -Is)"
} > "$OUT_DIR/provenance.txt"

# --- criterion 9's evidence, armed once around the whole sweep ------------
# `pilot-1` is why this is not optional: criterion 9 was structurally NOT_RUN on all
# twelve gates, NOT_RUN is red, and the driver exited 1 over twelve smoke gates whose
# other eight criteria were green. A criterion that is red on every ordinary run is one
# the reader learns to skip, which is how a never-armed canary read as a pass.
HERM_BEFORE="$OUT_DIR/.hermeticity-before.json"
if ! python -m earshot.tools.reset_manifest --verify-absent --when before > "$HERM_BEFORE"; then
  echo "WARN: could not record the pre-run hermeticity check — criterion 9 will be NOT_RUN"
  rm -f "$HERM_BEFORE"
fi

FAILED_RUNS=0
ZERO_YIELD=""
banner "[4/5] $N_ARMS arms x $N_SCENES scenes"
for i in "${!ARM_NAMES[@]}"; do
  arm="${ARM_NAMES[$i]}"
  echo ""
  echo "  --- arm $arm: ${ARM_WHY[$i]} ---"
  for scene in "${SCENE_LIST[@]}"; do
    run_dir="$OUT_DIR/$arm/$scene"
    echo "    $arm / $scene   ($(date +%H:%M:%S))"
    # `--detector oracle` and `--localization realizable` are the defaults and are passed
    # explicitly: the ORACLE STOP deletes the stop_miss half of the failure mass, so
    # these find numbers are an upper bound and the command line should say so rather
    # than the reader having to know the default.
    # shellcheck disable=SC2086
    python -m earshot \
      --run-dir "$run_dir" \
      --scene "$scene" \
      --n-episodes "$N_EPISODES" \
      --max-steps "$MAX_STEPS" \
      --seed "$SEED" \
      --localization realizable \
      --detector oracle \
      --anomaly-class "$ANOMALY_CLASS" \
      --sounding-policy fixed_steps \
      --sounding-steps "$SOUNDING_STEPS" \
      ${ARM_FLAGS[$i]} \
      > "$OUT_DIR/$arm-$scene.log" 2>&1
    status=$?
    if [ "$status" -ne 0 ]; then
      # Anything that is not zero yield is red.
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
    # stderr is NOT suppressed: `--verify-absent` prints "STILL PRESENT: <paths>" there
    # and that list is the entire diagnostic.
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
banner "[5/5] the ablation table"
GATE_FAILED=0
READ_FAILED=0
# Criterion 5 is a RATE criterion: green iff at least one episode reached CHECK and
# RESUME. `tally` still fails it at 0/n, and that floor exists to catch ADR-0014's
# VACUOUS ARM -- a loop that never once ran is a loop that is not wired.
#
# `abl-1` showed the floor is at the wrong GRAIN for an ablation. Ten scene/arm cells
# came back `RED -- criteria 5` and every one of them was in an arm built to cripple the
# controller: `scan-only` in seven scenes, one each in `no-climb`, `no-cue` and
# `anechoic`, and NONE in `full`. Identical episodes, identical harness, and the baseline
# completed the loop in all nineteen scenes -- so a crippled arm failing to close the loop
# in one room is the ablation working, not the loop being unwired, and it is a
# MEASUREMENT. Asking per scene turns the strongest result in the table into a red banner.
#
# So the floor moves to the arm: an ablation arm is vacuous only if it completes the loop
# in NO scene at all, which is still exactly the condition ADR-0014 wants caught. `full`
# is the baseline of record and keeps the per-scene bar, because there a zero IS a bug.
VACUOUS_CELLS=""
VACUOUS_BY_ARM=""
for arm in "${ARM_NAMES[@]}"; do
  arm_green=0
  arm_vacuous=0
  for scene in "${SCENE_LIST[@]}"; do
    [ -d "$OUT_DIR/$arm/$scene" ] || continue
    if is_zero_yield "$OUT_DIR/$arm/$scene"; then
      echo ""
      echo "  --- smoke gate: $arm / $scene --- SKIPPED, zero yield (see the list below)"
      continue
    fi
    echo ""
    echo "  --- smoke gate: $arm / $scene ---"
    gate_out="$(python -m earshot.task.smoke --run-dir "$OUT_DIR/$arm/$scene" 2>&1)"
    gate_rc=$?
    echo "$gate_out"
    gate_verdict "$arm" "$gate_rc" "$gate_out"
    case $? in
      0) arm_green=$((arm_green + 1)) ;;
      1) echo "      criterion 5 alone, and this is an ablation arm: 0 of the episodes in"
         echo "      this scene closed the loop. Recorded as a measurement, not a failure."
         arm_vacuous=$((arm_vacuous + 1))
         VACUOUS_CELLS="$VACUOUS_CELLS $arm/$scene" ;;
      *) GATE_FAILED=1 ;;
    esac
  done
  VACUOUS_BY_ARM="$VACUOUS_BY_ARM $arm=$arm_vacuous"
  if [ "$arm_green" -eq 0 ]; then
    echo ""
    echo "  RED: arm $arm closed the loop in NO scene. That is the vacuous arm ADR-0014"
    echo "       exists to catch, and no per-scene allowance covers it."
    GATE_FAILED=1
  fi
done

echo ""
echo "  --- every arm, side by side ---"
# The reader is `tools/window_report.py` and NOT a heredoc: `pilot-1` ran 120 episodes,
# wrote all of them, and printed "NO EPISODES ON DISK" three times because its reader was
# forty lines of Python inside a bash string and no test in the tree could see it. The
# layout here — <tag>/<arm>/<scene>/ — is the one that module already reads.
python -m earshot.tools.window_report "$OUT_DIR" --arms "${ARM_NAMES[*]}" || READ_FAILED=1

echo ""
echo "  --- each ablation arm against the baseline, PAIRED BY EPISODE ---"
echo "  Exact McNemar over the pairs. Report the scene-level sign test with it:"
echo "  a mechanism that was green and exact at the episode level has already come"
echo "  back null at the scene level in this repo, and neither test breaks the other."
BASELINE_DIR="$OUT_DIR/full"
if [ -d "$BASELINE_DIR" ]; then
  for arm in "${ARM_NAMES[@]}"; do
    [ "$arm" = "full" ] && continue
    [ -d "$OUT_DIR/$arm" ] || continue
    echo ""
    echo "  === full -> $arm ==="
    python -m earshot.tools.episode_diff "$BASELINE_DIR" "$OUT_DIR/$arm" 2>&1 | tail -n 30
  done
else
  echo "  SKIPPED: no baseline arm on disk. Every row below would be quoted against"
  echo "           nothing, which is not a smaller result — it is no result."
  READ_FAILED=1
fi

echo ""
echo "  logs:       $OUT_DIR/<arm>-<scene>.log"
echo "  provenance: $OUT_DIR/provenance.txt"
echo "  audits:     $OUT_DIR/<arm>/<scene>/"
if [ -n "$ZERO_YIELD" ]; then
  echo ""
  echo "  zero-yield scene/arm cells (measured, not failures):$ZERO_YIELD"
fi
if [ -n "$VACUOUS_CELLS" ]; then
  echo ""
  echo "  scenes where an ablation arm closed the loop ZERO times, by arm:$VACUOUS_BY_ARM"
  echo "  This is a second ordering of the same table and it belongs beside the SR column:"
  echo "  the baseline is absent from it by construction, and an arm that appears often is"
  echo "  an arm that stops working in whole rooms rather than losing episodes evenly."
  echo " $VACUOUS_CELLS"
fi

if [ "$FAILED_RUNS" -ne 0 ]; then
  echo ""
  echo "RED: $FAILED_RUNS run(s) failed for a reason other than zero yield. A sweep"
  echo "     missing runs is NOT_RUN for those cells, and NOT_RUN is red. The aggregates"
  echo "     above are over what survived."
  exit 1
fi
if [ "$READ_FAILED" -ne 0 ]; then
  echo ""
  echo "RED: the readout found no episode under any arm, or found no baseline to quote"
  echo "     against. Runs that produced nothing and a reader that cannot find what they"
  echo "     produced look identical from here: check $OUT_DIR/<arm>/<scene>/episodes/."
  exit 1
fi
if [ "$GATE_FAILED" -ne 0 ]; then
  echo ""
  echo "RED: at least one smoke gate did not pass. The measurements above stand as"
  echo "     measurements; what fails is the claim that the loop behaved."
  exit 1
fi
echo ""
echo "GREEN — every arm ran and every gate passed. The 'full' row is the paper's HM3D"
echo "        baseline; the other four are the ablation table. This sweep contains NO"
echo "        memory arm and is not ADR-0018's generalization matrix."
