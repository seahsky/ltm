#!/bin/bash
# earshot/tools/savnce_eval.sh — one SAVN-CE reproduced-reference run (ADR-0015).
#
#   bash earshot/tools/savnce_eval.sh --tag smoke1 --episodes 20
#   nrun bash earshot/tools/savnce_eval.sh --tag test1 --episodes 1000     # the real arm
#
# ONE DIRECTORY IS ONE RUN, so the tag must be fresh. Same rule as yield_sweep.sh, same
# reason: a run directory that two runs wrote to cannot be attributed to either.
#
# The deliverable is `runs/savnce-<tag>/` holding probe.json, gate.json, stdout.log,
# pip-freeze.txt, and their per-episode stats. The gate's tally is printed at the end and
# its exit code is this script's exit code.
#
# --- the released-checkpoint invocation, which their scripts do not show -------------
#
# `test.sh` uses `--eval-best` against a model dir produced by THEIR training, which we
# do not have. Running a RELEASED checkpoint needs `EVAL_CKPT_PATH_DIR` plus three
# `pretrained` flags off, and that combination appears only in the demo-video command in
# their README. Get it wrong and the eval runs happily on partly random weights and
# reports a number, which is the exact failure class this repo keeps paying for. Gate
# criterion 3 is the assertion that it did not happen.
#
# `EVAL.USE_CKPT_CONFIG` defaults to True, so the eval config comes from the checkpoint.
# Our overrides still win, but only because `_setup_eval_config` applies `eval_cmd_opts`
# last (`savnce_baselines/common/base_trainer.py:128`). Worth knowing before trusting a
# flag on this codebase.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

ENV_NAME="${SAVNCE_ENV_NAME:-savnce}"
SAVNCE_DIR="$REPO_ROOT/earshot/reference/savnce"
SAVNCE_DATA_ROOT="${SAVNCE_DATA_ROOT:-${HOME}/savnce-data}"

TAG=""
EPISODES=20
SPLIT="test"
SEED=0
# NUM_PROCESSES is 10 in their config, sized for a 128-thread box. This box has 4 cores,
# and audio propagation is CPU-side, so 10 workers on 4 cores is contention, not
# parallelism. Overridable; the smoke's measured throughput is what should set it.
PROCS=2
CKPT=""
FORCED="none"
CONFIG="savnce_baselines/magnet/config/mp3d/rgbd_ddppo_clean.yaml"

usage() {
  sed -n '2,12p' "$0"
  echo
  echo "  --tag NAME              required, must be fresh"
  echo "  --episodes N            default $EPISODES"
  echo "  --split NAME            default $SPLIT"
  echo "  --seed N                default $SEED"
  echo "  --procs N               default $PROCS (their config says 10; this box has 4 cores)"
  echo "  --ckpt PATH             default <data>/pretrained_ckpts/magnet/magnet_clean.pth"
  echo "  --forced-failure MODE   none | wrong-ckpt | empty-episodes"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --procs) PROCS="$2"; shift 2 ;;
    --ckpt) CKPT="$2"; shift 2 ;;
    --forced-failure) FORCED="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1"; usage; exit 2 ;;
  esac
done

[ -n "$TAG" ] || { echo "FATAL: --tag is required"; usage; exit 2; }
RUN_DIR="$REPO_ROOT/runs/savnce-$TAG"
[ -e "$RUN_DIR" ] && { echo "FATAL: $RUN_DIR exists. One directory is one run — use a fresh tag."; exit 2; }

[ -n "$CKPT" ] || CKPT="$SAVNCE_DATA_ROOT/pretrained_ckpts/magnet/magnet_clean.pth"

# ----------------------------------------------------------------------
# the forced-failure arms — ADR-0014 says a detector ships both
# ----------------------------------------------------------------------
EXTRA_OPTS=()
case "$FORCED" in
  none) ;;
  wrong-ckpt)
    # A VALID checkpoint that lacks magnet's submodules is the honest wrong-ckpt, so
    # prefer an av_nav one. If none is staged, a truncated file still fires criterion 3,
    # by a different route, and the log says which route was taken.
    ALT="$(find "$SAVNCE_DATA_ROOT/pretrained_ckpts/av_nav" -name '*.pth' 2>/dev/null | head -1)"
    if [ -n "$ALT" ]; then
      CKPT="$ALT"
      echo "FORCED FAILURE: using av_nav checkpoint $CKPT (valid, wrong submodules)"
    else
      mkdir -p "$RUN_DIR"
      CKPT="$RUN_DIR/not-a-checkpoint.pth"
      : > "$CKPT"
      echo "FORCED FAILURE: no av_nav checkpoint staged, using an empty file $CKPT (load failure)"
    fi
    ;;
  empty-episodes)
    EXTRA_OPTS+=(TASK_CONFIG.DATASET.CONTENT_SCENES '["__no_such_scene__"]')
    echo "FORCED FAILURE: restricting CONTENT_SCENES to a scene that does not exist"
    ;;
  *) echo "FATAL: unknown --forced-failure $FORCED"; exit 2 ;;
esac

MODEL_DIR="$RUN_DIR/model"
STATS_REL="model/tb/${SPLIT}_stats_${SEED}.json"
mkdir -p "$MODEL_DIR"

# ----------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1 && [ -x "${HOME}/miniconda3/bin/conda" ]; then
  eval "$("${HOME}/miniconda3/bin/conda" shell.bash hook)"
fi
eval "$(conda shell.bash hook)" && conda activate "$ENV_NAME" || {
  echo "FATAL: cannot activate '$ENV_NAME' — run: bash earshot/tools/savnce_bootstrap.sh"
  exit 1
}

echo "run dir  : $RUN_DIR"
echo "config   : $CONFIG"
echo "checkpoint: $CKPT"
echo "split=$SPLIT episodes=$EPISODES seed=$SEED procs=$PROCS forced=$FORCED"

# ----------------------------------------------------------------------
# probe FIRST, unconditionally. A run that dies mid-eval must still leave a probe the
# gate can judge red, rather than a missing file that reads as "not run yet".
# ----------------------------------------------------------------------
PYTHONPATH="$REPO_ROOT" python -m earshot.tools.savnce_probe pre \
  --run-dir "$RUN_DIR" --data-root "$SAVNCE_DATA_ROOT" --ckpt "$CKPT" \
  --split "$SPLIT" --episodes "$EPISODES" --stats-file "$STATS_REL"

python -m pip freeze > "$RUN_DIR/pip-freeze.txt" 2>/dev/null

# The Drive folder can change under us and leaves no version in the file. A CHANGED or
# MISSING recorded artefact stops the run; a merely unrecorded one is printed and
# continues, so a first run is never blocked by an empty manifest.
PYTHONPATH="$REPO_ROOT" python -m earshot.tools.savnce_artifacts verify \
  --data-root "$SAVNCE_DATA_ROOT" | tee "$RUN_DIR/artifacts.log"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "FATAL: a pinned artefact changed or went missing. The number this run would"
  echo "produce is not the number the manifest describes. Resolve before re-running."
  exit 1
fi

# ----------------------------------------------------------------------
# the eval. cwd is the submodule because every path in their configs is relative to it.
# ----------------------------------------------------------------------
export HABITAT_SIM_LOG='quiet'
START=$SECONDS
(
  cd "$SAVNCE_DIR" || exit 1
  PYTHONPATH="$SAVNCE_DIR" python savnce_baselines/magnet/run.py \
    --run-type test \
    --exp-config "$CONFIG" \
    --model-dir "$MODEL_DIR" \
    EVAL_CKPT_PATH_DIR "$CKPT" \
    EVAL.SPLIT "$SPLIT" \
    TEST_EPISODE_COUNT "$EPISODES" \
    NUM_PROCESSES "$PROCS" \
    SEED "$SEED" \
    RL.DDPPO.pretrained False \
    RL.PPO.GOAL_DESCRIPTOR.use_pretrained False \
    RL.PPO.SCENE_MEMORY_TRANSFORMER.use_pretrained False \
    "${EXTRA_OPTS[@]}"
) 2>&1 | tee "$RUN_DIR/stdout.log"
EVAL_STATUS=${PIPESTATUS[0]}
WALL=$((SECONDS - START))
echo "eval exited $EVAL_STATUS after ${WALL}s"

PYTHONPATH="$REPO_ROOT" python -m earshot.tools.savnce_probe post \
  --run-dir "$RUN_DIR" --wall-clock-s "$WALL"

# ----------------------------------------------------------------------
# the verdict
# ----------------------------------------------------------------------
PYTHONPATH="$REPO_ROOT" python -m earshot.tools.savnce_gate --run-dir "$RUN_DIR"
GATE_STATUS=$?

if [ "$FORCED" != "none" ]; then
  # The forced arms INVERT the meaning of the gate: red is the pass.
  if [ "$GATE_STATUS" -eq 0 ]; then
    echo "FORCED-FAILURE ARM FAILED: the gate went green on a run that was sabotaged."
    exit 1
  fi
  echo "FORCED-FAILURE ARM PASSED: the gate went red, as it must."
  exit 0
fi
exit "$GATE_STATUS"
