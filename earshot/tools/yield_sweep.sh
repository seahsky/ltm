#!/bin/bash
# earshot/tools/yield_sweep.sh — how much of HM3D can pose the task at all.
#
#   source earshot/tools/notify/notify-run.sh
#   nrun bash earshot/tools/yield_sweep.sh --tag yield-1
#
# Runs the S1 arm across every scene whose mesh is on this box and totals the builder's
# attrition. The number it produces is a DENOMINATOR: §2.1 refuses an episode whose scene
# cannot place an anomaly source far enough from every primary goal, on the same floor as
# both the anchor and the agent's start, at a real view point. Every refusal is correct,
# and the refusal RATE bounds every `n` an experiment matrix can quote. The smoke ran one
# scene and skipped 1 of 2 — a sample of one, consistent with anything.
#
# THIS IS NOT AN ABLATION. There is no memory in the tree (ADR-0008 deferred it), so every
# arm here is S1 and there is no S3 to difference against. It also runs the ORACLE STOP by
# §8's default, so its find numbers are an upper bound with roughly half the failure mass
# deleted — the run says so in its own notes. What it measures is yield, cost per step and
# the funnel at n >> 1, none of which need memory or a detector.
#
# CONTINUE-ON-FAILURE, deliberately: one scene that cannot load must not cost the other
# nineteen. Failures are listed at the end and the aggregate says how many scenes it is
# over, so a partial sweep cannot read as a complete one.
#
# Flags: --tag T, --n-episodes N (default 20), --max-steps M (default 250),
#        --scenes "a b c" (default: every scene with a mesh), --category C, --limit N,
#        --out-dir DIR (default runs/<tag>), --no-pull.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then :; else
  echo "ERROR: execute this script, don't source it — its exit calls would kill your shell." >&2
  return 1
fi

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

ENV_NAME="${SS2_ENV_NAME:-ss2}"
TAG="yield-$(date +%Y%m%d-%H%M%S)"
N_EPISODES=20
MAX_STEPS=250
SCENES=""
CATEGORY=""
LIMIT=0
OUT_DIR=""
NO_PULL=0

need_value() { [ "$1" -ge 2 ] || { echo "FATAL: $2 needs a value"; exit 2; }; }
while [ $# -gt 0 ]; do
  case "$1" in
    --tag)         need_value $# "$1"; TAG="$2";        shift 2 ;;
    --n-episodes)  need_value $# "$1"; N_EPISODES="$2"; shift 2 ;;
    --max-steps)   need_value $# "$1"; MAX_STEPS="$2";  shift 2 ;;
    --scenes)      need_value $# "$1"; SCENES="$2";     shift 2 ;;
    --category)    need_value $# "$1"; CATEGORY="$2";   shift 2 ;;
    --limit)       need_value $# "$1"; LIMIT="$2";      shift 2 ;;
    --out-dir)     need_value $# "$1"; OUT_DIR="$2";    shift 2 ;;
    --no-pull)     NO_PULL=1;                           shift ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "FATAL: unknown argument: $1"; exit 2 ;;
  esac
done
OUT_DIR="${OUT_DIR:-runs/$TAG}"

banner() { printf '\n========== %s ==========\n' "$1"; }

# --- 1. self-update by re-exec (bash runs the body it loaded, not the file) -
if [ "$NO_PULL" = 0 ]; then
  banner "[1/4] git pull --ff-only"
  _self_before="$(md5sum "$0" 2>/dev/null | awk '{print $1}')"
  git pull --ff-only || echo "WARN: git pull failed — running the checked-out copy"
  _self_after="$(md5sum "$0" 2>/dev/null | awk '{print $1}')"
  if [ -n "$_self_before" ] && [ "$_self_before" != "$_self_after" ] && [ -z "${_REEXEC:-}" ]; then
    echo "  this script changed in the pull — re-execing the new body"
    export _REEXEC=1
    exec bash "$0" --tag "$TAG" --n-episodes "$N_EPISODES" --max-steps "$MAX_STEPS" \
         ${SCENES:+--scenes "$SCENES"} ${CATEGORY:+--category "$CATEGORY"} \
         --limit "$LIMIT" --out-dir "$OUT_DIR"
  fi
else
  banner "[1/4] git pull SKIPPED (--no-pull)"
fi
echo "  commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

# --- 2. the env -----------------------------------------------------------
banner "[2/4] conda env: $ENV_NAME"
MINICONDA="${HOME}/miniconda3"
[ -x "$MINICONDA/bin/conda" ] || { echo "FATAL: $MINICONDA/bin/conda missing"; exit 1; }
# A DIRECTORY check, never `conda env list | grep -q`: under pipefail a matching grep
# exits early, SIGPIPEs conda, and turns found-it into a pipeline failure.
[ -d "$MINICONDA/envs/$ENV_NAME" ] || { echo "FATAL: env '$ENV_NAME' missing — run bootstrap_ss2.sh"; exit 1; }
set +u   # conda's compiler hooks dereference unset CONDA_BACKUP_* vars
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

# --- 3. discover the scenes, then run each one ----------------------------
banner "[3/4] scenes"
if [ -z "$SCENES" ]; then
  # Only scenes whose MESH is on this box: `available_scenes` lists content files, and a
  # content file without its .glb is a scene that will fail at load, not a scene with a
  # zero yield. Counting those as refusals would understate the denominator.
  SCENES="$(python - <<'PY'
# The same check task/runner._pick_scene makes, over every label instead of stopping at
# the first hit: load the content file, then test that its resolved mesh is on this box.
# NOTE: no apostrophes or backticks in this heredoc. It sits inside a $( ) command
# substitution, where bash scans for quote pairs even in a quoted heredoc, and a single
# apostrophe in a comment is enough to break the whole script with an EOF error.
# Reusing the loader rather than re-deriving the path layout: the last time this repo
# had two copies of a path rule, one of them quietly stopped being true.
import os
from earshot.task.episodes import available_scenes, find_scenes_dir, find_split_dir, load_scene

split_dir = find_split_dir("val", ".")
scenes_dir = find_scenes_dir(".")
have = []
for label in available_scenes(split_dir):
    try:
        dataset = load_scene(split_dir, label, scenes_dir=scenes_dir)
    except Exception:
        continue          # unreadable or empty content: not a scene with a zero yield
    if os.path.exists(dataset.scene_path):
        have.append(label)
print(" ".join(have))
PY
)" || { echo "FATAL: scene discovery failed"; exit 1; }
fi
set -- $SCENES
[ "$#" -gt 0 ] || { echo "FATAL: no scenes with a mesh on this box"; exit 1; }
if [ "$LIMIT" -gt 0 ] && [ "$#" -gt "$LIMIT" ]; then
  SCENES="$(echo "$SCENES" | tr ' ' '\n' | head -n "$LIMIT" | tr '\n' ' ')"
  set -- $SCENES
  echo "  --limit $LIMIT applied"
fi
echo "  $# scene(s): $*"
mkdir -p "$OUT_DIR"

FAILED=""
N_OK=0
for scene in "$@"; do
  banner "[3/4] $scene"
  python -m earshot --run-dir "$OUT_DIR/$scene" --scene "$scene" \
      --n-episodes "$N_EPISODES" --max-steps "$MAX_STEPS" \
      ${CATEGORY:+--category "$CATEGORY"}
  ec=$?
  if [ "$ec" -ne 0 ]; then
    echo "  WARN: $scene exited $ec — continuing"
    FAILED="$FAILED $scene"
  else
    N_OK=$((N_OK + 1))
  fi
done

# --- 4. the number --------------------------------------------------------
banner "[4/4] yield"
python -m earshot.tools.yield_report "$OUT_DIR"
echo
echo "  $N_OK of $# scene(s) completed."
if [ -n "$FAILED" ]; then
  echo "  FAILED (excluded from the totals above, so a partial sweep cannot read as a"
  echo "  complete one):$FAILED"
fi
echo "  records: $OUT_DIR/<scene>/summary.json"
exit 0
