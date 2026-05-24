#!/bin/bash
# RACE post-restart bootstrap.
#
# After a RACE pod restart, the `conda` CLI is gone from PATH but
# ~/miniconda3 survives. This script restores the conda command,
# activates the ltm-embodied env, exports the REMEMBR env vars (model
# names + Run-3 STOP stopgaps + Run-4 frontier-injection defaults),
# repairs the HM3D symlink if rsync left it dangling, and prints a
# status block.
#
# SOURCE it, don't execute it — `conda activate` only sticks in the
# current shell:
#
#     source scripts/race-setup.sh
#
# Re-run on every restart. Idempotent.

# Refuse to run when executed directly — conda activate would not
# propagate to the parent shell.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "ERROR: source this script, don't execute it."
  echo "  source scripts/race-setup.sh"
  exit 1
fi

MINICONDA="${HOME}/miniconda3"
ENV_NAME="${LTM_ENV_NAME:-ltm-embodied}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- 1. restore the conda command from the surviving miniconda3 dir -------
if [ ! -x "$MINICONDA/bin/conda" ]; then
  echo "ERROR: $MINICONDA/bin/conda missing. Re-install Miniconda before re-sourcing."
  return 1
fi

echo "[1/5] Loading conda hook from $MINICONDA"
eval "$("$MINICONDA/bin/conda" shell.bash hook)"

# --- 2. activate the env --------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[2/5] Activating env: $ENV_NAME"
  conda activate "$ENV_NAME"
else
  echo "ERROR: env '$ENV_NAME' missing. First-time create:"
  echo "  conda env create -f $REPO_ROOT/embodied_memory/environment.yml"
  return 1
fi

# --- 3. export REMEMBR env vars ------------------------------------------
# `:-` keeps any value the caller set in the parent shell.
echo "[3/5] Exporting REMEMBR env vars"
export REMEMBR_CAPTIONER_MODEL="${REMEMBR_CAPTIONER_MODEL:-Qwen/Qwen2-VL-2B-Instruct}"
export REMEMBR_PLANNER_MODEL="${REMEMBR_PLANNER_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
# Run-3 stopgaps: text-vs-text STOP path is too permissive at 0.25.
export REMEMBR_STOP_COS="${REMEMBR_STOP_COS:-0.40}"
export REMEMBR_STOP_MIN_STEP="${REMEMBR_STOP_MIN_STEP:-20}"
# Run-4 defaults — explicit so they're visible in `env`.
export REMEMBR_FRONTIER_INJECT="${REMEMBR_FRONTIER_INJECT:-3}"
export REMEMBR_MIN_WAYPOINT_DIST="${REMEMBR_MIN_WAYPOINT_DIST:-0.5}"

# --- 4. repair HM3D symlink if dangling ----------------------------------
# rsync from laptop copies an absolute symlink that doesn't resolve on
# RACE; the Phase-3 runbook documents re-creating it post-rsync.
HM3D_SCENE_DIR="$REPO_ROOT/data/hm3d/scene_datasets"
HM3D_LINK="$HM3D_SCENE_DIR/hm3d"
HM3D_TARGET="../versioned_data/hm3d-0.2/hm3d"
if [ -d "$HM3D_SCENE_DIR" ]; then
  if [ -L "$HM3D_LINK" ] && [ ! -e "$HM3D_LINK" ]; then
    echo "[4/5] HM3D symlink dangling — re-pointing to $HM3D_TARGET"
    (cd "$HM3D_SCENE_DIR" && ln -sfn "$HM3D_TARGET" hm3d)
  elif [ -e "$HM3D_LINK" ]; then
    echo "[4/5] HM3D symlink OK ($(readlink "$HM3D_LINK"))"
  else
    echo "[4/5] HM3D not yet rsynced — skip (run rsync from laptop first)"
  fi
else
  echo "[4/5] $HM3D_SCENE_DIR missing — skip"
fi

# --- 5. status block -----------------------------------------------------
echo "[5/5] Status"
echo "  python:   $(command -v python)"
echo "  env:      ${CONDA_DEFAULT_ENV:-?}"
echo "  repo:     $REPO_ROOT"
echo "  branch:   $(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
echo "  HEAD:     $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo
echo "  REMEMBR_CAPTIONER_MODEL=$REMEMBR_CAPTIONER_MODEL"
echo "  REMEMBR_PLANNER_MODEL=$REMEMBR_PLANNER_MODEL"
echo "  REMEMBR_STOP_COS=$REMEMBR_STOP_COS  REMEMBR_STOP_MIN_STEP=$REMEMBR_STOP_MIN_STEP"
echo "  REMEMBR_FRONTIER_INJECT=$REMEMBR_FRONTIER_INJECT  REMEMBR_MIN_WAYPOINT_DIST=$REMEMBR_MIN_WAYPOINT_DIST"
echo
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.free,memory.total \
             --format=csv,noheader | sed 's/^/  GPU: /'
else
  echo "  (nvidia-smi not on PATH)"
fi

echo
echo "Ready."
echo
echo "Run-5 oracle smoke (model-free env check — run this first):"
echo "  for sc in TEEsavR23oF wcojb4TFT35; do"
echo "    python -m embodied_memory.run_hm3d_pol --mode live --backbone oracle \\"
echo "        --setting 1 --scene \$sc --n-episodes 2 --target any --no-strict-pass \\"
echo "        --out-dir runs/oracle-smoke-\$sc"
echo "    python embodied_memory/scripts/verify_smoke_gate.py runs/oracle-smoke-\$sc"
echo "  done"
echo
echo "Run-5 densified-grid escape check (full stack, after oracle passes):"
echo "  python -m embodied_memory.run_hm3d_pol --mode live --backbone remembr \\"
echo "      --setting 3 --scene wcojb4TFT35 --n-episodes 2 --target any \\"
echo "      --out-dir runs/remembr-dense-smoke"
echo "  python embodied_memory/scripts/verify_smoke_gate.py runs/remembr-dense-smoke"
