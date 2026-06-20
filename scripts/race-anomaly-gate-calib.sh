#!/bin/bash
# scripts/race-anomaly-gate-calib.sh — the $0 go/no-go gate for Step 1 (the
# open-set CLAP normal-vs-anomaly gate, audio.is_anomaly). NO LLM spend: it only
# fetches a handful of ESC-50 clips and runs CLAP zero-shot cosines.
#
# Pipeline:
#   [1] git pull   [2] source race-setup (ltm-embodied)   [3] pre-verify tests
#   [4] FETCH ESC-50 anomaly + benign clips (fetch_anomaly_clips --include-benign)
#   [5] DIAGNOSE: diagnose_normal_anomaly_calib → per-clip margins + GO/STOP verdict
#
# Decision (printed as VERDICT): GO ⇒ build the runtime gate with RECOMMEND_DELTA;
# BORDERLINE ⇒ add clips / calibrate; STOP ⇒ CLAP can't separate these on our
# clips → Step-1 is a category-discrimination ceiling (honest $0 negative).
#
#   cd ~/ltm && git checkout lifelong-revisit-eval && git pull --ff-only \
#       && nrun bash scripts/race-anomaly-gate-calib.sh
#
# (Brand-new driver: the first run needs the manual `git pull` above because the
# driver self-pulls via its own open fd — see the RACE-testing memory.)
# EXECUTE (do NOT source). CLAP runs on CPU by default (--device cuda to use GPU).

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT" || exit 1
LTM_ENV="ltm-embodied"
DEVICE="cpu"; CLIP_INDEX="0"
while [ $# -gt 0 ]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --index)  CLIP_INDEX="$2"; shift 2 ;;   # which ESC-50 clip per class (0..39)
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

banner() { printf '\n========== %s ==========\n' "$1"; }

banner "[1/5] git pull --ff-only"
git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }

banner "[2/5] conda setup (source scripts/race-setup.sh → $LTM_ENV)"
set +u; source scripts/race-setup.sh || { echo "FATAL: race-setup.sh failed"; exit 1; }; set -u
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

banner "[3/5] pre-verify (free; abort before fetch)"
for t in test_audio test_audio_task test_diagnose_normal_anomaly_calib test_fetch_anomaly_clips; do
  python embodied_memory/scripts/$t.py \
    || { echo "FATAL: $t failed."; exit 1; }
done

banner "[4/5] fetch ESC-50 anomaly + benign clips (index=$CLIP_INDEX)"
python embodied_memory/scripts/fetch_anomaly_clips.py --include-benign --index "$CLIP_INDEX" \
  || { echo "FATAL: clip fetch failed (needs internet)."; exit 1; }

banner "[5/5] diagnose normal-vs-anomaly separation (device=$DEVICE)"
python embodied_memory/scripts/diagnose_normal_anomaly_calib.py --device "$DEVICE"
RC=$?
echo
echo "[calib] exit=$RC  (0 = GO/BORDERLINE, 1 = STOP, 2 = clips missing)"
exit $RC
