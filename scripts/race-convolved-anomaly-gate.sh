#!/bin/bash
# scripts/race-convolved-anomaly-gate.sh — the $0 Gate-0b: can the open-set CLAP
# gate (audio.is_anomaly) separate anomaly-vs-background on RIR-CONVOLVED (+
# mixed) audio? 0c proved the CLEAN-calibrated gate REJECTS the convolved alarm
# (onset only fired with --no-anomaly-gate). This recalibrates on the EXACT live
# signal (render_at_pose -> diotic bed mix -> is_anomaly) BEFORE any dataset/
# render/matrix build. NO LLM spend, no sim — reuses existing RIR grids + CLAP.
#
# $0 of *matrix*, NOT $0 of compute/network: loads CLAP + needs ESC-50 clips.
#
# Pipeline:
#   [1] git pull  [2] race-setup  [3] pre-verify  [4] fetch anomaly+benign clips
#   [5] locate existing RIR grids  [6] DIAGNOSE (convolved anomaly-vs-bg gate)
#
# Decision (printed as GATE_RESULT): GO => proceed to the mixture render + gate-ON
# build with RECOMMEND_DELTA/TAU/BG_GAIN; BORDERLINE => cheap ladder; STOP => the
# gate cannot survive room-IR convolution -> pivot (audio-prototype / temporally-
# separated / honest negative). See docs plan Phase 0.
#
#   cd ~/ltm && git checkout lifelong-revisit-eval && git pull --ff-only \
#       && nrun bash scripts/race-convolved-anomaly-gate.sh --device cuda
#
# (Brand-new driver: the first run needs the manual `git pull` above because the
# driver self-pulls via its own open fd — see the RACE-testing memory.)
# EXECUTE (do NOT source).

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT" || exit 1
LTM_ENV="ltm-embodied"
DEVICE="cpu"; CLIP_INDEX="0"
GRID_GLOB="runs/audiogoal/*_rir_grid.npz"
ONSET_RMS="0.05"; BG_GAINS="0.0 0.3 0.5 0.7 1.0"; MAX_CELLS="12"; MAX_GRIDS="6"
while [ $# -gt 0 ]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --index)  CLIP_INDEX="$2"; shift 2 ;;   # which ESC-50 clip per class (0..39)
    --rir-grid) GRID_GLOB="$2"; shift 2 ;;  # glob of RIR grids to calibrate over
    --onset-rms) ONSET_RMS="$2"; shift 2 ;;
    --bg-gains) BG_GAINS="$2"; shift 2 ;;
    --max-cells) MAX_CELLS="$2"; shift 2 ;;
    --max-grids) MAX_GRIDS="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

banner() { printf '\n========== %s ==========\n' "$1"; }

banner "[1/6] git pull --ff-only"
git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }

banner "[2/6] conda setup (source scripts/race-setup.sh -> $LTM_ENV)"
set +u; source scripts/race-setup.sh || { echo "FATAL: race-setup.sh failed"; exit 1; }; set -u
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

banner "[3/6] pre-verify (free; abort before fetch)"
for t in test_audio test_audio_task test_diagnose_normal_anomaly_calib \
         test_fetch_anomaly_clips test_diagnose_convolved_anomaly_calib; do
  python embodied_memory/scripts/$t.py \
    || { echo "FATAL: $t failed."; exit 1; }
done

banner "[4/6] fetch ESC-50 anomaly + benign clips (index=$CLIP_INDEX)"
python embodied_memory/scripts/fetch_anomaly_clips.py --include-benign --index "$CLIP_INDEX" \
  || { echo "FATAL: clip fetch failed (needs internet)."; exit 1; }

banner "[5/6] locate existing RIR grids ($GRID_GLOB)"
NGRID=$(ls $GRID_GLOB 2>/dev/null | wc -l | tr -d ' ')
echo "  found $NGRID grid(s)"
if [ "$NGRID" = "0" ]; then
  echo "FATAL: no RIR grids at '$GRID_GLOB'. Render one first, e.g. a prior"
  echo "  bash scripts/race-audiogoal.sh ... leaves runs/audiogoal/<scene>_<tag>_rir_grid.npz"
  exit 1
fi

banner "[6/6] diagnose convolved anomaly-vs-background gate (device=$DEVICE)"
# Two cell regimes: 'audible' (far-start, the real-eval regime) and 'loud' (the
# source-view_point start = loudest cells, where mix1 false-fired on the bed).
# The loud regime is the decisive one for the source==goal smoke.
RC=0
for REGIME in audible loud; do
  banner "  cell-regime=$REGIME"
  python embodied_memory/scripts/diagnose_convolved_anomaly_calib.py \
      --rir-grid "$GRID_GLOB" --device "$DEVICE" --onset-rms "$ONSET_RMS" \
      --bg-gains "$BG_GAINS" --max-cells "$MAX_CELLS" --max-grids "$MAX_GRIDS" \
      --cell-regime "$REGIME" || RC=$?
done
echo
echo "[gate0b] exit=$RC  — grep GATE_RESULT above (GO -> build Phase 1; STOP -> pivot)"
exit $RC
