#!/bin/bash
# scripts/race-room-clip.sh — one-shot RACE driver for the Stage-5 CLIP zero-shot
# ROOM classifier (the coarse-affordance head's dense room signal).
#
# WHY (coarse-1/coarse-2 closed the caption-only path): the coarse head grounds a
# position-free category->preferred_room prior (chair->living_room, bed->bedroom,
# toilet->bathroom) to the CURRENT scene to steer exploration. It NEVER fired
# (n_coarse_candidates ~= 0) because the only room signal was resolve_room(caption)
# keyword matching, which names the affordant room too rarely. HM3D has no room-type
# GT regions. Stage 5 adds a CLIP zero-shot room classifier (cosine of the keyframe
# CLIP IMAGE embedding vs CLIP-text("a photo of a {room}")) as a DENSE room signal.
#
# The adversarial review made ONE thing a hard gate: the abstain thresholds
# (min_cos/margin) must be SET FROM REAL CLIP cosines, not guessed — at the real
# ViT-B/32 image-text scale (~0.18-0.30) a wrong floor makes the gate a no-op
# (over-fire) or shut (inert). So this driver MEASURES first, then runs:
#
#   PHASE A (fast, sim+CLIP only, no 7B): diagnose_room_clip_cosines.py dumps the
#     top_cos / margin distributions + room histogram and prints a data-driven
#     RECOMMEND min_cos=.. margin=.. line. We export those as the gate thresholds.
#   PHASE B (the A/B, real backbone): the cross-env coarse run TWICE —
#     CLIP-on (tuned thresholds) vs caption-only (--no-room-clip baseline) — so we
#     can see whether the dense signal makes the head FIRE (coarse_chosen>0) AND
#     whether arrival (succ@1m / min_d2g) is helped, not just fire-rate.
#
# ACCEPTANCE (MF-2): coarse_chosen>0 AND warm succ@1m/min_d2g NOT regressed vs the
# caption-only baseline AND vs coarse-OFF. coarse_chosen>0 alone is NOT a pass.
# Also confirm the room histogram is NOT collapsed to one room and n_memory_chosen
# does not inflate (the instance-level over-fire trap).
#
# EXECUTE it (do NOT source). Because this file may be brand-new on RACE, the
# self-pull at step 1 runs the OLD copy via bash's open fd — so on the FIRST run
# `git pull` MANUALLY first:
#
#   git pull --ff-only && nrun bash scripts/race-room-clip.sh --tag clip1
#   nrun bash scripts/race-room-clip.sh --tag clip1 --calib-only   # PHASE A only (gate check)
#
# Flags: --tag <t> (run tag); --calib-only (stop after the cosine measurement);
#        --steps <n> (calibration walk length/scene, default 140); plus any
#        race-cross-env.sh flags are NOT forwarded — edit CATS/SCENES there.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

TAG="clip1"
CALIB_ONLY=0
STEPS=140
EPISODES_PATH="data/hm3d/datasets/objectnav/hm3d/v1/val_mini/val_mini.json.gz"

while [ $# -gt 0 ]; do
  case "$1" in
    --tag)         TAG="$2"; shift 2 ;;
    --calib-only)  CALIB_ONLY=1; shift ;;
    --steps)       STEPS="$2"; shift 2 ;;
    --episodes-path) EPISODES_PATH="$2"; shift 2 ;;
    *) echo "FATAL: unknown arg '$1'"; exit 1 ;;
  esac
done
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "FATAL: --tag must be alphanumeric/dash/underscore (got '$TAG')"; exit 1; }

banner() { printf '\n########## %s ##########\n' "$1"; }

# --- 1. git pull ---
banner "[1/5] git pull --ff-only"
git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }

# --- 2. conda setup ---
banner "[2/5] conda setup (source scripts/race-setup.sh)"
# shellcheck disable=SC1091
source scripts/race-setup.sh || { echo "FATAL: race-setup.sh failed"; exit 1; }

# --- 3. pre-test code verify (free; aborts before any paid run if broken) ---
banner "[3/5] pre-test verify (room classifier + wiring + proposer + analyzer)"
python embodied_memory/scripts/test_room_resolver.py    || { echo "FATAL: room_resolver tests failed."; exit 1; }
python embodied_memory/scripts/test_room_classifier.py  || { echo "FATAL: room_classifier tests failed."; exit 1; }
python embodied_memory/scripts/test_room_clip_wiring.py || { echo "FATAL: room-CLIP wiring tests failed."; exit 1; }
python embodied_memory/scripts/test_coarse_propose.py   || { echo "FATAL: coarse proposer tests failed."; exit 1; }
python embodied_memory/scripts/test_analyze_cross_env.py || { echo "FATAL: analyze_cross_env tests failed."; exit 1; }

# --- 4. PHASE A: measure real CLIP room cosines -> data-driven thresholds (MF-1) ---
banner "[4/5] PHASE A: calibrate CLIP room cosines (sim+CLIP only, no 7B)"
CALIB_LOG="runs/roomclip-calib-${TAG}.log"
mkdir -p runs
python embodied_memory/scripts/diagnose_room_clip_cosines.py --scene all \
    --episodes-path "$EPISODES_PATH" --n-scenes 2 --steps "$STEPS" 2>&1 \
  | tee "$CALIB_LOG"
rc=${PIPESTATUS[0]}
[ "$rc" = "0" ] || { echo "FATAL: calibration diagnostic crashed (rc=$rc)."; exit 1; }

REC_LINE="$(grep -E '^RECOMMEND ' "$CALIB_LOG" | tail -1)"
if [ -z "$REC_LINE" ]; then
  echo "FATAL: no RECOMMEND line in calibration output — cannot set thresholds."; exit 1
fi
echo ">> $REC_LINE"
REC_MIN="$(echo "$REC_LINE" | sed -E 's/.*min_cos=([0-9.]+).*/\1/')"
REC_MARGIN="$(echo "$REC_LINE" | sed -E 's/.*margin=([0-9.]+).*/\1/')"
REC_COLLAPSED="$(echo "$REC_LINE" | sed -E 's/.*collapsed=([0-9]+).*/\1/')"
REC_FIRE="$(echo "$REC_LINE" | sed -E 's/.*fire_rate=([0-9.]+).*/\1/')"
echo ">> data-driven thresholds: min_cos=$REC_MIN  margin=$REC_MARGIN  (default-gate fire_rate=$REC_FIRE, collapsed=$REC_COLLAPSED)"
if [ "${REC_COLLAPSED:-0}" = "1" ]; then
  echo "WARNING: the argmax room histogram is COLLAPSED to one room — the CLIP room"
  echo "         signal is likely uninformative on these scenes. The A/B below will"
  echo "         still run, but treat any coarse_chosen>0 with suspicion (it may all"
  echo "         steer to one room). Read the histogram in $CALIB_LOG."
fi

if [ "$CALIB_ONLY" = "1" ]; then
  banner "DONE (PHASE A only) — read the cosine band + RECOMMEND above; re-run without --calib-only for the A/B"
  exit 0
fi

# Export the measured thresholds so the child cross-env runs inherit them.
export LTM_ROOM_CLIP_MIN_COS="$REC_MIN"
export LTM_ROOM_CLIP_MARGIN="$REC_MARGIN"

# --- 5. PHASE B: the coarse A/B on the cross-env harness (real backbone) ---
# Reuse race-cross-env.sh (it self-pulls + sets up, idempotent). The exported
# thresholds propagate to the child. CLIP-on arm first, caption-only baseline next.
banner "[5/5] PHASE B: cross-env coarse A/B — CLIP-on (tuned) vs caption-only"
echo ">> ARM 1: CLIP-on  (min_cos=$LTM_ROOM_CLIP_MIN_COS margin=$LTM_ROOM_CLIP_MARGIN)"
bash scripts/race-cross-env.sh --coarse --tag "${TAG}-clipon" \
  || { echo "FATAL: CLIP-on cross-env arm failed."; exit 1; }

echo ">> ARM 2: caption-only baseline (LTM_COARSE_ROOM_CLIP=0)"
bash scripts/race-cross-env.sh --coarse --no-room-clip --tag "${TAG}-caponly" \
  || { echo "FATAL: caption-only cross-env arm failed."; exit 1; }

banner "DONE — compare the two arms"
cat <<EOF
Read across the two analyze_cross_env blocks above:
  ARM 1 (CLIP-on)      runs/${TAG}-clipon-s{1,3}
  ARM 2 (caption-only) runs/${TAG}-caponly-s{1,3}

ACCEPTANCE (MF-2): the CLIP arm passes iff
  (a) away n_coarse_chosen > 0          (the dense signal made the head FIRE), AND
  (b) warm/away succ@1m & min_d2g NOT regressed vs caption-only AND vs coarse-OFF, AND
  (c) coarse_room_hist is NOT collapsed to one room, AND
  (d) n_memory_chosen does NOT inflate vs caption-only (no instance over-fire).
coarse_chosen>0 ALONE is not a pass. Also eyeball each s3 summary.json for
n_coarse_room_clip_tagged vs n_coarse_room_caption_tagged vs n_coarse_room_abstained
and coarse_top_cos_max (must be <= 1.0; >1.0 = a normalization bug).
EOF
