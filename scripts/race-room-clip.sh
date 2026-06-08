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
#   PHASE B (real backbone): the CROSS-ENV coarse A/B — CLIP-on (tuned) vs
#     caption-only (--no-room-clip). Tests "does the dense signal make the head
#     FIRE (coarse_chosen>0) and HELP in a NEW scene (succ@1m/min_d2g)?".
#   PHASE C (real backbone): the WARM-REVISIT over-fire A/B — the documented
#     Phase-C +0.24 harness, coarse-OFF baseline (full S1/S2/S3) vs coarse-CLIP-ON
#     (S3 only, rerun on the SAME dataset). Tests "does the relaxed coarse gate
#     OVER-FIRE on warm same-scene episodes and dent the established +0.24?".
#
# This is a multi-hour run (PHASE C is the documented 48-episode revisit matrix +
# one extra S3 arm). PHASE A ~minutes; PHASE B ~1-3h; PHASE C ~several hours.
#
# ACCEPTANCE: the CLIP path passes iff
#   PHASE B: away coarse_chosen>0 AND away succ@1m/min_d2g NOT regressed vs
#            caption-only (coarse FIRES + HELPS in a new scene), AND
#   PHASE C: coarse-ON warm S3-S1 ~= coarse-OFF warm S3-S1 (~+0.24) — NO regression
#            (the relaxed gate does not over-fire on warm same-scene), AND
#   both:    coarse_room_hist NOT collapsed to one room; n_memory_chosen NOT
#            inflated (no instance over-fire); coarse_top_cos_max <= 1.0.
#
# EXECUTE it (do NOT source). Because this file may be brand-new on RACE, the
# self-pull at step 1 runs the OLD copy via bash's open fd — so on the FIRST run
# `git pull` MANUALLY first:
#
#   git pull --ff-only && nrun bash scripts/race-room-clip.sh --tag clip1
#   nrun bash scripts/race-room-clip.sh --tag clip1 --calib-only      # PHASE A only (gate check)
#   nrun bash scripts/race-room-clip.sh --tag clip1 --skip-revisit    # A+B only (faster)
#   nrun bash scripts/race-room-clip.sh --tag clip1 --skip-crossenv   # A+C only
#
# Flags: --tag <t>; --calib-only; --steps <n> (calib walk len, default 140);
#        --skip-crossenv / --skip-revisit; --rev-scenes/--rev-cats/--rev-nwarm
#        (PHASE C matrix; default = the documented chair+bed x 2 scenes x n-warm 3).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

TAG="clip1"
CALIB_ONLY=0
STEPS=140
EPISODES_PATH="data/hm3d/datasets/objectnav/hm3d/v1/val_mini/val_mini.json.gz"
SKIP_CROSSENV=0   # --skip-crossenv: omit PHASE B (the cross-env "does coarse FIRE+help" A/B)
SKIP_REVISIT=0    # --skip-revisit:  omit PHASE C (the warm-revisit "does coarse OVER-FIRE" A/B)
REV_SCENES="wcojb4TFT35 TEEsavR23oF"   # PHASE C: the documented Phase-C +0.24 harness
REV_CATS="chair bed"
REV_NWARM="3"

while [ $# -gt 0 ]; do
  case "$1" in
    --tag)           TAG="$2"; shift 2 ;;
    --calib-only)    CALIB_ONLY=1; shift ;;
    --steps)         STEPS="$2"; shift 2 ;;
    --episodes-path) EPISODES_PATH="$2"; shift 2 ;;
    --skip-crossenv) SKIP_CROSSENV=1; shift ;;
    --skip-revisit)  SKIP_REVISIT=1; shift ;;
    --rev-scenes)    REV_SCENES="$2"; shift 2 ;;
    --rev-cats)      REV_CATS="$2"; shift 2 ;;
    --rev-nwarm)     REV_NWARM="$2"; shift 2 ;;
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

# Export the measured thresholds so the child cross-env / revisit runs inherit them.
export LTM_ROOM_CLIP_MIN_COS="$REC_MIN"
export LTM_ROOM_CLIP_MARGIN="$REC_MARGIN"

# --- 5. PHASE B: cross-env coarse A/B (real backbone) — does coarse FIRE+HELP? ---
# Reuse race-cross-env.sh (self-pulls + sets up, idempotent). Exported thresholds
# propagate to the child. NOTE: --coarse only sets LTM_COARSE_AFFORDANCE inside the
# CHILD, so it never leaks into PHASE C's coarse-OFF baseline.
if [ "$SKIP_CROSSENV" = "0" ]; then
  banner "[5/6] PHASE B: cross-env coarse A/B — CLIP-on (tuned) vs caption-only"
  echo ">> ARM 1: CLIP-on  (min_cos=$LTM_ROOM_CLIP_MIN_COS margin=$LTM_ROOM_CLIP_MARGIN)"
  bash scripts/race-cross-env.sh --coarse --tag "${TAG}-clipon" \
    || { echo "FATAL: CLIP-on cross-env arm failed."; exit 1; }
  echo ">> ARM 2: caption-only baseline (LTM_COARSE_ROOM_CLIP=0)"
  bash scripts/race-cross-env.sh --coarse --no-room-clip --tag "${TAG}-caponly" \
    || { echo "FATAL: caption-only cross-env arm failed."; exit 1; }
else
  banner "[5/6] PHASE B: cross-env A/B SKIPPED (--skip-crossenv)"
fi

# --- 6. PHASE C: warm-revisit over-fire A/B (real backbone) — does coarse HURT? ---
# Baseline = the documented +0.24 harness with coarse OFF (full S1/S2/S3). The
# coarse-ON arm reruns ONLY S3 on the SAME dataset (--reuse-dataset) so it pairs
# against the baseline's exact S1 episodes. Compare coarse-ON warm S3-S1 vs the
# coarse-OFF baseline warm S3-S1 (~+0.24): a drop = the relaxed gate over-fires on
# warm same-scene; parity = safe.
REV_DS_DIR="data/hm3d/datasets/objectnav/hm3d/v1/revisit_${TAG}-revoff"
if [ "$SKIP_REVISIT" = "0" ]; then
  banner "[6/6] PHASE C: warm-revisit over-fire A/B — coarse-OFF baseline vs coarse-CLIP-ON"
  echo ">> ARM 1 (baseline, coarse OFF, full S1/S2/S3): the established +0.24 harness"
  bash scripts/race-revisit.sh --tag "${TAG}-revoff" \
      --scenes "$REV_SCENES" --cats "$REV_CATS" --n-warm "$REV_NWARM" \
    || { echo "FATAL: revisit baseline (coarse-OFF) arm failed."; exit 1; }
  echo ">> ARM 2 (coarse-CLIP-ON, S3 only, SAME dataset): does it dent the +0.24?"
  bash scripts/race-revisit.sh --tag "${TAG}-revon" --coarse --settings 3 \
      --reuse-dataset "$REV_DS_DIR" \
    || { echo "FATAL: revisit coarse-ON arm failed."; exit 1; }

  banner "PHASE C cross-arm pairing: coarse-ON S3 vs baseline S1/S2"
  # Pair the coarse-ON S3 against the baseline S1/S2 (SAME episodes) so this delta is
  # directly comparable to the baseline Gate-A block printed by ARM 1 above.
  if [ -d "runs/${TAG}-revoff-s1" ] && [ -d "runs/${TAG}-revon-s3" ]; then
    python embodied_memory/scripts/analyze_ablation.py --revisit \
        "runs/${TAG}-revoff-s1" "runs/${TAG}-revoff-s2" "runs/${TAG}-revon-s3" \
      || echo "WARN: coarse-ON pairing analysis failed (inspect the run dirs)."
  else
    echo "WARN: expected run dirs missing — skipping coarse-ON pairing."
  fi
else
  banner "[6/6] PHASE C: revisit over-fire A/B SKIPPED (--skip-revisit)"
fi

banner "DONE — read the ACCEPTANCE checklist"
cat <<EOF
Calibration band + RECOMMEND: $CALIB_LOG  (thresholds used: min_cos=$REC_MIN margin=$REC_MARGIN)

PHASE B (cross-env, does coarse FIRE+HELP in a NEW scene):
  ARM 1 CLIP-on      runs/${TAG}-clipon-s{1,3}
  ARM 2 caption-only runs/${TAG}-caponly-s{1,3}
  PASS iff away n_coarse_chosen>0 AND away succ@1m/min_d2g NOT regressed vs caption-only.

PHASE C (warm revisit, does coarse OVER-FIRE on same-scene):
  baseline (coarse OFF) runs/${TAG}-revoff-s{1,2,3}  -> warm S3-S1 should reproduce ~+0.24
  coarse-ON (S3)        runs/${TAG}-revon-s3          -> paired vs revoff-s1 above
  PASS iff coarse-ON warm S3-S1 ~= baseline warm S3-S1 (no regression).

BOTH: coarse_room_hist NOT collapsed to one room; n_memory_chosen NOT inflated vs
baseline (no instance over-fire); coarse_top_cos_max <= 1.0 (>1.0 = normalization bug).
Inspect each s3 summary.json: n_coarse_room_clip_tagged / _caption_tagged / _abstained.
EOF
