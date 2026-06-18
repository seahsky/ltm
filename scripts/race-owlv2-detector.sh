#!/bin/bash
# scripts/race-owlv2-detector.sh — Lever-3 RACE driver: A/B a REAL open-vocab
# detector (OWLv2) for precise <0.1 m goal localization against detector-OFF.
#
# WHY: binary SPL@0.1 m is localization-bound. The c7/c9 arc CLOSED detector-OFF
# because the Qwen2-VL *caption-grounding* detector picked the wrong INSTANCE
# (~half the time) — a bbox-source quality ceiling, not a radius knob. OWLv2 is a
# trained object detector (not a VLM grounding head), so its error mode is
# different; the hypothesis is it hallucinates wrong instances less. This driver
# measures it. The OWLv2 backend is already wired (goal_detector._infer_owlv2,
# dispatched on DETECTOR_BACKEND=owlv2) and TDD-covered (test_goal_detector.py);
# this driver just runs the binary-SPL matrix with that backend selected.
#
# Forked from scripts/archive/race-revisit-detector.sh (the closed Qwen-detector
# matrix). The ONLY behavioral change is `export DETECTOR_BACKEND=owlv2` — the
# detector-OFF arm is byte-identical to the c7/c9 baseline (DETECTOR_BACKEND is
# read per-call but the OFF arm never constructs the detector).
#
# EXECUTE it (do NOT source) — conda activates in its own process:
#
#   bash scripts/race-owlv2-detector.sh --tag owlv2-d1
#   bash scripts/race-owlv2-detector.sh --tag owlv2-d1 --owl-thresh 0.05   # looser
#
# A bare invocation reproduces the c7/c9 matrix scenes/cats with OWLv2 as the
# detector backend.
#
# Critical invariants (each cost a re-run before):
#   * --backbone remembr  — required by --detector (loads Qwen-VL handles; OWLv2
#     ignores them and lazily loads its own model, so they're loaded-but-unused)
#   * DETECTOR_BACKEND=owlv2 — exported BEFORE the runs (read per-call in locate())
#   * REMEMBR_STRICT=1     — stub fallback crashes instead of silently logging
#   * S1/S2/S3 x det/nodet in SEPARATE processes / out-dirs (LTM persists within
#     a process; mixing settings or det/nodet would corrupt it)
#   * --scene all + shuffle=False (pinned in habitat_env via episode_order)
#   * --target any         — runs all dataset episodes

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

# --- defaults (a bare run reproduces the c7/c9 matrix with the OWLv2 backend) ---
SCENES="wcojb4TFT35 TEEsavR23oF"
CATS="chair bed"
NWARM="3"
TAG=""
N_EPISODES=""
TARGET="any"
OWL_THRESH="0.1"   # DETECTOR_OWL_SCORE_THRESH; the scout flagged calibration risk

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --scenes) SCENES="$2"; shift 2 ;;
    --categories) CATS="$2"; shift 2 ;;
    --n-warm) NWARM="$2"; shift 2 ;;
    --n-episodes) N_EPISODES="$2"; shift 2 ;;
    --target) TARGET="$2"; shift 2 ;;
    --owl-thresh) OWL_THRESH="$2"; shift 2 ;;
    -h|--help) sed -n '1,46p' "$0"; exit 0 ;;
    *) echo "FATAL: unknown arg '$1'"; exit 1 ;;
  esac
done
CATS="${CATS//,/ }"
SCENES="${SCENES//,/ }"
[ -z "$TAG" ] && { echo "FATAL: --tag <name> required"; exit 1; }
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "FATAL: --tag must be alnum/dash/underscore"; exit 1; }

VALMINI="data/hm3d/datasets/objectnav/hm3d/v1/val_mini/content"
DS_DIR="data/hm3d/datasets/objectnav/hm3d/v1/revisit_${TAG}"
NAME="revisit_${TAG}"
DS="${DS_DIR}/${NAME}.json.gz"

banner() { printf '\n========== %s ==========\n' "$1"; }

# --- 1. git pull ---
banner "[1/7] git pull --ff-only"
git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }

# --- 2. conda setup ---
banner "[2/7] conda setup (source scripts/race-setup.sh)"
# shellcheck disable=SC1091
source scripts/race-setup.sh || { echo "FATAL: race-setup.sh failed"; exit 1; }

# --- 3. pre-test code verify ---
# NOTE: DETECTOR_BACKEND is deliberately NOT exported yet — the pre-tests
# (test_goal_detector.py) build a GoalDetector and would mis-dispatch to the
# OWLv2 path (loading the real model) if the env var leaked in here. The export
# happens AFTER this block, so the unit tests run in the clean qwen-default env.
banner "[3/7] pre-test code verify (analyzer + builder + SPL-guard + encoder + episode-order + analyze_ablation + goal_detector + episode_runner_detector)"
python embodied_memory/scripts/test_analyze_revisit.py \
  || { echo "FATAL: analyze_revisit sanity suite failed"; exit 1; }
python embodied_memory/scripts/test_make_revisit_smoke.py \
  || { echo "FATAL: make_revisit_smoke sanity suite failed"; exit 1; }
python embodied_memory/scripts/test_spl_guard.py \
  || { echo "FATAL: spl_guard sanity suite failed"; exit 1; }
python embodied_memory/scripts/test_text_encode_util.py \
  || { echo "FATAL: text_encode_util sanity suite failed"; exit 1; }
python embodied_memory/scripts/test_episode_order.py \
  || { echo "FATAL: episode_order sanity suite failed"; exit 1; }
python embodied_memory/scripts/test_analyze_ablation.py \
  || { echo "FATAL: analyze_ablation --revisit dispatch sanity suite failed"; exit 1; }
python embodied_memory/scripts/test_goal_detector.py \
  || { echo "FATAL: goal_detector sanity suite failed (incl. OWLv2 dispatch/threshold)"; exit 1; }
python embodied_memory/scripts/test_episode_runner_detector.py \
  || { echo "FATAL: episode_runner_detector sanity suite failed"; exit 1; }

# Lever-3: select the OWLv2 open-vocab backend for every detector call (read
# per-call in goal_detector._backend()). Exported HERE — after the pre-tests —
# so the qwen unit tests above ran clean. The detector-OFF arm never calls
# locate(), so it stays byte-identical to the c7/c9 baseline regardless.
export DETECTOR_BACKEND=owlv2
export DETECTOR_OWL_SCORE_THRESH="$OWL_THRESH"
echo "  [owlv2] DETECTOR_BACKEND=owlv2  DETECTOR_OWL_SCORE_THRESH=$OWL_THRESH"

# --- 4. build revisit dataset (same as race-revisit.sh) ---
banner "[4/7] build revisit dataset: scenes=[$SCENES] cats=[$CATS] n-warm=$NWARM -> $DS_DIR"
rm -rf "$DS_DIR"
for SCENE in $SCENES; do
  SRC="${VALMINI}/${SCENE}.json.gz"
  [ -f "$SRC" ] || { echo "FATAL: source episodes missing: $SRC"; exit 1; }
  # shellcheck disable=SC2086
  python embodied_memory/scripts/make_revisit_smoke.py \
      --src "$SRC" --scene "$SCENE" --categories $CATS --n-warm "$NWARM" \
      --out-dir "$DS_DIR" \
    || { echo "FATAL: dataset build failed for $SCENE"; exit 1; }
done
[ -f "$DS" ] || { echo "FATAL: top-level dataset not written: $DS"; exit 1; }
if [ -z "$N_EPISODES" ]; then
  N_EPISODES="$(python -c "import gzip,json,glob,sys; print(sum(len(json.load(gzip.open(f))['episodes']) for f in sorted(glob.glob(sys.argv[1]))))" "${DS_DIR}/content/*.json.gz")" \
    || { echo "FATAL: could not count dataset episodes"; exit 1; }
  echo "  auto n-episodes = $N_EPISODES (one pass over all built scenes)"
fi
[ "$N_EPISODES" -gt 0 ] 2>/dev/null || { echo "FATAL: n-episodes <=0"; exit 1; }

# --- 5. pre-flight OWLv2 detector smoke (1 episode, GO/NO-GO) ---
# Same GO/NO-GO as the qwen driver, now testing OWLv2 localization. If OWLv2
# never localizes, the det arm would be byte-identical to nodet — abort before
# spending the full matrix (the lesson from detector-c1).
banner "[5/7] pre-flight: setting=3 backbone=remembr --detector(owlv2)  scenes=wcojb4TFT35  n=1"
PREFLIGHT_DIR="runs/${TAG}-preflight"
REMEMBR_STRICT=1 python -m embodied_memory.run_hm3d_pol --mode live \
    --backbone remembr --detector --setting 3 --episodes-path "$DS" \
    --scene wcojb4TFT35 --target chair --n-episodes 1 \
    --out-dir "$PREFLIGHT_DIR" 2>&1 | tee "${PREFLIGHT_DIR}.log"
n_called="$(python -c "import json,sys; s=json.load(open(sys.argv[1])); print(s.get('n_detector_called', 0))" "${PREFLIGHT_DIR}/summary.json" 2>/dev/null || echo 0)"
n_localized="$(python -c "import json,sys; s=json.load(open(sys.argv[1])); print(s.get('n_detector_localized', 0))" "${PREFLIGHT_DIR}/summary.json" 2>/dev/null || echo 0)"
echo "preflight: n_detector_called=$n_called n_detector_localized=$n_localized"
DEBUG_LOG="${PREFLIGHT_DIR}/goal_detector_debug.log"
if [ -f "$DEBUG_LOG" ]; then
  n_fail="$(wc -l < "$DEBUG_LOG" | tr -d ' ')"
  echo "preflight goal_detector_debug.log: $n_fail failure entries (first 3 below)"
  head -3 "$DEBUG_LOG"
fi
if [ "$n_called" = "0" ]; then
  echo "FATAL: pre-flight — detector never called. Keyword-STOP didn't fire; rerun or diagnose."
  exit 1
fi
if [ "$n_localized" = "0" ]; then
  echo "FATAL: pre-flight — OWLv2 called but never localized. The debug log above"
  echo "shows the owl_no_detection / owl_below_threshold reasons. Running the matrix"
  echo "would make detector=ON byte-identical to detector=OFF. Lower --owl-thresh"
  echo "(default 0.1) or check OWLv2 loaded, then re-run."
  exit 1
fi

# --- 6. run 6 cells: S1/S2/S3 x detector OFF/ON in SEPARATE processes ---
OUT_DIRS_NODET=""
OUT_DIRS_DET=""
for FLAG in nodet det; do
  EXTRA=""
  [ "$FLAG" = "det" ] && EXTRA="--detector"
  for S in 1 2 3; do
    out_dir="runs/${TAG}-s${S}-${FLAG}"
    banner "[6/7] run: setting=$S detector=$FLAG (backend=owlv2 when det) -> $out_dir"
    # shellcheck disable=SC2086
    REMEMBR_STRICT=1 python -m embodied_memory.run_hm3d_pol --mode live \
        --backbone remembr $EXTRA --setting "$S" --episodes-path "$DS" \
        --scene all --target "$TARGET" --n-episodes "$N_EPISODES" \
        --out-dir "$out_dir" 2>&1 | tee "${out_dir}.log"
    rc=${PIPESTATUS[0]}
    completed="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['n_episodes_completed'])" "${out_dir}/summary.json" 2>/dev/null || echo 0)"
    if [ "$completed" != "$N_EPISODES" ]; then
      echo "WARN: setting $S/$FLAG completed ${completed}/${N_EPISODES} (exit $rc) — Gate contribution may be partial."
    fi
    if [ "$FLAG" = "nodet" ]; then
      OUT_DIRS_NODET="$OUT_DIRS_NODET $out_dir"
    else
      OUT_DIRS_DET="$OUT_DIRS_DET $out_dir"
    fi
  done
done

# --- 7. Gate analysis: paired bootstrap on warm visits for each condition ---
banner "[7/7] Gate analysis: detector OFF triple (the c7/c9 baseline that WON)"
# shellcheck disable=SC2086
python embodied_memory/scripts/analyze_ablation.py --revisit $OUT_DIRS_NODET

banner "[7/7] Gate analysis: detector ON triple (OWLv2 backend)"
# shellcheck disable=SC2086
python embodied_memory/scripts/analyze_ablation.py --revisit $OUT_DIRS_DET

banner "[7/7] Cross-condition contrast (OWLv2-det vs nodet; manual inspection)"
echo "Inspect the WARM binary S3-S1 means and CIs:"
echo "  WIN  if OWLv2 det WARM binary SPL >= nodet (OWLv2 fixed the wrong-instance ceiling)"
echo "  NULL/REGRESS if OWLv2 det <= nodet (open-vocab detector no better than caption-grounding"
echo "       at this success radius → binary SPL stays localization-bound; honest negative)"
echo "  Also: s3-nodet WARM soft-SPL S3-S1 should reproduce the LTM thesis (>= +0.15)."

banner "DONE — paste everything above (the two Gate blocks + the cross-condition summary)"
