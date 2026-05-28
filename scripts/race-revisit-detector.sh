#!/bin/bash
# scripts/race-revisit-detector.sh — RACE driver for the binary-SPL milestone:
# 6-cell ablation (S1/S2/S3 x detector ON/OFF) on the Phase-C revisit dataset.
#
# Mirrors race-revisit.sh (pull -> setup -> pre-verify -> build -> run -> analyze)
# but runs each setting twice: once with --detector and once without. Total
# 96 episodes / ~4 GPU-hours sequential on an L4.
#
# EXECUTE it (do NOT source) — conda is activated in its own process:
#
#   bash scripts/race-revisit-detector.sh --tag detector-c1
#
# A bare invocation reproduces the milestone's documented matrix.
#
# Critical invariants (each cost a re-run before):
#   * --backbone remembr  — required (--detector needs Qwen-VL handles)
#   * REMEMBR_STRICT=1     — stub fallback crashes instead of silently logging
#   * S1/S2/S3 x det/nodet in SEPARATE processes / out-dirs (LTM persists
#     within a process; mixing settings or det/nodet would corrupt it)
#   * --scene all + shuffle=False (pinned in habitat_env via episode_order)
#   * --target any         — runs all dataset episodes
#
# Aborts early if pull / conda / pre-verify / dataset build fails. Per-cell
# n_episodes_completed completeness is a WARN, not an abort.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

# --- defaults (a bare run reproduces the milestone matrix) ---
SCENES="wcojb4TFT35 TEEsavR23oF"
CATS="chair bed"
NWARM="3"
TAG=""
N_EPISODES=""
TARGET="any"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --scenes) SCENES="$2"; shift 2 ;;
    --categories) CATS="$2"; shift 2 ;;
    --n-warm) NWARM="$2"; shift 2 ;;
    --n-episodes) N_EPISODES="$2"; shift 2 ;;
    --target) TARGET="$2"; shift 2 ;;
    -h|--help) sed -n '1,40p' "$0"; exit 0 ;;
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
  || { echo "FATAL: goal_detector sanity suite failed"; exit 1; }
python embodied_memory/scripts/test_episode_runner_detector.py \
  || { echo "FATAL: episode_runner_detector sanity suite failed"; exit 1; }

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

# --- 5. pre-flight detector smoke (1 episode, GO/NO-GO) ---
banner "[5/7] pre-flight: setting=3 backbone=remembr --detector  scenes=wcojb4TFT35  n=1"
PREFLIGHT_DIR="runs/${TAG}-preflight"
REMEMBR_STRICT=1 python -m embodied_memory.run_hm3d_pol --mode live \
    --backbone remembr --detector --setting 3 --episodes-path "$DS" \
    --scene wcojb4TFT35 --target chair --n-episodes 1 \
    --out-dir "$PREFLIGHT_DIR" 2>&1 | tee "${PREFLIGHT_DIR}.log"
n_called="$(python -c "import json,sys; s=json.load(open(sys.argv[1])); print(s.get('n_detector_called', 0))" "${PREFLIGHT_DIR}/summary.json" 2>/dev/null || echo 0)"
n_localized="$(python -c "import json,sys; s=json.load(open(sys.argv[1])); print(s.get('n_detector_localized', 0))" "${PREFLIGHT_DIR}/summary.json" 2>/dev/null || echo 0)"
echo "preflight: n_detector_called=$n_called n_detector_localized=$n_localized"
if [ "$n_called" = "0" ]; then
  echo "FATAL: pre-flight — detector never called. Keyword-STOP didn't fire; rerun or diagnose."
  exit 1
fi
if [ "$n_localized" = "0" ]; then
  echo "WARN: pre-flight — detector called but never localized. Possible Qwen-VL grounding issue."
  echo "Proceeding to the 6-cell matrix anyway (the matrix itself will surface the rate)."
fi

# --- 6. run 6 cells: S1/S2/S3 x detector OFF/ON in SEPARATE processes ---
OUT_DIRS_NODET=""
OUT_DIRS_DET=""
for FLAG in nodet det; do
  EXTRA=""
  [ "$FLAG" = "det" ] && EXTRA="--detector"
  for S in 1 2 3; do
    out_dir="runs/${TAG}-s${S}-${FLAG}"
    banner "[6/7] run: setting=$S detector=$FLAG -> $out_dir"
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
banner "[7/7] Gate analysis: detector OFF triple"
# shellcheck disable=SC2086
python embodied_memory/scripts/analyze_ablation.py --revisit $OUT_DIRS_NODET

banner "[7/7] Gate analysis: detector ON triple"
# shellcheck disable=SC2086
python embodied_memory/scripts/analyze_ablation.py --revisit $OUT_DIRS_DET

banner "[7/7] Cross-condition contrast (det vs nodet; manual inspection)"
echo "Inspect the WARM binary S3-S1 means and CIs:"
echo "  Gate A: s1-det WARM binary SPL  vs  s1-nodet WARM binary SPL"
echo "  Gate B: s3-det WARM binary SPL (S3-S1)  vs  s1-det WARM binary SPL (S3-S1)"
echo "  Gate C: s3-det WARM binary SPL  vs  s1-nodet WARM binary SPL (HEADLINE; bar >= +0.3)"
echo "  Gate D: s3-nodet WARM soft-SPL S3-S1  reproduces Phase-C (>= +0.15, p<0.05)"

banner "DONE — paste everything above (the two Gate-A blocks + the cross-condition summary)"
