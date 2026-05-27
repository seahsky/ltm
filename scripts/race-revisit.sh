#!/bin/bash
# scripts/race-revisit.sh — one-shot RACE driver for the multi-scene,
# 3-setting lifelong/revisit ablation (Phase C).
#
# Phase 3 (Run 8) turned the revisit eval GREEN on a SINGLE scene (wcojb4TFT35,
# chair+bed, S1 vs S3). Phase C scales it to MULTIPLE scenes and adds S2
# (STM-only) so the gain can be attributed: S2-S1 = STM module, S3-S2 =
# consolidation+hierarchical-LTM+rerank (the proposal's novel part), S3-S1 =
# headline full system (the gate). Multiple scenes test the proposal's
# cross-environment (跨环境) claim.
#
# Mirrors race-smoke.sh (pull -> setup -> pre-verify -> build -> run -> analyze).
# EXECUTE it (do NOT source) — it activates conda in its own process:
#
#   bash scripts/race-revisit.sh --tag revisit-c1
#
# A bare invocation reproduces the documented Phase-C matrix: both val_mini
# scenes (wcojb4TFT35, TEEsavR23oF) x {chair, bed} x {S1, S2, S3}, n-warm 3.
#
# Critical invariants baked in (each cost a re-run before):
#   * --backbone remembr      — omitting it silently uses the 'frontier' stub.
#   * REMEMBR_STRICT=1         — a missing-weights/stub fallback CRASHES instead
#                                of silently logging a fake (stub_mode) run.
#   * S1/S2/S3 in SEPARATE processes / out-dirs — the LTM persists within a
#                                process, so mixing settings would corrupt it.
#   * --scene all + shuffle=False (pinned in habitat_env via episode_order) —
#                                each scene's COLD seed precedes its WARM visits.
#   * --target any            — runs all dataset episodes.
#
# Aborts early (before the paid run) if git pull, conda setup, the pre-test
# suite, or the dataset build fail. The final Gate-A verdict always prints.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

# --- defaults (a bare run reproduces the documented Phase-C matrix) ---
SCENES="wcojb4TFT35 TEEsavR23oF"
CATS="chair bed"
NWARM="3"
TAG="revisit-c1"
# Empty => auto: run each dataset episode exactly ONCE (one clean cold->warm
# pass across ALL scenes). Habitat wraps around when n_episodes > dataset size,
# which re-runs cold (start-on-goal) episodes and deflates the warm fire-rate.
N_EPISODES=""
TARGET="any"

# --- arg parse ---
while [ $# -gt 0 ]; do
  case "$1" in
    --scenes|--scene)    SCENES="$2"; shift 2 ;;
    --categories|--cats) CATS="$2"; shift 2 ;;
    --n-warm)            NWARM="$2"; shift 2 ;;
    --tag)               TAG="$2"; shift 2 ;;
    --n-episodes)        N_EPISODES="$2"; shift 2 ;;
    --target)            TARGET="$2"; shift 2 ;;
    *) echo "FATAL: unknown arg '$1'"; exit 1 ;;
  esac
done
CATS="${CATS//,/ }"        # accept comma- or space-separated lists
SCENES="${SCENES//,/ }"

VALMINI="data/hm3d/datasets/objectnav/hm3d/v1/val_mini/content"
DS_DIR="data/hm3d/datasets/objectnav/hm3d/v1/revisit_${TAG}"
NAME="revisit_${TAG}"
DS="${DS_DIR}/${NAME}.json.gz"

banner() { printf '\n========== %s ==========\n' "$1"; }

# --- 1. git pull ---
banner "[1/6] git pull --ff-only"
git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }

# --- 2. conda setup (sourced so the env persists in THIS process) ---
banner "[2/6] conda setup (source scripts/race-setup.sh)"
# shellcheck disable=SC1091
source scripts/race-setup.sh || { echo "FATAL: race-setup.sh failed"; exit 1; }

# --- 3. pre-test code verify (free; aborts before any paid run if broken) ---
# Standalone case_*/main() runners (assert-based, sys.exit), NOT pytest test_*.
banner "[3/6] pre-test code verify (analyzer + builder + SPL-guard + encoder + episode-order)"
python embodied_memory/scripts/test_analyze_revisit.py \
  || { echo "FATAL: analyze_revisit sanity suite failed — not spending on the live run."; exit 1; }
python embodied_memory/scripts/test_make_revisit_smoke.py \
  || { echo "FATAL: make_revisit_smoke sanity suite failed — not spending on the live run."; exit 1; }
python embodied_memory/scripts/test_spl_guard.py \
  || { echo "FATAL: spl_guard sanity suite failed — not spending on the live run."; exit 1; }
python embodied_memory/scripts/test_text_encode_util.py \
  || { echo "FATAL: text_encode_util sanity suite failed — not spending on the live run."; exit 1; }
python embodied_memory/scripts/test_episode_order.py \
  || { echo "FATAL: episode_order sanity suite failed — not spending on the live run."; exit 1; }

# --- 4. rebuild controlled-start dataset, ALL scenes into one shared dir ---
# make_revisit_smoke writes content/<scene>.json.gz per-scene (additive across
# calls) and rewrites the top-level <name>.json.gz (harmless: both val_mini
# scenes share the ObjectNav category map; the scene-annotation map is unused).
banner "[4/6] build revisit dataset: scenes=[$SCENES] cats=[$CATS] n-warm=$NWARM -> $DS_DIR"
rm -rf "$DS_DIR"   # fresh build so a stale content/ from an earlier tag can't inflate n-episodes
for SCENE in $SCENES; do
  SRC="${VALMINI}/${SCENE}.json.gz"
  [ -f "$SRC" ] || { echo "FATAL: source episodes missing: $SRC"; exit 1; }
  # shellcheck disable=SC2086
  python embodied_memory/scripts/make_revisit_smoke.py \
      --src "$SRC" --scene "$SCENE" --categories $CATS --n-warm "$NWARM" \
      --out-dir "$DS_DIR" \
    || { echo "FATAL: dataset build failed for scene $SCENE."; exit 1; }
done
[ -f "$DS" ] || { echo "FATAL: expected top-level dataset not written: $DS"; exit 1; }

# Default n-episodes = SUM of episodes across ALL content/*.json.gz (--scene all
# loads every scene; counting one file would truncate the others).
if [ -z "$N_EPISODES" ]; then
  N_EPISODES="$(python -c "import gzip,json,glob,sys; print(sum(len(json.load(gzip.open(f))['episodes']) for f in sorted(glob.glob(sys.argv[1]))))" "${DS_DIR}/content/*.json.gz")" \
    || { echo "FATAL: could not count dataset episodes."; exit 1; }
  echo "  auto n-episodes = $N_EPISODES (one pass over all built scenes)"
fi

# --- 5. run S1/S2/S3 in SEPARATE processes (--scene all over the built scenes) ---
OUT_DIRS=""
for S in 1 2 3; do
  out_dir="runs/${TAG}-s$S"
  banner "[5/6] run: setting=$S backbone=remembr scenes=all -> $out_dir"
  REMEMBR_STRICT=1 python -m embodied_memory.run_hm3d_pol --mode live \
      --backbone remembr --setting "$S" --episodes-path "$DS" \
      --scene all --target "$TARGET" --n-episodes "$N_EPISODES" \
      --out-dir "$out_dir" 2>&1 | tee "${out_dir}.log"
  OUT_DIRS="$OUT_DIRS $out_dir"
done

# --- 6. Gate-A verdict (warm-only paired soft-SPL + S2 decomposition) ---
banner "[6/6] Gate-A analysis: analyze_revisit.py$OUT_DIRS"
# shellcheck disable=SC2086
python embodied_memory/scripts/analyze_revisit.py $OUT_DIRS

banner "DONE — paste everything above (esp. the Gate-A block + S2 decomposition)"
