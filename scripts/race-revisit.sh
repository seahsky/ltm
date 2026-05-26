#!/bin/bash
# scripts/race-revisit.sh — one-shot RACE driver for the lifelong/revisit eval.
#
# Phase-2 closed with the LTM net-neutral on single-goal-per-episode val_mini
# (soft-SPL S3-S1 = -0.009, n.s.). The diagnosis was STRUCTURAL: ObjectNav
# never rewards recalling a past sighting. This driver runs the revisit eval —
# the same scene traversed repeatedly with the LTM carrying over, goals
# recurring, so a past sighting becomes retrievable and useful.
#
# Mirrors race-smoke.sh (pull -> setup -> pre-verify -> run -> analyze) but
# drives the controlled-start revisit dataset; race-smoke.sh can't be reused
# as-is because it doesn't forward --episodes-path.
#
# EXECUTE it (do NOT source) — it activates conda in its own process and runs
# everything there:
#
#   bash scripts/race-revisit.sh --scene wcojb4TFT35 --categories "chair bed" \
#       --n-warm 3 --tag revisit-b1
#
# A bare invocation reproduces the documented Gate-A smoke (chair+bed on
# wcojb4TFT35, 1 cold + 3 warm each = 8 eps × 2 settings).
#
# Critical invariants baked in (each cost a re-run before):
#   * --backbone remembr      — omitting it silently uses the 'frontier' stub.
#   * REMEMBR_STRICT=1         — a missing-weights/stub fallback CRASHES instead
#                                of silently logging a fake (stub_mode) run.
#   * S1 and S3 in SEPARATE processes / out-dirs — the LTM persists within a
#                                process, so mixing settings would corrupt the
#                                ablation.
#   * --target any            — runs all dataset episodes; the loader preserves
#                                cold-first order (no shuffle), so each category's
#                                cold sighting precedes its warm revisits.
#
# Aborts early (before the paid run) if git pull, conda setup, the pre-test
# suite, or the dataset build fail. The final Gate-A verdict is a valid result,
# not an abort — it always prints.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

# --- defaults (a bare run reproduces the documented Gate-A smoke) ---
SCENE="wcojb4TFT35"
CATS="chair bed"
NWARM="3"
TAG="revisit-b1"
# Empty => auto: run each dataset episode exactly ONCE (one clean cold->warm
# pass). Habitat wraps around when n_episodes > dataset size, which re-runs the
# cold (start-on-goal) episodes; the order-based analyzer then mislabels those
# repeats as "warm" and deflates the warm fire-rate. Pass --n-episodes N
# explicitly to cycle for lifelong-accumulation experiments.
N_EPISODES=""
TARGET="any"

# --- arg parse ---
while [ $# -gt 0 ]; do
  case "$1" in
    --scene)             SCENE="$2"; shift 2 ;;
    --categories|--cats) CATS="$2"; shift 2 ;;
    --n-warm)            NWARM="$2"; shift 2 ;;
    --tag)               TAG="$2"; shift 2 ;;
    --n-episodes)        N_EPISODES="$2"; shift 2 ;;
    --target)            TARGET="$2"; shift 2 ;;
    *) echo "FATAL: unknown arg '$1'"; exit 1 ;;
  esac
done
CATS="${CATS//,/ }"   # accept comma- or space-separated category lists

DS_DIR="data/hm3d/datasets/objectnav/hm3d/v1/revisit_${SCENE}"
SRC="data/hm3d/datasets/objectnav/hm3d/v1/val_mini/content/${SCENE}.json.gz"
DS="${DS_DIR}/revisit_${SCENE}.json.gz"

banner() { printf '\n========== %s ==========\n' "$1"; }

# --- 1. git pull ---
banner "[1/6] git pull --ff-only"
git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }

# --- 2. conda setup (sourced so the env persists in THIS process) ---
banner "[2/6] conda setup (source scripts/race-setup.sh)"
# shellcheck disable=SC1091
source scripts/race-setup.sh || { echo "FATAL: race-setup.sh failed"; exit 1; }

# --- 3. pre-test code verify (free; aborts before any paid run if broken) ---
# These suites are standalone case_*/main() runners (assert-based, sys.exit),
# NOT pytest test_* functions — `pytest` would collect zero and pass vacuously.
# Run them as scripts so a real failure returns non-zero and aborts here.
banner "[3/6] pre-test code verify (revisit analyzer + builder + SPL-guard suites)"
python embodied_memory/scripts/test_analyze_revisit.py \
  || { echo "FATAL: analyze_revisit sanity suite failed — not spending on the live run."; exit 1; }
python embodied_memory/scripts/test_make_revisit_smoke.py \
  || { echo "FATAL: make_revisit_smoke sanity suite failed — not spending on the live run."; exit 1; }
python embodied_memory/scripts/test_spl_guard.py \
  || { echo "FATAL: spl_guard sanity suite failed — not spending on the live run."; exit 1; }
python embodied_memory/scripts/test_text_encode_util.py \
  || { echo "FATAL: text_encode_util sanity suite failed — not spending on the live run."; exit 1; }

# --- 4. rebuild controlled-start dataset (gitignored -> not on RACE) ---
banner "[4/6] build revisit dataset: scene=$SCENE cats=[$CATS] n-warm=$NWARM -> $DS_DIR"
[ -f "$SRC" ] || { echo "FATAL: source episodes missing: $SRC"; exit 1; }
# shellcheck disable=SC2086
python embodied_memory/scripts/make_revisit_smoke.py \
    --src "$SRC" --scene "$SCENE" --categories $CATS --n-warm "$NWARM" \
    --out-dir "$DS_DIR" \
  || { echo "FATAL: dataset build failed."; exit 1; }
[ -f "$DS" ] || { echo "FATAL: expected dataset not written: $DS"; exit 1; }

# Default n-episodes = exactly the built episode count (one clean cold->warm
# pass, no Habitat wrap-around re-running the cold seeds). Honour an explicit
# --n-episodes override.
if [ -z "$N_EPISODES" ]; then
  N_EPISODES="$(python -c "import gzip,json,sys; print(len(json.load(gzip.open(sys.argv[1]))['episodes']))" "$DS_DIR/content/$SCENE.json.gz")" \
    || { echo "FATAL: could not count dataset episodes."; exit 1; }
  echo "  auto n-episodes = $N_EPISODES (one pass over the built dataset)"
fi

# --- 5. run S1 (memory-off) and S3 (full) in SEPARATE processes ---
# LTM persists within a process; settings must never share one out-dir/process.
OUT_DIRS=""
for S in 1 3; do
  out_dir="runs/${TAG}-s$S"
  banner "[5/6] run: setting=$S backbone=remembr scene=$SCENE -> $out_dir"
  REMEMBR_STRICT=1 python -m embodied_memory.run_hm3d_pol --mode live \
      --backbone remembr --setting "$S" --episodes-path "$DS" \
      --scene "$SCENE" --target "$TARGET" --n-episodes "$N_EPISODES" \
      --out-dir "$out_dir" 2>&1 | tee "${out_dir}.log"
  OUT_DIRS="$OUT_DIRS $out_dir"
done

# --- 6. Gate-A verdict (warm-only paired soft-SPL S3-S1 + fire-rate) ---
banner "[6/6] Gate-A analysis: analyze_revisit.py runs/${TAG}-s1 runs/${TAG}-s3"
# shellcheck disable=SC2086
python embodied_memory/scripts/analyze_revisit.py $OUT_DIRS

banner "DONE — paste everything above (esp. the Gate-A block)"
