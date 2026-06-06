#!/bin/bash
# scripts/race-multion.sh — one-shot RACE driver for the MultiON (sequential
# semantic ObjectNav) 3-setting ablation.
#
# Single-goal ObjectNav structurally under-tests the LTM (Run 7); MultiON
# chains K categories per episode so a c_{i+1} glimpsed while hunting c_i is
# recallable when the goal advances — the memory's value compounds. The
# hypothesis to test: S3 PPL >> S1 PPL, and the S3-S1 gap GROWS with sub-goal
# index. See docs/MULTION_PORT_PLAN.md.
#
# Mirrors race-revisit.sh (pull -> setup -> pre-verify -> build -> run ->
# analyze). EXECUTE it (do NOT source):
#
#   bash scripts/race-multion.sh --tag multion-a1
#
# A bare invocation runs both val_mini scenes x 4 K=3 orderings each x
# {S1, S2, S3}. For the cheap 1-scene S1-vs-S3 micro-smoke that gates the two
# runtime risks (finite distance_to_category; cursor advance + subgoals_found
# populating):
#
#   bash scripts/race-multion.sh --tag multion-micro --scenes wcojb4TFT35 \
#       --n-orderings 2 --settings "1 3"
#
# Critical invariants (same as race-revisit.sh; each cost a re-run before):
#   * --backbone remembr   — omitting it silently uses the 'frontier' stub.
#   * REMEMBR_STRICT=1      — stub fallback CRASHES instead of faking a run.
#   * S1/S2/S3 in SEPARATE processes / out-dirs — the LTM persists in-process.
#   * --target any          — the category filter must not drop K-chain
#                             episodes (object_category is c1 by design).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

# --- defaults ---
SCENES="wcojb4TFT35 TEEsavR23oF"
K="3"
N_ORDERINGS="4"
SEED="7"
FOUND_RADIUS="1.0"
TAG="multion-a1"
SETTINGS="1 2 3"
N_EPISODES=""
TARGET="any"
# K-scaled step budget: the 250-step single-goal default starves a K=3
# mission (multion-micro: 0 sub-goals reached, best min_d2g 4.5 m at cap).
MAX_STEPS="750"

# --- arg parse ---
while [ $# -gt 0 ]; do
  case "$1" in
    --scenes|--scene)  SCENES="$2"; shift 2 ;;
    --k)               K="$2"; shift 2 ;;
    --n-orderings)     N_ORDERINGS="$2"; shift 2 ;;
    --seed)            SEED="$2"; shift 2 ;;
    --found-radius)    FOUND_RADIUS="$2"; shift 2 ;;
    --tag)             TAG="$2"; shift 2 ;;
    --settings)        SETTINGS="$2"; shift 2 ;;
    --n-episodes)      N_EPISODES="$2"; shift 2 ;;
    --target)          TARGET="$2"; shift 2 ;;
    --max-steps)       MAX_STEPS="$2"; shift 2 ;;
    *) echo "FATAL: unknown arg '$1'"; exit 1 ;;
  esac
done
SCENES="${SCENES//,/ }"
SETTINGS="${SETTINGS//,/ }"
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "FATAL: --tag must be alphanumeric/dash/underscore (got '$TAG')"; exit 1; }

VALMINI="data/hm3d/datasets/objectnav/hm3d/v1/val_mini/content"
DS_DIR="data/hm3d/datasets/objectnav/hm3d/v1/multion_${TAG}"
NAME="multion_${TAG}"
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
# Standalone case_*/main() runners (assert-based), run as scripts on purpose.
banner "[3/6] pre-test code verify (multion suites + regression suites)"
for T in test_make_multion_smoke test_advance_subgoal test_analyze_multion \
         test_analyze_revisit test_analyze_ablation test_make_revisit_smoke \
         test_propose_candidates test_spl_guard test_episode_order \
         test_filter_near_candidates test_memory_bridge_consolidate \
         test_diagnose_propose_triggers; do
  python "embodied_memory/scripts/${T}.py" \
    || { echo "FATAL: ${T} failed — not spending on the live run."; exit 1; }
done

# --- 4. build the multion dataset, ALL scenes into one shared dir ---
banner "[4/6] build multion dataset: scenes=[$SCENES] K=$K n-orderings=$N_ORDERINGS seed=$SEED -> $DS_DIR"
rm -rf "$DS_DIR"   # fresh build so a stale content/ from an earlier tag can't inflate n-episodes
for SCENE in $SCENES; do
  SRC="${VALMINI}/${SCENE}.json.gz"
  [ -f "$SRC" ] || { echo "FATAL: source episodes missing: $SRC"; exit 1; }
  python embodied_memory/scripts/make_multion_smoke.py \
      --src "$SRC" --scene "$SCENE" --k "$K" --n-episodes "$N_ORDERINGS" \
      --seed "$SEED" --out-dir "$DS_DIR" \
    || { echo "FATAL: dataset build failed for scene $SCENE."; exit 1; }
done
[ -f "$DS" ] || { echo "FATAL: expected top-level dataset not written: $DS"; exit 1; }

# Default n-episodes = SUM across ALL content/*.json.gz (--scene all loads
# every scene; counting one file would truncate the others).
if [ -z "$N_EPISODES" ]; then
  N_EPISODES="$(python -c "import gzip,json,glob,sys; print(sum(len(json.load(gzip.open(f))['episodes']) for f in sorted(glob.glob(sys.argv[1]))))" "${DS_DIR}/content/*.json.gz")" \
    || { echo "FATAL: could not count dataset episodes."; exit 1; }
  echo "  auto n-episodes = $N_EPISODES (one pass over all built scenes)"
fi
[ "$N_EPISODES" -gt 0 ] 2>/dev/null || { echo "FATAL: episode count is '$N_EPISODES' (<=0 or non-numeric)."; exit 1; }

# --- 5. run the settings in SEPARATE processes (--scene all) ---
OUT_DIRS=""
for S in $SETTINGS; do
  out_dir="runs/${TAG}-s$S"
  banner "[5/6] run: setting=$S backbone=remembr K=$K found-radius=$FOUND_RADIUS max-steps=$MAX_STEPS -> $out_dir"
  REMEMBR_STRICT=1 python -m embodied_memory.run_hm3d_pol --mode live \
      --backbone remembr --setting "$S" --episodes-path "$DS" \
      --scene all --target "$TARGET" --n-episodes "$N_EPISODES" \
      --found-radius "$FOUND_RADIUS" --max-steps "$MAX_STEPS" \
      --out-dir "$out_dir" 2>&1 | tee "${out_dir}.log"
  rc=${PIPESTATUS[0]}
  # Judge completeness by episodes written, not exit code (S1/S2 can't meet
  # the full-system pass_conditions by design — same caveat as race-revisit).
  completed="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['n_episodes_completed'])" "${out_dir}/summary.json" 2>/dev/null || echo 0)"
  if [ "$completed" != "$N_EPISODES" ]; then
    echo "WARN: setting $S completed ${completed}/${N_EPISODES} episodes (exit $rc) — analysis may be partial."
  fi
  # Micro-smoke gate: the two runtime risks are (a) distance_to_category
  # finite at runtime, (b) the cursor actually advancing. Surface both.
  n_adv="$(python -c "
import glob, json, sys
n = sum(len(json.load(open(p)).get('subgoals_found') or [])
        for p in sorted(glob.glob(sys.argv[1] + '/episode_*.json'))
        if not p.endswith('_error.json'))
print(n)" "$out_dir" 2>/dev/null || echo "?")"
  echo "  setting $S: total subgoals_found events = ${n_adv}"
  # Per-episode digest (micro4 gates, readable straight off the emailed log):
  # thrash-gone = rerank << n_steps + n_propose_reached small; within-episode
  # memory = n_memory_candidates > 0 in the SAME episode (micro3: 0).
  echo "  per-episode digest (thrash + within-episode memory):"
  python -c "
import glob, json, sys
for p in sorted(glob.glob(sys.argv[1] + '/episode_*.json')):
    if p.endswith('_error.json'):
        continue
    e = json.load(open(p))
    cats = ','.join(e.get('target_categories') or [str(e.get('target_category'))])
    print('    ep%-3s %-22s rerank=%s/%s reached=%s filt_near=%s '
          'mem_cand=%s mem_chosen=%s adv=%s' % (
        e.get('episode_idx'), cats,
        e.get('rerank_calls'), e.get('n_steps'),
        e.get('n_propose_reached'), e.get('n_candidates_filtered_near'),
        e.get('n_memory_candidates'), e.get('n_memory_chosen'),
        len(e.get('subgoals_found') or [])))" "$out_dir" \
    || echo "  WARN: per-episode digest failed for $out_dir"
  OUT_DIRS="$OUT_DIRS $out_dir"
done

# --- 6. multion analysis (Progress/PPL + gap-by-sub-goal-index) ---
banner "[6/6] multion analysis: analyze_ablation.py --multion$OUT_DIRS"
# shellcheck disable=SC2086
python embodied_memory/scripts/analyze_ablation.py --multion $OUT_DIRS

banner "DONE — paste everything above (esp. per-setting summary, paired PPL, gap-by-index)"
