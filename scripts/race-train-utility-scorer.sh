#!/bin/bash
# scripts/race-train-utility-scorer.sh — train a goal-PROXIMITY head and inject
# it into the U slot (β term), then measure its effect on the revisit soft-SPL
# gain.
#
# WHY: the consolidator keeps only the top-k keyframes by importance
# I = αR + βU + γN; that top-k IS the fine LTM retrieval queries against.
# Retrieval is caption-goal cosine + position->waypoint (verified in
# propose_memory_candidates), so the importance signal that can beat the
# heuristic is "is this frame a GOOD WAYPOINT" = was it taken close to the goal.
# That is orthogonal to novelty (N), goal-mention (R/scorer-d3), and the
# forward-model surprise U (Runs 18 + e2, which regressed). This driver trains a
# scalar head (caption emb -> [0,1]) on a per-keyframe goal_proximity label
# (1 iff the frame was within GOAL_PROX_RADIUS_M of the goal) and wires it into
# the U slot (--utility-scorer-ckpt), then runs a 3-cell head-to-head:
#
#   S1            (memory off)              -> shared reference
#   S3 heuristic  (default U)               -> the reproduced baseline
#   S3 utility-U  (goal-proximity head)     -> the experiment
#
# NOTE: the proximity LABEL is per-step geodesic distance_to_goal, logged by the
# runner (episode_runner._serialize_step). Pre-Tier-3 run dirs do NOT carry it,
# so training data must be REGENERATED — --train-runs reuse is guarded by a
# distance-presence preflight.
#
# EXECUTE it (do NOT source):
#   bash scripts/race-train-utility-scorer.sh --tag utilscorer-p1 \
#       --dataset data/hm3d/datasets/objectnav/hm3d/v1/revisit_predictor-e1/revisit_predictor-e1.json.gz \
#       --reuse-baselines "runs/predictor-e1-s1 runs/predictor-e1-s3-heur"
#
# Critical invariants (mirror race-train-predictor.sh — each cost a re-run):
#   * --backbone remembr + REMEMBR_STRICT=1   (no silent 'frontier'/stub run)
#   * --encoder sbert for training            (MUST match the bridge's SBERT
#                                              caption index; the bridge RAISES
#                                              on a dim mismatch)
#   * each setting in a SEPARATE process       (LTM persists within a process)
#   * training data REGENERATED                (the goal_proximity label needs
#                                              per-step distance_to_goal)
#   * --dataset + --reuse-baselines together   (baseline cells are only paired
#                                              if the eval episodes are the
#                                              EXACT ones they ran on)

set -uo pipefail

# --- self-update guard: re-exec once after pull so a mid-run patch to THIS
# script can't run a half-old / half-new body. ---
if [ -z "${TRAIN_UTILSCORER_RELAUNCHED:-}" ]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }
  echo "========== [0/8] git pull --ff-only (pre-exec) =========="
  git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }
  export TRAIN_UTILSCORER_RELAUNCHED=1
  exec bash "$0" "$@"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

# --- defaults ---
SCENES="wcojb4TFT35 TEEsavR23oF"
CATS="chair bed sofa toilet tv_monitor plant"   # the Run-14 wide matrix
NWARM="3"
TAG="utilscorer-p1"
N_EPISODES=""          # eval: auto = one pass over the eval dataset
TARGET="any"
TRAIN_RUNS=""          # if set, reuse these run dirs (MUST carry distance_to_goal)
N_TRAIN="30"           # episodes to generate for training when TRAIN_RUNS empty
EPOCHS="8"
PROX_RADIUS="1.0"      # goal_proximity label radius (m); env GOAL_PROX_RADIUS_M
DATASET=""             # if set, reuse this revisit .json.gz (skip the build)
REUSE_BASELINES=""     # "S1DIR S3HEURDIR": skip re-running S1 + S3-heuristic
DEGENERATE_MAX="0.5"   # abort if > this fraction of train captions degenerate

while [ $# -gt 0 ]; do
  case "$1" in
    --scenes|--scene)    SCENES="$2"; shift 2 ;;
    --categories|--cats) CATS="$2"; shift 2 ;;
    --n-warm)            NWARM="$2"; shift 2 ;;
    --tag)               TAG="$2"; shift 2 ;;
    --n-episodes)        N_EPISODES="$2"; shift 2 ;;
    --target)            TARGET="$2"; shift 2 ;;
    --train-runs)        TRAIN_RUNS="$2"; shift 2 ;;
    --n-train)           N_TRAIN="$2"; shift 2 ;;
    --epochs)            EPOCHS="$2"; shift 2 ;;
    --prox-radius)       PROX_RADIUS="$2"; shift 2 ;;
    --dataset)           DATASET="$2"; shift 2 ;;
    --reuse-baselines)   REUSE_BASELINES="$2"; shift 2 ;;
    *) echo "FATAL: unknown arg '$1'"; exit 1 ;;
  esac
done
CATS="${CATS//,/ }"
SCENES="${SCENES//,/ }"
TRAIN_RUNS="${TRAIN_RUNS//,/ }"
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "FATAL: --tag must be alphanumeric/dash/underscore (got '$TAG')"; exit 1; }
if [ -n "$REUSE_BASELINES" ] && [ -z "$DATASET" ]; then
  echo "FATAL: --reuse-baselines requires --dataset (the baseline cells are only"
  echo "       paired if the eval episodes are the EXACT ones they ran on)."
  exit 1
fi
export GOAL_PROX_RADIUS_M="$PROX_RADIUS"

VALMINI="data/hm3d/datasets/objectnav/hm3d/v1/val_mini/content"
CKPT="models/embodied/utility-scorer-${TAG}.pt"

banner() { printf '\n========== %s ==========\n' "$1"; }

# --- 1. conda setup ---
banner "[1/8] conda setup (source scripts/race-setup.sh)"
# shellcheck disable=SC1091
source scripts/race-setup.sh || { echo "FATAL: race-setup.sh failed"; exit 1; }

# --- 2. pre-test code verify (free; aborts before any paid run) ---
banner "[2/8] pre-test code verify (predictor/utility wiring + revisit + analyzer suites)"
for T in test_predictor_wiring test_scorer_wiring test_analyze_revisit \
         test_make_revisit_smoke test_analyze_ablation; do
  python "embodied_memory/scripts/${T}.py" \
    || { echo "FATAL: ${T} failed — not spending on the live run."; exit 1; }
done

# --- 3. eval dataset: reuse --dataset, else build a fresh revisit set ---
if [ -n "$DATASET" ]; then
  DS="$DATASET"
  DS_DIR="$(dirname "$DS")"
  banner "[3/8] eval dataset: reuse $DS (no rebuild)"
  [ -f "$DS" ] || { echo "FATAL: --dataset not found: $DS"; exit 1; }
else
  DS_DIR="data/hm3d/datasets/objectnav/hm3d/v1/revisit_${TAG}"
  DS="${DS_DIR}/revisit_${TAG}.json.gz"
  banner "[3/8] build revisit eval dataset: scenes=[$SCENES] cats=[$CATS] n-warm=$NWARM"
  rm -rf "$DS_DIR"
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
fi
if [ -z "$N_EPISODES" ]; then
  N_EPISODES="$(python -c "import gzip,json,glob,sys; print(sum(len(json.load(gzip.open(f))['episodes']) for f in sorted(glob.glob(sys.argv[1]))))" "${DS_DIR}/content/*.json.gz")" \
    || { echo "FATAL: could not count dataset episodes."; exit 1; }
  echo "  auto eval n-episodes = $N_EPISODES"
fi
[ "$N_EPISODES" -gt 0 ] 2>/dev/null || { echo "FATAL: eval episode count '$N_EPISODES' <=0."; exit 1; }

# --- 4. training data: regenerate (the goal_proximity label needs per-step
#        distance_to_goal). --train-runs reuse is allowed but distance-checked. ---
if [ -n "$TRAIN_RUNS" ]; then
  banner "[4/8] training data: reuse $TRAIN_RUNS (will verify distance_to_goal present)"
  for d in $TRAIN_RUNS; do
    [ -d "$d" ] || { echo "FATAL: --train-runs dir missing: $d"; exit 1; }
  done
else
  TRAIN_RUNS="runs/${TAG}-train"
  banner "[4/8] generate training data: val_mini single-goal S3 x $N_TRAIN -> $TRAIN_RUNS"
  REMEMBR_STRICT=1 python -m embodied_memory.run_hm3d_pol --mode live \
      --backbone remembr --setting 3 --scene all --target any \
      --n-episodes "$N_TRAIN" --out-dir "$TRAIN_RUNS" 2>&1 | tee "${TRAIN_RUNS}.log"
  tc="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['n_episodes_completed'])" "${TRAIN_RUNS}/summary.json" 2>/dev/null || echo 0)"
  [ "$tc" -gt 0 ] 2>/dev/null || { echo "FATAL: training run wrote 0 episodes."; exit 1; }
  echo "  training run wrote $tc episodes"
fi

# --- 5a. caption-quality preflight (degenerate captions => a head of nothing) ---
banner "[5/8] caption-quality preflight on training data (max degenerate frac=$DEGENERATE_MAX)"
# shellcheck disable=SC2086
python - "$DEGENERATE_MAX" $TRAIN_RUNS <<'PY' || { echo "FATAL: caption preflight failed/aborted."; exit 1; }
import glob, json, os, sys
max_frac = float(sys.argv[1]); run_dirs = sys.argv[2:]
total = degen = 0
samples = []
for d in run_dirs:
    for f in sorted(glob.glob(os.path.join(d, "episode_*.json"))):
        try:
            ep = json.load(open(f))
        except Exception:
            continue
        for st in ep.get("steps", []):
            cap = (st.get("caption") or "").strip().lower()
            if not cap:
                continue
            total += 1
            if "room interior" in cap or cap.endswith("searching for"):
                degen += 1
            elif len(samples) < 5:
                samples.append(cap[:90])
if total == 0:
    print("  ABORT: no captions found in training runs."); sys.exit(1)
frac = degen / total
print(f"  captions: {total} total, {degen} degenerate ({frac:.1%})")
for s in samples:
    print(f"    sample: {s!r}")
if frac > max_frac:
    print(f"  ABORT: degenerate fraction {frac:.1%} > {max_frac:.0%} — captions are the "
          f"semantic-sensor fallback, not rich VLM output. Regenerate with --backbone remembr.")
    sys.exit(1)
print("  OK: captions are discriminative enough to train on.")
PY

# --- 5b. goal_proximity label preflight (distance present + has positives) ---
banner "[5/8] goal_proximity label preflight (radius=${PROX_RADIUS}m)"
# shellcheck disable=SC2086
python - "$PROX_RADIUS" $TRAIN_RUNS <<'PY' || { echo "FATAL: goal_proximity preflight failed/aborted."; exit 1; }
import glob, json, os, sys
radius = float(sys.argv[1]); run_dirs = sys.argv[2:]
total = with_dist = pos = 0
for d in run_dirs:
    for f in sorted(glob.glob(os.path.join(d, "episode_*.json"))):
        try:
            ep = json.load(open(f))
        except Exception:
            continue
        for st in ep.get("steps", []):
            if not (st.get("caption") or "").strip():
                continue
            total += 1
            dd = st.get("distance_to_goal")
            if dd is not None:
                with_dist += 1
                if float(dd) <= radius:
                    pos += 1
if total == 0:
    print("  ABORT: no captioned steps in training data."); sys.exit(1)
if with_dist == 0:
    print("  ABORT: no per-step distance_to_goal in training data — these are "
          "pre-Tier-3 logs. Regenerate (omit --train-runs) so the runner logs it.")
    sys.exit(1)
frac = pos / with_dist
print(f"  steps={total}  with_distance={with_dist}  positives(<= {radius}m)={pos} ({frac:.1%})")
if pos == 0:
    print(f"  ABORT: 0 positive labels within {radius}m — the goal_proximity head "
          f"would collapse. Raise --prox-radius / GOAL_PROX_RADIUS_M.")
    sys.exit(1)
print("  OK: goal_proximity label has positives.")
PY

# --- 6. train the goal-proximity head (SBERT space, per-keyframe binary) ---
banner "[6/8] train utility-scorer -> $CKPT (encoder=sbert label=goal_proximity epochs=$EPOCHS radius=${PROX_RADIUS}m)"
mkdir -p "$(dirname "$CKPT")"
# shellcheck disable=SC2086
python -m dialogue_memory.train_scorer --embodied $TRAIN_RUNS \
    --encoder sbert --label-mode goal_proximity --epochs "$EPOCHS" \
    --out "$CKPT" \
  || { echo "FATAL: utility-scorer training failed."; exit 1; }
[ -f "$CKPT" ] || { echo "FATAL: checkpoint not written: $CKPT"; exit 1; }

# --- 7. eval 3 cells on the revisit dataset (separate processes) ---
run_cell() {  # $1=setting  $2=out_dir  $3=extra args
  local S="$1" out="$2" extra="$3"
  banner "[7/8] eval: setting=$S -> $out ${extra}"
  # shellcheck disable=SC2086
  REMEMBR_STRICT=1 python -m embodied_memory.run_hm3d_pol --mode live \
      --backbone remembr --setting "$S" --episodes-path "$DS" \
      --scene all --target "$TARGET" --n-episodes "$N_EPISODES" \
      --out-dir "$out" $extra 2>&1 | tee "${out}.log"
  local c
  c="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['n_episodes_completed'])" "${out}/summary.json" 2>/dev/null || echo 0)"
  [ "$c" = "$N_EPISODES" ] || echo "WARN: setting $S completed ${c}/${N_EPISODES} episodes."
}
S3U="runs/${TAG}-s3-utility-u"
if [ -n "$REUSE_BASELINES" ]; then
  # S1 + S3-heuristic are U-head-independent; reuse a prior tag's cells to skip
  # ~2/3 of the GPU cost (only valid with the SAME --dataset).
  # shellcheck disable=SC2086
  set -- $REUSE_BASELINES
  S1="$1"; S3H="$2"
  [ -d "$S1" ] && [ -d "$S3H" ] || { echo "FATAL: --reuse-baselines dirs missing: '$S1' '$S3H'"; exit 1; }
  banner "[7/8] reuse baselines: S1=$S1  S3-heur=$S3H (skipping their re-run)"
else
  S1="runs/${TAG}-s1"
  S3H="runs/${TAG}-s3-heur"
  run_cell 1 "$S1" ""
  run_cell 3 "$S3H" ""
fi
run_cell 3 "$S3U" "--utility-scorer-ckpt $CKPT"

# --- 8. head-to-head Gate-A analysis (two calls; analyzer keys by setting) ---
banner "[8/8a] BASELINE (heuristic U): analyze_ablation --revisit $S1 $S3H"
python embodied_memory/scripts/analyze_ablation.py --revisit "$S1" "$S3H"
banner "[8/8b] GOAL-PROXIMITY U head: analyze_ablation --revisit $S1 $S3U"
python embodied_memory/scripts/analyze_ablation.py --revisit "$S1" "$S3U"

banner "DONE — compare warm soft-SPL S3−S1 across the two blocks above.
  baseline (heuristic U, wide matrix) ~ +0.115 p=0.005 (Run 14).
  goal-proximity U head: did the warm S3−S1 delta rise, tie, or regress?
  (forward-model U regressed to +0.061 — Runs 18/e2.)"
