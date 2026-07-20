#!/bin/bash
# scripts/race-r1-objectnav.sh — R1 / Table 1: backbone credibility.
#
# The direct answer to "44% Find-SR looks weak": plain HM3D ObjectNav on the
# FULL val split, memory OFF, two searcher arms —
#
#   S1   geometric frontier  (raw_score = 0.6*size + 0.4*dist, zero goal-semantics)
#   S1+  BLIP-2 ITM frontier (VLFM-style value map; the searcher VLFM used to reach
#                             HM3D ObjectNav SPL 0.304)
#
# Both arms are --setting 1 (memory off): the "+" is the frontier VALUE SIGNAL,
# nothing else. Comparable to VLFM's SPL 0.304 and the ~0.43 published SOTA.
#
# WHY THIS DRIVER EXISTS SEPARATELY: it touches NONE of the three broken audio
# things (no anomaly, no decoupling, no RIR grid, no feasibility gate — see
# docs/anomaly_response_buildplan_2026-07-16.md), so it can occupy the V100 while
# Phase F lands. It is the only unblocked work.
#
# 7B PLANNER KEPT (unlike race-blip2-frontier.sh, which swaps to Phi to fit the
# L4): this is the V100, where 7B + BLIP-2 co-fit (verify with the VRAM preflight
# FIRST: race-blip2-frontier.sh --tag r1pre --skip-ab --planner Qwen/Qwen2.5-7B-
# Instruct). Keeping the 7B makes R1's absolute SPL cross-quotable to the +0.171/
# +0.24/+0.2505 arc.
#
# VACUOUS-ARM GUARD (the load-bearing check): semantic_frontier=True is stamped on
# every candidate whenever the weight is on, so it cannot show the signal did
# anything. Only SPREAD reorders frontiers. A constant semantic value — 0.0 from an
# unobserved map, or a saturated one from a FLAT scorer (the CLIP 0.020 failure,
# measured 3x) — leaves raw_score a uniform rescale of geom_score, so S1+ ranks
# frontiers exactly like S1 while the run exits 0. check_semantic_arm FATALs on
# that, the same discipline the anomaly driver applies to a vacuous query-expansion
# arm. Without it, Table 1 cannot tell "BLIP-2 is flat" from "BLIP-2 never loaded"
# and would publish the first as the second.
#
# Pipeline: [1] pull [2] setup [3] pre-verify [4] episode count [5] S1 arm
#   [6] S1+ arm [6b] VACUOUS-ARM GATE [7] paired SPL (analyze_ablation).
#
# EXECUTE it (do NOT source):
#   nrun bash scripts/race-r1-objectnav.sh --tag r1v1
#   nrun bash scripts/race-r1-objectnav.sh --tag r1smoke --n-episodes 20   # stage
#   nrun bash scripts/race-r1-objectnav.sh --tag r1v1 --weight 0.4
#
# git pull FIRST (this driver + the semantic counters are new; the driver
# self-pulls at step 1 but that only takes effect on the 2nd invocation).

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT" || exit 1
MINICONDA="${HOME}/miniconda3"; LTM_ENV="ltm-embodied"

# --- defaults ---
TAG=""
SPLIT="val"                                    # full val split (20 scenes)
WEIGHT="0.5"                                   # S1+ LTM_SEMANTIC_FRONTIER blend weight
VALUE_MODEL="Salesforce/blip2-itm-vit-g"       # BLIP-2 ITM checkpoint (VLFM's)
PROMPT="Seems like there is a {goal} ahead."   # VLFM value prompt
N_EPISODES=""                                  # empty => all episodes in the split
MIN_SPREAD="0.01"                              # vacuous-arm floor (below = flat like CLIP)
BLIP2_CPU=""                                   # --blip2-cpu => value model on CPU (OOM hatch)
MAX_STEPS="500"                                # HM3D ObjectNav v1 benchmark budget. run_hm3d_pol
                                               # defaults to 250 (HALF); r1smoke ran at 250 and
                                               # ~50% of episodes were still exploring at the cap,
                                               # so 250 both understates reach and is not cross-
                                               # quotable to VLFM's 0.304 (measured at 500).
ANTISPIN="1"                                   # enable the anti-spin fixes (default ON for R1).
                                               # r1smoke/r1b500 ran with BOTH default-OFF, so the
                                               # follower spun on navmesh-unreachable frontiers
                                               # (n_waypoint_unreachable 440+/ep, n_unreachable_
                                               # escape=0, success 2/30 at BOTH 250 and 500 steps).
                                               # A searcher that cannot route to its own waypoints
                                               # is not a fair baseline. --no-antispin to A/B it.

while [ $# -gt 0 ]; do
  case "$1" in
    --tag)          TAG="$2"; shift 2 ;;
    --split)        SPLIT="$2"; shift 2 ;;
    --weight)       WEIGHT="$2"; shift 2 ;;
    --value-model)  VALUE_MODEL="$2"; shift 2 ;;
    --prompt)       PROMPT="$2"; shift 2 ;;
    --n-episodes)   N_EPISODES="$2"; shift 2 ;;
    --max-steps)    MAX_STEPS="$2"; shift 2 ;;
    --no-antispin)  ANTISPIN=""; shift ;;
    --min-spread)   MIN_SPREAD="$2"; shift 2 ;;
    --blip2-cpu)    BLIP2_CPU=1; shift ;;
    -h|--help)      sed -n '1,48p' "$0"; exit 0 ;;
    *) echo "FATAL: unknown arg '$1'"; exit 1 ;;
  esac
done
[ -z "$TAG" ] && { echo "FATAL: --tag <name> required"; exit 1; }
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "FATAL: --tag must be alnum/dash/underscore"; exit 1; }
case "$SPLIT" in val|val_mini|train) ;; *) echo "FATAL: --split must be val|val_mini|train (got '$SPLIT')"; exit 1 ;; esac

banner() { printf '\n========== %s ==========\n' "$1"; }

banner "[1/7] git pull --ff-only (+ self-heal the 2nd-invocation gotcha)"
[ -x "$MINICONDA/bin/conda" ] || { echo "FATAL: $MINICONDA/bin/conda missing"; exit 1; }
if git rev-parse --git-dir >/dev/null 2>&1; then
  # bash executes the body loaded at launch; `git pull` here updates the file on
  # DISK but not the RUNNING body, so a driver edit only takes effect on the 2nd
  # invocation. That silently wasted a 10h run (r1spin ran the pre-anti-spin body
  # at commit 32b3493, n_unreachable_escape=0). Self-heal: if the pull changed
  # THIS script, re-exec the new body once.
  _self_before="$(md5sum "$0" 2>/dev/null | awk '{print $1}')"
  git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }
  _self_after="$(md5sum "$0" 2>/dev/null | awk '{print $1}')"
  if [ -n "$_self_before" ] && [ "$_self_before" != "$_self_after" ] && [ -z "${_R1_REEXEC:-}" ]; then
    echo ">> driver self-updated on pull ($_self_before -> $_self_after); re-exec'ing the new body once."
    export _R1_REEXEC=1
    exec bash "$0" "$@"
  fi
fi
echo "  running commit: $(git rev-parse --short HEAD 2>/dev/null || echo '?')  antispin=${ANTISPIN:-off}"

banner "[2/7] conda setup (source scripts/race-setup.sh → $LTM_ENV; keeps the 7B planner)"
set +u; source scripts/race-setup.sh || { echo "FATAL: race-setup.sh failed"; exit 1; }; set -u
# race-setup does NOT set PYTHONPATH; every race driver exports it so the
# pre-verify tests (run as `python embodied_memory/scripts/<t>.py` from the repo
# root) can `import embodied_memory`.
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

banner "[3/7] pre-verify (free; abort before spend)"
# The semantic-frontier path end to end: the spread diag on the planner, the
# summary counters it rolls into, and the vacuous-arm verdict the gate below reads.
for t in test_semantic_frontier; do
  python embodied_memory/$t.py \
    || { echo "FATAL: $t failed — not spending on the live run."; exit 1; }
done
for t in test_summary_semantic_frontier test_check_semantic_arm; do
  python embodied_memory/scripts/$t.py \
    || { echo "FATAL: $t failed — not spending on the live run."; exit 1; }
done

banner "[4/7] episode count (split=$SPLIT)"
CONTENT_DIR="data/hm3d/datasets/objectnav/hm3d/v1/${SPLIT}/content"
[ -d "$CONTENT_DIR" ] || { echo "FATAL: content dir missing: $CONTENT_DIR"; exit 1; }
if [ -z "$N_EPISODES" ]; then
  N_EPISODES="$(python -c "import gzip,json,glob,sys
tot=0
for f in sorted(glob.glob(sys.argv[1]+'/*.json.gz')):
    tot += len(json.load(gzip.open(f))['episodes'])
print(tot)" "$CONTENT_DIR")" || { echo "FATAL: could not count episodes"; exit 1; }
fi
[ "$N_EPISODES" -gt 0 ] 2>/dev/null || { echo "FATAL: episode count '$N_EPISODES' <= 0"; exit 1; }
N_SCENES=$(ls "$CONTENT_DIR"/*.json.gz 2>/dev/null | wc -l | tr -d ' ')
echo "  split=$SPLIT scenes=$N_SCENES n_episodes=$N_EPISODES"

banner "[4b/7] MESH PREFLIGHT — every split scene must have a .basis.glb"
# r1v1 burned 3m53s crashing 100/100 episodes because `--split val` references 20
# scenes but only the 2 minival-overlap scenes have meshes (val->minival symlink);
# every missing-mesh episode dies at sim init with ESP_CHECK. Fail fast HERE with
# the fix, before any GPU spend, instead of per-episode crashes → 0 completed →
# FATAL. USABLE_SCENES lets a partial run proceed only when explicitly intended.
INV="$(python embodied_memory/scripts/inventory_hm3d_meshes.py --split "$SPLIT")" \
  || { echo "FATAL: mesh inventory failed"; exit 1; }
echo "$INV"
N_MISSING=$(printf '%s\n' "$INV" | sed -n 's/^split=.*missing=\([0-9]*\).*/\1/p')
if [ "${N_MISSING:-0}" -gt 0 ]; then
  echo "FATAL: $N_MISSING/${N_SCENES} scenes in split '$SPLIT' have NO mesh — R1 needs the"
  echo "  full split, so this is NOT a runnable Table-1 baseline. Download the full mesh"
  echo "  split on the VM (needs the Matterport token in .env):"
  echo "    rm -f data/hm3d/scene_datasets/hm3d/val   # drop the val->minival symlink"
  echo "    HM3D_SCENE_GROUP=hm3d_val_full bash embodied_memory/scripts/download_hm3d.sh"
  echo "  Or run the de-risking smoke on the scenes that DO have meshes:"
  echo "    bash scripts/race-r1-objectnav.sh --tag r1smoke --split val_mini"
  exit 1
fi

# Both arms share these. --setting 1 = memory OFF; --scene all discovers every
# scene in the split; --target any disables the per-episode category filter.
COMMON=(--mode live --backbone remembr --setting 1 --split "$SPLIT"
        --scene all --target any --n-episodes "$N_EPISODES" --max-steps "$MAX_STEPS")
echo "  max_steps=$MAX_STEPS (HM3D ObjectNav benchmark budget; VLFM 0.304 is at 500)"

# Anti-spin: both arms are navigation-quality fixes orthogonal to the frontier-value
# A/B, so they apply to S1 AND S1+. NAVMESH_FRONTIER drops unreachable frontiers at
# proposal; ANTITHRASH_SINGLEGOAL adds the blacklist + snap-escape when the follower
# reports unreachable. Set once here so both arm subprocesses inherit them.
if [ -n "$ANTISPIN" ]; then
  export REMEMBR_ANTITHRASH_SINGLEGOAL=1 REMEMBR_NAVMESH_FRONTIER=1
  echo "  anti-spin ON (REMEMBR_ANTITHRASH_SINGLEGOAL + REMEMBR_NAVMESH_FRONTIER)"
else
  unset REMEMBR_ANTITHRASH_SINGLEGOAL REMEMBR_NAVMESH_FRONTIER
  echo "  anti-spin OFF (legacy spin behaviour — for the A/B against r1b500)"
fi
S1_DIR="runs/${TAG}-s1"
S1PLUS_DIR="runs/${TAG}-s1plus"

banner "[5/7] S1 arm (geometric frontier) -> $S1_DIR"
rm -f "$S1_DIR/summary.json"
REMEMBR_STRICT=1 python -m embodied_memory.run_hm3d_pol "${COMMON[@]}" \
    --semantic-frontier-weight 0.0 --out-dir "$S1_DIR" 2>&1 | tee "${S1_DIR}.log"
s1_done="$(python -c "import json,sys
try:
    s=json.load(open(sys.argv[1])); print(1 if s.get('n_episodes_completed',0)>0 else 0)
except Exception: print(0)" "$S1_DIR/summary.json" 2>/dev/null || echo 0)"
[ "$s1_done" = 1 ] || { echo "FATAL: S1 arm produced no completed episode at $S1_DIR (crash/OOM). See ${S1_DIR}.log."; exit 1; }

banner "[6/7] S1+ arm (BLIP-2 ITM frontier, weight=$WEIGHT) -> $S1PLUS_DIR"
rm -f "$S1PLUS_DIR/summary.json"
LTM_SEMANTIC_FRONTIER_PROMPT="$PROMPT" \
REMEMBR_STRICT=1 python -m embodied_memory.run_hm3d_pol "${COMMON[@]}" \
    --semantic-frontier-weight "$WEIGHT" --semantic-frontier-backend blip2 \
    --value-model "$VALUE_MODEL" ${BLIP2_CPU:+--blip2-cpu} \
    --out-dir "$S1PLUS_DIR" 2>&1 | tee "${S1PLUS_DIR}.log"

banner "[6b/7] VACUOUS-ARM GATE (check_semantic_arm) — did BLIP-2 actually reorder frontiers?"
python embodied_memory/scripts/check_semantic_arm.py "$S1PLUS_DIR/summary.json" \
    --min-spread "$MIN_SPREAD"
gate_rc=$?
if [ "$gate_rc" -ne 0 ]; then
  echo "FATAL: S1+ is VACUOUS (see verdict above) — it is byte-equivalent to S1, so the"
  echo "  A/B is meaningless. Either BLIP-2 never loaded (n_semantic_scored=0), or its"
  echo "  scores are FLAT (spread < $MIN_SPREAD) exactly like the CLIP 0.020 signal. Not"
  echo "  quoting Table 1 on a vacuous arm. Check ${S1PLUS_DIR}.log for the BLIP-2 load line."
  exit 1
fi

banner "[7/7] paired SPL: S1 vs S1+ (analyze_ablation, plain — pairs on scene_id/episode_id)"
python embodied_memory/scripts/analyze_ablation.py "$S1_DIR" "$S1PLUS_DIR" \
    2>&1 | tee "runs/${TAG}-r1-analysis.log"

echo
echo "DONE. R1 / Table 1 for split=$SPLIT ($N_SCENES scenes, $N_EPISODES episodes)."
echo "  S1 (geometric) vs S1+ (BLIP-2 ITM frontier), memory OFF, 7B planner (cross-quotable)."
echo "  Compare mean SPL to VLFM 0.304 / VLingNav 0.429. Analysis: runs/${TAG}-r1-analysis.log"
exit 0
