#!/bin/bash
# scripts/race-all-levers.sh — run all 3 round-3 lever re-runs in ONE batch.
#
# Sequentially invokes the three existing, tested child drivers (NOT a
# reimplementation), each in its OWN process so conda activation, the real-model
# GPU allocation, and the LTM all live and die per-child — the next lever starts
# with a clean GPU. A failed lever does NOT abort the batch (so a crash in one
# still yields the other two verdicts); each rc + duration is captured and a final
# summary is printed. Exit code = number of HARD-failed levers (0 = all clean),
# so the nrun email subject is meaningful.
#
#   L1  caption-rerank A/B   -> scripts/race-audiogoal-matrix.sh --caption-rerank
#   L2  changed-world        -> scripts/race-changed-world.sh
#   L3  OWLv2 detector       -> scripts/race-owlv2-detector.sh  (swaps to a small
#       planner internally + runs OWLv2 on cuda; --owl-planner overrides the model.
#       L1/L2 stay on the published 7B — they source race-setup.sh fresh.)
#
# git pull FIRST (this is a new file; the child drivers self-pull at their step 1).
#
#   nrun bash scripts/race-all-levers.sh                              # all 3, defaults
#   nrun bash scripts/race-all-levers.sh --tag r3 --owl-thresh 0.05
#   nrun bash scripts/race-all-levers.sh --skip-l3                    # just L1 + L2
#   nrun bash scripts/race-all-levers.sh --only-l2                    # one lever
#   nrun bash scripts/race-all-levers.sh --only-l3 --owl-planner Qwen/Qwen2.5-3B-Instruct
#
# Runtime: L1 is the long pole (~4-5 h, S3-only over 6 cells, REUSES the cached
# baseline m3-* matrix — that must exist); L2 ~30 m; L3 a few minutes.
#
# EXECUTE it (do NOT source) — each child activates conda in its own process.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

TAG="r3"
OWL_THRESH="0.1"          # DETECTOR_OWL_SCORE_THRESH for L3; drop to 0.05 if OWLv2 localizes 0x
OWL_PLANNER=""            # optional: override L3's planner (--planner passthrough); "" => L3 default (small)
CELLS="baby_cry:bed alarm:toilet glass_break:chair"   # L1 cells (match the cached m3-* baseline)
RUN_L1=1; RUN_L2=1; RUN_L3=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)         TAG="$2"; shift 2 ;;
    --owl-thresh)  OWL_THRESH="$2"; shift 2 ;;
    --owl-planner) OWL_PLANNER="$2"; shift 2 ;;
    --cells)       CELLS="$2"; shift 2 ;;
    --skip-l1)     RUN_L1=0; shift ;;
    --skip-l2)     RUN_L2=0; shift ;;
    --skip-l3)     RUN_L3=0; shift ;;
    --only-l1)     RUN_L1=1; RUN_L2=0; RUN_L3=0; shift ;;
    --only-l2)     RUN_L1=0; RUN_L2=1; RUN_L3=0; shift ;;
    --only-l3)     RUN_L1=0; RUN_L2=0; RUN_L3=1; shift ;;
    -h|--help)     sed -n '1,30p' "$0"; exit 0 ;;
    *) echo "FATAL: unknown arg '$1'"; exit 1 ;;
  esac
done
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "FATAL: --tag must be alnum/dash/underscore"; exit 1; }

banner() { printf '\n############################## %s ##############################\n' "$1"; }

declare -a NAMES STATUS
run_lever() {   # $1=label  $2..=command (with args)
  local label="$1"; shift
  banner "LEVER ${label}  ::  $*"
  local t0 t1 rc
  t0=$SECONDS
  "$@"; rc=$?
  t1=$SECONDS
  NAMES+=("$label"); STATUS+=("rc=${rc}  ($(( (t1 - t0) / 60 ))m $(( (t1 - t0) % 60 ))s)")
  if [ "$rc" -eq 0 ]; then
    echo ">>> LEVER ${label} DONE (rc=0)"
  else
    echo ">>> LEVER ${label} FAILED (rc=${rc}) — continuing to the next lever"
  fi
  return 0   # never abort the batch on a single lever
}

[ "$RUN_L1" -eq 1 ] && run_lever "L1-caption-rerank" \
  bash scripts/race-audiogoal-matrix.sh --caption-rerank --cells "$CELLS"
[ "$RUN_L2" -eq 1 ] && run_lever "L2-changed-world" \
  bash scripts/race-changed-world.sh --tag "cw-${TAG}"
# L3 swaps to a small planner internally (frees VRAM for OWLv2 on cuda); --owl-planner
# overrides that. L1/L2 are untouched — they source race-setup.sh fresh => 7B default.
# Explicit if/else (not an empty array passthrough) to stay safe under `set -u`.
if [ "$RUN_L3" -eq 1 ]; then
  if [ -n "$OWL_PLANNER" ]; then
    run_lever "L3-owlv2-detector" \
      bash scripts/race-owlv2-detector.sh --tag "owlv2-${TAG}" --owl-thresh "$OWL_THRESH" --planner "$OWL_PLANNER"
  else
    run_lever "L3-owlv2-detector" \
      bash scripts/race-owlv2-detector.sh --tag "owlv2-${TAG}" --owl-thresh "$OWL_THRESH"
  fi
fi

banner "ALL-LEVERS SUMMARY"
[ "${#NAMES[@]}" -eq 0 ] && { echo "  (no levers selected)"; exit 0; }
fails=0
for i in "${!NAMES[@]}"; do
  s="${STATUS[$i]}"
  printf '  %-20s %s\n' "${NAMES[$i]}" "$s"
  [[ "$s" == rc=0* ]] || fails=$((fails + 1))
done
echo
echo "  Verdict logs to read:"
echo "    L1  runs/m3c-caption-rerank-compare.log      (warm B-A; expect tie + mem_chosen back ~271)"
echo "    L2  runs/cw-${TAG}-cw-ablation.log           (Q1 warm S3-S1: does stale memory hurt?)"
echo "    L2  runs/cw-${TAG}-cw-temporal-compare.log   (Q2 warm B-A: does recency mitigate?)"
echo "    L3  race-owlv2-detector tail above           (n_detector_called>0, no OOM = preflight passed)"
echo
if [ "$fails" -eq 0 ]; then
  echo "DONE — all requested levers ran clean (rc=0)."
else
  echo "DONE — ${fails} lever(s) HARD-FAILED (crash/preflight/missing-data; soft pass-condition misses are rc=0)."
fi
exit "$fails"
