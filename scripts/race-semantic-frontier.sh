#!/bin/bash
# scripts/race-semantic-frontier.sh — the LTM_SEMANTIC_FRONTIER lever: gate first,
# then a conditional S1/S3 A/B.
#
# WHY: the absolute soft-SPL ceiling (~0.39) is path-length-bound (L_b) — the
# memory-OFF agent wanders because frontier scoring is PURELY GEOMETRIC
# (frontier_planner.py raw_score = 0.6*size + 0.4*dist, zero goal-semantics). The
# lever blends a VLFM-style CLIP goal-value map into the frontier raw_score so
# exploration biases toward goal-affording regions. The SAME experiment also tests
# the deep risk that the +0.2505 delta is partly a WEAK-BASELINE artifact: a
# semantic frontier is the stronger baseline a reviewer would demand.
#
# GATE FIRST ($0-of-matrix): does CLIP cos(image, "a photo of a {goal}") even
# DISCRIMINATE goal-facing views on HM3D sim renders? Prior work measured CLIP
# FLAT here (0.25 vs 0.228; OWLv2 noise-floor), so a HOLD is plausible — and a HOLD
# is PROTECTIVE (no cheap semantic explorer beats the geometric baseline => the
# +0.2505 stands). Only on GATE=GO do we spend the A/B.
#
# Pipeline: [1] pull [2] setup [3] pre-verify (unit tests) [4] GATE (CLIP
# separation) [5] (GO only) S1/S3 A/B: baseline (semantic OFF) vs semantic ON, on
# the category gradient extremes (toilet = predicted biggest shrink, chair =
# smallest), 7B backbone (cross-quotable). Decision rule printed at the end.
#
#   cd ~/ltm && git checkout lifelong-revisit-eval && git pull
#   nrun bash scripts/race-semantic-frontier.sh                       # gate (+ A/B if GO)
#   nrun bash scripts/race-semantic-frontier.sh --skip-ab             # gate only
#   nrun bash scripts/race-semantic-frontier.sh --weight 0.4 --margin 0.05
#
# EXECUTE (do NOT source).

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT" || exit 1

WEIGHT="0.5"                                   # LTM_SEMANTIC_FRONTIER blend weight
CEILING="0.45"                                 # LTM_SEMANTIC_FRONTIER_CEILING renorm
MARGIN="0.05"                                  # min CLIP separation for GATE=GO
GATE_SCENES="wcojb4TFT35 TEEsavR23oF"
GATE_CATS="chair bed sofa toilet"
NVP=6
AB_SCENES="wcojb4TFT35 TEEsavR23oF"
AB_CATS="toilet chair"                         # gradient extremes (biggest/smallest shrink)
NWARM=3
TAG="semfront"
RUN_AB=1
while [ $# -gt 0 ]; do
  case "$1" in
    --weight)       WEIGHT="$2"; shift 2 ;;
    --ceiling)      CEILING="$2"; shift 2 ;;
    --margin)       MARGIN="$2"; shift 2 ;;
    --gate-scenes)  GATE_SCENES="$2"; shift 2 ;;
    --gate-cats)    GATE_CATS="$2"; shift 2 ;;
    --n-viewpoints) NVP="$2"; shift 2 ;;
    --ab-scenes)    AB_SCENES="$2"; shift 2 ;;
    --ab-cats)      AB_CATS="$2"; shift 2 ;;
    --n-warm)       NWARM="$2"; shift 2 ;;
    --tag)          TAG="$2"; shift 2 ;;
    --skip-ab)      RUN_AB=0; shift ;;
    -h|--help)      sed -n '1,40p' "$0"; exit 0 ;;
    *) echo "FATAL: unknown arg $1"; exit 1 ;;
  esac
done
OUT_DIR="runs/${TAG}"; GATE_LOG="${OUT_DIR}/gate.log"
banner() { printf '\n########## %s ##########\n' "$1"; }

banner "[1/5] git pull --ff-only"
if [ -n "${RACE_SKIP_PULL:-}" ]; then echo "  RACE_SKIP_PULL set — skipping"; else
  git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }
fi

banner "[2/5] conda setup (source race-setup.sh -> ltm-embodied)"
set +u; source scripts/race-setup.sh || { echo "FATAL: race-setup.sh failed"; exit 1; }; set -u
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
mkdir -p "$OUT_DIR"

banner "[3/5] pre-verify (free unit tests)"
python embodied_memory/test_semantic_frontier.py \
  || { echo "FATAL: test_semantic_frontier failed."; exit 1; }
python embodied_memory/scripts/test_diagnose_clip_frontier_separation.py \
  || { echo "FATAL: test_diagnose_clip_frontier_separation failed."; exit 1; }

banner "[4/5] CLIP-separation GATE (does CLIP discriminate goal-facing views here?)"
# shellcheck disable=SC2086
python embodied_memory/scripts/diagnose_clip_frontier_separation.py \
    --scenes $GATE_SCENES --categories $GATE_CATS --n-viewpoints "$NVP" \
    --margin "$MARGIN" --out "${OUT_DIR}/clip_separation.json" 2>&1 | tee "$GATE_LOG"
grc=${PIPESTATUS[0]}
RESULT="$(grep -oE 'GATE_RESULT=[A-Z]+' "$GATE_LOG" | tail -1 | cut -d= -f2)"
[ "$grc" -eq 0 ] && [ -n "$RESULT" ] \
  || { echo "FATAL: gate failed (rc=$grc) or no GATE_RESULT marker — see $GATE_LOG"; exit 1; }

if [ "$RESULT" != "GO" ]; then
  banner "VERDICT: GATE=$RESULT — STOP (no A/B)"
  if [ "$RESULT" = "HOLD" ]; then
    echo "  HOLD -> CLIP cosine is non-discriminative on this renderer (matches the"
    echo "          project's twice-measured flatness). The cheap semantic-frontier"
    echo "          lever is DEAD on arrival — do NOT build further. This is PROTECTIVE:"
    echo "          no cheap semantic explorer beats the geometric baseline, so the"
    echo "          +0.2505 headline stands against the 'weak-baseline' reviewer attack."
  else
    echo "  INSUFFICIENT -> thin/NaN separation; check the render + $GATE_LOG."
  fi
  echo "  gate: $GATE_LOG   data: ${OUT_DIR}/clip_separation.json"
  exit 0
fi

banner "GATE=GO — CLIP discriminates; running the S1/S3 A/B (semantic OFF vs ON)"
if [ "$RUN_AB" -eq 0 ]; then
  echo "  --skip-ab set: gate GREEN but A/B skipped by request."; exit 0
fi
export LTM_SEMANTIC_FRONTIER_CEILING="$CEILING"

banner "[5/5] A/B — arm A: baseline (semantic OFF), arm B: semantic ON (weight=$WEIGHT)"
# Arm A: geometric baseline (LTM_SEMANTIC_FRONTIER unset -> weight 0.0).
# shellcheck disable=SC2086
LTM_SEMANTIC_FRONTIER=0 bash scripts/race-revisit.sh --backbone remembr \
    --settings "1 3" --scenes "$AB_SCENES" --categories "$AB_CATS" \
    --n-warm "$NWARM" --tag "${TAG}-base" 2>&1 | tee "${OUT_DIR}/ab_base.log"
arc=${PIPESTATUS[0]}
# Arm B: semantic frontier ON.
# shellcheck disable=SC2086
LTM_SEMANTIC_FRONTIER="$WEIGHT" bash scripts/race-revisit.sh --backbone remembr \
    --settings "1 3" --scenes "$AB_SCENES" --categories "$AB_CATS" \
    --n-warm "$NWARM" --tag "${TAG}-on" 2>&1 | tee "${OUT_DIR}/ab_on.log"
brc=${PIPESTATUS[0]}

banner "VERDICT (read both arms): GATE=GO, A/B complete (base rc=$arc, on rc=$brc)"
cat <<EOF
  DECISION RULE (pre-registered):
    (i)  S1(on) soft-SPL > S1(base)  -> the semantic value map cuts L_b (exploration
         improves). If flat -> the value map doesn't help here; STOP.
    (ii) warm S3-S1(on) still > +0.12 AND n_memory_chosen(on) >> 0 -> the +0.2505
         delta SURVIVES a stronger explorer (publishable strength) + higher absolute.
         If S3-S1 collapses -> the delta was partly a weak-baseline artifact; REFRAME
         the paper to "memory adds on top of semantic exploration (delta +X)".
  Compare the per-arm summaries:
    base S1/S3:  runs/${TAG}-base-*/   (semantic OFF)
    on   S1/S3:  runs/${TAG}-on-*/     (semantic ON, weight=$WEIGHT, ceiling=$CEILING)
  Check n_memory_chosen in the S3 summaries stays >> 0 (renorm guardrail held).
  gate: $GATE_LOG
EOF
exit 0
