#!/bin/bash
# scripts/race-caprl-gate.sh — Phase 0: the $0-MATRIX captioner-swap GATE.
#
# Decides — BEFORE any GPU ablation — whether swapping the keyframe captioner
# Qwen2-VL-2B -> internlm/CapRL-3B is worth a full revisit matrix. It renders
# HM3D keyframes labeled by physical instance, captions each with BOTH models,
# and measures within- vs between-instance SBERT separation per captioner:
#   GATE=GO   -> CapRL widens the instance separation -> the caption WAS the
#               bottleneck; proceed to Phase 1 (fit-smoke) + Phase 2 (held A/B).
#   GATE=HOLD -> no widening -> the ceiling is the embedding/query, not the
#               caption; do NOT spend the matrix; pivot to a retriever fix.
#
# Runs ENTIRELY in ltm-embodied (habitat_sim render + transformers captioners +
# SBERT) — NO soundspaces env switch. The captioning is a CHEAP one-off pass over
# a few hundred frames (two ~2-3 GB VLMs, sequential), not a 50 h ablation.
#
# Pipeline: [1] pull  [2] setup  [3] pre-verify  [4] build corpus (render +
# dual-caption)  [5] GATE (diagnose_sbert_cosines --compare-captions).
#
#   cd ~/ltm && git checkout lifelong-revisit-eval && git pull
#   nrun bash scripts/race-caprl-gate.sh                         # val_mini default
#   nrun bash scripts/race-caprl-gate.sh --scenes "wcojb4TFT35" --n-viewpoints 8
#
# EXECUTE (do NOT source).

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT" || exit 1

SCENES="wcojb4TFT35 TEEsavR23oF"
CATEGORIES="chair bed sofa toilet"
NVP=6; SPLIT="val_mini"; TAG="caprl-gate"
BASELINE="qwen2-vl-2b"; BASELINE_MODEL="Qwen/Qwen2-VL-2B-Instruct"
CANDIDATE="caprl-3b";   CANDIDATE_MODEL="internlm/CapRL-3B"
MARGIN="0.02"
while [ $# -gt 0 ]; do
  case "$1" in
    --scenes) SCENES="$2"; shift 2 ;;
    --categories) CATEGORIES="$2"; shift 2 ;;
    --n-viewpoints) NVP="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --candidate-model) CANDIDATE_MODEL="$2"; shift 2 ;;
    --margin) MARGIN="$2"; shift 2 ;;
    *) echo "FATAL: unknown arg $1"; exit 1 ;;
  esac
done
OUT_DIR="runs/${TAG}"; CAPS="${OUT_DIR}/captions.json"; GATE_LOG="${OUT_DIR}/gate.log"
banner() { printf '\n########## %s ##########\n' "$1"; }

banner "[1/5] git pull --ff-only"
if [ -n "${RACE_SKIP_PULL:-}" ]; then echo "  RACE_SKIP_PULL set — skipping"; else
  git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }
fi

banner "[2/5] conda setup (source race-setup.sh -> ltm-embodied)"
set +u; source scripts/race-setup.sh || { echo "FATAL: race-setup.sh failed"; exit 1; }; set -u
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
mkdir -p "$OUT_DIR"

banner "[3/5] pre-verify (free; abort before spend)"
for t in test_build_instance_caption_corpus test_diagnose_sbert_cosines; do
  python embodied_memory/scripts/$t.py \
    || { echo "FATAL: $t failed — not spending."; exit 1; }
done

banner "[4/5] build instance-caption corpus (render + caption with BOTH models)"
echo "  scenes=[$SCENES] categories=[$CATEGORIES] n_viewpoints=$NVP"
echo "  baseline=$BASELINE ($BASELINE_MODEL)  candidate=$CANDIDATE ($CANDIDATE_MODEL)"
# shellcheck disable=SC2086
python embodied_memory/scripts/build_instance_caption_corpus.py \
    --scenes $SCENES --categories $CATEGORIES --n-viewpoints "$NVP" --split "$SPLIT" \
    --captioners "${BASELINE}=${BASELINE_MODEL}" "${CANDIDATE}=${CANDIDATE_MODEL}" \
    --out "$CAPS" 2>&1 | tee "${OUT_DIR}/build.log"
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] && [ -f "$CAPS" ] \
  || { echo "FATAL: corpus build failed (rc=$rc); see ${OUT_DIR}/build.log. If the CANDIDATE model failed to load (stub/OOM), that is itself a NO-GO signal."; exit 1; }

banner "[5/5] GATE — within-vs-between instance separation, CapRL vs Qwen"
python embodied_memory/scripts/diagnose_sbert_cosines.py \
    --compare-captions "$CAPS" --baseline "$BASELINE" --candidate "$CANDIDATE" \
    --margin "$MARGIN" 2>&1 | tee "$GATE_LOG"
drc=${PIPESTATUS[0]}
# Read the machine-readable marker (GATE_RESULT=GO|HOLD|INSUFFICIENT), NOT the prose.
RESULT="$(grep -oE 'GATE_RESULT=[A-Z]+' "$GATE_LOG" | tail -1 | cut -d= -f2)"
[ "$drc" -eq 0 ] && [ -n "$RESULT" ] \
  || { echo "FATAL: gate diagnostic failed (rc=$drc) or emitted no GATE_RESULT marker — see $GATE_LOG"; exit 1; }

echo
echo "########## PHASE-0 VERDICT: $RESULT ##########"
case "$RESULT" in
  GO)
    echo "  GO  -> CapRL widens instance separation. Proceed to Phase 1:"
    echo "        REMEMBR_CAPTIONER_MODEL=$CANDIDATE_MODEL bash scripts/race-planner-fit-smoke.sh --scene wcojb4TFT35 --category chair"
    echo "        then a held A/B via scripts/race-revisit.sh (CapRL arm vs the +0.2505 baseline)." ;;
  HOLD)
    echo "  HOLD -> CapRL did NOT widen separation. The ceiling is the embedding/query,"
    echo "          NOT the caption. Do NOT run the matrix; pivot to a read-side query /"
    echo "          instance-aware retriever fix. (The 6th instance lever staying closed —"
    echo "          exactly what the \$0 gate is meant to catch.)" ;;
  INSUFFICIENT)
    echo "  INSUFFICIENT -> the corpus is too thin OR a captioner emitted no captions"
    echo "          (model load/stub failure — a load failure of $CANDIDATE is itself a NO-GO)."
    echo "          The gate is meaningless; inspect ${OUT_DIR}/build.log + $CAPS and re-run." ;;
esac
echo "  corpus: $CAPS   gate: $GATE_LOG"
exit 0
