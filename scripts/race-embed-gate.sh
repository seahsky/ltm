#!/bin/bash
# scripts/race-embed-gate.sh — Phase 3a: the $0 ENCODER-swap gate.
#
# Phase 0 showed the CAPTIONER is not the instance-discrimination bottleneck
# (CapRL-3B = HOLD). This gate tests the READ side: with the captions FIXED (the
# production Qwen2-VL-2B captions), does a stronger TEXT EMBEDDER widen the
# within-vs-between instance separation over the current SBERT all-MiniLM-L6-v2?
#   GATE=GO   -> the text embedding was a limiter; swap the fine-index encoder.
#   GATE=HOLD -> not the text encoder either; next lever is a VISUAL instance
#               embedder (DINOv3) on the keyframe image, or the instance-aware query.
#
# REUSES the existing runs/caprl-gate/captions.json corpus (no re-render / re-caption
# — just re-embeds the same captions with candidate encoders). Builds a qwen-only
# corpus only if that file is absent. CPU/SBERT after the first encoder download;
# runs in MINUTES. ltm-embodied env (sentence-transformers), NO soundspaces.
#
#   cd ~/ltm && git checkout lifelong-revisit-eval && git pull
#   nrun bash scripts/race-embed-gate.sh
#   nrun bash scripts/race-embed-gate.sh --encoders all-MiniLM-L6-v2=sentence-transformers/all-MiniLM-L6-v2 bge-large=BAAI/bge-large-en-v1.5
#
# EXECUTE (do NOT source).

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT" || exit 1

CORPUS="runs/caprl-gate/captions.json"   # reuse the Phase-0 corpus by default
CORPUS_EXPLICIT=""                        # set when the user passes --corpus
CAPTIONER="qwen2-vl-2b"                   # fix the production captioner
BASELINE_ENC="all-MiniLM-L6-v2"          # the production fine-index encoder
ENCODERS=""                              # empty -> diagnose's default shortlist
MARGIN="0.02"; TAG="embed-gate"
# fallback corpus-build knobs (only used if $CORPUS is missing)
SCENES="wcojb4TFT35 TEEsavR23oF"; CATEGORIES="chair bed sofa toilet"; NVP=6
while [ $# -gt 0 ]; do
  case "$1" in
    --corpus) CORPUS="$2"; CORPUS_EXPLICIT=1; shift 2 ;;
    --captioner) CAPTIONER="$2"; shift 2 ;;
    --baseline-encoder) BASELINE_ENC="$2"; shift 2 ;;
    --encoders) ENCODERS="$2"; shift 2 ;;
    --margin) MARGIN="$2"; shift 2 ;;
    --scenes) SCENES="$2"; shift 2 ;;
    --n-viewpoints) NVP="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
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

banner "[3/5] pre-verify (free)"
python embodied_memory/scripts/test_diagnose_sbert_cosines.py \
  || { echo "FATAL: test_diagnose_sbert_cosines failed."; exit 1; }

banner "[4/5] corpus (reuse $CORPUS if present, else build a ${CAPTIONER}-only one)"
if [ -f "$CORPUS" ]; then
  echo "  reusing existing corpus: $CORPUS"
elif [ -n "$CORPUS_EXPLICIT" ]; then
  echo "FATAL: --corpus path does not exist: $CORPUS (omit --corpus to build a fresh one)"; exit 1
else
  echo "  $CORPUS absent — building a ${CAPTIONER}-only corpus (render + caption once)"
  CORPUS="${OUT_DIR}/captions.json"
  # shellcheck disable=SC2086
  python embodied_memory/scripts/build_instance_caption_corpus.py \
      --scenes $SCENES --categories $CATEGORIES --n-viewpoints "$NVP" \
      --captioners "${CAPTIONER}=Qwen/Qwen2-VL-2B-Instruct" --out "$CORPUS" \
      2>&1 | tee "${OUT_DIR}/build.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] && [ -f "$CORPUS" ] \
    || { echo "FATAL: corpus build failed; see ${OUT_DIR}/build.log"; exit 1; }
fi

banner "[5/5] ENCODER-swap gate (fixed captions, varying the fine-index encoder)"
ENC_ARG=""; [ -n "$ENCODERS" ] && ENC_ARG="--encoders $ENCODERS"
# shellcheck disable=SC2086
python embodied_memory/scripts/diagnose_sbert_cosines.py \
    --compare-encoders "$CORPUS" --captioner "$CAPTIONER" \
    --baseline-encoder "$BASELINE_ENC" --margin "$MARGIN" $ENC_ARG 2>&1 | tee "$GATE_LOG"
drc=${PIPESTATUS[0]}
RESULT="$(grep -oE 'GATE_RESULT=[A-Z]+' "$GATE_LOG" | tail -1 | cut -d= -f2)"
[ "$drc" -eq 0 ] && [ -n "$RESULT" ] \
  || { echo "FATAL: gate diagnostic failed (rc=$drc) or no GATE_RESULT marker — see $GATE_LOG"; exit 1; }

echo
echo "########## PHASE-3a VERDICT: $RESULT ##########"
case "$RESULT" in
  GO)   echo "  GO  -> a stronger TEXT encoder widens instance separation. Swap the SBERT"
        echo "        fine-index encoder (dialogue_memory.encoder) to the winner + A/B the"
        echo "        warm-revisit matrix vs the +0.2505 baseline." ;;
  HOLD) echo "  HOLD -> no text encoder separates the instances better. The captions don't"
        echo "          carry separable instance signal for these categories. Next lever ="
        echo "          a VISUAL instance embedder (DINOv3) on the keyframe IMAGE, or the"
        echo "          instance-aware query — NOT another text encoder." ;;
  INSUFFICIENT) echo "  INSUFFICIENT -> baseline or all candidate encoders failed to load; check $GATE_LOG." ;;
esac
echo "  corpus: $CORPUS   gate: $GATE_LOG"
exit 0
