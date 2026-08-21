#!/bin/bash
# earshot/tools/clap_gate.sh — can CLAP tell the sounding classes apart through HM3D reverb?
#
#   source earshot/tools/notify/notify-run.sh
#   nrun bash earshot/tools/clap_gate.sh --tag clapgate-1
#
# THIS IS A GATE, NOT AN ABLATION. It runs no episode, no controller and no memory. It
# renders the candidate sounding vocabulary through live IRs at a spread of geodesic
# distances in every scene whose mesh is on this box, classifies each render with CLAP, and
# reports what separates from what.
#
# WHY IT RUNS FIRST. ADR-0017 pivots the task to sound-source finding and ADR-0018 makes the
# goal class INFERRED — the agent is told nothing and CLAP names what it heard. Every other
# decision in that pair is downstream of CLAP working on this renderer, and CLAP working on
# this renderer is unmeasured. `audio/clap.py` says so in its own source: `ANOMALY_GATE_DELTA`
# and `ANOMALY_GATE_TAU` were calibrated on a grid render convolved OFFLINE, carried across
# on an inference that "the domain should match", and the one arc that exercised the gate
# live had it reject 0 of 8 — which is also what a gate that discriminates nothing does.
#
# BOTH ARMS, OR IT IS RED. The healthy arm is the candidate vocabulary. The forced-failure
# arm is `vocabulary.ABSENT_CLASSES` — eight sounds with no room in a home, staged as audio
# and never placed in the prompt bank. clapgate-2 showed the arm is not uniform: `airplane`
# rejects at 1.000 and `chainsaw` at 0.351, because a chainsaw sounds like the `vacuum_cleaner`
# in the bank. Read the per-class rejection, not just the EER. A run missing either arm
# raises in `separation.summarise` rather than reporting the half that ran, on CLAUDE.md's
# rule that a criterion which could not be evaluated is never green.
#
# THE OUTPUT IS A CURVE, NOT A SCALAR. Top-1 accuracy banded by distance is the number that
# decides what happens next. If CLAP holds near and collapses far, the design survives and
# `AudioConfig.audible_band_m` moves; a single accuracy over the whole 1–8 m band cannot say
# that, and would be the same shape of number this repo has twice mistaken for a result.
#
# DO NOT HAND-PRUNE THE VOCABULARY. The candidate set is deliberately generous and carries
# weak anchors on purpose. `separation.prune` cuts by measured ANCHOR recall, so the surviving
# vocabulary is an artefact of this run rather than an author's guess. A class dropped for
# want of DATA, for want of SEPARATION and for want of AFFINITY are three different findings
# and the report prints all three counts.
#
# ONE DIRECTORY IS ONE RUN, enforced before any work starts, for yield-1's reason: a reused
# tag mixed two invocations into one pool with nothing on disk saying so.
#
# Flags: --tag T (required), --scenes "a b c" (default: every scene with a mesh),
#        --n-sources N (default 2), --n-poses N (default 6), --n-recordings N (default 4),
#        --n-bands N (default 4), --n-per-class N (default 8, staging only),
#        --clip-start N (default 0; ESC-50 ships 40 per class, so --clip-start 8
#          against the default 8 gives HELD-OUT recordings and the only prune-
#          unbiased accuracy this design can produce without new audio),
#        --min-recall R (default 0.50, the prune bar), --seed N, --split S,
#        --indirect-ray-count N (unset keeps ACOUSTICS_PRESET's 500), --no-pull, --no-stage.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then :; else
  echo "ERROR: execute this script, don't source it — its exit calls would kill your shell." >&2
  return 1
fi

set -u

TAG=""
SCENES=""
SPLIT="val"
N_SOURCES=2
N_POSES=6
N_RECORDINGS=4
N_BANDS=4
N_PER_CLASS=8
CLIP_START=0
MIN_RECALL=0.50
SEED=20260820
RAYS=""
PULL=1
STAGE=1

need_value() { [ "$1" -ge 2 ] || { echo "FATAL: $2 needs a value" >&2; exit 2; }; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    --tag)                need_value $# "$1"; TAG="$2";          shift 2 ;;
    --scenes)             need_value $# "$1"; SCENES="$2";       shift 2 ;;
    --split)              need_value $# "$1"; SPLIT="$2";        shift 2 ;;
    --n-sources)          need_value $# "$1"; N_SOURCES="$2";    shift 2 ;;
    --n-poses)            need_value $# "$1"; N_POSES="$2";      shift 2 ;;
    --n-recordings)       need_value $# "$1"; N_RECORDINGS="$2"; shift 2 ;;
    --n-bands)            need_value $# "$1"; N_BANDS="$2";      shift 2 ;;
    --n-per-class)        need_value $# "$1"; N_PER_CLASS="$2";  shift 2 ;;
    --clip-start)         need_value $# "$1"; CLIP_START="$2";   shift 2 ;;
    --min-recall)         need_value $# "$1"; MIN_RECALL="$2";   shift 2 ;;
    --seed)               need_value $# "$1"; SEED="$2";         shift 2 ;;
    --indirect-ray-count) need_value $# "$1"; RAYS="$2";         shift 2 ;;
    --no-pull)            PULL=0;  shift ;;
    --no-stage)           STAGE=0; shift ;;
    *) echo "FATAL: unknown flag $1" >&2; exit 2 ;;
  esac
done

[ -n "$TAG" ] || { echo "FATAL: --tag is required" >&2; exit 2; }

OUT_DIR="runs/$TAG"
banner() { echo; echo "=== $* ==="; }

# --- 1. one directory is one run, checked FIRST ---------------------------
banner "[1/4] preflight"
if [ -e "$OUT_DIR" ] && [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]; then
  echo "FATAL: $OUT_DIR already exists and is not empty."
  echo "  One directory is one run. Reusing a tag mixes two invocations into one pool"
  echo "  with nothing on disk saying so — that is how yield-1 reported 41% over a pool"
  echo "  of two runs. Pick a fresh --tag."
  exit 2
fi
mkdir -p "$OUT_DIR" || { echo "FATAL: cannot create $OUT_DIR"; exit 2; }

if [ "$PULL" -eq 1 ]; then
  git pull --ff-only || { echo "FATAL: git pull failed"; exit 2; }
fi

# Provenance, written before the work rather than after it: a finished sweep that cannot
# say what code produced it is the failure 6561434 fixed.
{
  echo "tag=$TAG"
  echo "commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  echo "dirty=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  echo "split=$SPLIT"
  echo "scenes=${SCENES:-<every scene with a mesh>}"
  echo "n_sources=$N_SOURCES n_poses=$N_POSES n_recordings=$N_RECORDINGS n_bands=$N_BANDS"
  echo "n_per_class=$N_PER_CLASS clip_start=$CLIP_START min_recall=$MIN_RECALL seed=$SEED"
  echo "indirect_ray_count=${RAYS:-<ACOUSTICS_PRESET default 500>}"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUT_DIR/provenance.txt"
cat "$OUT_DIR/provenance.txt"

# --- 2. stage the corpus --------------------------------------------------
banner "[2/4] staging the sound corpus"
if [ "$STAGE" -eq 1 ]; then
  # Stages BOTH the candidate vocabulary and the absent classes. They are staged together
  # on purpose: a gate run needs both arms, and staging them in separate steps is how one
  # of them ends up missing on the night it matters.
  python -m earshot.audio.clips --vocabulary --n-per-class "$N_PER_CLASS" \
    --clip-start "$CLIP_START" \
    || { echo "FATAL: staging failed — the gate cannot run on a partial corpus"; exit 2; }
else
  echo "  --no-stage: reusing whatever is already under data/sound_corpus and data/absent_corpus"
fi

# The CLAP checkpoint itself, converted to safetensors ONCE. `laion/clap-htsat-unfused`
# ships only pytorch_model.bin, and transformers >= 4.52 refuses torch.load on a .bin below
# torch 2.6 (CVE-2025-32434) — while this box pins torch 2.2.2+cu118 because cu118 is the
# last CUDA line where the V100's sm_70 is first-class. Upgrading torch would cost the GPU;
# converting the checkpoint costs nothing and changes no pin. Idempotent, so it is safe to
# leave in the path — the second run prints "already staged" and returns.
echo
echo "  CLAP checkpoint:"
python -m earshot.task.models \
  || { echo "FATAL: could not stage the CLAP checkpoint — the gate cannot classify"; exit 2; }

# --- 3. the gate ----------------------------------------------------------
banner "[3/4] rendering and classifying"
python -m earshot.task.clap_gate \
  --run-dir "$OUT_DIR" \
  --split "$SPLIT" \
  ${SCENES:+--scenes "$SCENES"} \
  --n-sources "$N_SOURCES" \
  --n-poses "$N_POSES" \
  --n-recordings "$N_RECORDINGS" \
  --n-bands "$N_BANDS" \
  --seed "$SEED" \
  ${RAYS:+--indirect-ray-count "$RAYS"} \
  2>&1 | tee "$OUT_DIR/gate.log"
GATE_STATUS=${PIPESTATUS[0]}

if [ "$GATE_STATUS" -ne 0 ]; then
  echo
  echo "FATAL: the gate exited $GATE_STATUS. Its own raises are the diagnosis — a missing"
  echo "  arm, an unstaged class or a scene that produced no rows all stop the run rather"
  echo "  than reporting the half that worked."
  exit "$GATE_STATUS"
fi

# --- 4. the pruned vocabulary --------------------------------------------
#
# `earshot.tools.anchor_report` and not an inline prune. This stage used to reimplement the
# cut in a heredoc, and the two implementations diverged exactly as you would expect: the
# heredoc cut on CLASS recall and ignored the affinity rule, so `clapsmoke-3` kept three weak
# classes and dropped `pouring_water` at 0.354 despite an anchor recall of 1.000. One
# implementation, called both by the live run and by a re-score of it.
banner "[4/4] pruning"
python -m earshot.tools.anchor_report "$OUT_DIR" --min-recall "$MIN_RECALL"
PRUNE_STATUS=$?

banner "done"
echo "  artefacts: $OUT_DIR/{provenance.txt,rows.jsonl,separation.json,pruned_vocabulary.json,gate.log}"
echo "  finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ "$PRUNE_STATUS" -ne 0 ]; then
  echo "  EXIT NONZERO: at least one scene failed to load, so this run is incomplete."
fi
exit "$PRUNE_STATUS"
