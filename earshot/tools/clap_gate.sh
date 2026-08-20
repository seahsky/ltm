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
# arm is `vocabulary.ABSENT_CLASSES` — chainsaw, helicopter, airplane, church bells, sea
# waves — staged as audio and never placed in the prompt bank. A run missing either arm
# raises in `separation.summarise` rather than reporting the half that ran, on CLAUDE.md's
# rule that a criterion which could not be evaluated is never green.
#
# THE OUTPUT IS A CURVE, NOT A SCALAR. Top-1 accuracy banded by distance is the number that
# decides what happens next. If CLAP holds near and collapses far, the design survives and
# `AudioConfig.audible_band_m` moves; a single accuracy over the whole 1–8 m band cannot say
# that, and would be the same shape of number this repo has twice mistaken for a result.
#
# DO NOT HAND-PRUNE THE VOCABULARY. The candidate set is deliberately generous and carries
# weak anchors on purpose. `separation.prune` cuts by measured recall, so the surviving
# vocabulary is an artefact of this run rather than an author's guess. A class dropped for
# want of DATA and a class dropped for want of SEPARATION are different findings and the
# report prints both counts.
#
# ONE DIRECTORY IS ONE RUN, enforced before any work starts, for yield-1's reason: a reused
# tag mixed two invocations into one pool with nothing on disk saying so.
#
# Flags: --tag T (required), --scenes "a b c" (default: every scene with a mesh),
#        --n-sources N (default 2), --n-poses N (default 6), --n-recordings N (default 4),
#        --n-bands N (default 4), --n-per-class N (default 8, staging only),
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
  echo "n_per_class=$N_PER_CLASS min_recall=$MIN_RECALL seed=$SEED"
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
    || { echo "FATAL: staging failed — the gate cannot run on a partial corpus"; exit 2; }
else
  echo "  --no-stage: reusing whatever is already under data/sound_corpus and data/absent_corpus"
fi

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
banner "[4/4] pruning"
python - "$OUT_DIR/separation.json" "$MIN_RECALL" <<'PY'
import json
import sys

path, min_recall = sys.argv[1], float(sys.argv[2])
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)

rows = payload["per_class"]
kept, cut_recall, cut_n = [], [], []
for item in rows:
    if item["n"] < 8:
        cut_n.append(item)
    elif item["recall"] < min_recall:
        cut_recall.append(item)
    else:
        kept.append(item)

print("")
print("PRUNED VOCABULARY at min_recall={:.2f}".format(min_recall))
print("  kept {} of {} candidate class(es)".format(len(kept), len(rows)))
print("")
for item in sorted(kept, key=lambda entry: -entry["recall"]):
    print("  KEEP  {:18s} {:8s} n={:4d} recall={:.3f}".format(
        item["name"], item["affinity"], item["n"], item["recall"]))
for item in sorted(cut_recall, key=lambda entry: -entry["recall"]):
    print("  cut   {:18s} {:8s} n={:4d} recall={:.3f}  (separation)".format(
        item["name"], item["affinity"], item["n"], item["recall"]))
for item in cut_n:
    print("  cut   {:18s} {:8s} n={:4d} recall={:.3f}  (TOO FEW ROWS — not a separation "
          "finding)".format(item["name"], item["affinity"], item["n"], item["recall"]))

# The strong-versus-weak read. The affinity grades are declared judgements (ADR-0018), so
# this says whether the gate agreed with them — which is a check on the vocabulary's design
# rather than on CLAP.
print("")
print("BY DECLARED AFFINITY (does the gate agree with the table?)")
for grade in ("strong", "moderate", "weak", "unknown"):
    at_grade = [item for item in rows if item["affinity"] == grade]
    if not at_grade:
        continue
    mean = sum(item["recall"] for item in at_grade) / len(at_grade)
    survived = len([item for item in at_grade if item in kept])
    print("  {:8s} n_classes={:2d}  mean recall={:.3f}  survived={}/{}".format(
        grade, len(at_grade), mean, survived, len(at_grade)))

reject = payload["rejection"]
print("")
print("FORCED-FAILURE ARM: EER {:.3f}  (0.500 = the two arms are on top of each other)".format(
    reject["eer"]))

failed = payload.get("scenes_failed") or []
if failed:
    print("")
    print("SCENES THAT FAILED TO LOAD ({}): {}".format(len(failed), " ".join(failed)))
    print("  Continuing is not passing — this run is INCOMPLETE.")

with open(path.replace("separation.json", "pruned_vocabulary.json"), "w", encoding="utf-8") as sink:
    json.dump(
        {
            "min_recall": min_recall,
            "kept": [item["name"] for item in kept],
            "cut_for_separation": [item["name"] for item in cut_recall],
            "cut_for_too_few_rows": [item["name"] for item in cut_n],
        },
        sink,
        indent=2,
        sort_keys=True,
    )

sys.exit(1 if failed else 0)
PY
PRUNE_STATUS=$?

banner "done"
echo "  artefacts: $OUT_DIR/{provenance.txt,rows.jsonl,separation.json,pruned_vocabulary.json,gate.log}"
echo "  finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ "$PRUNE_STATUS" -ne 0 ]; then
  echo "  EXIT NONZERO: at least one scene failed to load, so this run is incomplete."
fi
exit "$PRUNE_STATUS"
