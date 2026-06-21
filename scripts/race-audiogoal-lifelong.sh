#!/bin/bash
# scripts/race-audiogoal-lifelong.sh — OVERNIGHT lifelong cross-visit A/B matrix
# (Step 2 audio→LTM write, oracle-source upper bound). For each (scene,class,
# category) cell it runs the write-ON vs write-OFF A/B on ONE shared lifelong
# dataset, then summarizes directly from each arm's summary.json:
#   * seed n_audio_writes  — did the audio→LTM write finally FIRE?
#   * recall n_audio_event_recalled — recalled from a distance (not deduped)?
#   * recall soft-SPL / succ@1m paired B−A — did the oracle write HELP or is it
#     REDUNDANT with the seed's own visual sighting (the documented LOS risk)?
#
# Per cell: arm A (write-OFF) builds the --lifelong dataset (+ its $0 construction
# gate) + renders the grid + runs S3; arm B (write-ON, --reuse-dataset) reuses the
# dataset+grid and runs S3 with LTM_AUDIO_WRITE=1. RESILIENT: a cell that fails to
# build/render/run is logged and SKIPPED — the batch keeps going.
#
#   cd ~/ltm && git checkout lifelong-revisit-eval && git pull --ff-only \
#       && nrun bash scripts/race-audiogoal-lifelong.sh
#   # tune:  --n-warm 8   --cells "wcojb4TFT35:alarm:bed TEEsavR23oF:glass_break:chair"
#
# EXECUTE (do NOT source). Wrap in nrun for the emailed report. Estimated runtime
# ~3-4 h for the 4-cell default at --n-warm 6 (7B backbone). No paid run starts
# until the offline pre-verify passes.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT" || exit 1
LTM_ENV="ltm-embodied"

# 2 val_mini scenes × {glass_break:chair, alarm:bed}. The $0 gate + resilient loop
# drop any cell whose source is off-navmesh / has no goal instance.
CELLS_DEFAULT="wcojb4TFT35:glass_break:chair wcojb4TFT35:alarm:bed TEEsavR23oF:glass_break:chair TEEsavR23oF:alarm:bed"
CELLS="${CELLS:-$CELLS_DEFAULT}"
NWARM="${NWARM:-6}"
CONSUME_SG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --cells) CELLS="$2"; shift 2 ;;
    --n-warm) NWARM="$2"; shift 2 ;;
    # Over-fire fix confirmation: ungate the reached-memory consumption +
    # anti-thrash filters for single-goal AudioGoal (export INSIDE the driver so
    # it reaches python without ambient nrun inheritance). Expect the write-ON
    # over-fire to damp -> B-A moves from -0.170 toward ~0 (REDUNDANT).
    --consume-singlegoal) CONSUME_SG=1; shift ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done
[ -n "$CONSUME_SG" ] && export REMEMBR_CONSUME_SINGLEGOAL=1

banner() { printf '\n========== %s ==========\n' "$1"; }
N_CELLS=$(set -- $CELLS; echo $#)

banner "[1/4] git pull --ff-only"
git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }

banner "[2/4] conda setup (source scripts/race-setup.sh → $LTM_ENV) + pre-verify"
set +u; source scripts/race-setup.sh || { echo "FATAL: race-setup.sh failed"; exit 1; }; set -u
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
for t in test_make_audiogoal_smoke test_audio_write test_audio_task \
         test_analyze_lifelong_ab test_diagnose_normal_anomaly_calib; do
  python embodied_memory/scripts/$t.py \
    || { echo "FATAL: $t failed — not spending on the matrix."; exit 1; }
done
echo "  cells ($N_CELLS): $CELLS"
echo "  n-warm (recall episodes/cell) = $NWARM  → ~$((N_CELLS * 2 * (1 + NWARM))) episodes total"
[ -n "$CONSUME_SG" ] && echo "  [over-fire fix] REMEMBR_CONSUME_SINGLEGOAL=1 — single-goal AudioGoal memory consumption ON (expect write-ON over-fire to damp → B−A → ~0)"

# race-audiogoal.sh self-pulls at step 1; we already pulled once → skip its pull.
export RACE_SKIP_PULL=1

banner "[3/4] run the A/B matrix (arm A=write-OFF, arm B=write-ON reuse)"
ALL_A=""; ALL_B=""; FAILED=""
for cell in $CELLS; do
  scene="${cell%%:*}"; rest="${cell#*:}"; class="${rest%%:*}"; cat="${rest##*:}"
  tagA="llA-${scene}-${class}"; tagB="llB-${scene}-${class}"
  dirA="runs/${tagA}-${class}-s3"; dirB="runs/${tagB}-${class}-s3"

  banner "CELL ${scene} ${class}:${cat} — arm A (write-OFF) [builds dataset+grid]"
  if ! bash scripts/race-audiogoal.sh --lifelong --settings 3 --n-warm "$NWARM" \
        --scene "$scene" --class "$class" --category "$cat" --tag "$tagA" --fetch-audio; then
    echo "WARN: cell ${cell} arm A failed — skipping cell"; FAILED="$FAILED ${cell}(A)"; continue
  fi

  banner "CELL ${scene} ${class}:${cat} — arm B (write-ON, reuse dataset+grid)"
  if ! LTM_AUDIO_WRITE=1 bash scripts/race-audiogoal.sh --lifelong --settings 3 --n-warm "$NWARM" \
        --scene "$scene" --class "$class" --category "$cat" \
        --tag "$tagA" --out-tag "$tagB" --reuse-dataset --audio-write --fetch-audio; then
    echo "WARN: cell ${cell} arm B failed — skipping cell"; FAILED="$FAILED ${cell}(B)"; continue
  fi

  ALL_A="$ALL_A $dirA"; ALL_B="$ALL_B $dirB"
  echo "  cell ${cell} DONE: A=$dirA  B=$dirB"
done

banner "[4/4] lifelong write-ON vs write-OFF A/B summary"
if [ -n "$ALL_A" ]; then
  # shellcheck disable=SC2086
  python embodied_memory/scripts/analyze_lifelong_ab.py --a $ALL_A --b $ALL_B || true
else
  echo "  no cell completed BOTH arms — nothing to summarize"
fi
[ -n "$FAILED" ] && echo "  FAILED arms:$FAILED"
echo
echo "DONE. Lifelong A/B matrix. Read the [4/4] table: seedW>0 = the write fired;"
echo "rcl>0 = recalled from a distance; dB-A = oracle write's recall soft-SPL lift"
echo "(≈0 = REDUNDANT, the seed already visually mapped the source → needs a non-LOS seed)."
exit 0
