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
REUSE_NONLOS=""; NONLOS_DIR="runs/nonlos-gate"
while [ $# -gt 0 ]; do
  case "$1" in
    --cells) CELLS="$2"; shift 2 ;;
    --n-warm) NWARM="$2"; shift 2 ;;
    # Over-fire fix confirmation: ungate the reached-memory consumption +
    # anti-thrash filters for single-goal AudioGoal (export INSIDE the driver so
    # it reaches python without ambient nrun inheritance). Expect the write-ON
    # over-fire to damp -> B-A moves from -0.170 toward ~0 (REDUNDANT).
    --consume-singlegoal) CONSUME_SG=1; shift ;;
    # Consume the $0 non-LOS seed gate (race-nonlos-seed-gate.sh) instead of
    # letting each cell rebuild an LOS seed + re-render a random source. STAGE the
    # gate's lifelong dataset (<dir>/audiogoal.json.gz + content/<scene>.json.gz +
    # source_manifest.json) into each cell's tagA DS_DIR and run BOTH arms with
    # --reuse-dataset (so neither rm-rf-rebuilds nor re-renders; the gate grid at
    # runs/audiogoal/<scene>_<class>_rir_grid.npz is already the canonical $GRID).
    # This makes the A/B the "sound-on wins on a NON-LOS seed" experiment, not the
    # closed redundant-LOS one. Opt-in; default path is byte-identical. Pins
    # REMEMBR_CONSUME_SINGLEGOAL=1 on BOTH arms (the over-fire damping must be held
    # fixed so the only difference between arms is LTM_AUDIO_WRITE). Optional arg =
    # the gate --out-dir (default runs/nonlos-gate).
    --reuse-nonlos)
      REUSE_NONLOS=1
      case "${2:-}" in ""|--*) ;; *) NONLOS_DIR="$2"; shift ;; esac
      shift ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done
# --reuse-nonlos REQUIRES the single-goal anti-thrash machinery on BOTH arms (the
# gate's single saturating-cosine GT-source waypoint over-fires otherwise → the
# documented mem_chosen=188 / replan_stuck=147 attractor swamps the B−A signal).
[ -n "$REUSE_NONLOS" ] && CONSUME_SG=1
[ -n "$CONSUME_SG" ] && export REMEMBR_CONSUME_SINGLEGOAL=1

banner() { printf '\n========== %s ==========\n' "$1"; }
N_CELLS=$(set -- $CELLS; echo $#)

# Stage the $0 non-LOS seed gate's lifelong dataset into the per-cell tagA DS_DIR
# that race-audiogoal.sh --reuse-dataset reads. The gate (race-nonlos-seed-gate.sh)
# wrote, under $NONLOS_DIR (default runs/nonlos-gate):
#   audiogoal.json.gz, content/<scene>.json.gz, source_manifest.json
# race-audiogoal.sh --tag <tagA> --reuse-dataset expects, under
#   DS_DIR=data/hm3d/datasets/objectnav/hm3d/v1/audiogoal_<tagA>:
#   audiogoal_<tagA>.json.gz, content/<scene>.json.gz, source_manifest.json
# (the grid runs/audiogoal/<scene>_<class>_rir_grid.npz is already at $GRID, so the
# render is reused as-is). Returns 1 (and prints why) if a gate input is missing.
stage_nonlos_dataset() {  # $1=tagA $2=scene
  local tag="$1" scene="$2"
  local g_top="$NONLOS_DIR/audiogoal.json.gz"
  local g_content="$NONLOS_DIR/content/${scene}.json.gz"
  local g_manifest="$NONLOS_DIR/source_manifest.json"
  local ds_dir="data/hm3d/datasets/objectnav/hm3d/v1/audiogoal_${tag}"
  for f in "$g_top" "$g_content" "$g_manifest"; do
    [ -f "$f" ] || { echo "  STAGE FAIL: missing gate file $f (run race-nonlos-seed-gate.sh --out-dir $NONLOS_DIR first)"; return 1; }
  done
  mkdir -p "$ds_dir/content" || return 1
  cp -f "$g_top"      "$ds_dir/audiogoal_${tag}.json.gz" || return 1
  cp -f "$g_content"  "$ds_dir/content/${scene}.json.gz" || return 1
  cp -f "$g_manifest" "$ds_dir/source_manifest.json"     || return 1
  echo "  STAGED non-LOS gate dataset -> $ds_dir (audiogoal_${tag}.json.gz + content/${scene}.json.gz + source_manifest.json)"
  return 0
}

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
if [ -n "$REUSE_NONLOS" ]; then
  echo "  [reuse-nonlos] consuming the \$0 non-LOS seed gate at $NONLOS_DIR — BOTH arms run --reuse-dataset (no rebuild, no re-render); arm B adds LTM_AUDIO_WRITE=1 only"
  # Fail fast (before any spend) if the gate hasn't been run / its dir is wrong.
  for cell in $CELLS; do
    scene="${cell%%:*}"; rest="${cell#*:}"; class="${rest%%:*}"
    g_top="$NONLOS_DIR/audiogoal.json.gz"; g_content="$NONLOS_DIR/content/${scene}.json.gz"
    g_grid="runs/audiogoal/${scene}_${class}_rir_grid.npz"
    for f in "$g_top" "$g_content" "$NONLOS_DIR/source_manifest.json" "$g_grid"; do
      [ -f "$f" ] || { echo "FATAL: --reuse-nonlos but gate output missing: $f — run race-nonlos-seed-gate.sh (--out-dir $NONLOS_DIR) + render the grid first."; exit 1; }
    done
  done
  echo "  [reuse-nonlos] gate dataset + source-targeted grid present for all cells"
fi

# race-audiogoal.sh self-pulls at step 1; we already pulled once → skip its pull.
export RACE_SKIP_PULL=1

banner "[3/4] run the A/B matrix (arm A=write-OFF, arm B=write-ON reuse)"
ALL_A=""; ALL_B=""; FAILED=""
for cell in $CELLS; do
  scene="${cell%%:*}"; rest="${cell#*:}"; class="${rest%%:*}"; cat="${rest##*:}"
  tagA="llA-${scene}-${class}"; tagB="llB-${scene}-${class}"
  dirA="runs/${tagA}-${class}-s3"; dirB="runs/${tagB}-${class}-s3"

  # --reuse-nonlos: stage the gate dataset into tagA's DS_DIR and run arm A with
  # --reuse-dataset too (so it consumes the non-LOS seed + the source-targeted grid
  # instead of rebuilding an LOS seed at a random +0.5m source). Default path
  # (REUSE_NONLOS unset) is byte-identical: arm A is the builder, no extra flag.
  ARM_A_REUSE=""; ARM_A_NOTE=" [builds dataset+grid]"
  if [ -n "$REUSE_NONLOS" ]; then
    if ! stage_nonlos_dataset "$tagA" "$scene"; then
      echo "WARN: cell ${cell} non-LOS staging failed — skipping cell"; FAILED="$FAILED ${cell}(stage)"; continue
    fi
    ARM_A_REUSE="--reuse-dataset"; ARM_A_NOTE=" [reuse non-LOS gate dataset+grid]"
  fi

  banner "CELL ${scene} ${class}:${cat} — arm A (write-OFF)${ARM_A_NOTE}"
  # shellcheck disable=SC2086
  if ! bash scripts/race-audiogoal.sh --lifelong --settings 3 --n-warm "$NWARM" \
        --scene "$scene" --class "$class" --category "$cat" --tag "$tagA" --fetch-audio $ARM_A_REUSE; then
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
