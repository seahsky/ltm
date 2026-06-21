#!/bin/bash
# scripts/race-rerender-grid.sh — re-render an existing SOURCE-TARGETED RIR grid
# at its OWN source with the updated render_rir_grid.py so it carries
# cell_geodesics.
#
# WHY: grids rendered before commit 55a87ed lack cell_geodesics, so the non-LOS
# seed gate hard-errors ("--rir-grid ... has no cell_geodesics"). This driver
# does NOT move the source and does NOT run any LLM (model-free, ~minutes/scene);
# it only re-renders the same source-targeted grid so the geodesic field exists.
# (The listener cells are re-sampled at the renderer's default seed, which is
# fine — the gate only needs cell_geodesics + audible occluded cells, not the
# exact legacy cells.)
#
# Runs the render in the dedicated `soundspaces-spike` env (habitat_sim audio
# build), reads the source straight out of the existing .npz, and is SAFE:
# renders to <grid>.regrid.npz and only replaces the original after verifying
# the new grid actually carries cell_geodesics — a failed render never destroys
# a working grid.
#
#   cd ~/ltm && git checkout lifelong-revisit-eval && git pull \
#       && bash scripts/race-rerender-grid.sh \
#            runs/audiogoal/wcojb4TFT35_glass_break_rir_grid.npz
#
# Multiple grids may be passed; with none, the two standard cells are used.
# Knobs: --n-cells N (default 24; bump if the gate's Tier-2 finds no occluded
# audible cell), --ss-env NAME (default soundspaces-spike).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

MINICONDA="${HOME}/miniconda3"
SS_ENV="${SS_ENV:-soundspaces-spike}"
LTM_ENV="${LTM_ENV:-ltm-embodied}"
N_CELLS="${N_CELLS:-24}"

GRIDS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --n-cells) N_CELLS="$2"; shift 2;;
    --ss-env)  SS_ENV="$2";  shift 2;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    -*) echo "unknown flag: $1"; exit 2;;
    *) GRIDS+=("$1"); shift;;
  esac
done
if [ "${#GRIDS[@]}" -eq 0 ]; then
  GRIDS=( runs/audiogoal/wcojb4TFT35_glass_break_rir_grid.npz \
          runs/audiogoal/TEEsavR23oF_alarm_rir_grid.npz )
fi

banner() { printf '\n========== %s ==========\n' "$1"; }
[ -x "$MINICONDA/bin/conda" ] || { echo "FATAL: $MINICONDA/bin/conda missing"; exit 1; }
eval "$("$MINICONDA/bin/conda" shell.bash hook)"

banner "activate $SS_ENV (render env)"
set +u
conda activate "$SS_ENV" || {
  echo "FATAL: conda activate $SS_ENV failed — build it first: bash scripts/race-soundspaces-spike.sh"
  exit 1
}
set -u
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

fail=0
for GRID in "${GRIDS[@]}"; do
  banner "re-render $GRID"
  if [ ! -f "$GRID" ]; then echo "RED: grid not found: $GRID"; fail=1; continue; fi

  # Pull source xyz + scene id + whether it already has cell_geodesics (plain numpy).
  mapfile -t META < <(python -c "
import numpy as np
d = np.load('$GRID')
sp = d['source_position']
sid = str(d['scene_id']) if 'scene_id' in d.files else ''
print(','.join('%.6f' % float(v) for v in sp))
print(sid)
print('yes' if 'cell_geodesics' in d.files else 'no')
") || { echo "RED: could not read $GRID"; fail=1; continue; }
  SRC_XYZ="${META[0]:-}"; SCENE="${META[1]:-}"; HASGEO="${META[2]:-}"
  if [ -z "$SRC_XYZ" ] || [ -z "$SCENE" ]; then
    echo "RED: could not parse source/scene from $GRID"; fail=1; continue
  fi
  echo "  scene=$SCENE  source=$SRC_XYZ  already_has_cell_geodesics=$HASGEO"

  GLB="$(find data/hm3d -name "${SCENE}.basis.glb" 2>/dev/null | head -1)"
  [ -n "$GLB" ] || GLB="$(find data/hm3d -name "*${SCENE}*.glb" 2>/dev/null | grep -v semantic | head -1)"
  [ -n "$GLB" ] || { echo "RED: no .glb for $SCENE under data/hm3d"; fail=1; continue; }

  TMP="${GRID%.npz}.regrid.npz"
  rm -f "$TMP"
  # '=' form for --source: HM3D coords start with '-' and contain commas, so a
  # space-separated value is misread by argparse as a flag (mirrors race-audiogoal.sh).
  python embodied_memory/scripts/render_rir_grid.py \
      --scene "$GLB" --source="$SRC_XYZ" --out "$TMP" --n-cells "$N_CELLS" \
      2>&1 | tee "${GRID%.npz}_regrid.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ] || [ ! -f "$TMP" ]; then
    echo "RED: render failed for $SCENE (rc=$rc) — ORIGINAL grid untouched. See ${GRID%.npz}_regrid.log"
    rm -f "$TMP"; fail=1; continue
  fi

  ok="$(python -c "
import numpy as np
d = np.load('$TMP')
g = d['cell_geodesics'] if 'cell_geodesics' in d.files else None
print('yes' if (g is not None and len(g) > 0) else 'no')
")"
  if [ "$ok" != "yes" ]; then
    echo "RED: re-rendered $TMP still lacks cell_geodesics — ORIGINAL untouched (is render_rir_grid.py at >= 55a87ed?)"
    rm -f "$TMP"; fail=1; continue
  fi
  mv -f "$TMP" "$GRID"
  echo "  GREEN: $GRID now carries cell_geodesics (re-rendered at its own source)"
done

set +u; conda deactivate 2>/dev/null || true; set -u

banner "RE-RENDER"
if [ "$fail" -eq 0 ]; then
  echo "GREEN: grids re-rendered with cell_geodesics. Re-run the non-LOS gate:"
  echo "  bash scripts/race-nonlos-seed-gate.sh --scene <S> --src <val_mini>.json.gz \\"
  echo "      --category <cat> --class <class> --rir-grid <grid>.npz"
  echo "  (if Tier-2 still RED 'no occluded audible cell': re-run this with --n-cells 32,"
  echo "   or re-render the grid with the source tucked behind a doorway via race-audiogoal.sh)"
else
  echo "RED: one or more grids failed — see logs above. Originals are untouched."
fi
exit "$fail"
