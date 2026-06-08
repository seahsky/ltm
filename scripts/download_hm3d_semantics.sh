#!/usr/bin/env bash
# Download the HM3D-SEMANTICS annotations (region + object labels) for the scenes
# already on disk, so the coarse-affordance head's GT-region grounding works
# (sim.semantic_scene.regions). The geometry (.basis.glb) is already present from
# download_hm3d.sh / hm3d_minival_full; this adds the missing semantic files
# (.semantic.glb, .semantic.txt, .basis.scn / scene-descriptor configs) that the
# sim looks for and currently fails to load ("SSD Load Failure ... .basis.scn does
# not exist" -> empty regions; see diagnose_hm3d_regions.py).
#
# Prereq: the Matterport HM3D token in .env (same as download_hm3d.sh):
#   MATTERPORT_TOKEN_ID=...  MATTERPORT_TOKEN_SECRET=...
#
# Usage (on RACE, env active):
#   bash scripts/download_hm3d_semantics.sh            # list uids, then download defaults
#   HM3D_SEMANTIC_UIDS="hm3d_semantic_annots_v0.2 hm3d_semantic_configs_v0.2" \
#     bash scripts/download_hm3d_semantics.sh          # override the uids
#
# After it finishes, RE-CONFIRM regions appear:
#   python3 embodied_memory/scripts/diagnose_hm3d_regions.py --scene all \
#     --episodes-path data/hm3d/datasets/objectnav/hm3d/v1/val_mini/val_mini.json.gz

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

set -a
[[ -f "${REPO_ROOT}/.env" ]] && source "${REPO_ROOT}/.env"
set +a

DEST_DIR="${1:-data/hm3d}"
# HM3D-Semantics v0.2 uids are SPLIT-SPECIFIC (confirmed via --list). Our scenes
# live under minival/ (the val->minival symlink; TEEsavR23oF/wcojb4TFT35 are
# minival scenes), so default to the MINIVAL semantic split. For the full 20-scene
# val matrix later, override:
#   HM3D_SEMANTIC_UIDS="hm3d_val_semantic_annots_v0.2 hm3d_val_semantic_configs_v0.2"
SEMANTIC_UIDS="${HM3D_SEMANTIC_UIDS:-hm3d_minival_semantic_annots_v0.2 hm3d_minival_semantic_configs_v0.2}"

if [[ -z "${MATTERPORT_TOKEN_ID:-}" || -z "${MATTERPORT_TOKEN_SECRET:-}" ]]; then
  echo "ERROR: MATTERPORT_TOKEN_ID / MATTERPORT_TOKEN_SECRET not set (need them in .env)." >&2
  exit 1
fi
if ! python -c "import habitat_sim" 2>/dev/null; then
  echo "ERROR: habitat-sim not importable — activate the env first (source scripts/race-setup.sh)." >&2
  exit 1
fi

echo "========== available HM3D uids in this habitat-sim =========="
python -m habitat_sim.utils.datasets_download --list 2>&1 | grep -iE "hm3d.*seman|seman.*hm3d|hm3d_(minival|val).*full" || \
  python -m habitat_sim.utils.datasets_download --list 2>&1 | grep -i hm3d || true
echo "============================================================="
echo ">> downloading semantic uids: ${SEMANTIC_UIDS}  -> ${DEST_DIR}"

for uid in ${SEMANTIC_UIDS}; do
  echo ">> --- ${uid} ---"
  if ! python -m habitat_sim.utils.datasets_download \
        --username "${MATTERPORT_TOKEN_ID}" --password "${MATTERPORT_TOKEN_SECRET}" \
        --uids "${uid}" --data-path "${DEST_DIR}"; then
    echo "WARN: uid '${uid}' failed — check it against the --list output above and re-run with"
    echo "      HM3D_SEMANTIC_UIDS set to the correct semantic-annotation uid(s)." >&2
  fi
done

# Keep the val -> minival symlink (ObjectNav val episodes resolve through it). If
# the semantic files landed under minival/, the existing symlink already exposes
# them at val/. Re-assert it defensively.
SCENE_HM3D_DIR="${DEST_DIR}/scene_datasets/hm3d"
if [[ -d "${SCENE_HM3D_DIR}/minival" && ! -e "${SCENE_HM3D_DIR}/val" ]]; then
  ln -sfn minival "${SCENE_HM3D_DIR}/val"
fi

echo ">> Done. Verify the semantic files now exist for a scene, e.g.:"
echo "   ls ${SCENE_HM3D_DIR}/val/00802-wcojb4TFT35/"
echo "   (expect a .semantic.glb / .semantic.txt and a .basis.scn or info_semantic.json)"
echo ">> Then RE-CONFIRM regions populate:"
echo "   python3 embodied_memory/scripts/diagnose_hm3d_regions.py --scene all \\"
echo "     --episodes-path data/hm3d/datasets/objectnav/hm3d/v1/val_mini/val_mini.json.gz"
