#!/usr/bin/env bash
# Download a small slice of HM3D-Semantics val-mini for the proof-of-life run.
#
# Prereq (manual, one-time): sign the Matterport HM3D academic agreement at
#   https://matterport.com/habitat-matterport-3d-research-dataset
# Then copy .env.example to .env at the repo root and fill in:
#   MATTERPORT_TOKEN_ID=...
#   MATTERPORT_TOKEN_SECRET=...
#
# Usage:
#   bash embodied_memory/scripts/download_hm3d.sh [DEST_DIR]
#
# Default DEST_DIR is data/hm3d at the repo root.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

set -a
if [[ -f "${ENV_FILE}" ]]; then
  source "${ENV_FILE}"
fi
set +a

DEST_DIR="${1:-data/hm3d}"

# Datasource group from habitat-sim's downloader catalog. `hm3d_minival_full`
# pulls the 10-scene val-mini split + GLBs + semantic annots/configs — the
# smallest slice that exercises a real ObjectNav episode loop. Switch to
# `hm3d_val_full` for the full 36-scene val split.
SCENE_GROUP="${HM3D_SCENE_GROUP:-hm3d_minival_full}"

# ObjectNav HM3D v1 episodes are not in the habitat-sim downloader catalog;
# they ship as a single zip from habitat-lab's public CDN.
OBJECTNAV_URL="https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip"

if [[ -z "${MATTERPORT_TOKEN_ID:-}" || -z "${MATTERPORT_TOKEN_SECRET:-}" ]]; then
  echo "ERROR: MATTERPORT_TOKEN_ID and MATTERPORT_TOKEN_SECRET are not set." >&2
  echo "Sign the agreement at https://matterport.com/habitat-matterport-3d-research-dataset," >&2
  echo "then 'cp .env.example .env' and fill in your token + secret." >&2
  exit 1
fi

mkdir -p "${DEST_DIR}"

if ! python -c "import habitat_sim" 2>/dev/null; then
  echo "ERROR: habitat-sim is not importable. Activate the conda env first:" >&2
  echo "  conda env create -f embodied_memory/environment.yml  # one-time" >&2
  echo "  conda activate ltm-embodied" >&2
  exit 1
fi

echo ">> Downloading HM3D scene group '${SCENE_GROUP}' to ${DEST_DIR}"
python -m habitat_sim.utils.datasets_download \
  --username "${MATTERPORT_TOKEN_ID}" \
  --password "${MATTERPORT_TOKEN_SECRET}" \
  --uids "${SCENE_GROUP}" \
  --data-path "${DEST_DIR}"

# ObjectNav HM3D v1 episode JSONs hardcode scene paths as `val/<scene>/...`,
# but the `hm3d_minival_full` group lays meshes under `minival/<scene>/...`.
# The minival 10 scenes are a strict subset of val, so a symlink resolves
# both paths to the same physical assets without re-downloading.
SCENE_HM3D_DIR="${DEST_DIR}/scene_datasets/hm3d"
if [[ -d "${SCENE_HM3D_DIR}/minival" && ! -e "${SCENE_HM3D_DIR}/val" ]]; then
  echo ">> Linking ${SCENE_HM3D_DIR}/val -> minival (so ObjectNav val episodes resolve)"
  ln -sfn minival "${SCENE_HM3D_DIR}/val"
fi

echo ">> Downloading ObjectNav HM3D v1 episodes from ${OBJECTNAV_URL}"
EPISODES_DIR="${DEST_DIR}/datasets/objectnav/hm3d/v1"
mkdir -p "${EPISODES_DIR}"
TMP_ZIP="$(mktemp -t objectnav_hm3d_v1.XXXXXX.zip)"
trap 'rm -f "${TMP_ZIP}"' EXIT
curl -fL --retry 3 -o "${TMP_ZIP}" "${OBJECTNAV_URL}"
# Zip layout is `objectnav_hm3d_v1/{train,val,val_mini,minival}/...`; strip
# the top-level dir so files land directly under v1/.
unzip -q -o "${TMP_ZIP}" -d "${EPISODES_DIR}/_unzipped"
SRC_DIR="$(find "${EPISODES_DIR}/_unzipped" -maxdepth 2 -type d -name "val*" -o -name "train" -o -name "minival" | head -n 1 | xargs -I{} dirname {})"
if [[ -n "${SRC_DIR}" ]]; then
  cp -R "${SRC_DIR}/"* "${EPISODES_DIR}/"
fi
rm -rf "${EPISODES_DIR}/_unzipped"

echo ">> Done. Verify with:"
echo "   ls ${DEST_DIR}/scene_datasets/hm3d | head"
echo "   ls ${EPISODES_DIR}"
