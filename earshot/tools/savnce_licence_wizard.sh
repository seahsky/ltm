#!/bin/bash
# earshot/tools/savnce_licence_wizard.sh — stage the data SAVN-CE needs (ADR-0015).
#
#   bash earshot/tools/savnce_licence_wizard.sh
#
# This walks the steps NO AGENT CAN DO FOR YOU, and does the mechanical parts around
# them. The first step is a dataset licence: MP3D is not a wget, it is a Terms of Use you
# agree to as a person, and the record of having agreed matters.
#
# Safe to re-run. Every step checks whether it is already done and says so.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SAVNCE_DATA_ROOT="${SAVNCE_DATA_ROOT:-${HOME}/savnce-data}"
RECORD="$SAVNCE_DATA_ROOT/.licence-record"

MP3D_PAGE="https://niessner.github.io/Matterport/"
SCENE_CONFIG_URL="http://dl.fbaipublicfiles.com/habitat/mp3d/config_v1/mp3d.scene_dataset_config.json"
SOUNDS_URL="http://dl.fbaipublicfiles.com/SoundSpaces/sounds.tar.xz"
DRIVE_FOLDER="https://drive.google.com/drive/folders/1tz92HS9JsWmZuSnFjK513eu-Q0NYnTHT"

step() { printf '\n\033[1m--- %s ---\033[0m\n' "$1"; }
ask()  { local reply; read -r -p "$1 [y/N] " reply; [ "$reply" = "y" ] || [ "$reply" = "Y" ]; }

mkdir -p "$SAVNCE_DATA_ROOT"
echo "data root: $SAVNCE_DATA_ROOT   (override with SAVNCE_DATA_ROOT)"

# ----------------------------------------------------------------------
step "1 of 5   the Matterport3D licence"
# ----------------------------------------------------------------------
if [ -s "$RECORD" ]; then
  echo "already recorded:"
  cat "$RECORD"
elif [ -d "$SAVNCE_DATA_ROOT/scene_datasets/mp3d" ] && [ "$(find "$SAVNCE_DATA_ROOT/scene_datasets/mp3d" -name '*.glb' 2>/dev/null | wc -l)" -gt 5 ]; then
  echo "MP3D scenes are already staged; skipping the licence prompt."
else
  cat <<TEXT
MP3D is licensed, not public. You need the Matterport3D Terms of Use signed and the
per-user download script 'download_mp.py' that follows from it.

  1. Open  $MP3D_PAGE
  2. Follow its instructions to agree to the Terms of Use.
  3. They send you 'download_mp.py'.

This is days, not minutes. Nothing else in this wizard is blocked by it, so run the
remaining steps now and come back.
TEXT
  if ask "Do you already hold download_mp.py?"; then
    read -r -p "  path to download_mp.py: " SCRIPT_PATH
    if [ -f "$SCRIPT_PATH" ]; then
      printf 'agreed_recorded_at=%s\ndownload_mp=%s\n' "$(date -Iseconds)" "$SCRIPT_PATH" > "$RECORD"
      echo "recorded in $RECORD"
      echo
      echo "Run this yourself — it is long, and it is yours to supervise:"
      echo "  python2 $SCRIPT_PATH --task habitat -o $SAVNCE_DATA_ROOT/scene_datasets"
      echo "(DATASETS.md notes download_mp.py needs python 2.7. Only the habitat archive"
      echo " is needed, not the full Matterport3D release.)"
    else
      echo "  not found at $SCRIPT_PATH — nothing recorded."
    fi
  else
    echo "Fine. Come back to step 1 when it arrives."
  fi
fi

# ----------------------------------------------------------------------
step "2 of 5   the habitat scene-dataset config"
# ----------------------------------------------------------------------
CONFIG_DEST="$SAVNCE_DATA_ROOT/scene_datasets/mp3d/mp3d.scene_dataset_config.json"
if [ -f "$CONFIG_DEST" ]; then
  echo "already present: $CONFIG_DEST"
elif [ -d "$SAVNCE_DATA_ROOT/scene_datasets/mp3d" ]; then
  wget -q -O "$CONFIG_DEST" "$SCENE_CONFIG_URL" && echo "downloaded $CONFIG_DEST" \
    || echo "download failed — fetch it by hand: $SCENE_CONFIG_URL"
else
  echo "skipped: no scene_datasets/mp3d yet (step 1 first)"
fi

# ----------------------------------------------------------------------
step "3 of 5   the SoundSpaces sound assets"
# ----------------------------------------------------------------------
if [ -d "$SAVNCE_DATA_ROOT/sounds" ]; then
  echo "already present: $SAVNCE_DATA_ROOT/sounds"
elif ask "Download sounds.tar.xz now?"; then
  ( cd "$SAVNCE_DATA_ROOT" && wget -q --show-progress "$SOUNDS_URL" && tar xf sounds.tar.xz ) \
    && echo "unpacked into $SAVNCE_DATA_ROOT/sounds" \
    || echo "failed — the host has been unreliable; fetch by hand: $SOUNDS_URL"
fi

# ----------------------------------------------------------------------
step "4 of 5   episodes and released checkpoints (Google Drive)"
# ----------------------------------------------------------------------
if [ -d "$SAVNCE_DATA_ROOT/pretrained_ckpts" ] && [ -d "$SAVNCE_DATA_ROOT/datasets" ]; then
  echo "already present."
else
  cat <<TEXT
Their episode dataset and released checkpoints live in one Drive FOLDER:
  $DRIVE_FOLDER

gdown first. If it hits a quota wall — which it does — download in a browser and scp the
tree to $SAVNCE_DATA_ROOT. Step 5's checksums make that hand-carried path safe.
TEXT
  if ask "Try gdown now?"; then
    python -m pip install -q gdown
    ( cd "$SAVNCE_DATA_ROOT" && python -m gdown --folder "$DRIVE_FOLDER" ) \
      || echo "gdown failed. Use the browser route; step 5 will verify what arrives."
  fi
fi
# Their docs say `savnce-dataset`, their config reads `savnce_dataset`. The config wins.
if [ -d "$SAVNCE_DATA_ROOT/datasets/savnce-dataset" ] && [ ! -e "$SAVNCE_DATA_ROOT/datasets/savnce_dataset" ]; then
  ln -s "$SAVNCE_DATA_ROOT/datasets/savnce-dataset" "$SAVNCE_DATA_ROOT/datasets/savnce_dataset" \
    && echo "reconciled the hyphen/underscore mismatch between their docs and their config"
fi

# ----------------------------------------------------------------------
step "5 of 5   pin what arrived"
# ----------------------------------------------------------------------
echo "Checkpoints and episode files are hashed; scene meshes are recorded by size."
if ask "Record checksums into earshot/tools/savnce-artifacts.json now?"; then
  PYTHONPATH="$REPO_ROOT" python -m earshot.tools.savnce_artifacts record --data-root "$SAVNCE_DATA_ROOT"
  echo
  echo "COMMIT that manifest. It is the only record of which Drive revision produced the number."
else
  PYTHONPATH="$REPO_ROOT" python -m earshot.tools.savnce_artifacts verify --data-root "$SAVNCE_DATA_ROOT"
fi

printf '\nNext:  bash earshot/tools/savnce_bootstrap.sh\n'
