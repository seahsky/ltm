#!/usr/bin/env bash
# One-shot: repair HM3D semantic-file placement (so sim.semantic_scene.regions
# populates) and re-confirm with the region diagnostic. Run after the minival
# semantic download (download_hm3d_semantics.sh) when regions are still 0.
#
#   git pull --ff-only
#   bash scripts/fix_hm3d_semantics.sh
#
# Step 1 (filesystem, instant): inventory geometry vs semantic files per scene and
#   symlink the semantic files next to the geometry the sim reads (the likely
#   minival/->val/ placement mismatch). Idempotent.
# Step 2 (sim, ~seconds/scene): re-run diagnose_hm3d_regions and show ONLY the
#   region result (GL/plugin noise filtered).

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

echo "########## STEP 1: inspect + repair semantic-file placement ##########"
python3 embodied_memory/scripts/fix_hm3d_semantics.py --root data/hm3d

echo ""
echo "########## STEP 2: re-confirm regions (diagnostic, noise filtered) ##########"
python3 embodied_memory/scripts/diagnose_hm3d_regions.py --scene all \
    --episodes-path data/hm3d/datasets/objectnav/hm3d/v1/val_mini/val_mini.json.gz 2>&1 \
  | grep -vE "PluginManager|GL_[A-Z]|Renderer:|OpenGL version|Using (optional features|driver workarounds)|^    (no-|nv-|GL_)|Gym has been|migration guide|gymnasium|duplicate static plugin|Initializing (dataset|sim|task)|Lighting Layout|^\s*$" \
  || true

echo ""
echo "########## DONE — read the region count + names above ##########"
echo "If 'regions: N>0' with room names -> paste them; I'll write the room-type mapping."
echo "If still 0 + a '.basis.scn missing' WARNING -> it's a config/naming issue, paste STEP 1 output."
