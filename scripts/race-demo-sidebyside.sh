#!/bin/bash
# scripts/race-demo-sidebyside.sh — a SIDE-BY-SIDE memory-contrast demo clip in
# ONE command: memory-OFF (S1) on the LEFT, memory-ON (S3) on the RIGHT, of the
# SAME warm episode, so the memory effect is visually obvious (left wanders and
# never arrives; right recalls the prior sighting, routes to the goal, STOPs).
#
# This is COMPLEMENTARY to scripts/race-demo-video.sh (which makes the two-act
# WITH-SOUND AudioGoal story). This one is the silent A/B memory contrast on
# ObjectNav — the clearest single visual of the +0.24/+0.39 memory result. 7B
# backbone (the solid, published arc), NOT the experimental Phi/BLIP-2 stack.
#
# Pipeline: [1] pull  [2] setup  [3] S1-vs-S3 revisit with --save-video  [4] auto-
# pick the biggest-contrast warm episode + stitch the two clips side-by-side
# (make_demo_sidebyside.py) -> runs/<tag>/demo_sidebyside.mp4.
#
#   cd ~/ltm && git checkout lifelong-revisit-eval && git pull
#   nrun bash scripts/race-demo-sidebyside.sh                      # wcojb chair, n-warm 2
#   nrun bash scripts/race-demo-sidebyside.sh --categories toilet --scenes TEEsavR23oF
#
# Runtime ~30-50 min (a few episodes; warm ones can run to the 249-step cap).
# EXECUTE (do NOT source).
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT" || exit 1

SCENES="wcojb4TFT35"; CATEGORIES="chair"; NWARM=2; FPS=8; TAG="demo"
while [ $# -gt 0 ]; do
  case "$1" in
    --scenes)     SCENES="$2"; shift 2 ;;
    --categories) CATEGORIES="$2"; shift 2 ;;
    --n-warm)     NWARM="$2"; shift 2 ;;
    --fps)        FPS="$2"; shift 2 ;;
    --tag)        TAG="$2"; shift 2 ;;
    -h|--help)    sed -n '1,21p' "$0"; exit 0 ;;
    *) echo "FATAL: unknown arg $1"; exit 1 ;;
  esac
done
banner() { printf '\n########## %s ##########\n' "$1"; }

banner "[1/4] git pull --ff-only"
if [ -n "${RACE_SKIP_PULL:-}" ]; then echo "  RACE_SKIP_PULL set — skipping"; else
  git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }
fi

banner "[2/4] conda setup (for the stitch step; race-revisit re-sources in its own process)"
set +u; source scripts/race-setup.sh || { echo "FATAL: race-setup.sh failed"; exit 1; }; set -u
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

banner "[3/4] S1-vs-S3 revisit with --save-video (7B) — scenes=[$SCENES] cats=[$CATEGORIES] n-warm=$NWARM"
# race-revisit hardcodes --backbone remembr internally; do NOT pass --backbone.
bash scripts/race-revisit.sh --settings "1 3" --scenes "$SCENES" --categories "$CATEGORIES" \
    --n-warm "$NWARM" --save-video --video-fps "$FPS" --tag "$TAG" \
  || { echo "FATAL: race-revisit (--save-video) failed"; exit 1; }

S1_DIR="runs/${TAG}-s1"; S3_DIR="runs/${TAG}-s3"
banner "[4/4] pick the biggest-contrast warm episode + stitch side-by-side"
# Auto-pick the warm episode (n_steps>=20, i.e. NOT a start-at-goal cold) where S3
# succeeds and the S3-minus-S1 soft-SPL contrast is largest; resolve both clips.
SEL="$(python - "$S1_DIR" "$S3_DIR" <<'PY'
import json, os, sys
s1d, s3d = sys.argv[1], sys.argv[2]
def load(d):
    p = os.path.join(d, "summary.json")
    if not os.path.isfile(p): return {}
    return {int(e["episode_idx"]): e for e in (json.load(open(p)).get("episodes") or [])}
def vid(d, e):
    vp = e.get("video_path")
    if vp and os.path.isfile(os.path.join(d, vp)): return os.path.join(d, vp)
    for ext in (".mp4", ".gif"):
        p = os.path.join(d, "video", "episode_%03d%s" % (int(e["episode_idx"]), ext))
        if os.path.isfile(p): return p
    return None
e1, e3 = load(s1d), load(s3d)
shared = sorted(set(e1) & set(e3))
best, bestc = None, None
for idx in shared:
    a, b = e1[idx], e3[idx]
    if not b.get("success"): continue
    if int(b.get("n_steps", 0)) < 20: continue
    c = float(b.get("soft_spl", 0) or 0) - float(a.get("soft_spl", 0) or 0)
    if bestc is None or c > bestc: bestc, best = c, idx
if best is None:  # fallback: any successful S3 episode, highest soft_spl
    cand = [(float(e3[i].get("soft_spl", 0) or 0), i) for i in shared if e3[i].get("success")]
    best = max(cand)[1] if cand else (shared[-1] if shared else None)
if best is None:
    print("|||"); sys.exit(0)
L, R = vid(s1d, e1[best]), vid(s3d, e3[best])
note = "S3 soft_spl=%.3f vs S1 soft_spl=%.3f" % (
    float(e3[best].get("soft_spl", 0) or 0), float(e1[best].get("soft_spl", 0) or 0))
print("%d|%s|%s|%s" % (best, L or "", R or "", note))
PY
)"
IDX="${SEL%%|*}"; REST="${SEL#*|}"; LEFT="${REST%%|*}"; REST="${REST#*|}"; RIGHT="${REST%%|*}"; NOTE="${REST#*|}"
if [ -z "$IDX" ] || [ -z "$LEFT" ] || [ -z "$RIGHT" ]; then
  echo "  could not auto-pick a clean contrast episode. The per-episode clips are under:"
  echo "    $S1_DIR/video/   and   $S3_DIR/video/"
  echo "  Pick a matching episode_NNN and stitch manually:"
  echo "    python embodied_memory/scripts/make_demo_sidebyside.py \\"
  echo "      --left $S1_DIR/video/episode_NNN.mp4 --right $S3_DIR/video/episode_NNN.mp4 \\"
  echo "      --out runs/${TAG}/demo_sidebyside.mp4"
  exit 0
fi
echo "  picked episode $IDX  ($NOTE)"
echo "    left  (S1): $LEFT"
echo "    right (S3): $RIGHT"
OUT="runs/${TAG}/demo_sidebyside.mp4"; mkdir -p "runs/${TAG}"
python embodied_memory/scripts/make_demo_sidebyside.py \
    --left "$LEFT" --right "$RIGHT" --out "$OUT" --fps "$FPS" \
    --left-label "memory OFF (S1): wanders" --right-label "memory ON (S3): recall -> goal" \
  || { echo "FATAL: stitch failed"; exit 1; }
echo
echo "########## DONE — side-by-side memory demo ##########"
echo "  clip: $OUT"
echo "  (raw per-episode clips: $S1_DIR/video/  and  $S3_DIR/video/)"
exit 0
