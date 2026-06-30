#!/usr/bin/env python
"""make_demo_sidebyside — stitch two first-person episode clips into ONE
side-by-side demo video (e.g. memory-OFF | memory-ON of the SAME warm episode),
to visually display the memory effect.

Reads with imageio (the same dependency ``--save-video`` already uses) and writes
with ``embodied_memory.video_recorder`` (mp4, GIF fallback) — so it needs no
system ffmpeg CLI. The frame compositing (pad/letterbox/hstack/label) is pure
numpy/PIL and is unit-tested without any video files.

  python embodied_memory/scripts/make_demo_sidebyside.py \
      --left runs/demo-s1/video/episode_005.mp4 \
      --right runs/demo-s3/video/episode_005.mp4 \
      --out runs/demo/demo_sidebyside.mp4 \
      --left-label "memory OFF (S1)" --right-label "memory ON (S3)"
"""
import argparse
import os
import sys
from typing import List, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def pad_to_length(frames: Sequence[np.ndarray], n: int) -> List[np.ndarray]:
    """Repeat the last frame until the clip has ``n`` frames (freeze-frame hold
    so the shorter clip stays on its final state while the longer one finishes).
    Never truncates; empty input stays empty."""
    out = list(frames)
    if not out or len(out) >= n:
        return out
    out.extend([out[-1]] * (n - len(out)))
    return out


def _match_height(img: np.ndarray, h: int, fill: int = 0) -> np.ndarray:
    """Letterbox ``img`` vertically (centered) to height ``h``."""
    ih = img.shape[0]
    if ih == h:
        return img
    if ih > h:  # rare (same camera ⇒ same H); center-crop down
        top = (ih - h) // 2
        return img[top:top + h]
    pad = h - ih
    top = pad // 2
    return np.pad(img, ((top, pad - top), (0, 0), (0, 0)), mode="constant", constant_values=fill)


def stack_pair(left: np.ndarray, right: np.ndarray, gap_px: int = 8,
               gap_color=(30, 30, 30)) -> np.ndarray:
    """Horizontally stack two frames with a vertical divider bar; heights matched
    by letterboxing the shorter to the taller."""
    left = np.asarray(left, dtype=np.uint8)
    right = np.asarray(right, dtype=np.uint8)
    h = max(left.shape[0], right.shape[0])
    left = _match_height(left, h)
    right = _match_height(right, h)
    if gap_px > 0:
        gap = np.empty((h, gap_px, 3), dtype=np.uint8)
        gap[:, :] = np.asarray(gap_color, dtype=np.uint8)
        return np.concatenate([left, gap, right], axis=1)
    return np.concatenate([left, right], axis=1)


def stack_clips(left_frames: Sequence[np.ndarray], right_frames: Sequence[np.ndarray],
                gap_px: int = 8) -> List[np.ndarray]:
    """Pad both clips to the same length (freeze-frame hold) then stack each pair."""
    n = max(len(left_frames), len(right_frames))
    left = pad_to_length(left_frames, n)
    right = pad_to_length(right_frames, n)
    return [stack_pair(lf, rf, gap_px=gap_px) for lf, rf in zip(left, right)]


def add_label_bar(frame: np.ndarray, left_text: str, right_text: str,
                  bar_h: int = 30) -> np.ndarray:
    """Prepend a label bar (left text on the left half, right text on the right
    half) above ``frame``. PIL-rendered; if PIL is unavailable the bar is blank
    but the composite height still grows by ``bar_h``."""
    frame = np.asarray(frame, dtype=np.uint8)
    w = frame.shape[1]
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(bar)
        d = ImageDraw.Draw(im)
        y = max(0, bar_h // 2 - 6)
        d.text((8, y), left_text, fill=(255, 255, 255))
        d.text((w // 2 + 8, y), right_text, fill=(120, 255, 120))
        bar = np.asarray(im, dtype=np.uint8)
    except Exception:
        pass
    return np.concatenate([bar, frame], axis=0)


def read_frames(path: str) -> List[np.ndarray]:
    import imageio.v2 as iio
    return [np.asarray(f, dtype=np.uint8)[..., :3] for f in iio.mimread(path, memtest=False)]


def build_sidebyside(left_path: str, right_path: str, *, left_label: str, right_label: str,
                     gap_px: int = 8, bar_h: int = 30) -> List[np.ndarray]:
    stacked = stack_clips(read_frames(left_path), read_frames(right_path), gap_px=gap_px)
    if bar_h > 0:
        stacked = [add_label_bar(f, left_label, right_label, bar_h=bar_h) for f in stacked]
    return stacked


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stitch two episode clips side-by-side.")
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--left-label", default="memory OFF (S1)")
    ap.add_argument("--right-label", default="memory ON (S3)")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--gap", type=int, default=8)
    ap.add_argument("--bar-h", type=int, default=30)
    a = ap.parse_args(argv)
    for p in (a.left, a.right):
        if not os.path.isfile(p):
            print(f"FATAL: input not found: {p}")
            return 1
    frames = build_sidebyside(a.left, a.right, left_label=a.left_label,
                              right_label=a.right_label, gap_px=a.gap, bar_h=a.bar_h)
    from embodied_memory import video_recorder
    out = video_recorder.write_video(frames, a.out, fps=a.fps)
    if not out:
        print("FATAL: no video written (imageio missing?)")
        return 1
    print(f"wrote side-by-side demo: {out}  ({len(frames)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
