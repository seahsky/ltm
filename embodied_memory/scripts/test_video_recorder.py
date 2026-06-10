"""
Sanity test for ``video_recorder.write_video`` — the helper that turns a list of
first-person RGB frames (the ``Step.rgb`` stream collected during an episode) into
a watchable clip for the ``--save-video`` option of ``run_hm3d_pol``.

Why it matters: the episode loop already has ``step.rgb`` ((H,W,3) uint8) in hand
every step but discards it. ``write_video`` is the one place that persists those
frames, so it must (a) no-op cleanly on an empty buffer, (b) write a real,
non-empty file, and (c) degrade gracefully — an mp4 request must fall back to an
animated GIF when the ffmpeg backend is missing (true on dev boxes without
imageio-ffmpeg) rather than crashing the whole run.

imageio-only (no faiss/habitat). The mp4 path is environment-dependent (ffmpeg
backend present on the RACE habitat env, absent locally), so the mp4 case accepts
EITHER an .mp4 or a fallback .gif — it asserts a file was written, not which.

Invoke with::

    python embodied_memory/scripts/test_video_recorder.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

# Import the module directly (top-level, not via the embodied_memory package) so
# we don't trigger the package __init__'s faiss import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import video_recorder  # noqa: E402


def _frames(n=4, hw=32):
    """n distinct (hw,hw,3) uint8 frames (distinct so the writer can't collapse)."""
    out = []
    for i in range(n):
        f = np.zeros((hw, hw, 3), dtype=np.uint8)
        f[:, :, 0] = (i * 37) % 256          # vary red per frame
        f[i % hw, :, 1] = 255                # a moving green row
        out.append(f)
    return out


def case_empty_returns_none():
    assert video_recorder.write_video([], "/tmp/should_not_exist.mp4") is None
    assert video_recorder.write_video(None, "/tmp/should_not_exist.mp4") is None
    print("  case empty_returns_none: OK")


def case_writes_gif():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "clip.gif")
        out = video_recorder.write_video(_frames(4), path, fps=8)
        assert out == path, out
        assert os.path.exists(out), out
        assert os.path.getsize(out) > 0, "gif is empty"
    print("  case writes_gif: OK")


def case_creates_missing_parent_dir():
    with tempfile.TemporaryDirectory() as d:
        # nested dir that does not exist yet -> write_video must mkdir it
        path = os.path.join(d, "video", "episode_000.gif")
        out = video_recorder.write_video(_frames(3), path, fps=4)
        assert out == path, out
        assert os.path.exists(out), out
    print("  case creates_missing_parent_dir: OK")


def case_mp4_writes_or_falls_back_to_gif():
    """mp4 request: real mp4 where ffmpeg exists (RACE), else a .gif fallback.

    Either way a non-empty file must exist and the returned path must point at it.
    """
    with tempfile.TemporaryDirectory() as d:
        req = os.path.join(d, "episode_001.mp4")
        out = video_recorder.write_video(_frames(5), req, fps=8)
        assert out is not None, "mp4 request returned None (no fallback happened)"
        assert out.endswith(".mp4") or out.endswith(".gif"), out
        assert os.path.exists(out), out
        assert os.path.getsize(out) > 0, "written video is empty"
        # If ffmpeg was missing, the fallback must be the SAME stem as requested.
        if out.endswith(".gif"):
            assert os.path.splitext(out)[0] == os.path.splitext(req)[0], out
    print("  case mp4_writes_or_falls_back_to_gif: OK")


def case_roundtrip_frame_count():
    """Frames written are readable back (count preserved for the gif path)."""
    import imageio.v2 as iio

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rt.gif")
        n = 6
        video_recorder.write_video(_frames(n), path, fps=10)
        read = iio.mimread(path)
        assert len(read) == n, f"expected {n} frames, read {len(read)}"
    print("  case roundtrip_frame_count: OK")


def main():
    print("test_video_recorder")
    case_empty_returns_none()
    case_writes_gif()
    case_creates_missing_parent_dir()
    case_mp4_writes_or_falls_back_to_gif()
    case_roundtrip_frame_count()
    print("ALL PASS")


if __name__ == "__main__":
    main()
