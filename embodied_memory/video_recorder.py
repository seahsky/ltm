"""
video_recorder — persist a first-person RGB stream to a watchable clip.

The episode loop has ``Step.rgb`` ((H,W,3) uint8) in hand every step but normally
discards it. When ``run_hm3d_pol --save-video`` is set, ``EpisodeRunner`` buffers
those frames per episode and hands them here at episode end.

Design goals:
- **No hard dependency surprises.** imageio is imported lazily; if it's missing,
  we warn and write nothing rather than crashing a multi-hour ablation.
- **Graceful container fallback.** mp4/webm/... need the ffmpeg backend
  (imageio-ffmpeg), which is present on the RACE habitat env but often absent on
  dev boxes. If the requested container can't be written, we fall back to an
  animated GIF at the SAME stem and return that path. The caller logs whatever
  path comes back, so the run never dies over a missing codec.

``write_video`` returns the path actually written (which may differ from the
requested one when it falls back to GIF), or ``None`` if nothing was written.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

# Containers that require the imageio ffmpeg plugin.
_FFMPEG_EXTS = {".mp4", ".webm", ".avi", ".mov", ".mkv"}


def write_video(
    frames: Optional[Sequence],
    path: str,
    fps: int = 8,
) -> Optional[str]:
    """Write ``frames`` (list of (H,W,3) uint8 arrays) to ``path``.

    Tries the requested container; on a missing ffmpeg backend, falls back to an
    animated GIF at the same stem. Returns the written path or ``None``.
    """
    if frames is None or len(frames) == 0:
        return None

    try:
        import imageio.v2 as iio
    except Exception as e:  # imageio absent / broken — don't take the run down
        print(f"[video] imageio unavailable ({type(e).__name__}: {e}); no video written")
        return None

    import numpy as np

    seq: List = [np.ascontiguousarray(np.asarray(f, dtype=np.uint8)) for f in frames]
    fps = max(1, int(fps))

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    def _write_gif(p: str) -> None:
        # The pillow GIF plugin takes per-frame ``duration`` (seconds) on some
        # imageio versions and ``fps`` on others — try fps, then duration.
        try:
            iio.mimsave(p, seq, fps=fps)
        except TypeError:
            iio.mimsave(p, seq, duration=1.0 / float(fps))

    ext = os.path.splitext(path)[1].lower()

    if ext in _FFMPEG_EXTS:
        try:
            # macro_block_size=None avoids ffmpeg silently resizing odd dims.
            iio.mimsave(path, seq, fps=fps, macro_block_size=None)
            return path
        except Exception as e:
            gif_path = os.path.splitext(path)[0] + ".gif"
            print(
                f"[video] {ext} writer unavailable ({type(e).__name__}: {e}); "
                f"falling back to GIF -> {os.path.basename(gif_path)}"
            )
            try:
                _write_gif(gif_path)
                return gif_path
            except Exception as e2:
                print(
                    f"[video] GIF fallback also failed "
                    f"({type(e2).__name__}: {e2}); no video written"
                )
                return None

    # Non-ffmpeg container (.gif or other pillow format).
    try:
        _write_gif(path)
        return path
    except Exception as e:
        print(f"[video] failed to write {path} ({type(e).__name__}: {e}); no video written")
        return None
