#!/usr/bin/env python
"""
make_two_act_demo — stitch ONE two-act story video from a single S3 audiogoal run.

The driver (``scripts/race-demo-video.sh``) runs a COLD seed episode (the robot is
spawned at the bed, observes & memorizes it) followed by one or more WARM recall
episodes (the alarm fires, the robot recalls where the bed was and navigates to
it). Both fire the alarm (the driver passes ``t_anom=5`` to every episode) so both
have an audible soundtrack and concat cleanly.

This script turns two chosen episodes into a single legible demo:

  ACT 1 = the COLD seed episode  ("FIRST VISIT: finds & memorizes the bed")
  ACT 2 = the WARM recall episode ("RETURN VISIT: alarm fires; recalls & navigates")

For each act it:
  (a) reads the act's SILENT first-person mp4 (already HUD-overlaid by the live
      runner) from ``ep["video_path"]``,
  (b) burns a full-width TITLE BANNER across the very top of every frame (so the
      two-act story is readable; the band sits above the existing HUD block),
  (c) writes a banner_<out_name>.mp4 (still silent),
  (d) muxes the pose-conditioned anomaly soundtrack onto THAT banner video via
      :func:`render_demo_audio_track.process_episode` (reused verbatim, with the
      new ``silent_mp4_override``),
then ffmpeg-CONCATENATES the two muxed act clips (concat FILTER → re-encode, robust
to minor stream differences) into ``runs/<...>/demo_two_act.mp4``.

Two-env split preserved: like ``render_demo_audio_track`` this imports only
numpy / scipy (via ``embodied_memory.audio``) + PIL + imageio — never habitat_sim.

Graceful degrade (mirrors ``render_demo_audio_track``): if ffmpeg is absent we
still write each muxed per-act mp4 and print the manual concat command (exit 0); we
exit non-zero only on a genuine failure — and a ``SilentSoundtrackError`` from
either act ALWAYS propagates so a silent act can never become a shipped demo.

The pure parts (``burn_banner``, ``concat_command``) take plain numpy / lists and
are unit-tested WITHOUT habitat, ffmpeg, or imageio file I/O.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from embodied_memory.scripts import render_demo_audio_track
from embodied_memory.scripts.render_demo_audio_track import (
    SilentSoundtrackError,
    _find_ffmpeg,
)


# ----------------------------------------------------------------------
# pure banner draw (no habitat, no ffmpeg, no I/O) — mirrors video_overlay.py
# ----------------------------------------------------------------------


def _font(size: int):
    """Best-effort truetype font at ``size`` px; falls back to Pillow's bitmap
    default so there is NO hard font dependency (mirrors video_overlay._load_font's
    fallback chain)."""
    from PIL import ImageFont  # lazy

    for path in ("DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


# Height of the title band, in pixels. ~28 px is the spec; kept >= a small floor
# for tiny test frames so the painted strip is always non-degenerate.
_BANNER_H = 28


def _band_height(frame_h: int) -> int:
    """The banner band height for a frame this tall (never taller than the frame)."""
    return max(8, min(_BANNER_H, max(1, frame_h // 3)))


def burn_banner(
    frames: Union[List[np.ndarray], np.ndarray],
    text: str,
    *,
    where: str = "top",
) -> List[np.ndarray]:
    """Draw a solid dark title strip with ``text`` across the TOP of every frame.

    The band sits at the very top (a full-width dark bar + white text), ABOVE the
    live runner's HUD block (whose own dark box starts at y=0 but is narrower) so
    the act title is always legible without overwriting the HUD content lower down.

    Pure & defensive (mirrors ``video_overlay.draw_overlay``):
      * never mutates the caller's frames (copies each),
      * preserves frame COUNT, ORDER, dtype (uint8) and SHAPE,
      * on any failure (PIL missing, odd shape) returns the frame unchanged.

    Accepts a list of (H,W,3) uint8 arrays OR a single (T,H,W,3) array; always
    returns a list. ``where`` is accepted for symmetry; only "top" is drawn (the
    HUD lives top-left, so a full-width top band is the safe non-colliding band).
    """
    if isinstance(frames, np.ndarray) and frames.ndim == 4:
        seq = [frames[i] for i in range(frames.shape[0])]
    else:
        seq = list(frames)

    try:
        from PIL import Image, ImageDraw
    except Exception:  # noqa: BLE001 — no PIL: return untouched copies
        return [np.array(f, dtype=np.uint8, copy=True) for f in seq]

    out: List[np.ndarray] = []
    for f in seq:
        try:
            arr = np.ascontiguousarray(f)
            if arr.ndim != 3 or arr.shape[2] != 3:
                out.append(np.array(f, dtype=np.uint8, copy=True))
                continue
            arr = arr.astype(np.uint8, copy=True)
            h, w = arr.shape[:2]
            band = _band_height(h)

            img = Image.fromarray(arr)
            draw = ImageDraw.Draw(img)
            # full-width dark band across the top
            draw.rectangle([0, 0, w, band], fill=(0, 0, 0))
            font = _font(max(10, band - 8))
            pad = max(2, w // 200)
            # vertically centre the text within the band
            try:
                tb = draw.textbbox((0, 0), text, font=font)
                th = tb[3] - tb[1]
            except Exception:  # noqa: BLE001 — very old Pillow
                th = band - 4
            ty = max(0, (band - th) // 2)
            draw.text((pad, ty), text, font=font, fill=(255, 255, 255))
            out.append(np.ascontiguousarray(np.asarray(img, dtype=np.uint8)))
        except Exception:  # noqa: BLE001 — a bad frame never breaks the stitch
            out.append(np.array(f, dtype=np.uint8, copy=True))
    return out


# ----------------------------------------------------------------------
# ffmpeg concat-filter command (pure, testable string builder)
# ----------------------------------------------------------------------


def concat_command(clips: Sequence[str], out: str, ffmpeg: str = "ffmpeg") -> List[str]:
    """ffmpeg argv that concatenates N (video+audio) ``clips`` → ``out``.

    Uses the concat FILTER (``-filter_complex …concat=n=N:v=1:a=1``) which
    RE-ENCODES, so it is robust to the minor stream differences between
    independently-muxed act clips (the concat *demuxer* requires identical codecs/
    timebases and would fail/desync here). Each clip is one ``-i`` input; the
    filter consumes ``[i:v][i:a]`` for every input and emits ``[outv][outa]`` which
    are mapped to the output."""
    cmd: List[str] = [ffmpeg, "-y"]
    for c in clips:
        cmd += ["-i", c]
    n = len(clips)
    streams = "".join(f"[{i}:v:0][{i}:a:0]" for i in range(n))
    filt = f"{streams}concat=n={n}:v=1:a=1[outv][outa]"
    cmd += [
        "-filter_complex", filt,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        out,
    ]
    return cmd


# ----------------------------------------------------------------------
# imageio frame I/O (impure) — mirrors video_recorder's imageio usage
# ----------------------------------------------------------------------


def read_frames(mp4: str) -> List[np.ndarray]:
    """Read all frames of ``mp4`` as a list of (H,W,3) uint8 arrays."""
    import imageio.v2 as iio

    out: List[np.ndarray] = []
    rdr = iio.get_reader(mp4)
    try:
        for fr in rdr:
            out.append(np.asarray(fr, dtype=np.uint8))
    finally:
        rdr.close()
    return out


def write_frames(mp4: str, frames: Sequence[np.ndarray], fps: float) -> str:
    """Write ``frames`` to ``mp4`` (macro_block_size=None, as video_recorder)."""
    import imageio.v2 as iio

    parent = os.path.dirname(os.path.abspath(mp4))
    os.makedirs(parent, exist_ok=True)
    seq = [np.ascontiguousarray(np.asarray(f, dtype=np.uint8)) for f in frames]
    iio.mimsave(mp4, seq, fps=max(1, int(round(fps))), macro_block_size=None)
    return mp4


# ----------------------------------------------------------------------
# per-act stitch (impure: reads/writes video, muxes soundtrack)
# ----------------------------------------------------------------------


def stitch_act(
    run_dir: str,
    episode_json: str,
    rir_grid: str,
    anomaly_class: Optional[str],
    t_anom: Optional[int],
    fps: float,
    anomaly_clip: Optional[str],
    banner_text: str,
    out_name: str,
) -> Dict[str, Any]:
    """Banner-overlay one act's silent clip, then mux its anomaly soundtrack.

    (a) read the act's silent mp4 (``ep["video_path"]``, relative to ``run_dir``),
    (b) ``burn_banner`` onto its frames,
    (c) write a ``banner_<out_name>.mp4`` (still silent),
    (d) ``render_demo_audio_track.process_episode(..., silent_mp4_override=<banner
        silent mp4>, out_name=out_name)`` so the soundtrack muxes onto the bannered
        video.

    Returns the process_episode result dict (muxed mp4 path + track_rms + n_frames).
    A ``SilentSoundtrackError`` from the mux step propagates (refuse a silent act).
    """
    with open(episode_json, "r", encoding="utf-8") as f:
        ep = json.load(f)
    vid_rel = ep.get("video_path")
    if not vid_rel:
        raise FileNotFoundError(
            f"{episode_json} has no video_path — was --save-video set on the run?")
    silent_mp4 = os.path.join(run_dir, vid_rel)
    if not os.path.isfile(silent_mp4):
        idx = int(ep.get("episode_idx", 0))
        guess = os.path.join(run_dir, "video", f"episode_{idx:03d}.mp4")
        if os.path.isfile(guess):
            silent_mp4 = guess
        else:
            raise FileNotFoundError(
                f"silent mp4 not found for act ({silent_mp4}); see {episode_json}")

    frames = read_frames(silent_mp4)
    bannered = burn_banner(frames, banner_text, where="top")
    banner_mp4 = os.path.join(run_dir, f"banner_{out_name}")
    write_frames(banner_mp4, bannered, fps)

    # mux the anomaly soundtrack onto the BANNER video (override), keeping the
    # poses / onset / RMS-silence guard all driven by the same episode JSON.
    res = render_demo_audio_track.process_episode(
        run_dir, rir_grid,
        episode_json=episode_json,
        anomaly_clip=anomaly_clip,
        anomaly_class=anomaly_class,
        t_anom=t_anom,
        fps=fps,
        out_name=out_name,
        silent_mp4_override=banner_mp4,
    )
    res["banner_mp4"] = banner_mp4
    res["banner_text"] = banner_text
    return res


# ----------------------------------------------------------------------
# CLI / orchestration
# ----------------------------------------------------------------------

_DEFAULT_ACT1 = "ACT 1 - FIRST VISIT: the robot finds & memorizes the bed"
_DEFAULT_ACT2 = ("ACT 2 - RETURN VISIT: alarm fires; the robot recalls the bed "
                 "& navigates to it")


def _est_duration(res: Dict[str, Any], fps: float) -> float:
    n = int(res.get("n_frames", 0) or 0)
    return (n / float(fps)) if fps else 0.0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Stitch a two-act (cold-seed + warm-recall) demo video")
    p.add_argument("--run-dir", required=True,
                   help="The S3 run output dir (holds episode_NNN.json + video/)")
    p.add_argument("--act1-episode", required=True,
                   help="Cold seed episode_NNN.json (ACT 1 — finds/memorizes)")
    p.add_argument("--act2-episode", required=True,
                   help="Warm recall episode_NNN.json (ACT 2 — recalls/navigates)")
    p.add_argument("--rir-grid", required=True,
                   help="The rendered RIR grid .npz (same one the run used)")
    p.add_argument("--anomaly-clip", default=None,
                   help="Explicit anomaly .wav (else resolved from --anomaly-class)")
    p.add_argument("--anomaly-class", default=None,
                   choices=["baby_cry", "alarm", "glass_break"],
                   help="Resolve data/anomaly_audio/<class>.wav when clip unset")
    p.add_argument("--t-anom", type=int, default=None,
                   help="Run-level onset step (silence before it).")
    p.add_argument("--fps", type=float, default=4.0,
                   help="Video fps the silent mp4s were written at (match --video-fps)")
    p.add_argument("--out-name", default="demo_two_act.mp4")
    p.add_argument("--act1-label", default=_DEFAULT_ACT1)
    p.add_argument("--act2-label", default=_DEFAULT_ACT2)
    args = p.parse_args(argv)

    acts = [
        ("act1.mp4", args.act1_episode, args.act1_label),
        ("act2.mp4", args.act2_episode, args.act2_label),
    ]

    results: List[Dict[str, Any]] = []
    try:
        for out_name, ep_json, label in acts:
            res = stitch_act(
                run_dir=args.run_dir,
                episode_json=ep_json,
                rir_grid=args.rir_grid,
                anomaly_class=args.anomaly_class,
                t_anom=args.t_anom,
                fps=args.fps,
                anomaly_clip=args.anomaly_clip,
                banner_text=label,
                out_name=out_name,
            )
            results.append(res)
            dur = _est_duration(res, args.fps)
            print(f"[two-act] {label}\n         -> {res.get('muxed_mp4')}  "
                  f"({res.get('n_frames')} frames, ~{dur:.1f}s, "
                  f"RMS={res.get('track_rms', 0.0):.4f})")
    except SilentSoundtrackError as e:
        print(f"FATAL: a demo act is SILENT (refusing to ship): {e}",
              file=sys.stderr)
        return 2

    act_clips = [r.get("muxed_mp4") for r in results]
    if any(c is None or not os.path.isfile(c) for c in act_clips):
        # an act produced no muxed mp4 (e.g. ffmpeg absent at the per-act mux):
        # the per-act .wav + silent banner clip are intact — degrade, don't crash.
        print("[two-act] ffmpeg absent for the per-act mux — wrote each act's "
              "soundtrack + banner clip; cannot concat. Per-act outputs:")
        for r in results:
            print(f"  banner clip: {r.get('banner_mp4')}  wav: {r.get('wav')}")
        return 0

    out_path = os.path.join(args.run_dir, args.out_name)
    ffmpeg = _find_ffmpeg()
    cmd = concat_command(act_clips, out_path, ffmpeg=ffmpeg or "ffmpeg")
    manual = " ".join(cmd)

    if ffmpeg is None:
        print("[two-act] ffmpeg not found — wrote both per-act mp4s. Concat them "
              "manually with:\n  " + manual)
        for r, (out_name, _, _) in zip(results, acts):
            print(f"  act: {r.get('muxed_mp4')}")
        return 0

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:  # noqa: BLE001 — concat failed: per-act clips intact
        print(f"[two-act] ffmpeg concat failed ({type(e).__name__}: {e}); the "
              f"per-act mp4s are intact. Concat manually:\n  " + manual)
        for r in results:
            print(f"  act: {r.get('muxed_mp4')}")
        return 1

    total = sum(_est_duration(r, args.fps) for r in results)
    print(f"[two-act] wrote {out_path}")
    for r, (_, _, label) in zip(results, acts):
        print(f"  {label}: {r.get('n_frames')} frames, "
              f"~{_est_duration(r, args.fps):.1f}s, RMS={r.get('track_rms', 0.0):.4f}")
    print(f"TWO_ACT_OUTPUT={out_path}")
    print(f"TWO_ACT_SECONDS={total:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
