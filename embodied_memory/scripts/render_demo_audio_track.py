#!/usr/bin/env python
"""
render_demo_audio_track — POST-HOC, pose-conditioned demo SOUNDTRACK + mux.

The live runner records a SILENT first-person mp4 (``video_recorder.write_video``
writes RGB frames only — no audio track) but persists everything needed to
RE-SYNTHESIZE the anomaly soundtrack the agent "heard":

  * per-keyframe agent poses        ``episode_NNN.json["steps"][i]["agent_pos"]``
  * the anomaly source xyz          ``episode_NNN.json["source_position"]``
  * the silent video                ``episode_NNN.json["video_path"]``
  * onset boundary                  ``--t-anom`` (run-level) or the first step
                                    with non-zero ``audio_energy`` (auto-detect)

Given those + the RIR grid ``.npz`` (rendered offline) + the resolved anomaly
clip, this script rebuilds a binaural soundtrack — silence before onset, then
``audio.render_at_pose`` at each frame's pose so the alarm SWELLS as the agent
nears the source (per-cell IR energy is a monotone proxy for audibility) — writes
``demo_track.wav``, and FFMPEG-MUXES it onto the silent mp4 → ``demo_with_sound.mp4``.

Two-env split preserved: this imports only ``numpy``/``scipy`` via
``embodied_memory.audio`` (``RIRGrid`` + ``render_at_pose`` never touch
habitat_sim). The grid was rendered in the soundspaces env; here we only convolve.

Graceful degrade: if ffmpeg is absent we still write the .wav (+ keep the silent
.mp4) and PRINT the exact manual mux command.

The soundtrack-assembly maths (pose→chunk→concatenate, fps alignment, onset
gating) lives in pure functions (``build_soundtrack`` / ``mux_command`` /
``onset_from_steps``) that take a grid + clip + a plain pose list, so they are
unit-testable WITHOUT habitat or ffmpeg.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# audio.py imports only numpy + scipy (the live, no-habitat side of the split).
from embodied_memory.audio import RIRGrid, render_at_pose
from embodied_memory.audio_task import build_anomaly_clip, resolve_anomaly_clip


# ----------------------------------------------------------------------
# pure soundtrack assembly (no habitat, no ffmpeg, no I/O)
# ----------------------------------------------------------------------


def samples_per_frame(sample_rate: int, fps: float) -> int:
    """How many audio samples cover one video frame at ``fps``. >= 1."""
    return max(1, int(round(float(sample_rate) / float(fps))))


def onset_from_steps(
    steps: Sequence[Dict[str, Any]], t_anom: Optional[int]
) -> int:
    """Resolve the onset *frame index* (into ``steps``) at which the anomaly
    starts. An explicit ``t_anom`` (the run-level onset STEP) wins: the first
    frame whose ``step_idx >= t_anom`` is the onset frame. Else auto-detect from
    the first frame with non-zero ``audio_energy`` (silence reads 0.0 before
    onset). Falls back to 0 (whole track audible) when neither is available."""
    if t_anom is not None:
        for i, s in enumerate(steps):
            if int(s.get("step_idx", i)) >= int(t_anom):
                return i
        return len(steps)  # anomaly never reached within the recorded frames
    for i, s in enumerate(steps):
        e = s.get("audio_energy")
        if e is not None and float(e) > 0.0:
            return i
    return 0


def build_soundtrack(
    grid: RIRGrid,
    clip: np.ndarray,
    poses: Sequence[Sequence[float]],
    *,
    sample_rate: int,
    fps: float,
    onset_frame: int,
    clip_offset_samples: int = 0,
) -> np.ndarray:
    """Assemble a ``(2, n_frames * samples_per_frame)`` binaural soundtrack.

    For each frame i:
      * i < ``onset_frame`` → a silent chunk (pre-anomaly).
      * else → ``render_at_pose(grid, poses[i], window)`` for a window of the
        clip, truncated/padded to exactly ``samples_per_frame`` samples. The
        per-cell IR energy makes nearer poses louder automatically, so the
        soundtrack swells as the agent approaches the source.

    ``clip_offset_samples`` lets the caller advance the clip read-head across
    frames so a short clip is heard continuously (looped) rather than re-onset
    every frame. The default 0 renders the clip's leading window at every frame
    (each frame is the clip's onset convolved at that pose) — still energy-
    correct; the driver uses a per-frame advance for a continuous sound.
    """
    spf = samples_per_frame(sample_rate, fps)
    n = len(poses)
    out = np.zeros((2, n * spf), dtype=np.float32)
    clip = np.asarray(clip, dtype=np.float32).reshape(-1)
    if clip.size == 0:
        return out
    for i in range(n):
        if i < onset_frame:
            continue  # silence before the anomaly
        # window of the clip for this frame (loop the clip by offset)
        start = (clip_offset_samples * (i - onset_frame)) % clip.size
        # take spf samples, wrapping the clip so a short clip is continuous
        idx = (np.arange(spf) + start) % clip.size
        window = clip[idx]
        chunk = render_at_pose(grid, poses[i], window, max_len=spf)
        # render_at_pose returns (2, <=spf); pad to spf if convolution is short
        L = chunk.shape[1]
        if L >= spf:
            out[:, i * spf:(i + 1) * spf] = chunk[:, :spf]
        else:
            out[:, i * spf:i * spf + L] = chunk
    return out


def mux_command(
    silent_mp4: str, wav: str, out_mp4: str, ffmpeg: str = "ffmpeg"
) -> List[str]:
    """The ffmpeg argv that muxes ``wav`` onto ``silent_mp4`` → ``out_mp4``.

    ``-c:v copy`` keeps the video stream byte-identical; the audio is AAC.
    ``-shortest`` trims to the shorter of the two streams so a slightly longer
    soundtrack doesn't extend a black tail. ``-map`` pins video from input 0 and
    audio from input 1."""
    return [
        ffmpeg, "-y",
        "-i", silent_mp4,
        "-i", wav,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        out_mp4,
    ]


# ----------------------------------------------------------------------
# I/O glue (impure: reads JSON, writes wav, shells ffmpeg)
# ----------------------------------------------------------------------


def write_wav(path: str, binaural: np.ndarray, sample_rate: int) -> None:
    """Write a ``(2, L)`` float32 binaural array to a 16-bit stereo WAV."""
    from scipy.io import wavfile

    x = np.asarray(binaural, dtype=np.float32)
    if x.ndim == 1:
        x = np.stack([x, x], axis=0)
    # peak-normalize defensively so the mux doesn't clip, then to int16.
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.0:
        x = x / peak
    interleaved = x.T  # (L, 2) — wavfile wants (n_samples, n_channels)
    pcm = np.clip(interleaved, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    wavfile.write(path, int(sample_rate), pcm)


def _find_ffmpeg() -> Optional[str]:
    """ffmpeg on PATH, else the imageio-ffmpeg bundled binary, else None."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _resolve_episode_json(run_dir: str, episode_json: Optional[str]) -> str:
    if episode_json:
        return episode_json
    # default: the first episode_NNN.json under the run dir
    cands = sorted(
        f for f in os.listdir(run_dir)
        if f.startswith("episode_") and f.endswith(".json")
        and "_error" not in f
    )
    if not cands:
        raise FileNotFoundError(
            f"no episode_NNN.json under {run_dir}; pass --episode-json")
    return os.path.join(run_dir, cands[0])


def process_episode(
    run_dir: str,
    grid_path: str,
    *,
    episode_json: Optional[str] = None,
    anomaly_clip: Optional[str] = None,
    anomaly_class: Optional[str] = None,
    t_anom: Optional[int] = None,
    fps: float = 8.0,
    out_name: str = "demo_with_sound.mp4",
) -> Dict[str, Any]:
    """End-to-end for one episode: load JSON + grid + clip → build soundtrack →
    write wav → mux. Returns a dict of the paths written + a status."""
    ep_path = _resolve_episode_json(run_dir, episode_json)
    with open(ep_path, "r", encoding="utf-8") as f:
        ep = json.load(f)

    steps = ep.get("steps") or []
    poses = [s["agent_pos"] for s in steps if s.get("agent_pos") is not None]
    if not poses:
        raise ValueError(f"{ep_path} has no per-step agent_pos to render from")

    grid = RIRGrid.load(grid_path)
    sr = int(grid.sample_rate)

    clip_path = resolve_anomaly_clip(anomaly_class, anomaly_clip)
    clip = build_anomaly_clip(clip_path, sr)

    onset = onset_from_steps(steps, t_anom)
    spf = samples_per_frame(sr, fps)
    track = build_soundtrack(
        grid, clip, poses,
        sample_rate=sr, fps=fps, onset_frame=onset,
        clip_offset_samples=spf,  # advance one frame of clip per frame → continuous
    )

    wav_path = os.path.join(run_dir, "demo_track.wav")
    write_wav(wav_path, track, sr)

    # locate the silent mp4 (ep_log["video_path"] is relative to the run dir).
    vid_rel = ep.get("video_path")
    silent_mp4 = os.path.join(run_dir, vid_rel) if vid_rel else None
    if not silent_mp4 or not os.path.isfile(silent_mp4):
        # fall back to the conventional location
        idx = int(ep.get("episode_idx", 0))
        guess = os.path.join(run_dir, "video", f"episode_{idx:03d}.mp4")
        silent_mp4 = guess if os.path.isfile(guess) else None

    result: Dict[str, Any] = {
        "episode_json": ep_path,
        "wav": wav_path,
        "n_frames": len(poses),
        "onset_frame": onset,
        "track_seconds": track.shape[1] / float(sr),
        "silent_mp4": silent_mp4,
        "muxed_mp4": None,
        "manual_mux_cmd": None,
    }

    if silent_mp4 is None:
        print("[demo-audio] no silent .mp4 found (was --save-video set on the "
              "run?). Wrote the soundtrack only: " + wav_path)
        return result

    out_mp4 = os.path.join(run_dir, out_name)
    ffmpeg = _find_ffmpeg()
    cmd = mux_command(silent_mp4, wav_path, out_mp4, ffmpeg=ffmpeg or "ffmpeg")
    result["manual_mux_cmd"] = " ".join(cmd)

    if ffmpeg is None:
        print("[demo-audio] ffmpeg not found — wrote the soundtrack + kept the "
              "silent mp4. Mux them manually with:\n  " + result["manual_mux_cmd"])
        return result

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result["muxed_mp4"] = out_mp4
        print(f"[demo-audio] wrote {out_mp4} (video {silent_mp4} + sound {wav_path})")
    except Exception as e:  # ffmpeg failed — degrade, don't crash
        print(f"[demo-audio] ffmpeg mux failed ({type(e).__name__}: {e}); "
              f"soundtrack + silent mp4 are intact. Manual mux:\n  "
              + result["manual_mux_cmd"])
    return result


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Post-hoc pose-conditioned demo soundtrack + ffmpeg mux")
    p.add_argument("--run-dir", required=True,
                   help="The run output dir (holds episode_NNN.json + video/)")
    p.add_argument("--rir-grid", required=True,
                   help="The rendered RIR grid .npz (same one the run used)")
    p.add_argument("--episode-json", default=None,
                   help="Specific episode_NNN.json (default: first in --run-dir)")
    p.add_argument("--anomaly-clip", default=None,
                   help="Explicit anomaly .wav (else resolved from --anomaly-class)")
    p.add_argument("--anomaly-class", default=None,
                   choices=["baby_cry", "alarm", "glass_break"],
                   help="Resolve data/anomaly_audio/<class>.wav when --anomaly-clip unset")
    p.add_argument("--t-anom", type=int, default=None,
                   help="Run-level onset step (silence before it). If unset, "
                        "auto-detect from the first non-zero audio_energy frame.")
    p.add_argument("--fps", type=float, default=8.0,
                   help="Video fps the silent mp4 was written at (match --video-fps)")
    p.add_argument("--out-name", default="demo_with_sound.mp4")
    args = p.parse_args(argv)

    res = process_episode(
        args.run_dir, args.rir_grid,
        episode_json=args.episode_json,
        anomaly_clip=args.anomaly_clip,
        anomaly_class=args.anomaly_class,
        t_anom=args.t_anom,
        fps=args.fps,
        out_name=args.out_name,
    )
    final = res.get("muxed_mp4") or res.get("wav")
    print(f"DEMO_AUDIO_OUTPUT={final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
