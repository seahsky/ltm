#!/usr/bin/env python
"""
TDD for render_demo_audio_track — the post-hoc pose-conditioned demo soundtrack.

All pure: a synthetic RIRGrid (one loud cell at the source, quieter with
distance) + a synthetic clip + a fake per-step pose list. No habitat_sim, no
ffmpeg (the one mux test asserts the COMMAND string only; the I/O path is
exercised with a monkeypatched subprocess + a temp dir).
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)))

from embodied_memory.audio import RIRGrid  # noqa: E402
from embodied_memory.scripts import render_demo_audio_track as R  # noqa: E402


SR = 48000
FPS = 8.0


def _grid():
    """A line of 4 cells along +x, source at x=0. Each cell's binaural IR is a
    single impulse whose amplitude DECAYS with distance (1/(1+d)), so the
    convolved energy is a monotone proxy for proximity. Right ear slightly louder
    (a fixed ILD) so lateral_sign is well-defined."""
    src = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    xs = [0.0, 1.0, 2.0, 3.0]
    cells = np.array([[x, 0.0, 0.0] for x in xs], dtype=np.float32)
    T = 64
    irs = np.zeros((len(xs), 2, T), dtype=np.float32)
    for i, x in enumerate(xs):
        amp = 1.0 / (1.0 + abs(x))  # decays with distance to source at x=0
        irs[i, 0, 0] = amp * 0.9     # left
        irs[i, 1, 0] = amp * 1.0     # right (louder → +1 lateral sign)
    return RIRGrid(cells, src, irs, SR, "TESTSCENE")


def _clip(n=None):
    """A deterministic non-silent clip (one frame long by default)."""
    n = n or R.samples_per_frame(SR, FPS)
    rng = np.random.default_rng(0)
    return (rng.standard_normal(n).astype(np.float32) * 0.5)


def _steps(positions, t_anom_step=0, audio_on_from=None):
    """Build a fake steps[] list. ``positions`` are agent_pos; step_idx counts
    up from 0. If ``audio_on_from`` is set, audio_energy is 0.0 before that frame
    and 1.0 at/after it (for the auto-detect onset test)."""
    out = []
    for i, p in enumerate(positions):
        e = None
        if audio_on_from is not None:
            e = 0.0 if i < audio_on_from else 1.0
        out.append({"step_idx": i, "agent_pos": list(p), "audio_energy": e})
    return out


class TestSamplesPerFrame(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(R.samples_per_frame(48000, 8.0), 6000)

    def test_min_one(self):
        self.assertGreaterEqual(R.samples_per_frame(10, 1000.0), 1)


class TestOnsetResolution(unittest.TestCase):
    def test_explicit_t_anom_maps_to_first_frame_at_or_after(self):
        steps = _steps([[3, 0, 0]] * 10)
        # step_idx == frame index here; onset step 4 → frame 4
        self.assertEqual(R.onset_from_steps(steps, t_anom=4), 4)

    def test_t_anom_beyond_recorded_returns_len(self):
        steps = _steps([[3, 0, 0]] * 5)
        self.assertEqual(R.onset_from_steps(steps, t_anom=100), 5)

    def test_autodetect_from_first_nonzero_energy(self):
        steps = _steps([[3, 0, 0]] * 8, audio_on_from=3)
        self.assertEqual(R.onset_from_steps(steps, t_anom=None), 3)

    def test_autodetect_defaults_zero_when_no_signal(self):
        steps = _steps([[3, 0, 0]] * 4)  # audio_energy all None
        self.assertEqual(R.onset_from_steps(steps, t_anom=None), 0)


class TestBuildSoundtrack(unittest.TestCase):
    def test_length_equals_nframes_times_spf(self):
        grid = _grid()
        clip = _clip()
        poses = [[3, 0, 0], [2, 0, 0], [1, 0, 0]]
        spf = R.samples_per_frame(SR, FPS)
        track = R.build_soundtrack(
            grid, clip, poses, sample_rate=SR, fps=FPS, onset_frame=0)
        self.assertEqual(track.shape, (2, len(poses) * spf))

    def test_silence_before_onset(self):
        grid = _grid()
        clip = _clip()
        poses = [[3, 0, 0]] * 5
        spf = R.samples_per_frame(SR, FPS)
        onset = 2
        track = R.build_soundtrack(
            grid, clip, poses, sample_rate=SR, fps=FPS, onset_frame=onset)
        # frames before onset are exactly silent...
        pre = track[:, : onset * spf]
        self.assertEqual(float(np.max(np.abs(pre))), 0.0)
        # ...and at/after onset there is signal
        post = track[:, onset * spf:]
        self.assertGreater(float(np.max(np.abs(post))), 0.0)

    def test_energy_rises_as_agent_approaches_source(self):
        grid = _grid()
        clip = _clip()
        # agent walks from the far cell (x=3) toward the source (x=0)
        poses = [[3, 0, 0], [2, 0, 0], [1, 0, 0], [0, 0, 0]]
        spf = R.samples_per_frame(SR, FPS)
        track = R.build_soundtrack(
            grid, clip, poses, sample_rate=SR, fps=FPS, onset_frame=0)
        # per-frame RMS energy must be monotonically NON-DECREASING toward source
        energies = []
        for i in range(len(poses)):
            seg = track[:, i * spf:(i + 1) * spf]
            energies.append(float(np.sqrt(np.mean(seg ** 2))))
        for a, b in zip(energies, energies[1:]):
            self.assertLess(a, b, f"energy should rise approaching source: {energies}")

    def test_empty_clip_is_silent(self):
        grid = _grid()
        poses = [[1, 0, 0], [0, 0, 0]]
        track = R.build_soundtrack(
            grid, np.zeros(0, dtype=np.float32), poses,
            sample_rate=SR, fps=FPS, onset_frame=0)
        self.assertEqual(float(np.max(np.abs(track))), 0.0)

    def test_binaural_two_channels(self):
        grid = _grid()
        track = R.build_soundtrack(
            grid, _clip(), [[1, 0, 0]], sample_rate=SR, fps=FPS, onset_frame=0)
        self.assertEqual(track.shape[0], 2)


class TestMuxCommand(unittest.TestCase):
    def test_well_formed(self):
        cmd = R.mux_command("a.mp4", "b.wav", "out.mp4", ffmpeg="ffmpeg")
        self.assertEqual(cmd[0], "ffmpeg")
        # inputs present in order
        self.assertIn("a.mp4", cmd)
        self.assertIn("b.wav", cmd)
        self.assertEqual(cmd[-1], "out.mp4")
        # video copied, audio re-encoded, shortest trim
        self.assertIn("copy", cmd)
        self.assertIn("-shortest", cmd)
        # maps video from input 0, audio from input 1
        self.assertIn("0:v:0", cmd)
        self.assertIn("1:a:0", cmd)

    def test_input_order_video_then_audio(self):
        cmd = R.mux_command("vid.mp4", "snd.wav", "o.mp4")
        i_idx = [i for i, t in enumerate(cmd) if t == "-i"]
        self.assertEqual(cmd[i_idx[0] + 1], "vid.mp4")
        self.assertEqual(cmd[i_idx[1] + 1], "snd.wav")


class TestWriteWav(unittest.TestCase):
    def test_writes_stereo_int16(self):
        import tempfile
        from scipy.io import wavfile
        track = np.stack([np.ones(1000), -np.ones(1000)], axis=0).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.wav")
            R.write_wav(path, track, SR)
            sr, data = wavfile.read(path)
            self.assertEqual(sr, SR)
            self.assertEqual(data.shape, (1000, 2))
            self.assertEqual(data.dtype, np.int16)


class TestProcessEpisodeDegrade(unittest.TestCase):
    """End-to-end with the grid loader real (synthetic npz) but ffmpeg mocked /
    absent — asserts graceful degrade writes the wav and prints a manual cmd."""

    def _write_grid_npz(self, path):
        from embodied_memory.audio import save_rir_grid
        g = _grid()
        save_rir_grid(path, cell_positions=g.cell_positions,
                      source_position=g.source_position, irs=g.irs,
                      sample_rate=g.sample_rate, scene_id=g.scene_id)

    def test_no_silent_mp4_writes_wav_only(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            grid_path = os.path.join(d, "grid.npz")
            self._write_grid_npz(grid_path)
            ep = {
                "episode_idx": 0,
                "source_position": [0.0, 0.0, 0.0],
                "steps": _steps([[3, 0, 0], [2, 0, 0], [1, 0, 0], [0, 0, 0]]),
                # no video_path, no video/ dir → silent mp4 absent
            }
            with open(os.path.join(d, "episode_000.json"), "w") as f:
                json.dump(ep, f)
            res = R.process_episode(
                d, grid_path, anomaly_class="alarm", t_anom=0, fps=FPS)
            self.assertTrue(os.path.isfile(res["wav"]))
            self.assertIsNone(res["muxed_mp4"])

    def test_ffmpeg_absent_prints_manual_cmd(self):
        import json
        import tempfile
        from unittest import mock
        with tempfile.TemporaryDirectory() as d:
            grid_path = os.path.join(d, "grid.npz")
            self._write_grid_npz(grid_path)
            # make a fake silent mp4 so the mux branch is reached
            os.makedirs(os.path.join(d, "video"), exist_ok=True)
            silent = os.path.join(d, "video", "episode_000.mp4")
            with open(silent, "wb") as f:
                f.write(b"\x00\x00\x00\x18ftypmp42")  # not a real mp4, just a file
            ep = {
                "episode_idx": 0,
                "video_path": "video/episode_000.mp4",
                "source_position": [0.0, 0.0, 0.0],
                "steps": _steps([[3, 0, 0], [1, 0, 0]]),
            }
            with open(os.path.join(d, "episode_000.json"), "w") as f:
                json.dump(ep, f)
            with mock.patch.object(R, "_find_ffmpeg", return_value=None):
                res = R.process_episode(
                    d, grid_path, anomaly_class="alarm", t_anom=0, fps=FPS)
            self.assertTrue(os.path.isfile(res["wav"]))
            self.assertIsNone(res["muxed_mp4"])
            self.assertIsNotNone(res["manual_mux_cmd"])
            self.assertIn("ffmpeg", res["manual_mux_cmd"])
            self.assertIn(silent, res["manual_mux_cmd"])

    def test_mux_invoked_when_ffmpeg_present(self):
        import json
        import tempfile
        from unittest import mock
        with tempfile.TemporaryDirectory() as d:
            grid_path = os.path.join(d, "grid.npz")
            self._write_grid_npz(grid_path)
            os.makedirs(os.path.join(d, "video"), exist_ok=True)
            silent = os.path.join(d, "video", "episode_000.mp4")
            with open(silent, "wb") as f:
                f.write(b"fake")
            ep = {
                "episode_idx": 0,
                "video_path": "video/episode_000.mp4",
                "steps": _steps([[1, 0, 0], [0, 0, 0]]),
            }
            with open(os.path.join(d, "episode_000.json"), "w") as f:
                json.dump(ep, f)
            calls = {}

            def fake_run(cmd, **kw):
                calls["cmd"] = cmd
                # simulate ffmpeg producing the output file
                open(cmd[-1], "wb").write(b"muxed")

                class _R:
                    returncode = 0
                return _R()

            with mock.patch.object(R, "_find_ffmpeg", return_value="ffmpeg"), \
                    mock.patch.object(R.subprocess, "run", side_effect=fake_run):
                res = R.process_episode(
                    d, grid_path, anomaly_class="alarm", t_anom=0, fps=FPS)
            self.assertIsNotNone(res["muxed_mp4"])
            self.assertTrue(os.path.isfile(res["muxed_mp4"]))
            self.assertEqual(calls["cmd"][0], "ffmpeg")


if __name__ == "__main__":
    unittest.main()
