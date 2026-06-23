#!/usr/bin/env python
"""
TDD for make_two_act_demo — the two-act (cold-seed + warm-recall) demo stitcher.

All pure / no habitat / no real ffmpeg / no imageio file I/O:
  * ``burn_banner`` runs on synthetic ``np.zeros`` frame lists (asserts it
    preserves count/shape/dtype and actually paints the top band).
  * ``concat_command`` asserts the ffmpeg argv is well-formed (names both inputs,
    uses the concat FILTER, maps the muxed output).
  * the ``stitch_act`` / ``main`` orchestration is exercised with
    ``render_demo_audio_track.process_episode``, ``read_frames``/``write_frames``
    and the ffmpeg ``subprocess.run`` all monkeypatched — asserting act1 → act2 →
    concat ordering and that a ``SilentSoundtrackError`` from an act propagates
    (so a silent act can never become a shipped demo).
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)))

from embodied_memory.scripts import make_two_act_demo as M  # noqa: E402
from embodied_memory.scripts import render_demo_audio_track as R  # noqa: E402


def _frames(n=3, h=64, w=96):
    """A list of n distinct mid-gray frames (so a top-band paint is detectable)."""
    out = []
    for i in range(n):
        f = np.full((h, w, 3), 100 + i, dtype=np.uint8)
        out.append(f)
    return out


class TestBurnBanner(unittest.TestCase):
    def test_preserves_count_shape_dtype(self):
        frames = _frames(4, 48, 80)
        out = M.burn_banner(frames, "ACT 1 - HELLO")
        self.assertEqual(len(out), len(frames))
        for a, b in zip(frames, out):
            self.assertEqual(b.shape, a.shape)
            self.assertEqual(b.dtype, np.uint8)

    def test_modifies_top_band(self):
        frames = _frames(2, 64, 120)
        before = frames[0].copy()
        out = M.burn_banner(frames, "ACT 1 - FIRST VISIT")
        # the very top rows must change (a dark band is painted there)...
        top = out[0][:20, :, :]
        before_top = before[:20, :, :]
        self.assertFalse(np.array_equal(top, before_top),
                         "top band should be painted")
        # ...and a band lower than the painted strip should be untouched.
        lower = out[0][40:, :, :]
        before_lower = before[40:, :, :]
        self.assertTrue(np.array_equal(lower, before_lower),
                        "rows below the banner band must be unchanged")

    def test_does_not_mutate_input(self):
        frames = _frames(2, 40, 60)
        snapshot = [f.copy() for f in frames]
        M.burn_banner(frames, "X")
        for orig, snap in zip(frames, snapshot):
            self.assertTrue(np.array_equal(orig, snap),
                            "burn_banner must not mutate the caller's frames")

    def test_accepts_single_ndarray(self):
        # a single (T,H,W,3) array is accepted and yields one list per frame
        arr = np.stack(_frames(3, 32, 48), axis=0)
        out = M.burn_banner(arr, "Y")
        self.assertEqual(len(out), 3)
        for f in out:
            self.assertEqual(f.shape, (32, 48, 3))
            self.assertEqual(f.dtype, np.uint8)

    def test_preserves_frame_order(self):
        # distinct base colors → order is recoverable from the untouched bottom band
        frames = _frames(3, 64, 64)
        out = M.burn_banner(frames, "Z")
        for i, f in enumerate(out):
            # bottom row keeps the per-frame base color (banner is at the top)
            self.assertEqual(int(f[-1, 0, 0]), 100 + i)


class TestConcatCommand(unittest.TestCase):
    def test_well_formed(self):
        cmd = M.concat_command(["a.mp4", "b.mp4"], "out.mp4", "ffmpeg")
        self.assertEqual(cmd[0], "ffmpeg")
        # both inputs referenced
        self.assertIn("a.mp4", cmd)
        self.assertIn("b.mp4", cmd)
        # output last
        self.assertEqual(cmd[-1], "out.mp4")
        joined = " ".join(cmd)
        # uses the concat FILTER (re-encode), not the demuxer
        self.assertIn("concat", joined)
        self.assertIn("-filter_complex", cmd)
        # maps the filtered video + audio out
        self.assertIn("-map", cmd)

    def test_two_inputs_two_i_flags(self):
        cmd = M.concat_command(["x.mp4", "y.mp4"], "z.mp4")
        i_positions = [i for i, t in enumerate(cmd) if t == "-i"]
        self.assertEqual(len(i_positions), 2)
        self.assertEqual(cmd[i_positions[0] + 1], "x.mp4")
        self.assertEqual(cmd[i_positions[1] + 1], "y.mp4")

    def test_three_inputs(self):
        cmd = M.concat_command(["a.mp4", "b.mp4", "c.mp4"], "o.mp4")
        joined = " ".join(cmd)
        # concat filter must declare n=3
        self.assertIn("n=3", joined)
        for clip in ("a.mp4", "b.mp4", "c.mp4"):
            self.assertIn(clip, cmd)


class _FakeProcessEpisode:
    """Records process_episode calls; writes a fake muxed mp4 for each act so the
    concat step has real files to reference."""

    def __init__(self):
        self.calls = []

    def __call__(self, run_dir, grid_path, **kw):
        self.calls.append(dict(run_dir=run_dir, grid_path=grid_path, **kw))
        out_name = kw.get("out_name", "demo_with_sound.mp4")
        muxed = os.path.join(run_dir, out_name)
        with open(muxed, "wb") as f:
            f.write(b"muxed")
        return {
            "muxed_mp4": muxed,
            "wav": os.path.join(run_dir, "demo_track.wav"),
            "track_rms": 0.05,
            "n_frames": 10,
            "silent_mp4": kw.get("silent_mp4_override"),
        }


def _fake_read_frames(_path):
    return _frames(3, 32, 48)


def _fake_write_frames(path, frames, fps):
    with open(path, "wb") as f:
        f.write(b"banner-silent")
    return path


class TestStitchAct(unittest.TestCase):
    def _episode_json(self, d, idx=0):
        import json
        os.makedirs(os.path.join(d, "video"), exist_ok=True)
        silent = os.path.join(d, "video", f"episode_{idx:03d}.mp4")
        with open(silent, "wb") as f:
            f.write(b"silent")
        ep = {
            "episode_idx": idx,
            "video_path": f"video/episode_{idx:03d}.mp4",
            "source_position": [0.0, 0.0, 0.0],
            "steps": [{"step_idx": i, "agent_pos": [i, 0, 0]} for i in range(6)],
        }
        ep_path = os.path.join(d, f"episode_{idx:03d}.json")
        with open(ep_path, "w") as f:
            json.dump(ep, f)
        return ep_path, silent

    def test_stitch_act_banners_then_muxes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ep_path, _ = self._episode_json(d, idx=0)
            fake_pe = _FakeProcessEpisode()
            with mock.patch.object(M, "read_frames", _fake_read_frames), \
                    mock.patch.object(M, "write_frames", _fake_write_frames), \
                    mock.patch.object(M.render_demo_audio_track,
                                      "process_episode", fake_pe):
                res = M.stitch_act(
                    run_dir=d, episode_json=ep_path, rir_grid="grid.npz",
                    anomaly_class="alarm", t_anom=5, fps=4.0,
                    anomaly_clip=None, banner_text="ACT 1 - X",
                    out_name="act1.mp4")
            # process_episode got the banner-overlaid silent mp4 as the override
            self.assertEqual(len(fake_pe.calls), 1)
            call = fake_pe.calls[0]
            self.assertIsNotNone(call["silent_mp4_override"])
            self.assertTrue(os.path.isfile(call["silent_mp4_override"]))
            self.assertEqual(call["out_name"], "act1.mp4")
            # returns the muxed path + rms
            self.assertTrue(os.path.isfile(res["muxed_mp4"]))
            self.assertGreater(res["track_rms"], 0.0)

    def test_silent_act_propagates(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ep_path, _ = self._episode_json(d, idx=0)

            def boom(*a, **k):
                raise R.SilentSoundtrackError("silent")

            with mock.patch.object(M, "read_frames", _fake_read_frames), \
                    mock.patch.object(M, "write_frames", _fake_write_frames), \
                    mock.patch.object(M.render_demo_audio_track,
                                      "process_episode", boom):
                with self.assertRaises(R.SilentSoundtrackError):
                    M.stitch_act(
                        run_dir=d, episode_json=ep_path, rir_grid="grid.npz",
                        anomaly_class="alarm", t_anom=5, fps=4.0,
                        anomaly_clip=None, banner_text="ACT 1 - X",
                        out_name="act1.mp4")


class TestMainOrchestration(unittest.TestCase):
    def _two_episodes(self, d):
        import json
        os.makedirs(os.path.join(d, "video"), exist_ok=True)
        paths = []
        for idx in (0, 1):
            silent = os.path.join(d, "video", f"episode_{idx:03d}.mp4")
            with open(silent, "wb") as f:
                f.write(b"silent")
            ep = {
                "episode_idx": idx,
                "video_path": f"video/episode_{idx:03d}.mp4",
                "source_position": [0.0, 0.0, 0.0],
                "steps": [{"step_idx": i, "agent_pos": [i, 0, 0]}
                          for i in range(6)],
            }
            ep_path = os.path.join(d, f"episode_{idx:03d}.json")
            with open(ep_path, "w") as f:
                json.dump(ep, f)
            paths.append(ep_path)
        return paths

    def test_main_stitches_act1_act2_then_concat(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ep0, ep1 = self._two_episodes(d)
            fake_pe = _FakeProcessEpisode()
            order = []

            def fake_concat_run(cmd, **kw):
                order.append(("concat", cmd))
                open(cmd[-1], "wb").write(b"final")

                class _R:
                    returncode = 0
                return _R()

            def tracking_pe(run_dir, grid_path, **kw):
                order.append(("act", kw.get("out_name")))
                return fake_pe(run_dir, grid_path, **kw)

            with mock.patch.object(M, "read_frames", _fake_read_frames), \
                    mock.patch.object(M, "write_frames", _fake_write_frames), \
                    mock.patch.object(M.render_demo_audio_track,
                                      "process_episode", tracking_pe), \
                    mock.patch.object(M, "_find_ffmpeg", return_value="ffmpeg"), \
                    mock.patch.object(M.subprocess, "run",
                                      side_effect=fake_concat_run):
                rc = M.main([
                    "--run-dir", d,
                    "--act1-episode", ep0,
                    "--act2-episode", ep1,
                    "--rir-grid", "grid.npz",
                    "--anomaly-class", "alarm",
                    "--t-anom", "5",
                    "--fps", "4.0",
                    "--out-name", "demo_two_act.mp4",
                ])
            self.assertEqual(rc, 0)
            # order: act1, act2, then concat
            kinds = [o[0] for o in order]
            self.assertEqual(kinds, ["act", "act", "concat"])
            self.assertEqual(order[0][1], "act1.mp4")
            self.assertEqual(order[1][1], "act2.mp4")
            # concat references both act clips and writes the final
            concat_cmd = order[2][1]
            self.assertTrue(any("act1.mp4" in t for t in concat_cmd))
            self.assertTrue(any("act2.mp4" in t for t in concat_cmd))
            self.assertTrue(os.path.isfile(os.path.join(d, "demo_two_act.mp4")))

    def test_main_propagates_silent_act(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ep0, ep1 = self._two_episodes(d)

            def boom(*a, **k):
                raise R.SilentSoundtrackError("silent act")

            with mock.patch.object(M, "read_frames", _fake_read_frames), \
                    mock.patch.object(M, "write_frames", _fake_write_frames), \
                    mock.patch.object(M.render_demo_audio_track,
                                      "process_episode", boom):
                rc = M.main([
                    "--run-dir", d,
                    "--act1-episode", ep0,
                    "--act2-episode", ep1,
                    "--rir-grid", "grid.npz",
                    "--anomaly-class", "alarm",
                    "--t-anom", "5",
                ])
            # a silent act must make the CLI exit non-zero (never a silent demo)
            self.assertNotEqual(rc, 0)

    def test_main_degrades_when_ffmpeg_absent(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ep0, ep1 = self._two_episodes(d)
            fake_pe = _FakeProcessEpisode()
            with mock.patch.object(M, "read_frames", _fake_read_frames), \
                    mock.patch.object(M, "write_frames", _fake_write_frames), \
                    mock.patch.object(M.render_demo_audio_track,
                                      "process_episode", fake_pe), \
                    mock.patch.object(M, "_find_ffmpeg", return_value=None):
                rc = M.main([
                    "--run-dir", d,
                    "--act1-episode", ep0,
                    "--act2-episode", ep1,
                    "--rir-grid", "grid.npz",
                    "--anomaly-class", "alarm",
                    "--t-anom", "5",
                ])
            # both per-act mp4s exist; concat skipped but not a hard failure
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.isfile(os.path.join(d, "act1.mp4")))
            self.assertTrue(os.path.isfile(os.path.join(d, "act2.mp4")))


def _ep(idx, *, n_steps, video=True, success_1m=False, n_memory_chosen=0,
        n_arrival_stop=0, n_stop_signals=0, distance_to_goal=None,
        n_remembr_chosen=0, min_distance_to_goal=None):
    """An episode_NNN.json-shaped dict (only the picker-relevant ep_log fields)."""
    ep = {
        "episode_idx": idx,
        "n_steps": n_steps,
        "success_1m": success_1m,
        "n_memory_chosen": n_memory_chosen,
        "n_arrival_stop": n_arrival_stop,
        "n_stop_signals": n_stop_signals,
        "n_remembr_chosen": n_remembr_chosen,
        "distance_to_goal": distance_to_goal,
        "min_distance_to_goal": min_distance_to_goal,
        "_path": f"episode_{idx:03d}.json",
    }
    if video:
        ep["video_path"] = f"video/episode_{idx:03d}.mp4"
    return ep


class TestPickTwoActs(unittest.TestCase):
    def test_act1_is_smallest_idx_with_video(self):
        eps = [_ep(2, n_steps=40, success_1m=True, n_memory_chosen=3,
                   n_arrival_stop=1),
               _ep(0, n_steps=22),
               _ep(1, n_steps=120)]
        res = M.pick_two_acts(eps, t_anom=5)
        self.assertEqual(res["act1"]["episode_idx"], 0)  # cold seed = smallest idx

    def test_arrived_and_memfired_beats_arrived_only(self):
        # ep1: arrived only (no memory). ep2: arrived AND memory fired AND stop.
        # The clean recall story (ep2) must win even though it is LONGER here.
        eps = [
            _ep(0, n_steps=22),
            _ep(1, n_steps=18, success_1m=True, n_memory_chosen=0),
            _ep(2, n_steps=40, success_1m=True, n_memory_chosen=3, n_arrival_stop=1),
        ]
        res = M.pick_two_acts(eps, t_anom=5)
        self.assertEqual(res["status"], "OK")
        self.assertEqual(res["act2"]["episode_idx"], 2)

    def test_arrived_beats_longest_wander(self):
        # ep1 = a 250-step timeout (longest, never arrived). ep2 = a short arrival.
        # The OLD longest-wins picker took ep1; the new one must take the arrival.
        eps = [
            _ep(0, n_steps=22),
            _ep(1, n_steps=250),  # wander / timeout — longest
            _ep(2, n_steps=30, success_1m=True),  # arrived
        ]
        res = M.pick_two_acts(eps, t_anom=5)
        self.assertEqual(res["status"], "ARRIVED")
        self.assertEqual(res["act2"]["episode_idx"], 2)

    def test_shortest_among_equal_tier(self):
        # two clean recall stories → prefer the SHORTER (crisper) path.
        eps = [
            _ep(0, n_steps=22),
            _ep(1, n_steps=45, success_1m=True, n_memory_chosen=2, n_stop_signals=1),
            _ep(2, n_steps=20, success_1m=True, n_memory_chosen=4, n_arrival_stop=1),
        ]
        res = M.pick_two_acts(eps, t_anom=5)
        self.assertEqual(res["status"], "OK")
        self.assertEqual(res["act2"]["episode_idx"], 2)  # shorter clean recall

    def test_distance_to_goal_under_1m_counts_as_arrived(self):
        # no success_1m flag but final distance_to_goal < 1.0 → tier 1 (arrived).
        eps = [
            _ep(0, n_steps=22),
            _ep(1, n_steps=250),  # wander
            _ep(2, n_steps=33, distance_to_goal=0.6),  # arrived via d2g
        ]
        res = M.pick_two_acts(eps, t_anom=5)
        self.assertEqual(res["status"], "ARRIVED")
        self.assertEqual(res["act2"]["episode_idx"], 2)

    def test_nofire_falls_back_to_longest_past_onset(self):
        # NO warm episode arrived → NOFIRE; among non-arrivals prefer the LONGEST
        # that ran past onset (so the soundtrack is non-silent), not a pre-onset stub.
        eps = [
            _ep(0, n_steps=22),
            _ep(1, n_steps=4),    # ended before onset (t_anom+3 = 8) → not eligible
            _ep(2, n_steps=120),  # past onset, longest wander
            _ep(3, n_steps=60),   # past onset, shorter wander
        ]
        res = M.pick_two_acts(eps, t_anom=5)
        self.assertEqual(res["status"], "NOFIRE")
        self.assertEqual(res["act2"]["episode_idx"], 2)

    def test_nofire_all_pre_onset_uses_longest_overall(self):
        eps = [
            _ep(0, n_steps=22),
            _ep(1, n_steps=4),
            _ep(2, n_steps=6),
        ]
        res = M.pick_two_acts(eps, t_anom=5)
        self.assertEqual(res["status"], "NOFIRE")
        self.assertEqual(res["act2"]["episode_idx"], 2)  # longest overall

    def test_solo_when_only_one_video(self):
        eps = [_ep(0, n_steps=22), _ep(1, n_steps=120, video=False)]
        res = M.pick_two_acts(eps, t_anom=5)
        self.assertEqual(res["status"], "SOLO")
        self.assertIsNone(res["act2"])
        self.assertEqual(res["act1"]["episode_idx"], 0)

    def test_none_when_no_video(self):
        eps = [_ep(0, n_steps=22, video=False), _ep(1, n_steps=30, video=False)]
        res = M.pick_two_acts(eps, t_anom=5)
        self.assertEqual(res["status"], "NONE")
        self.assertIsNone(res["act1"])
        self.assertIsNone(res["act2"])

    def test_reads_only_documented_fields(self):
        # The picker must only consult fields that EXIST in episode_NNN.json (ep_log).
        # (Guards against the n_audio_onset_fired dead-field bug recurring.)
        allowed = {
            "episode_idx", "n_steps", "video_path", "success_1m",
            "n_memory_chosen", "n_arrival_stop", "n_stop_signals",
            "distance_to_goal", "_path",
        }
        sentinel = _ep(1, n_steps=30, success_1m=True, n_memory_chosen=2,
                       n_arrival_stop=1)

        class _Dict(dict):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.seen = set()

            def get(self, key, default=None):
                self.seen.add(key)
                return super().get(key, default)

        tracked = _Dict(sentinel)
        M.pick_two_acts([_ep(0, n_steps=22), tracked], t_anom=5)
        leaked = tracked.seen - allowed
        self.assertEqual(leaked, set(),
                         f"picker read undocumented ep field(s): {leaked}")


class TestPickMainCLI(unittest.TestCase):
    def _write_run(self, d, episodes):
        import json
        os.makedirs(os.path.join(d, "video"), exist_ok=True)
        for ep in episodes:
            idx = ep["episode_idx"]
            if ep.get("video_path"):
                with open(os.path.join(d, "video", f"episode_{idx:03d}.mp4"),
                          "wb") as f:
                    f.write(b"silent")
            payload = {k: v for k, v in ep.items() if k != "_path"}
            with open(os.path.join(d, f"episode_{idx:03d}.json"), "w") as f:
                json.dump(payload, f)

    def test_pick_main_emits_pick_line_and_table(self):
        import io
        import tempfile
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as d:
            self._write_run(d, [
                _ep(0, n_steps=22),
                _ep(1, n_steps=250),  # wander
                _ep(2, n_steps=30, success_1m=True, n_memory_chosen=3,
                    n_arrival_stop=1, min_distance_to_goal=0.4),
            ])
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = M.main(["--pick", "--run-dir", d, "--t-anom", "5"])
            out = buf.getvalue()
        self.assertEqual(rc, 0)
        # a machine-readable PICK line the bash driver parses
        pick_lines = [ln for ln in out.splitlines() if ln.startswith("PICK\t")]
        self.assertEqual(len(pick_lines), 1)
        parts = pick_lines[0].split("\t")
        self.assertEqual(parts[1], "OK")
        self.assertTrue(parts[2].endswith("episode_000.json"))  # act1 = cold seed
        self.assertTrue(parts[3].endswith("episode_002.json"))  # act2 = recall story
        # the diagnostic table is present (per-warm-episode lines)
        self.assertIn("steps=250", out)
        self.assertIn("success_1m=True", out)

    def test_pick_main_none_exit_3(self):
        import io
        import tempfile
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as d:
            self._write_run(d, [_ep(0, n_steps=22, video=False)])
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = M.main(["--pick", "--run-dir", d])
            out = buf.getvalue()
        self.assertEqual(rc, 3)
        self.assertIn("PICK\tNONE", out)


if __name__ == "__main__":
    unittest.main()
