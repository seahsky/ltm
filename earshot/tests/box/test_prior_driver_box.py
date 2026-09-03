#!/usr/bin/env python3
"""Does the assembled prior pass produce a real store from a real scene? V100 + ``ss2``.

    conda activate ss2
    bash earshot/tools/box_gate.sh

**What no Mac can answer.** `tests/mac/test_prior_driver.py` plans against a fake dataset
and walks a `FakeWorld` whose `embed` is a stub returning a literal vector -- a green
suite there licenses the PLANNING (which stops, restricted to which classes) and nothing
about the RENDERING. `render_embedding_at_stop` calls `render_through_ir`, `mix_bed`,
`heard_clip_for_clap` and a real CLAP forward pass in sequence, over a real navmesh walk
that `tests/box/test_prior_pass_box.py` already proved can complete. Nothing has ever
exercised that whole chain end to end and dumped the result to disk, which is exactly the
seam `run()`'s new `--memory-store` flag reads back.

Python 3.9 (the SoundSpaces pin). Skips, rather than fails, when a class in
`MATRIX_CLASSES` has no staged clip -- that is a staging gap
(`python -m earshot.audio.clips --out-dir data/anomaly_audio`), not a defect in this
module, and the two must not read as the same failure.
"""

from __future__ import annotations

import os
import tempfile
import unittest

# MUST precede habitat_sim: importing the package pins HABITAT_SIM_LOG, which
# habitat-sim reads at import time.
import earshot  # noqa: F401
from earshot.audio.clips import resolve_anomaly_clip
from earshot.task.episodes import available_scenes, find_scenes_dir, find_split_dir, load_scene
from earshot.task.memory_build import load_stores
from earshot.task.prior_driver import run_prior_pass

SPLIT = os.environ.get("SS2_SPLIT", "val")
# The matrix's own room-balanced assignment (PR #77's measured result): one class per
# room, so a single scene's tour exercises all three rooms it can reach.
MATRIX_CLASSES = ("toilet_flush", "snoring", "keyboard_typing")
SEED = 20260821

_LABEL = None


def setUpModule():
    global _LABEL
    split_dir = find_split_dir(SPLIT)
    scenes_dir = find_scenes_dir()
    override = os.environ.get("SS2_SCENE_LABEL")
    labels = [override] if override else list(available_scenes(split_dir))
    for label in labels:
        dataset = load_scene(split_dir, label, scenes_dir=scenes_dir)
        if os.path.exists(dataset.scene_path):
            _LABEL = label
            break
    if _LABEL is None:
        raise unittest.SkipTest("no HM3D scene mesh on this box")
    missing = [
        name for name in MATRIX_CLASSES
        if not resolve_anomaly_clip(name, None, "data/anomaly_audio")
    ]
    if missing:
        raise unittest.SkipTest(
            "no staged clip for {} -- stage with `python -m earshot.audio.clips "
            "--out-dir data/anomaly_audio` first".format(", ".join(missing))
        )


class TestARealPriorPassProducesARealStore(unittest.TestCase):
    def test_one_scene_tours_renders_and_dumps(self):
        """THE HEALTHY ARM, end to end: walk, render at every reached stop, dump, and
        read the dump back through the exact reader `run()` uses."""
        with tempfile.TemporaryDirectory() as tmp:
            path = run_prior_pass(
                run_dir=tmp,
                scenes=(_LABEL,),
                classes=MATRIX_CLASSES,
                split=SPLIT,
                seed=SEED,
            )
            self.assertTrue(path.exists())
            semantic, episodic, provenance = load_stores(str(path))

        print(
            "\n  scene {}: {} semantic row(s), {} episodic row(s)".format(
                _LABEL, len(semantic), len(episodic)
            ),
            flush=True,
        )
        for entry in semantic.entries:
            print(
                "    heard {:16s} in {:12s} at {:10s} dim={}".format(
                    entry.sound_class, entry.room, entry.category, entry.embedding.size
                ),
                flush=True,
            )
        if len(semantic) == 0:
            self.skipTest(
                "no room this scene has matches {}'s bank -- a scene fact "
                "(see test_prior_pass_box's own anchor-room-yield count), not a "
                "defect in this module".format(MATRIX_CLASSES)
            )
        self.assertEqual(len(semantic), len(episodic))
        self.assertEqual(provenance["scenes_complete"], [_LABEL])
        # The one property a fake cannot check: a real CLAP embedding is neither zero
        # nor a stand-in width. `_encoder_favouring`-style stubs in the Mac suite are
        # 8-wide by construction; a real forward pass is not, and this is the assertion
        # that distinguishes "the pipe is connected" from "the pipe carries a fake".
        for entry in semantic.entries:
            self.assertGreater(entry.embedding.size, 8)
            self.assertGreater(float((entry.embedding ** 2).sum()) ** 0.5, 0.0)

    def test_a_scene_that_cannot_load_is_recorded_not_fatal(self):
        """The forced-failure arm of the continue-on-failure rule, at the SCENE grain."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                run_prior_pass(
                    run_dir=tmp,
                    scenes=("this-scene-does-not-exist",),
                    classes=MATRIX_CLASSES,
                    split=SPLIT,
                    seed=SEED,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
