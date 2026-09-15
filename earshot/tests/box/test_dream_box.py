#!/usr/bin/env python3
"""What a DREAM step COSTS, measured before a night is spent on one.

    conda activate ss2
    python -m earshot.task.models          # stages CLAP and CLIP
    bash earshot/tools/box_gate.sh

**THE NUMBER THIS FILE EXISTS FOR IS `f_u` PER STEP, AND IT HAS NEVER BEEN MEASURED.**
The tree runs CLAP ONCE per episode, at the onset. DREAM's eq. 6 asks for `z^u_t` at every
step, which is a 153.5 M-parameter forward pass the runner has never paid per step. PR
#108 measured `f_v` at 11.6 ms mean over 20 real frames and that was enough to license
encoding every step; there is no equivalent number for `f_u`, so this produces one.

**AND CRITERION 7 WILL NOT CATCH IT.** The smoke gate audits `audio_render_s` -- the
render, the convolution, the bed mix, the guard. `f_v` and `f_u` sit OUTSIDE that bracket,
so a DREAM run could double its per-step cost with every criterion green. `dream.observe`
times itself and the runner writes `dream_step_s_mean` / `dream_step_s_worst` precisely
because nothing else would.

The ceiling is criterion 7's 0.5 s and `matrix-2` measured its worst step at 0.3356 s, so
the headroom a DREAM step has to fit inside is about 0.164 s. This test prints what it
actually uses and FAILS if it does not fit, because a sweep launched over that budget
produces a night of episodes the gate will refuse.

**These tests print their measurements** (ADR-0014).

Python 3.9 (the SoundSpaces pin).
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from earshot.audio.clap import audio_embedding, heard_clip_for_clap
from earshot.audio.clips import render_through_ir, synthetic_burst
from earshot.audio.config import AudioConfig
from earshot.audio.sensor import AudioSensorHandle
from earshot.audio.spec import audio_sensor_spec
from earshot.memory.consolidate import ImportanceWeights
from earshot.task.dream import (
    DreamContext,
    DreamKnobs,
    begin_episode,
    consolidate_episode,
    empty_memory,
    observe,
)
from earshot.task.plan import PlanWeights
from earshot.task.dataset import available_scenes, find_scenes_dir, find_split_dir, load_scene

SPLIT = "val"
PLACEMENT_SEED = 20260821
STEPS = 20
GOAL_RADIUS = 0.3

# criterion 7's ceiling, and what `matrix-2` left under it.
CRITERION_7_CEILING_S = 0.5
MATRIX_2_WORST_STEP_S = 0.3356
HEADROOM_S = CRITERION_7_CEILING_S - MATRIX_2_WORST_STEP_S

_DATASET = None


def setUpModule():
    global _DATASET
    split_dir = find_split_dir(SPLIT)
    scenes_dir = find_scenes_dir()
    override = os.environ.get("SS2_SCENE_LABEL")
    for label in ([override] if override else list(available_scenes(split_dir))):
        dataset = load_scene(split_dir, label, scenes_dir=scenes_dir)
        if os.path.exists(dataset.scene_path):
            _DATASET = dataset
            break
    if _DATASET is None:
        raise unittest.SkipTest("no ObjectNav {} scene has its mesh on this box".format(SPLIT))
    print("\n  scene: {}".format(_DATASET.scene_label), flush=True)


def _knobs():
    return DreamKnobs(
        stm_horizon=8, stm_decay=0.8, present_weight=0.7,
        coherence=0.995, min_segment=2, max_segment=8,
        importance=ImportanceWeights(alpha=1.0, beta=1.0, gamma=1.0),
        eta=0.5, min_support=2,
        k_experience=3, k_pattern=2, k_knowledge=1, temperature=0.5,
        plan_weights=PlanWeights(plan=1.0, memory=0.5, feasibility=0.5),
    )


class TestWhatADreamStepCosts(unittest.TestCase):
    """One real walk, the whole per-step DREAM path, timed."""

    @classmethod
    def setUpClass(cls):
        from earshot.sim.world import World, audio_spec_parts, camera_sensor_specs
        from earshot.task.models import load_clap_encoder, load_clip_encoder
        from earshot.types import NoRouteError

        config = AudioConfig()
        spec, binaural = audio_spec_parts()
        audio_sensor_spec(spec, config, binaural)
        cls.world = World(
            _DATASET.scene_path, camera_sensor_specs(width=256, height=256) + [spec]
        )
        cls.world.seed_navmesh(PLACEMENT_SEED)
        cls.clap = load_clap_encoder()
        cls.context = DreamContext(
            knobs=_knobs(), memory=empty_memory(), clip_encoder=load_clip_encoder()
        )
        source = cls.world.random_navigable_point()
        handle = AudioSensorHandle(
            cls.world.sensor_handle(str(spec.uuid)), cls.world.observe, source,
            uuid=str(spec.uuid),
        )
        clip = synthetic_burst(config.sample_rate)
        cls.world.set_pose(cls.world.random_navigable_point())
        target = cls.world.random_navigable_point()
        follow = cls.world.follower(goal_radius=GOAL_RADIUS)

        cls.state = begin_episode(cls.context)
        cls.seconds, cls.clap_seconds = [], []
        for index in range(STEPS):
            observation, _guard = handle.observe()
            mono, rate = heard_clip_for_clap(
                render_through_ir(handle.audio_of(observation), clip),
                config.sample_rate,
            )
            import time

            started = time.perf_counter()
            audio = audio_embedding(mono, rate, cls.clap)
            cls.clap_seconds.append(time.perf_counter() - started)
            outcome = observe(
                cls.state, cls.context,
                frame=np.asarray(observation["rgb"]),
                audio=audio,
                pose=cls.world.pose(),
                prev_action=None if index == 0 else "move_forward",
                belief=None,
            )
            cls.state = outcome.episode
            cls.seconds.append(outcome.seconds)
            try:
                action = follow(target)
            except NoRouteError:
                break
            if action is None:
                break
            cls.world.step(action)
        print("  walked {} step(s)".format(len(cls.state)), flush=True)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "world", None) is not None:
            cls.world.close()

    def test_the_walk_was_long_enough_to_time(self):
        self.assertGreaterEqual(len(self.seconds), 5)

    def test_the_whole_dream_step_fits_under_criterion_sevens_headroom(self):
        """**THE MEASUREMENT THIS FILE EXISTS FOR.** `observe`'s own bracket covers `f_v`,
        the fuse, the STM push, `q_t` and all three retrievals. `f_u` is timed separately
        beside it because the runner calls it outside `observe` -- it owns the CLAP
        encoder and the heard signal."""
        worst = max(self.seconds)
        mean = sum(self.seconds) / len(self.seconds)
        clap_worst = max(self.clap_seconds)
        clap_mean = sum(self.clap_seconds) / len(self.clap_seconds)
        print("  f_u (CLAP) per step:      mean {:.4f} s  worst {:.4f} s".format(
            clap_mean, clap_worst))
        print("  observe() (f_v + eq19-24) mean {:.4f} s  worst {:.4f} s".format(
            mean, worst))
        print("  whole DREAM step:         mean {:.4f} s  worst {:.4f} s".format(
            mean + clap_mean, worst + clap_worst))
        print("  headroom under criterion 7: {:.4f} s "
              "(ceiling {:.3f} - matrix-2's worst step {:.4f})".format(
                  HEADROOM_S, CRITERION_7_CEILING_S, MATRIX_2_WORST_STEP_S))
        print("  -> a DREAM step would land at about {:.4f} s ({:.0f}% of the "
              "ceiling)".format(
                  MATRIX_2_WORST_STEP_S + worst + clap_worst,
                  100.0 * (MATRIX_2_WORST_STEP_S + worst + clap_worst)
                  / CRITERION_7_CEILING_S))
        self.assertLess(
            worst + clap_worst, HEADROOM_S,
            "a DREAM step costs {:.4f} s against {:.4f} s of headroom; a sweep launched "
            "over this budget produces a night of episodes criterion 7 will refuse, and "
            "the fix is a design change rather than a retry".format(
                worst + clap_worst, HEADROOM_S),
        )

    def test_f_u_is_the_dominant_new_cost_or_it_is_not(self):
        """Printed rather than asserted: which half dominates decides what a cheaper
        design would have to change. `f_v` was priced at 11.6 ms in PR #108 and `f_u` has
        never been priced per step at all."""
        clap_share = sum(self.clap_seconds) / (sum(self.clap_seconds) + sum(self.seconds))
        print("  f_u is {:.0f}% of the new per-step cost".format(100.0 * clap_share))
        self.assertGreater(sum(self.seconds), 0.0)

    def test_consolidation_is_paid_once_per_episode_and_is_cheap(self):
        """Eq. 8-16 runs at the episode boundary, not per step, so it is outside every
        per-step budget. Printed so a 500-step episode's boundary cost is on record."""
        import time

        started = time.perf_counter()
        memory, scores = consolidate_episode(
            self.state, self.context,
            sound_concept="toilet_flush", target_concept="toilet", room_concept=None,
            reached=True, final_gap_m=0.5,
        )
        elapsed = time.perf_counter() - started
        print("  consolidation of {} step(s): {:.4f} s -> {} segment(s) scored, "
              "{} M^E row(s), {} pattern(s)".format(
                  len(self.state), elapsed, len(scores),
                  len(memory.experience), len(memory.pattern)))
        if scores:
            print("  I_j: min {:.4f} max {:.4f} (eta {:.3f})".format(
                min(scores), max(scores), self.context.knobs.eta))
        self.assertLess(
            elapsed, 1.0,
            "consolidating one episode took {:.3f} s; at 500 episodes that is real time "
            "and the rebuild of M^P is the thing to look at".format(elapsed),
        )

    def test_the_retrieval_is_live_once_the_memory_has_rows(self):
        """A second episode against a filled `M^L`, which is the state every episode
        after the first runs in. An empty memory retrieves nothing, so timing the first
        episode alone would price the cheapest case."""
        memory, _scores = consolidate_episode(
            self.state, self.context,
            sound_concept="toilet_flush", target_concept="toilet", room_concept=None,
            reached=True, final_gap_m=0.5,
        )
        filled = self.context.with_memory(memory)
        if not len(memory.experience):
            self.skipTest("nothing was retained, so there is no filled memory to time")
        state = begin_episode(filled)
        timings, informed = [], 0
        for index in range(5):
            observation = self.world.observe()
            outcome = observe(
                state, filled,
                frame=np.asarray(observation["rgb"]),
                # A fixed waveform: this test times the RETRIEVAL against a filled
                # memory, and the audio half's content is not what varies here. The
                # per-step cost of a real heard signal is the test above.
                audio=audio_embedding(
                    synthetic_burst(48000, seconds=0.5), 48000, self.clap
                ),
                pose=self.world.pose(),
                prev_action=None if index == 0 else "move_forward",
                belief=None,
            )
            state = outcome.episode
            timings.append(outcome.seconds)
            informed += int(bool(outcome.context))
        print("  against {} M^E row(s) / {} pattern(s): observe() mean {:.4f} s, "
              "{}/5 step(s) retrieved something".format(
                  len(memory.experience), len(memory.pattern),
                  sum(timings) / len(timings), informed))
        self.assertEqual(
            informed, 5,
            "a filled M^L retrieved nothing on some step, so omega_t was undefined "
            "exactly where the hypothesis needs it",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
