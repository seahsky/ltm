#!/usr/bin/env python
"""Unit tests for the BLIP-2 ITM value-scorer (the semantic-frontier "blip2"
value SIGNAL that replaces the flat CLIP cosine) — the pure logic, NO torch /
transformers / habitat.

Covers:
  * Blip2ITMScorer._match_prob — softmax over the 2-logit ITM head → match-class
    column, clamped to [0,1]; the ONCE-softmax (not the model-card double-softmax)
    convention; degenerate (1,) ITC fallback → sigmoid; empty → 0.0.
  * Blip2ITMScorer.score — delegates to the heavy _itm_logits seam (monkeypatched
    GPU-free) and applies _match_prob; goal-prompt independence (score re-runs per
    frame, the prompt is just passed through).
  * EpisodeRunner._observe_semantic_value backend dispatch (the one logic swap):
      - off (weight<=0) => no observe_value call (byte-identical default).
      - backend "clip" (default) => uses keyframe.visual_embedding + CLIP text.
      - backend "blip2" => uses RAW step.rgb via self._value_scorer.score, NOT
        the keyframe embedding; no-op when self._value_scorer is None.

Run: /opt/anaconda3/envs/ltm-embodied/bin/python embodied_memory/test_value_scorer.py
"""
import math
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embodied_memory.perception import Blip2ITMScorer  # noqa: E402
from embodied_memory.episode_runner import EpisodeRunner  # noqa: E402


# ----------------------------------------------------------------------
# Blip2ITMScorer._match_prob (the softmax->[:,1] mapping)
# ----------------------------------------------------------------------

def case_match_prob_softmax_match_column():
    # logits [non-match, match] = [0, ln(3)] -> softmax = [0.25, 0.75]
    p = Blip2ITMScorer._match_prob(np.array([0.0, math.log(3.0)], dtype=np.float32))
    assert abs(p - 0.75) < 1e-6, p


def case_match_prob_is_in_unit_interval():
    for lo, hi in [(-50.0, 50.0), (50.0, -50.0), (0.0, 0.0), (3.2, 3.2)]:
        p = Blip2ITMScorer._match_prob(np.array([lo, hi], dtype=np.float32))
        assert 0.0 <= p <= 1.0, (lo, hi, p)


def case_match_prob_higher_match_logit_higher_prob():
    low = Blip2ITMScorer._match_prob(np.array([1.0, 0.0]))   # match weaker
    high = Blip2ITMScorer._match_prob(np.array([0.0, 1.0]))  # match stronger
    assert high > low, (high, low)
    assert high > 0.5 > low, (high, low)


def case_match_prob_single_logit_itc_fallback_sigmoid():
    # A degenerate (1,) logit (ITC head) -> sigmoid squash, monotone, in [0,1].
    p0 = Blip2ITMScorer._match_prob(np.array([0.0]))
    assert abs(p0 - 0.5) < 1e-6, p0
    pbig = Blip2ITMScorer._match_prob(np.array([10.0]))
    psmall = Blip2ITMScorer._match_prob(np.array([-10.0]))
    assert pbig > 0.99 and psmall < 0.01, (pbig, psmall)


def case_match_prob_empty_is_zero():
    assert Blip2ITMScorer._match_prob(np.array([])) == 0.0


def case_match_prob_numerically_stable_large_logits():
    # Without the max-subtraction this overflows; must still be a clean float.
    p = Blip2ITMScorer._match_prob(np.array([1000.0, 1001.0]))
    assert 0.0 <= p <= 1.0 and not math.isnan(p), p
    assert p > 0.5, p   # match column is larger


# ----------------------------------------------------------------------
# Blip2ITMScorer.score (delegates to the monkeypatched seam)
# ----------------------------------------------------------------------

def case_score_uses_itm_logits_seam():
    sc = Blip2ITMScorer()
    calls = []

    def fake_logits(rgb, text):
        calls.append((rgb.shape, text))
        return np.array([0.0, math.log(3.0)], dtype=np.float32)  # -> 0.75

    sc._itm_logits = fake_logits  # monkeypatch the heavy seam (no GPU)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    v = sc.score(rgb, "Seems like there is a chair ahead.")
    assert abs(v - 0.75) < 1e-6, v
    assert calls == [((4, 4, 3), "Seems like there is a chair ahead.")], calls


def case_score_clamped_and_passes_prompt_through():
    sc = Blip2ITMScorer()
    seen = {}

    def fake_logits(rgb, text):
        seen["text"] = text
        return np.array([5.0, -5.0], dtype=np.float32)  # match weak -> ~0

    sc._itm_logits = fake_logits
    v = sc.score(np.zeros((2, 2, 3), dtype=np.uint8), "a photo of a bed")
    assert 0.0 <= v <= 1.0 and v < 0.01, v
    assert seen["text"] == "a photo of a bed"


# ----------------------------------------------------------------------
# EpisodeRunner._observe_semantic_value backend dispatch
# ----------------------------------------------------------------------

class _FakePlanner:
    def __init__(self, weight):
        self.semantic_frontier_weight = weight
        self.observed = []  # list of (pos, yaw, value)

    def observe_value(self, pos, yaw, value):
        self.observed.append((pos, yaw, float(value)))


class _FakeClip:
    """Returns a unit-vector text embedding so CLIP-path dot products are
    predictable; records that encode_text was called (blip2 path must NOT)."""
    def __init__(self):
        self.text_calls = []

    def encode_text(self, text):
        self.text_calls.append(text)
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


class _FakeValueScorer:
    def __init__(self, ret=0.83):
        self.ret = ret
        self.calls = []  # (rgb_shape, text)

    def score(self, rgb, text):
        self.calls.append((np.asarray(rgb).shape, text))
        return self.ret


def _fake_keyframe(emb):
    return types.SimpleNamespace(visual_embedding=np.asarray(emb, dtype=np.float32))


def _fake_step(rgb):
    return types.SimpleNamespace(
        rgb=rgb,
        agent_state=types.SimpleNamespace(
            position=np.array([1.0, 0.0, 2.0], dtype=np.float32),
            rotation_yaw=0.3,
        ),
    )


def _runner(weight, clip=None, value_scorer=None):
    """A bare object exposing only the attributes _observe_semantic_value reads,
    so we exercise the dispatch logic without building a full EpisodeRunner."""
    r = types.SimpleNamespace(
        planner=_FakePlanner(weight),
        clip_encoder=clip,
        _value_scorer=value_scorer,
        _semantic_goal=None,
        _goal_text_emb=None,
    )
    return r


def _observe(runner, step, keyframe, goal):
    # call the unbound method against our duck-typed runner
    return EpisodeRunner._observe_semantic_value(runner, step, keyframe, goal)


def _clear_backend_env():
    os.environ.pop("LTM_SEMANTIC_FRONTIER_BACKEND", None)
    os.environ.pop("LTM_SEMANTIC_FRONTIER_PROMPT", None)


def case_observe_noop_when_weight_off():
    _clear_backend_env()
    r = _runner(weight=0.0, clip=_FakeClip(), value_scorer=_FakeValueScorer())
    _observe(r, _fake_step(np.zeros((4, 4, 3), np.uint8)), _fake_keyframe([1, 0, 0]), "chair")
    assert r.planner.observed == [], r.planner.observed  # default path untouched


def case_observe_clip_default_backend_uses_keyframe_embedding():
    _clear_backend_env()  # backend defaults to "clip"
    clip = _FakeClip()
    vs = _FakeValueScorer()
    r = _runner(weight=0.5, clip=clip, value_scorer=vs)
    # keyframe emb aligned with the clip text emb [1,0,0] -> dot = 1.0
    _observe(r, _fake_step(np.zeros((4, 4, 3), np.uint8)), _fake_keyframe([1, 0, 0]), "chair")
    assert len(r.planner.observed) == 1, r.planner.observed
    _, _, v = r.planner.observed[0]
    assert abs(v - 1.0) < 1e-6, v
    assert clip.text_calls == ["a photo of a chair"], clip.text_calls
    assert vs.calls == [], "blip2 scorer must NOT be called on the clip path"


def case_observe_blip2_backend_uses_raw_rgb_not_embedding():
    _clear_backend_env()
    os.environ["LTM_SEMANTIC_FRONTIER_BACKEND"] = "blip2"
    try:
        clip = _FakeClip()
        vs = _FakeValueScorer(ret=0.83)
        r = _runner(weight=0.5, clip=clip, value_scorer=vs)
        rgb = np.zeros((7, 9, 3), dtype=np.uint8)
        _observe(r, _fake_step(rgb), _fake_keyframe([1, 0, 0]), "bed")
        assert len(r.planner.observed) == 1, r.planner.observed
        _, _, v = r.planner.observed[0]
        assert abs(v - 0.83) < 1e-6, v
        # used RAW rgb (7,9,3) + the VLFM default prompt; did NOT touch CLIP.
        assert vs.calls == [((7, 9, 3), "Seems like there is a bed ahead.")], vs.calls
        assert clip.text_calls == [], "clip must NOT be used on the blip2 path"
    finally:
        _clear_backend_env()


def case_observe_blip2_noop_when_scorer_missing():
    _clear_backend_env()
    os.environ["LTM_SEMANTIC_FRONTIER_BACKEND"] = "blip2"
    try:
        r = _runner(weight=0.5, clip=_FakeClip(), value_scorer=None)
        _observe(r, _fake_step(np.zeros((4, 4, 3), np.uint8)), _fake_keyframe([1, 0, 0]), "chair")
        assert r.planner.observed == [], r.planner.observed
    finally:
        _clear_backend_env()


def case_observe_blip2_respects_prompt_override():
    _clear_backend_env()
    os.environ["LTM_SEMANTIC_FRONTIER_BACKEND"] = "blip2"
    os.environ["LTM_SEMANTIC_FRONTIER_PROMPT"] = "is this a {goal}?"
    try:
        vs = _FakeValueScorer()
        r = _runner(weight=0.5, clip=_FakeClip(), value_scorer=vs)
        _observe(r, _fake_step(np.zeros((4, 4, 3), np.uint8)), _fake_keyframe([1, 0, 0]), "sofa")
        assert vs.calls[0][1] == "is this a sofa?", vs.calls
    finally:
        _clear_backend_env()


def case_observe_noop_when_no_goal():
    _clear_backend_env()
    os.environ["LTM_SEMANTIC_FRONTIER_BACKEND"] = "blip2"
    try:
        vs = _FakeValueScorer()
        r = _runner(weight=0.5, clip=_FakeClip(), value_scorer=vs)
        _observe(r, _fake_step(np.zeros((4, 4, 3), np.uint8)), _fake_keyframe([1, 0, 0]), "")
        assert r.planner.observed == [] and vs.calls == []
    finally:
        _clear_backend_env()


def main():
    case_match_prob_softmax_match_column()
    case_match_prob_is_in_unit_interval()
    case_match_prob_higher_match_logit_higher_prob()
    case_match_prob_single_logit_itc_fallback_sigmoid()
    case_match_prob_empty_is_zero()
    case_match_prob_numerically_stable_large_logits()
    case_score_uses_itm_logits_seam()
    case_score_clamped_and_passes_prompt_through()
    case_observe_noop_when_weight_off()
    case_observe_clip_default_backend_uses_keyframe_embedding()
    case_observe_blip2_backend_uses_raw_rgb_not_embedding()
    case_observe_blip2_noop_when_scorer_missing()
    case_observe_blip2_respects_prompt_override()
    case_observe_noop_when_no_goal()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
