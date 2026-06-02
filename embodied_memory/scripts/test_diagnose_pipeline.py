"""
Sanity tests for diagnose_pipeline.py (pipeline log-mining diagnostics).

Pure-function tests only — no Habitat, no models, no file I/O. The module
parses episode_*.json logs that the runner already writes; here we exercise
the parsing/aggregation helpers on synthetic dicts.

Invoke with::

    python embodied_memory/scripts/test_diagnose_pipeline.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_EMB_DIR = Path(__file__).resolve().parent.parent  # …/embodied_memory


def _load():
    path = _EMB_DIR / "scripts" / "diagnose_pipeline.py"
    spec = importlib.util.spec_from_file_location("diagnose_pipeline", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["diagnose_pipeline"] = mod
    spec.loader.exec_module(mod)
    return mod


dp = _load()


# ---- caption_mentions: word-boundary keyword + synonyms ----

def case_caption_mentions_direct():
    assert dp.caption_mentions("a wooden chair and a small table", "chair") is True
    print("  case_caption_mentions_direct: OK")


def case_caption_mentions_plural():
    assert dp.caption_mentions("two chairs by the window", "chair") is True
    print("  case_caption_mentions_plural: OK")


def case_caption_mentions_synonym():
    assert dp.caption_mentions("a comfy armchair in the corner", "chair") is True
    assert dp.caption_mentions("a mattress on the floor", "bed") is True
    print("  case_caption_mentions_synonym: OK")


def case_caption_mentions_absent():
    assert dp.caption_mentions("a hallway with white walls", "chair") is False
    print("  case_caption_mentions_absent: OK")


def case_caption_mentions_word_boundary():
    # 'bedroom' must NOT count as observing a 'bed'; 'chairman' not a 'chair'.
    assert dp.caption_mentions("a cozy bedroom with blue walls", "bed") is False
    assert dp.caption_mentions("the chairman gave a speech", "chair") is False
    print("  case_caption_mentions_word_boundary: OK")


# ---- classify_visits: cold-first per (scene, category) ----

def case_classify_visits_cold_first():
    eps = [
        {"episode_idx": 0, "scene_id": "A", "target_category": "chair"},
        {"episode_idx": 1, "scene_id": "A", "target_category": "chair"},
        {"episode_idx": 2, "scene_id": "A", "target_category": "bed"},
        {"episode_idx": 3, "scene_id": "A", "target_category": "bed"},
    ]
    cold = dp.classify_visits(eps)
    assert cold == [True, False, True, False], cold
    print("  case_classify_visits_cold_first: OK")


def case_classify_visits_separates_scenes():
    eps = [
        {"episode_idx": 0, "scene_id": "A", "target_category": "chair"},
        {"episode_idx": 1, "scene_id": "B", "target_category": "chair"},
    ]
    assert dp.classify_visits(eps) == [True, True]
    print("  case_classify_visits_separates_scenes: OK")


# ---- episode_observation: did the target appear in keyframe captions ----

def case_episode_observation_counts():
    steps = [
        {"caption": "a hallway", "agent_pos": [0, 0, 0]},
        {"caption": "a wooden chair", "agent_pos": [1, 0, 1]},
        {"caption": "a kitchen", "agent_pos": [2, 0, 2]},
    ]
    observed, frac, n = dp.episode_observation(steps, "chair")
    assert observed is True
    assert abs(frac - 1 / 3) < 1e-9, frac
    assert n == 3
    print("  case_episode_observation_counts: OK")


def case_episode_observation_none():
    steps = [{"caption": "a hallway", "agent_pos": [0, 0, 0]}]
    observed, frac, n = dp.episode_observation(steps, "chair")
    assert observed is False and frac == 0.0 and n == 1
    print("  case_episode_observation_none: OK")


# ---- nearest_caption: match a memory world_xy back to a keyframe caption ----

def case_nearest_caption_picks_closest():
    index = [
        ((0.0, 0.0), "a hallway"),
        ((5.0, 5.0), "a wooden chair"),
    ]
    cap = dp.nearest_caption((4.8, 5.1), index)
    assert cap == "a wooden chair", cap
    print("  case_nearest_caption_picks_closest: OK")


# ---- memory_candidate_audit: are retrieved memories on-target? ----

def case_memory_audit_relevance():
    index = [
        ((0.0, 0.0), "a long hallway"),
        ((5.0, 5.0), "a wooden chair by a table"),
    ]
    decisions = [
        {"candidates": [
            {"source": "memory", "world_xy": [5.0, 5.0], "raw_score": 0.61},
            {"source": "memory", "world_xy": [0.0, 0.0], "raw_score": 0.20},
            {"source": "frontier", "world_xy": [9.0, 9.0], "raw_score": 0.7},
        ]},
    ]
    audit = dp.memory_candidate_audit(decisions, index, "chair")
    assert audit["n_memory"] == 2, audit
    # one of the two memory candidates' nearest caption mentions chair
    assert abs(audit["on_target_rate"] - 0.5) < 1e-9, audit
    assert abs(audit["cos_max"] - 0.61) < 1e-9, audit
    print("  case_memory_audit_relevance: OK")


def main() -> int:
    print("diagnose_pipeline sanity tests")
    case_caption_mentions_direct()
    case_caption_mentions_plural()
    case_caption_mentions_synonym()
    case_caption_mentions_absent()
    case_caption_mentions_word_boundary()
    case_classify_visits_cold_first()
    case_classify_visits_separates_scenes()
    case_episode_observation_counts()
    case_episode_observation_none()
    case_nearest_caption_picks_closest()
    case_memory_audit_relevance()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
