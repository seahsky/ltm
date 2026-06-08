"""
Sanity tests for ``EmbodiedMemoryBridge.consolidate_subgoal_boundary`` — the
within-episode (sub-goal event-boundary) consolidation added after
multion-micro3 showed end-of-episode-only consolidation makes same-episode
LTM recall structurally impossible.

Run on the REAL bridge (faiss + dialogue_memory), unlike the stub-level
runner tests in ``test_advance_subgoal.py``. The invariants under test are
the S2/S3 attribution-purity guarantees:

  * S1 (disable_stm + disable_ltm): boundary call is a no-op; no
    ``modules_invoked`` flag flips.
  * S2 (disable_ltm): boundary call drains ``_pending`` WITHOUT bumping the
    episode counters ``consolidate()`` bumps — STM-only stays pure.
  * S3: the fine layer grows mid-episode, ``_episodes_seen`` is untouched,
    mid-layer clustering is NOT triggered, and entries are stamped
    ``episode_success=False`` (outcome unknown mid-episode) with the
    embodied metadata (agent_position / scene_id) patched on.

SKIP-prints (exit 0) when the heavy deps (faiss et al.) are unavailable —
mirrors how RACE-only suites degrade locally.

Invoke with::

    python embodied_memory/scripts/test_memory_bridge_consolidate.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from embodied_memory.memory_bridge import EmbodiedMemoryBridge
except ImportError as e:  # faiss / torch / transformers missing locally
    print(f"SKIP test_memory_bridge_consolidate: heavy deps unavailable ({e})")
    sys.exit(0)


_DIM = 16


def _encode(text: str) -> np.ndarray:
    """Deterministic, content-sensitive stand-in for SBERT."""
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.standard_normal(_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _mk_bridge(**toggles) -> EmbodiedMemoryBridge:
    return EmbodiedMemoryBridge(
        text_embed_dim=_DIM,
        visual_embed_dim=_DIM,
        text_encode_fn=_encode,
        **toggles,
    )


def _kf(step_idx: int, caption: str):
    """Duck-typed Keyframe; observe_keyframe only reads these fields."""
    return SimpleNamespace(
        step_idx=step_idx,
        caption=caption,
        text_embedding=_encode(caption),
        visual_embedding=np.zeros(_DIM, dtype=np.float32),
        agent_position=np.array([1.0, 0.0, float(step_idx)], dtype=np.float32),
        agent_yaw=0.25,
    )


def _observe(bridge, n=3):
    bridge.begin_episode("ep-test", scene_id="SC")
    for i in range(n):
        bridge.observe_keyframe(
            _kf(i, f"a long detailed caption number {i} with my favorite chair "
                   f"and what I like about this room"),
            action=1, reward=0.0)


def case_s1_boundary_is_noop():
    bridge = _mk_bridge(disable_stm=True, disable_ltm=True, disable_rerank=True)
    _observe(bridge)  # disable_stm: nothing buffers
    assert bridge._pending == []
    out = bridge.consolidate_subgoal_boundary(episode_idx=0)
    assert out == {"fine": [], "mid": [], "coarse": []}
    assert len(bridge.ltm.fine) == 0
    assert bridge._episodes_seen == 0
    assert not any(bridge.modules_invoked.values()), bridge.modules_invoked
    print("  case_s1_boundary_is_noop: OK")


def case_s2_boundary_drains_without_counter_bumps():
    bridge = _mk_bridge(disable_ltm=True)
    _observe(bridge)
    assert len(bridge._pending) == 3
    out = bridge.consolidate_subgoal_boundary(episode_idx=0)
    assert out == {"fine": [], "mid": [], "coarse": []}
    assert bridge._pending == [], "boundary must drain STM under disable_ltm"
    # The S2-purity guarantee: unlike consolidate(), NO episode-counter bumps
    # and no consolidation flag (nothing was written).
    assert bridge._episodes_seen == 0
    assert bridge._successful_episodes_seen == 0
    assert bridge.modules_invoked["consolidation"] is False
    assert len(bridge.ltm.fine) == 0
    # Contrast: end-of-episode consolidate() DOES bump the episode counter.
    _observe(bridge)
    bridge.consolidate(episode_success=False, episode_idx=0)
    assert bridge._episodes_seen == 1
    print("  case_s2_boundary_drains_without_counter_bumps: OK")


def case_s3_boundary_writes_fine_without_episode_bookkeeping():
    bridge = _mk_bridge()
    _observe(bridge)
    out = bridge.consolidate_subgoal_boundary(episode_idx=0)
    assert len(out["fine"]) > 0, "fine layer must grow mid-episode"
    assert len(bridge.ltm.fine) == len(out["fine"])
    assert bridge._pending == [], "pending must drain at the boundary"
    assert bridge.modules_invoked["consolidation"] is True
    assert bridge.modules_invoked["ltm_fine"] is True
    # No episode bookkeeping: counters untouched, NO mid-cluster trigger.
    assert bridge._episodes_seen == 0
    assert bridge._successful_episodes_seen == 0
    assert len(bridge.ltm.mid) == 0
    # Entries are stamped episode_success=False (outcome unknown mid-episode)
    # with the embodied metadata patched on.
    for entry in bridge.ltm.fine.entries:
        assert entry.metadata.get("episode_success") is False, entry.metadata
        assert entry.metadata.get("scene_id") == "SC", entry.metadata
        assert entry.metadata.get("agent_position") is not None, entry.metadata
    print("  case_s3_boundary_writes_fine_without_episode_bookkeeping: OK")


def case_s3_boundary_then_end_consolidate_counts_one_episode():
    bridge = _mk_bridge()
    _observe(bridge)
    n_boundary = len(bridge.consolidate_subgoal_boundary(episode_idx=0)["fine"])
    # Second hunt segment within the same episode, then the normal episode end.
    for i in range(3, 6):
        bridge.observe_keyframe(
            _kf(i, f"another rich caption {i} with my favorite bed and what "
                   f"I like near the window"),
            action=1, reward=0.0)
    bridge.consolidate(episode_success=False, episode_idx=0)
    assert bridge._episodes_seen == 1, "episode counted exactly once (at END)"
    assert len(bridge.ltm.fine) > n_boundary, "end consolidate must also write"
    print("  case_s3_boundary_then_end_consolidate_counts_one_episode: OK")


def case_s3_empty_pending_boundary_sets_no_flags():
    bridge = _mk_bridge()
    bridge.begin_episode("ep-empty", scene_id="SC")
    out = bridge.consolidate_subgoal_boundary(episode_idx=0)
    assert out == {"fine": [], "mid": [], "coarse": []}
    assert bridge.modules_invoked["consolidation"] is False
    print("  case_s3_empty_pending_boundary_sets_no_flags: OK")


def case_freeze_scene_skips_fine_write():
    """LTM_FREEZE_SCENE=<scene> makes consolidate() SKIP the fine-layer write while
    in that scene (cross-env isolation: each away/query episode then sees only the
    earlier home sightings, no within-away cross-episode accumulation). Episode
    bookkeeping still bumps; pending still drains; other scenes are unaffected."""
    import os as _os
    bridge = _mk_bridge()  # S3
    # home scene: consolidate writes to fine
    bridge.begin_episode("ep-home", scene_id="HOME")
    for i in range(3):
        bridge.observe_keyframe(
            _kf(i, f"a long detailed caption {i} with a chair and a table near the window"),
            action=1, reward=0.0)
    bridge.consolidate(episode_success=False, episode_idx=0)
    n_home = len(bridge.ltm.fine)
    assert n_home > 0, "home write should populate fine"
    seen_before = bridge._episodes_seen

    # away scene with LTM_FREEZE_SCENE set: the fine write is skipped
    _os.environ["LTM_FREEZE_SCENE"] = "AWAY"
    try:
        bridge.begin_episode("ep-away", scene_id="AWAY")
        for i in range(3):
            bridge.observe_keyframe(
                _kf(i, f"a long detailed caption {i} with a bed and a lamp near the door"),
                action=1, reward=0.0)
        out = bridge.consolidate(episode_success=False, episode_idx=1)
        assert len(bridge.ltm.fine) == n_home, "frozen away scene must NOT add fine entries"
        assert out["fine"] == [], out
        assert bridge._pending == [], "pending must still drain"
        assert bridge._episodes_seen == seen_before + 1, "episode still counted when frozen"
    finally:
        _os.environ.pop("LTM_FREEZE_SCENE", None)

    # a different (non-frozen) scene writes again
    bridge.begin_episode("ep-other", scene_id="OTHER")
    for i in range(3):
        bridge.observe_keyframe(
            _kf(i, f"a long detailed caption {i} with a sofa and a rug near a bookshelf"),
            action=1, reward=0.0)
    bridge.consolidate(episode_success=False, episode_idx=2)
    assert len(bridge.ltm.fine) > n_home, "non-frozen scene must write again"
    print("  case_freeze_scene_skips_fine_write: OK")


def main() -> int:
    print("memory_bridge consolidate_subgoal_boundary sanity tests")
    case_s1_boundary_is_noop()
    case_s2_boundary_drains_without_counter_bumps()
    case_s3_boundary_writes_fine_without_episode_bookkeeping()
    case_s3_boundary_then_end_consolidate_counts_one_episode()
    case_s3_empty_pending_boundary_sets_no_flags()
    case_freeze_scene_skips_fine_write()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
