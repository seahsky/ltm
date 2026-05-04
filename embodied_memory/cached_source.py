"""
CachedEpisodeSource — replay pre-recorded HM3D trajectories.

Bundle format (a single .npz):

    rgb         : (T, H, W, 3) uint8
    depth       : (T, H, W)    float32
    semantic    : (T, H, W)    int32   (or omitted if not available)
    agent_pos   : (T, 3)       float32
    agent_yaw   : (T,)         float32
    actions     : (T,)         int32   (action that produced obs_t; first is -1)
    rewards     : (T,)         float32
    dones       : (T,)         bool
    episode_id  : str          (saved as a 0-d unicode array)
    scene_id    : str
    target_cat  : str

Multiple episodes are stored as a npz with an index file ``episodes.json`` in
the same directory listing the bundle paths. Or, for a single-episode bundle,
just point ``--cached-bundle`` at the npz directly and we'll loop it
``--n-episodes`` times (proof-of-life: same trajectory multiple times so the
LTM has something to retrieve from on episode 2+).
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import numpy as np

from .episode_source import AgentState, Episode, EpisodeSource, Step


class CachedEpisodeSource(EpisodeSource):
    """Replay a single .npz bundle as N episodes.

    The runner still drives the loop (calls ``step(action)`` each tick) but
    the cached source ignores the action and just returns the next frame in
    the bundle. This lets us exercise the entire downstream pipeline without
    a live habitat-sim install.
    """

    def __init__(self, bundle_path: str, n_episodes: int = 5):
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"Cached bundle not found: {bundle_path}")
        self.bundle_path = bundle_path
        self.n_episodes = n_episodes

        data = np.load(bundle_path, allow_pickle=False)
        self._rgb = data["rgb"]
        self._depth = data["depth"]
        self._semantic = data["semantic"] if "semantic" in data.files else None
        self._agent_pos = data["agent_pos"]
        self._agent_yaw = data["agent_yaw"]
        self._actions = data["actions"] if "actions" in data.files else None
        self._rewards = data["rewards"] if "rewards" in data.files else None
        self._dones = data["dones"] if "dones" in data.files else None
        self._episode_id = (
            str(data["episode_id"]) if "episode_id" in data.files else "cached_0"
        )
        self._scene_id = (
            str(data["scene_id"]) if "scene_id" in data.files else "cached_scene"
        )
        self._target_cat = (
            str(data["target_cat"]) if "target_cat" in data.files else "chair"
        )

        self._t = 0
        self._T = int(self._rgb.shape[0])
        self._current_episode_idx = 0

    # ------------------------------------------------------------------
    # EpisodeSource interface
    # ------------------------------------------------------------------

    def num_episodes(self) -> int:
        return int(self.n_episodes)

    def reset(self, episode_idx: int) -> Tuple[Step, Episode]:
        self._current_episode_idx = int(episode_idx)
        self._t = 0
        ep = Episode(
            episode_id=f"{self._episode_id}#{episode_idx}",
            scene_id=self._scene_id,
            target_category=self._target_cat,
            target_position=None,
            metadata={"source": "cached", "bundle": os.path.basename(self.bundle_path)},
        )
        return self._make_step(action=None), ep

    def step(self, action: int) -> Step:
        self._t = min(self._t + 1, self._T - 1)
        return self._make_step(action=int(action))

    @property
    def supports_actions(self) -> bool:
        return False

    def _make_step(self, action: Optional[int]) -> Step:
        t = self._t
        rgb = np.asarray(self._rgb[t], dtype=np.uint8)
        depth = np.asarray(self._depth[t], dtype=np.float32)
        sem = (
            np.asarray(self._semantic[t], dtype=np.int32)
            if self._semantic is not None
            else None
        )
        pos = np.asarray(self._agent_pos[t], dtype=np.float32)
        yaw = float(self._agent_yaw[t])
        agent_state = AgentState(position=pos, rotation_yaw=yaw)
        reward = float(self._rewards[t]) if self._rewards is not None else 0.0
        # Cached bundles end when t hits T-1 (or the recorded `dones` flag).
        if self._dones is not None:
            done = bool(self._dones[t])
        else:
            done = bool(t >= self._T - 1)
        return Step(
            step_idx=t,
            rgb=rgb,
            depth=depth,
            semantic=sem,
            agent_state=agent_state,
            action=action,
            reward=reward,
            done=done,
            info={"source": "cached"},
        )


def write_synthetic_bundle(path: str, T: int = 60, hw: Tuple[int, int] = (64, 64)) -> str:
    """Build a tiny synthetic bundle so the cached pipeline is testable
    without any HM3D download. Used by unit-test smoke runs."""
    h, w = hw
    rng = np.random.RandomState(0)
    rgb = (rng.rand(T, h, w, 3) * 255).astype(np.uint8)
    depth = (rng.rand(T, h, w) * 5.0).astype(np.float32)
    semantic = (rng.randint(0, 20, size=(T, h, w))).astype(np.int32)
    agent_pos = np.stack(
        [np.linspace(0, 2, T), np.zeros(T), np.linspace(0, -3, T)], axis=1
    ).astype(np.float32)
    agent_yaw = np.linspace(0, 1.0, T).astype(np.float32)
    actions = np.ones(T, dtype=np.int32)
    rewards = np.zeros(T, dtype=np.float32)
    dones = np.zeros(T, dtype=bool)
    dones[-1] = True
    np.savez(
        path,
        rgb=rgb,
        depth=depth,
        semantic=semantic,
        agent_pos=agent_pos,
        agent_yaw=agent_yaw,
        actions=actions,
        rewards=rewards,
        dones=dones,
        episode_id=np.array("synthetic"),
        scene_id=np.array("synthetic_scene"),
        target_cat=np.array("chair"),
    )
    return path
