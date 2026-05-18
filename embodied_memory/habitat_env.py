"""
Live Habitat ObjectNav env wrapper.

Wraps habitat-lab's ObjectNav-v1 task on HM3D into an EpisodeSource. All
habitat-* imports are lazy so this module is importable without the env
installed (e.g. when running --mode cached on a vanilla Python install).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .episode_source import AgentState, Episode, EpisodeSource, Step


# Discrete action ids must match the order produced by habitat-lab's
# DiscreteActionSpaceConfiguration. We pin them here so the planner is
# decoupled from the env config.
_ACTION_NAMES = ["stop", "move_forward", "turn_left", "turn_right", "look_up", "look_down"]


class HabitatObjectNavSource(EpisodeSource):
    """ObjectNav on HM3D via habitat-lab.

    Args:
        scene_id: HM3D scene id, e.g. ``"00800-TEEsavR23oF"``. Must exist under
            the configured scene dataset path.
        scene_dataset_path: path to ``hm3d_annotated_basis.scene_dataset_config.json``
            (under ``data/hm3d/scene_datasets/hm3d/...``). If None, we fall back
            to habitat-lab's default search.
        episodes_path: gz JSON of ObjectNav episodes (HM3D val episodes).
        n_episodes: cap on how many episodes from the dataset to expose.
        max_steps: per-episode step cap.
        target_category: if set, filter dataset episodes to this object category.
        image_hw: (height, width) of RGB/depth/semantic sensors. Default 256x256
            keeps CPU rendering tractable on Apple Silicon.
    """

    def __init__(
        self,
        scene_id,  # str | List[str]
        scene_dataset_path: Optional[str] = None,
        episodes_path: Optional[str] = None,
        n_episodes: int = 5,
        max_steps: int = 250,
        target_category: Optional[str] = "chair",
        image_hw: Tuple[int, int] = (256, 256),
    ):
        # scene_id can be a single id (legacy) or a list — passed straight to
        # habitat's `dataset.content_scenes`, which cycles episodes across all
        # listed scenes. Keep the public attribute as a string for backwards
        # compat (joined with commas) and stash the resolved list separately.
        if isinstance(scene_id, (list, tuple)):
            self._scene_ids: List[str] = [str(s) for s in scene_id]
        else:
            self._scene_ids = [str(scene_id)]
        self.scene_id = ",".join(self._scene_ids)

        self.scene_dataset_path = scene_dataset_path or self._default_scene_dataset_path()
        self.episodes_path = episodes_path or self._default_episodes_path()
        self.n_episodes = n_episodes
        self.max_steps = max_steps
        self.target_category = target_category
        self.image_hw = image_hw

        self._env = None
        self._current_episode: Optional[Episode] = None
        self._step_count = 0

    @staticmethod
    def _default_scene_dataset_path() -> Optional[str]:
        # Conventional layout produced by embodied_memory/scripts/download_hm3d.sh.
        # Returned only if it actually exists on disk; otherwise None so habitat-lab's
        # own default lookup runs unchanged.
        candidates = [
            "data/hm3d/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json",
            "data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json",
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    @staticmethod
    def _default_episodes_path() -> Optional[str]:
        # Prefer val_mini (smallest), then val, then train. Matches the splits
        # downloaded by embodied_memory/scripts/download_hm3d.sh.
        candidates = [
            "data/hm3d/datasets/objectnav/hm3d/v1/val_mini/val_mini.json.gz",
            "data/hm3d/datasets/objectnav/hm3d/v1/val/val.json.gz",
            "data/hm3d/datasets/objectnav/hm3d/v1/train/train.json.gz",
            "data/datasets/objectnav/hm3d/v1/val_mini/val_mini.json.gz",
            "data/datasets/objectnav/hm3d/v1/val/val.json.gz",
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    # ------------------------------------------------------------------
    # lazy env construction
    # ------------------------------------------------------------------

    def _build_env(self):
        if self._env is not None:
            return self._env

        try:
            import habitat
            from habitat.config import read_write
            from habitat.config.default import get_config
            from habitat.config.default_structured_configs import HabitatSimSemanticSensorConfig
        except ImportError as e:
            raise RuntimeError(
                "habitat-lab is not importable. Activate the conda env "
                "(see embodied_memory/README.md) or use --mode cached."
            ) from e

        # We use the canonical ObjectNav HM3D config and override scene + sensor
        # resolution. habitat-lab ships this config under benchmark/nav/objectnav.
        config = get_config("benchmark/nav/objectnav/objectnav_hm3d.yaml")

        h, w = self.image_hw
        with read_write(config):
            sim_sensors = config.habitat.simulator.agents.main_agent.sim_sensors
            sim_sensors.rgb_sensor.height = h
            sim_sensors.rgb_sensor.width = w
            sim_sensors.depth_sensor.height = h
            sim_sensors.depth_sensor.width = w

            if "semantic_sensor" not in sim_sensors:
                sim_sensors.semantic_sensor = HabitatSimSemanticSensorConfig()
            sim_sensors.semantic_sensor.height = h
            sim_sensors.semantic_sensor.width = w

            if self.scene_dataset_path:
                config.habitat.simulator.scene_dataset = self.scene_dataset_path
                # habitat-sim resolves each episode's relative scene path against
                # `dataset.scenes_dir` (default ``data/scene_datasets``). Our
                # downloads live under ``data/hm3d/scene_datasets`` so we point
                # `scenes_dir` at the parent of the scene_dataset_config file.
                # Walk up two levels: <root>/scene_datasets/hm3d/<config>.json
                #                  -> <root>/scene_datasets   (i.e. scenes_dir)
                cfg_dir = os.path.dirname(self.scene_dataset_path)
                scenes_root = os.path.dirname(cfg_dir)
                if scenes_root:
                    config.habitat.dataset.scenes_dir = scenes_root
            if self.episodes_path:
                config.habitat.dataset.data_path = self.episodes_path
            # ObjectNav HM3D v1 names per-scene episode files by the bare scene
            # name (e.g. ``TEEsavR23oF.json.gz``), not the prefixed form
            # (``00800-TEEsavR23oF``). Accept either input by stripping a
            # leading ``NNNNN-`` prefix when present.
            bare_scene_ids: List[str] = []
            for sid in self._scene_ids:
                head, sep, tail = sid.partition("-")
                bare_scene_ids.append(tail if sep and head.isdigit() else sid)
            config.habitat.dataset.content_scenes = bare_scene_ids
            config.habitat.environment.max_episode_steps = int(self.max_steps)

            # Override habitat-sim's GPU device selection when the host's EGL
            # stack has no CUDA-aware device (e.g. compute-only containers
            # that ship Mesa software EGL but no libEGL_nvidia.so). Default
            # 0 matches habitat-lab's normal CUDA+EGL interop path.
            gpu_dev = int(os.environ.get("HABITAT_SIM_GPU_DEVICE_ID", "0"))
            config.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu_dev

        self._env = habitat.Env(config=config)
        return self._env

    # ------------------------------------------------------------------
    # EpisodeSource interface
    # ------------------------------------------------------------------

    def num_episodes(self) -> int:
        env = self._build_env()
        # habitat-lab's Env exposes total episodes via the dataset.
        return min(self.n_episodes, len(env.episodes))

    def reset(self, episode_idx: int) -> Tuple[Step, Episode]:
        env = self._build_env()
        ep_count = len(env.episodes)
        if ep_count == 0:
            raise RuntimeError(
                f"No episodes available for scene {self.scene_id}. "
                "Check episodes_path and scene_dataset_path."
            )

        # Skip past episodes that don't match the target category.
        for _ in range(ep_count):
            obs = env.reset()
            ep = env.current_episode
            if (
                self.target_category is None
                or getattr(ep, "object_category", None) == self.target_category
            ):
                break
        else:
            raise RuntimeError(
                f"No episode in scene {self.scene_id} matches target "
                f"category={self.target_category}."
            )

        self._step_count = 0
        agent_state = self._read_agent_state(env)
        target_pos = None
        goals = getattr(env.current_episode, "goals", None)
        if goals:
            try:
                target_pos = np.array(goals[0].position, dtype=np.float32)
            except Exception:
                target_pos = None

        # When multiple scenes are loaded, habitat cycles through them — tag
        # each episode with the scene it actually came from (extracted from
        # the episode's glb path) so paired analysis can join on it.
        ep_scene = getattr(env.current_episode, "scene_id", None)
        if isinstance(ep_scene, str) and ep_scene:
            base = os.path.basename(ep_scene)
            scene_label = base.split(".", 1)[0]
        else:
            scene_label = self._scene_ids[0]
        episode = Episode(
            episode_id=str(getattr(env.current_episode, "episode_id", episode_idx)),
            scene_id=scene_label,
            target_category=getattr(env.current_episode, "object_category", "unknown"),
            target_position=target_pos,
            metadata={"source": "habitat_live", "max_steps": self.max_steps},
        )
        step = self._make_step(obs, action=None, reward=0.0, done=False, info={})
        self._current_episode = episode
        return step, episode

    def step(self, action: int) -> Step:
        env = self._build_env()
        action_name = _ACTION_NAMES[action] if 0 <= action < len(_ACTION_NAMES) else "stop"
        out = env.step(action_name)
        # habitat-lab returns Observations + reward/done via env.get_metrics().
        # In gym-style wrapper it's (obs, reward, done, info); core Env it's just obs.
        if isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
        else:
            obs = out
            reward = 0.0
            done = env.episode_over
            info = env.get_metrics() if hasattr(env, "get_metrics") else {}

        self._step_count += 1
        return self._make_step(obs, action=action, reward=float(reward), done=bool(done), info=info)

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _make_step(self, obs: Any, action: Optional[int], reward: float, done: bool, info: Dict[str, Any]) -> Step:
        rgb = np.asarray(obs.get("rgb"), dtype=np.uint8)
        depth_raw = obs.get("depth")
        depth = np.asarray(depth_raw, dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        semantic_raw = obs.get("semantic")
        semantic = None
        if semantic_raw is not None:
            semantic = np.asarray(semantic_raw, dtype=np.int32)
            if semantic.ndim == 3 and semantic.shape[-1] == 1:
                semantic = semantic[..., 0]

        agent_state = self._read_agent_state(self._env)
        return Step(
            step_idx=self._step_count,
            rgb=rgb,
            depth=depth,
            semantic=semantic,
            agent_state=agent_state,
            action=action,
            reward=reward,
            done=done,
            info=dict(info or {}),
        )

    @staticmethod
    def _read_agent_state(env) -> AgentState:
        try:
            sim = env.sim if hasattr(env, "sim") else env._sim
            state = sim.get_agent_state()
            pos = np.asarray(state.position, dtype=np.float32)
            # rotation is a quaternion; pull yaw from the y-axis component.
            q = state.rotation
            # habitat returns numpy quaternion (w,x,y,z); yaw = atan2(2*(w*y+x*z), 1-2*(y*y+z*z))
            w, x, y, z = float(q.w), float(q.x), float(q.y), float(q.z)
            yaw = float(np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z)))
            return AgentState(position=pos, rotation_yaw=yaw)
        except Exception:
            return AgentState(position=np.zeros(3, dtype=np.float32), rotation_yaw=0.0)
