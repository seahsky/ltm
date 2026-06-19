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


def _category_viewpoints_from_content(
    content: Dict[str, Any], category: str
) -> List[List[float]]:
    """All goal view_point positions for ``category`` from an ObjectNav
    content dict (``goals_by_category`` keys look like
    ``<scene>.basis.glb_<category>`` — suffix-matched so multi-token
    categories like ``tv_monitor`` resolve correctly). Pure: no I/O."""
    suffix = f"_{category}"
    out: List[List[float]] = []
    for key, instances in (content.get("goals_by_category") or {}).items():
        if not key.endswith(suffix):
            continue
        for inst in instances or []:
            for vp in inst.get("view_points") or []:
                pos = (vp.get("agent_state") or {}).get("position")
                if pos:
                    out.append(list(pos))
    return out


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
        task: str = "objectnav",
        rir_grid_path: Optional[str] = None,
        t_anom: int = 0,
        anomaly_clip_path: Optional[str] = None,
        target_norm_rms_db: float = -20.0,
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
        # MultiON per-category viewpoint cache, keyed (scene_label, category).
        # Loaded lazily from the dataset's content/<scene>.json.gz — we do NOT
        # depend on habitat-lab attaching goals_by_category to the episode
        # object. Viewpoints are static per scene, so the cache persists.
        self._cat_vps_cache: Dict[Tuple[str, str], List[List[float]]] = {}
        self._scene_content_cache: Dict[str, Optional[Dict[str, Any]]] = {}
        self._current_scene_label: Optional[str] = None

        # AudioGoal task (render side only — the cached .npz RIR grid is consumed
        # here, the audio SIMULATOR is never imported). All None / inert unless
        # task=="audiogoal", so the objectnav/revisit/multion paths are unchanged.
        self.task = str(task)
        self._rir_grid_path = rir_grid_path
        self._audio_t_anom = int(t_anom)
        self._anomaly_clip_path = anomaly_clip_path
        self._target_norm_rms_db = float(target_norm_rms_db)
        self._rir_grid = None              # lazy-loaded at reset (scene-matched)
        self._anomaly_clip_norm = None     # normalized FSD50K clip, loaded once
        self._audio_render_cfg = None      # AudioTaskConfig, built at reset

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

        # The revisit eval's cold seeding episodes start the agent ON the goal
        # viewpoint (start_end_distance == 0), so Habitat's SoftSPL/SPL would
        # raise ZeroDivisionError mid-step and abort the seed. Guard the measures
        # to yield metric=0.0 on that degenerate division (cold SPL is unused by
        # Gate A). No-op if the measure classes can't be imported.
        try:
            from habitat.tasks.nav.nav import SPL, SoftSPL

            from .spl_guard import guard_zero_start_distance
            guard_zero_start_distance([SPL, SoftSPL])
        except Exception:
            pass

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
            # HM3D ObjectNav's depth sensor defaults to normalize_depth=True,
            # which returns depth in [0,1] (depth_m / max_depth) rather than
            # meters. The frontier planner's occupancy splat assumes *metric*
            # depth — normalized depth collapses every ray's ground range
            # (a 3 m wall reads ~0.3), so the height gate marks nearly every
            # endpoint OCCUPIED and carves almost no FREE cells (Run-5 oracle
            # smoke: cells_free≈4 on wcojb4TFT35). Return true meters so the
            # densified splat and height gate work as designed.
            sim_sensors.depth_sensor.normalize_depth = False

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

            # Pin the episode iterator (shuffle=False + group_by_scene=True) so a
            # multi-scene revisit run yields each scene's COLD seed episode before
            # its WARM visits (the LTM must hold the cold sighting before a warm
            # visit recalls it; the analyzer also labels visit order by processing
            # order).
            from .episode_order import pin_episode_order
            pin_episode_order(config)

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
        self._current_scene_label = scene_label
        metadata: Dict[str, Any] = {
            "source": "habitat_live", "max_steps": self.max_steps,
        }

        ep_info = getattr(env.current_episode, "info", None)

        # AudioGoal: lazy-load the scene-matched RIR grid + normalized clip ONCE.
        # Per-episode t_anom (episode.info — M2 writes a high value for cold-
        # silent mapping passes and a low value for warm-fires episodes) overrides
        # the run-level default. Gated on task=="audiogoal"; otherwise inert.
        ep_t_anom = self._audio_t_anom
        if self.task == "audiogoal" and self._rir_grid_path:
            from .audio import RIRGrid
            from .audio_task import AudioTaskConfig, resolve_t_anom
            ep_t_anom = resolve_t_anom(ep_info, self._audio_t_anom)
            if self._rir_grid is None or self._rir_grid.scene_id != scene_label:
                grid = RIRGrid.load(self._rir_grid_path)
                if grid.scene_id != scene_label:
                    raise ValueError(
                        f"RIR grid scene_id {grid.scene_id!r} != current scene "
                        f"{scene_label!r} — audio is rendered per (scene, source)."
                    )
                self._rir_grid = grid
                self._anomaly_clip_norm = None   # re-normalize at the grid's sr
            if self._anomaly_clip_norm is None:
                self._anomaly_clip_norm = self._load_anomaly_clip()
            self._audio_render_cfg = AudioTaskConfig(
                enabled=True, t_anom=ep_t_anom,
                sample_rate=int(self._rir_grid.sample_rate),
            )

        # MultiON: surface the ordered category chain (written by
        # make_multion_smoke into the episode's info dict) so the runner's
        # sub-goal cursor can read it from Episode.metadata. Absent the key,
        # single-goal episodes carry no chain — behaviour unchanged.
        if isinstance(ep_info, dict) and ep_info.get("object_categories"):
            metadata["object_categories"] = [
                str(c) for c in ep_info["object_categories"]
            ]

        # AudioGoal: surface the anomaly config (source_position / anomaly_class /
        # anomaly_object / per-episode t_anom written by the M2 dataset builder
        # into episode.info) for the runner + analysis.
        if self.task == "audiogoal":
            src_pos = None
            anom_cls = None
            anom_obj = None
            if isinstance(ep_info, dict):
                if ep_info.get("source_position") is not None:
                    src_pos = [float(v) for v in ep_info["source_position"]]
                if ep_info.get("anomaly_class"):
                    anom_cls = str(ep_info["anomaly_class"])
                if ep_info.get("anomaly_object"):
                    anom_obj = str(ep_info["anomaly_object"])
            metadata["audio_config"] = {
                "task": self.task,
                "scene_id": scene_label,
                "t_anom": ep_t_anom,
                "rir_grid_available": self._rir_grid is not None,
                "sample_rate": int(self._rir_grid.sample_rate) if self._rir_grid is not None else None,
                "source_position": src_pos,
                "anomaly_class": anom_cls,
                "anomaly_object": anom_obj,
            }

        episode = Episode(
            episode_id=str(getattr(env.current_episode, "episode_id", episode_idx)),
            scene_id=scene_label,
            target_category=getattr(env.current_episode, "object_category", "unknown"),
            target_position=target_pos,
            metadata=metadata,
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

    def get_sim(self):
        """Return the underlying habitat-sim Simulator (for the oracle
        ShortestPathFollower), or None if the env hasn't been built yet."""
        if self._env is None:
            return None
        return getattr(self._env, "sim", None) or getattr(self._env, "_sim", None)

    # ------------------------------------------------------------------
    # MultiON per-sub-goal distance seam
    # ------------------------------------------------------------------

    def _scene_content(self, scene_label: str) -> Optional[Dict[str, Any]]:
        """Load (and cache) the dataset content dict for ``scene_label`` from
        ``content/<scene>.json.gz`` next to ``episodes_path``."""
        if scene_label in self._scene_content_cache:
            return self._scene_content_cache[scene_label]
        content: Optional[Dict[str, Any]] = None
        if self.episodes_path:
            path = os.path.join(
                os.path.dirname(self.episodes_path), "content",
                f"{scene_label}.json.gz",
            )
            if os.path.exists(path):
                import gzip
                import json
                try:
                    with gzip.open(path, "rt", encoding="utf-8") as f:
                        content = json.load(f)
                except Exception:
                    content = None
        self._scene_content_cache[scene_label] = content
        return content

    def _category_viewpoints(self, category: str) -> List[List[float]]:
        scene = self._current_scene_label or self._scene_ids[0]
        key = (scene, category)
        if key not in self._cat_vps_cache:
            content = self._scene_content(scene)
            self._cat_vps_cache[key] = (
                _category_viewpoints_from_content(content, category)
                if content else []
            )
        return self._cat_vps_cache[key]

    def distance_to_category(self, agent_pos, category: str) -> Optional[float]:
        """Geodesic distance to the nearest goal view_point of ``category``
        via the sim's multi-goal shortest path. ``inf`` (unreachable), no sim,
        or no viewpoints -> None (a None never advances a sub-goal)."""
        sim = self.get_sim()
        vps = self._category_viewpoints(category)
        if sim is None or not vps:
            return None
        try:
            d = sim.geodesic_distance(
                np.asarray(agent_pos, dtype=np.float32),
                [np.asarray(v, dtype=np.float32) for v in vps],
            )
        except Exception:
            return None
        if d is None or not np.isfinite(float(d)):
            return None
        return float(d)

    def nearest_category_viewpoint(self, agent_pos, category: str):
        """``(geodesic_distance, viewpoint_position)`` of the nearest
        view_point of ``category`` (the L_opt leg anchor), or None."""
        sim = self.get_sim()
        vps = self._category_viewpoints(category)
        if sim is None or not vps:
            return None
        best_d, best_vp = None, None
        for vp in vps:
            try:
                d = sim.geodesic_distance(
                    np.asarray(agent_pos, dtype=np.float32),
                    np.asarray(vp, dtype=np.float32),
                )
            except Exception:
                continue
            if d is None or not np.isfinite(float(d)):
                continue
            if best_d is None or float(d) < best_d:
                best_d, best_vp = float(d), list(vp)
        if best_d is None:
            return None
        return best_d, best_vp

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

        # AudioGoal RENDER ONLY: nearest-cell lookup + fftconvolve of the cached
        # RIR with the clip (silence before t_anom). No decision logic here — the
        # runner calls audio_task on Step.audio. Never imports the audio sim.
        audio = None
        if (self.task == "audiogoal" and self._rir_grid is not None
                and self._anomaly_clip_norm is not None and self._audio_render_cfg is not None):
            from .audio_task import render_step_audio
            audio = render_step_audio(
                self._rir_grid, agent_state.position, self._anomaly_clip_norm,
                self._step_count, self._audio_render_cfg,
            )

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
            audio=audio,
        )

    def _load_anomaly_clip(self) -> np.ndarray:
        """Load + RMS-normalize the anomaly clip at the grid's sample rate. A real
        FSD50K .wav if ``anomaly_clip_path`` is set, else a deterministic synthetic
        broadband burst. Delegates to ``audio_task.build_anomaly_clip`` so the live
        render and the onset-calibration diagnostic share one energy scale."""
        from .audio_task import build_anomaly_clip
        grid_sr = int(self._rir_grid.sample_rate) if self._rir_grid is not None else 48000
        return build_anomaly_clip(self._anomaly_clip_path, grid_sr, self._target_norm_rms_db)

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
