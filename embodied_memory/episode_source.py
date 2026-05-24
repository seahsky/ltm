"""
EpisodeSource interface and dataclasses.

The runner consumes episodes through this interface so live Habitat-sim and
pre-recorded cached trajectories are interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np


@dataclass
class AgentState:
    """Agent pose at a single step. Habitat conventions: y-up, meters, radians."""
    position: np.ndarray  # shape (3,) world xyz
    rotation_yaw: float   # heading in radians around y-axis

    def as_dict(self) -> Dict[str, Any]:
        return {
            "position": self.position.tolist(),
            "rotation_yaw": float(self.rotation_yaw),
        }


@dataclass
class Step:
    """Single environment step.

    Fields are deliberately the union of what live and cached modes can supply.
    Cached mode may leave `semantic` as None and pass pre-computed captions in
    `info["caption"]`; downstream consumers must tolerate either.
    """
    step_idx: int
    rgb: np.ndarray              # (H, W, 3) uint8
    depth: np.ndarray            # (H, W) float32, meters
    semantic: Optional[np.ndarray]  # (H, W) int32 instance ids, or None
    agent_state: AgentState
    action: Optional[int]        # action that produced this obs (None for first obs)
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """One ObjectNav episode."""
    episode_id: str
    scene_id: str
    target_category: str
    target_position: Optional[np.ndarray]  # (3,) goal location if known, else None
    steps: List[Step] = field(default_factory=list)
    success: bool = False
    spl: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.steps)


class EpisodeSource:
    """Abstract source of episodes.

    Implementations:
    - HabitatObjectNavSource (live): drives a habitat-lab env, streams steps.
    - CachedEpisodeSource: replays npz bundles produced by a prior run.

    The runner does NOT call .step() directly; instead the source produces a
    callable that the runner invokes with the action chosen by the planner.
    This keeps the planner-in-the-loop semantics consistent across modes.
    """

    def reset(self, episode_idx: int) -> Tuple[Step, Episode]:
        """Reset to episode `episode_idx`, return the initial observation
        plus a mostly-empty Episode struct (steps list will grow)."""
        raise NotImplementedError

    def step(self, action: int) -> Step:
        """Apply `action` and return the next Step."""
        raise NotImplementedError

    def num_episodes(self) -> int:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def get_sim(self):
        """Return the underlying simulator if the source has one (live Habitat),
        else None. Used by the oracle backbone's ShortestPathFollower."""
        return None

    # Convenience: iterate fully scripted episodes (cached mode shortcut).
    def iter_steps(self, episode_idx: int) -> Iterator[Step]:
        """For cached sources only: replay an episode without action input."""
        raise NotImplementedError

    @property
    def supports_actions(self) -> bool:
        """True if .step(action) actually drives the env. Cached mode is False."""
        return True

    @property
    def action_space(self) -> List[int]:
        """Discrete action ids the planner may emit. Default ObjectNav set."""
        # 0: stop, 1: move_forward, 2: turn_left, 3: turn_right, 4: look_up, 5: look_down
        return [0, 1, 2, 3, 4, 5]
