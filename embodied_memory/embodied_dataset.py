"""
Dataset adapters for training the predictor/scorer on embodied episode logs.

The dialogue-side ``train_predictor.py`` and ``train_scorer.py`` consume two
dataset shapes from MSC:

- predictor:  ``__getitem__`` -> ``(history_emb, next_emb)``  (FloatTensors)
- scorer:     ``__getitem__`` -> ``(segment_emb, label_tensor)``

This module produces the same shapes from ``runs/<dir>/episode_*.json``
written by ``EpisodeRunner``, so the dialogue trainers can be reused
unchanged. Captions are re-encoded at ``__getitem__`` time via the same
encoder protocol the dialogue trainers expect (any object exposing
``.encode(str) -> np.ndarray``).

Per-episode JSON layout the runner writes:

    {
      "episode_idx": int,
      "episode_id": str,
      "scene_id": str,
      "target_category": str,
      "success": bool,
      "spl": float,
      "soft_spl": float,
      "n_steps": int,
      "steps": [
        {"step_idx": int, "agent_pos": [...], "agent_yaw": float,
         "caption": str, "action": int|null, "reward": float, "done": bool},
        ...
      ]
    }

Note the per-step record carries the *caption* but not the visual embedding.
Train-time re-encoding keeps the dataset cheap to load and avoids having to
serialize CLIP vectors into every run's per-episode JSON.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover — torch is in the env
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[misc,assignment]


# ----------------------------------------------------------------------
# raw sample loader
# ----------------------------------------------------------------------


@dataclass
class EmbodiedSample:
    """One step's worth of information lifted out of an episode JSON."""
    episode_idx: int
    episode_id: str
    scene_id: str
    target_category: Optional[str]
    success: bool
    spl: float
    step_idx: int
    caption: str
    agent_pos: np.ndarray
    agent_yaw: float
    action: Optional[int] = None
    reward: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


def iter_episode_files(run_dirs: Sequence[str]) -> Iterator[str]:
    """Yield ``episode_*.json`` paths under one or more run directories,
    skipping error files (``episode_*_error.json``).
    """
    for run in run_dirs:
        for path in sorted(glob.glob(os.path.join(run, "episode_*.json"))):
            base = os.path.basename(path)
            if base.endswith("_error.json"):
                continue
            yield path


def load_episode_samples(run_dirs: Sequence[str]) -> List[EmbodiedSample]:
    """Flatten every step in every episode under ``run_dirs`` into a list
    of ``EmbodiedSample``. Episodes that fail to parse are skipped."""
    out: List[EmbodiedSample] = []
    for ep_path in iter_episode_files(run_dirs):
        try:
            with open(ep_path, "r", encoding="utf-8") as f:
                ep = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        success = bool(ep.get("success", False))
        spl = float(ep.get("spl", 1.0 if success else 0.0))
        ep_idx = int(ep.get("episode_idx", -1))
        ep_id = str(ep.get("episode_id", ""))
        scene = str(ep.get("scene_id", ""))
        target = ep.get("target_category")

        for stp in ep.get("steps") or []:
            cap = stp.get("caption")
            if not cap:
                continue
            pos = np.asarray(stp.get("agent_pos", [0.0, 0.0, 0.0]), dtype=np.float32)
            out.append(EmbodiedSample(
                episode_idx=ep_idx,
                episode_id=ep_id,
                scene_id=scene,
                target_category=str(target) if target is not None else None,
                success=success,
                spl=spl,
                step_idx=int(stp.get("step_idx", 0)),
                caption=str(cap),
                agent_pos=pos,
                agent_yaw=float(stp.get("agent_yaw", 0.0)),
                action=stp.get("action"),
                reward=float(stp.get("reward", 0.0)),
                extra={"soft_spl": float(ep.get("soft_spl", spl))},
            ))
    return out


# ----------------------------------------------------------------------
# Predictor: (history, next) pairs
# ----------------------------------------------------------------------


class EmbodiedPredictionDataset(Dataset):
    """Predictor adapter — yields ``(history_emb, next_emb)``.

    The predictor's job is forward modeling: given the agent's recent
    caption stream, anticipate the next observation's embedding. The
    historical "utterance" here is the concatenation of the last
    ``max_history_len`` captions within the same episode; the target is
    the next caption.
    """

    def __init__(
        self,
        samples_or_run_dirs: Sequence[Any],
        encoder: Any,
        max_history_len: int = 5,
    ):
        if not samples_or_run_dirs:
            raise ValueError("samples_or_run_dirs must be non-empty")
        if isinstance(samples_or_run_dirs[0], (str, os.PathLike)):
            samples = load_episode_samples(list(map(str, samples_or_run_dirs)))
        else:
            samples = list(samples_or_run_dirs)
        self.encoder = encoder
        self.max_history_len = int(max_history_len)
        self._pairs: List[Tuple[str, str, EmbodiedSample]] = []
        self._group_by_episode_and_build_pairs(samples)

    def _group_by_episode_and_build_pairs(self, samples: List[EmbodiedSample]):
        # Group by (scene_id, episode_id) preserving order.
        buckets: Dict[Tuple[str, str], List[EmbodiedSample]] = {}
        for s in samples:
            buckets.setdefault((s.scene_id, s.episode_id), []).append(s)
        for key, group in buckets.items():
            group.sort(key=lambda s: s.step_idx)
            for i in range(1, len(group)):
                hist_start = max(0, i - self.max_history_len)
                history_caps = [s.caption for s in group[hist_start:i]]
                next_cap = group[i].caption
                history_str = " ".join(history_caps)
                self._pairs.append((history_str, next_cap, group[i]))

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx):
        history_str, next_cap, _ = self._pairs[idx]
        history_emb = np.asarray(self.encoder.encode(history_str), dtype=np.float32)
        next_emb = np.asarray(self.encoder.encode(next_cap), dtype=np.float32)
        if torch is None:
            return history_emb, next_emb
        return torch.from_numpy(history_emb), torch.from_numpy(next_emb)


# ----------------------------------------------------------------------
# Scorer: (segment, label) pairs
# ----------------------------------------------------------------------


class EmbodiedImportanceDataset(Dataset):
    """Scorer adapter — yields ``(segment_emb, label_tensor)``.

    Label semantics ('``label_mode``'):
        - ``"success"``    : 1.0 if the episode succeeded, else 0.0.
        - ``"soft_spl"``   : episode's soft-SPL as a continuous regression target
                             (useful when binary SPL is sparse — Phase-1 baseline
                             ran 90 episodes with zero successes).
        - ``"spl"``        : episode's SPL (typically binary in HM3D-ObjectNav).

    Default is ``"success"`` so the trained scorer mirrors the
    "will-be-mentioned" supervision of the MSC scorer (a binary classification).
    """

    SUPPORTED_LABELS = ("success", "soft_spl", "spl")

    def __init__(
        self,
        samples_or_run_dirs: Sequence[Any],
        encoder: Any,
        label_mode: str = "success",
        keep_per_episode_top_k: Optional[int] = None,
    ):
        if label_mode not in self.SUPPORTED_LABELS:
            raise ValueError(
                f"label_mode={label_mode!r} not in {self.SUPPORTED_LABELS}"
            )
        if not samples_or_run_dirs:
            raise ValueError("samples_or_run_dirs must be non-empty")
        if isinstance(samples_or_run_dirs[0], (str, os.PathLike)):
            samples = load_episode_samples(list(map(str, samples_or_run_dirs)))
        else:
            samples = list(samples_or_run_dirs)

        self.encoder = encoder
        self.label_mode = label_mode

        if keep_per_episode_top_k is not None:
            samples = self._downsample_per_episode(samples, keep_per_episode_top_k)

        self._samples = samples

    @staticmethod
    def _downsample_per_episode(
        samples: List[EmbodiedSample], k: int
    ) -> List[EmbodiedSample]:
        buckets: Dict[Tuple[str, str], List[EmbodiedSample]] = {}
        for s in samples:
            buckets.setdefault((s.scene_id, s.episode_id), []).append(s)
        out: List[EmbodiedSample] = []
        for group in buckets.values():
            group.sort(key=lambda s: s.step_idx)
            # Uniform stride keeps both early and late observations.
            if len(group) <= k:
                out.extend(group)
                continue
            stride = len(group) / float(k)
            picks = [group[int(i * stride)] for i in range(k)]
            out.extend(picks)
        return out

    def _label(self, s: EmbodiedSample) -> float:
        if self.label_mode == "success":
            return 1.0 if s.success else 0.0
        if self.label_mode == "spl":
            return float(s.spl)
        return float(s.extra.get("soft_spl", s.spl))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx):
        s = self._samples[idx]
        emb = np.asarray(self.encoder.encode(s.caption), dtype=np.float32)
        label = self._label(s)
        if torch is None:
            return emb, np.array([label], dtype=np.float32)
        return torch.from_numpy(emb), torch.tensor([label], dtype=torch.float32)

    def get_stats(self) -> Dict[str, Any]:
        labels = [self._label(s) for s in self._samples]
        total = len(labels)
        positive = sum(1 for v in labels if v > 0.0)
        return {
            "total": total,
            "label_mode": self.label_mode,
            "positive": positive,
            "negative": total - positive,
            "positive_ratio": positive / total if total > 0 else 0.0,
            "mean_label": float(np.mean(labels)) if labels else 0.0,
        }


# ----------------------------------------------------------------------
# Convenience: a single CLI entry point users can `python -m` for sanity
# ----------------------------------------------------------------------


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Embodied dataset stats")
    parser.add_argument("run_dirs", nargs="+", help="One or more runs/<dir>/")
    parser.add_argument("--label-mode", default="success",
                        choices=EmbodiedImportanceDataset.SUPPORTED_LABELS)
    args = parser.parse_args(argv)

    samples = load_episode_samples(args.run_dirs)
    print(f"loaded {len(samples)} per-step samples from {len(args.run_dirs)} run dir(s)")
    if not samples:
        return 1

    # Distinct (scene, episode) groups
    keys = {(s.scene_id, s.episode_id) for s in samples}
    successes = sum(1 for s in samples if s.success)
    print(f"  distinct (scene, episode) pairs: {len(keys)}")
    print(f"  successful steps: {successes}  /  failed steps: {len(samples) - successes}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main(sys.argv[1:]))
