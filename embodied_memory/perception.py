"""
Perception layer for the embodied loop.

Two pieces:
- ``CLIPKeyframeEncoder`` — open_clip ViT-B/32 on MPS (or CPU fallback) → 512-d
  visual embedding per RGB keyframe.
- ``SemanticCaptioner`` — turns Habitat's semantic-sensor output into a short
  English caption ("sees: chair, table, door"), so the existing SBERT-based
  text encoder in dialogue_memory.encoder can produce a paired text embedding.

Both are intentionally light. The proof-of-life slice substitutes them for
ReMEmbR's full vision-language stack; swapping ReMEmbR in later only needs a
new ``KeyframeEncoder`` implementing the same encode() signature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Keyframe dataclass
# ----------------------------------------------------------------------


@dataclass
class Keyframe:
    """One keyframe summarises a sub-trajectory of M raw steps.

    Both visual and text vectors are stored so the bridge can index either
    space. The textual caption is human-readable for log inspection.
    """
    step_idx: int
    rgb: np.ndarray                       # (H, W, 3) uint8
    visual_embedding: np.ndarray          # CLIP image vector
    caption: str                          # short English description
    text_embedding: np.ndarray            # SBERT vector for the caption
    agent_position: np.ndarray            # (3,) world xyz at this frame
    agent_yaw: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
# CLIP keyframe encoder
# ----------------------------------------------------------------------


class CLIPKeyframeEncoder:
    """open_clip ViT-B/32 image tower.

    Tries MPS first (Apple Silicon GPU), falls back to CPU on any error.
    Loaded lazily on first encode().
    """

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: Optional[str] = None):
        self.model_name = model_name
        self.pretrained = pretrained
        self._requested_device = device
        self._model = None
        self._preprocess = None
        self._device = None
        self._embed_dim = 512  # ViT-B/32 output

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def device(self) -> str:
        if self._device is None:
            self._lazy_load()
        return self._device

    def _pick_device(self) -> str:
        if self._requested_device is not None:
            return self._requested_device
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _lazy_load(self):
        if self._model is not None:
            return
        try:
            import torch
            import open_clip
        except ImportError as e:
            raise RuntimeError(
                "open_clip_torch and torch are required for CLIPKeyframeEncoder. "
                "Install via embodied_memory/environment.yml or pip install open_clip_torch torch."
            ) from e

        device = self._pick_device()
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained
            )
            model = model.to(device).eval()
        except Exception:
            # MPS sometimes fails on first weight cast; retry on CPU.
            device = "cpu"
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained
            )
            model = model.to(device).eval()

        self._model = model
        self._preprocess = preprocess
        self._device = device

    def encode(self, rgb: np.ndarray) -> np.ndarray:
        """Encode a single uint8 (H, W, 3) RGB frame to a 512-d float32 vector."""
        self._lazy_load()
        import torch
        from PIL import Image

        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        img = Image.fromarray(rgb)
        tensor = self._preprocess(img).unsqueeze(0).to(self._device)
        with torch.no_grad():
            feats = self._model.encode_image(tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return feats.squeeze(0).detach().cpu().float().numpy()

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a text string into the same joint CLIP space as encode().

        Used by the embodied LTM to:
        - seed the coarse layer with category priors ("a photo of a chair")
        - build goal-directed queries at decision time (using the per-episode
          target category) against a fine-layer indexed on visual embeddings.
        """
        self._lazy_load()
        import torch
        import open_clip

        tokenizer = open_clip.get_tokenizer(self.model_name)
        tokens = tokenizer([text]).to(self._device)
        with torch.no_grad():
            feats = self._model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return feats.squeeze(0).detach().cpu().float().numpy()


# ----------------------------------------------------------------------
# Semantic captioner
# ----------------------------------------------------------------------


# Minimal HM3D-Semantics → English label map. We deliberately keep this small;
# the goal is a short caption, not a full scene graph.
_DEFAULT_HM3D_LABELS: Tuple[str, ...] = (
    "wall", "floor", "ceiling", "door", "window", "chair", "sofa", "couch",
    "bed", "table", "desk", "cabinet", "shelf", "tv", "monitor", "lamp",
    "plant", "rug", "mirror", "sink", "toilet", "bathtub", "shower",
    "refrigerator", "microwave", "stove", "oven", "counter", "stairs",
    "fireplace", "picture", "book", "vase", "cushion", "towel", "curtain",
    "appliance", "object",
)


class SemanticCaptioner:
    """Turn Habitat semantic-sensor output into a short English caption.

    Habitat HM3D-Semantics returns per-pixel instance ids; the simulator's
    semantic_scene maps each instance to a category name. We pick the top-K
    most-pixel-coverage categories (ignoring background classes) and produce
    a template caption.

    For the proof-of-life slice we don't *need* the live scene's instance map
    to be wired in — if it's missing or empty, we fall back to a generic
    "agent at (x,y) sees the room" caption so the rest of the pipeline still
    runs. The caller can pass an explicit ``id_to_category`` map if available.
    """

    def __init__(
        self,
        top_k: int = 4,
        ignore_categories: Tuple[str, ...] = ("wall", "floor", "ceiling", "unknown", "void"),
        id_to_category: Optional[Dict[int, str]] = None,
    ):
        self.top_k = top_k
        self.ignore_categories = set(ignore_categories)
        self.id_to_category = id_to_category or {}

    def set_scene_categories(self, id_to_category: Dict[int, str]):
        """Refresh the instance-id → category-name map (e.g. on env reset)."""
        self.id_to_category = dict(id_to_category)

    def caption(self, semantic: Optional[np.ndarray], agent_pos: np.ndarray, target: Optional[str] = None) -> str:
        x, _, z = float(agent_pos[0]), float(agent_pos[1]), float(agent_pos[2])
        loc = f"agent at ({x:.1f}, {z:.1f})"

        cats = self._top_categories(semantic)
        if cats:
            cats_str = ", ".join(cats)
            base = f"{loc} sees: {cats_str}"
        else:
            base = f"{loc} sees: room interior"

        if target:
            base += f" | searching for {target}"
        return base

    def _top_categories(self, semantic: Optional[np.ndarray]) -> List[str]:
        if semantic is None or semantic.size == 0:
            return []

        # If we have no instance->category map, fall back to treating instance
        # ids modulo a small label set so log lines aren't empty. This is
        # intentionally a hack for the POL slice.
        if not self.id_to_category:
            ids, counts = np.unique(semantic, return_counts=True)
            order = np.argsort(-counts)
            picks: List[str] = []
            for idx in order[: self.top_k * 2]:
                inst = int(ids[idx])
                if inst < 0:
                    continue
                label = _DEFAULT_HM3D_LABELS[inst % len(_DEFAULT_HM3D_LABELS)]
                if label in self.ignore_categories:
                    continue
                if label not in picks:
                    picks.append(label)
                if len(picks) >= self.top_k:
                    break
            return picks

        # With a real id->category map, count pixel coverage per category.
        ids, counts = np.unique(semantic, return_counts=True)
        cat_counts: Dict[str, int] = {}
        for inst, cnt in zip(ids.tolist(), counts.tolist()):
            cat = self.id_to_category.get(int(inst))
            if cat is None:
                continue
            cat = cat.lower()
            if cat in self.ignore_categories:
                continue
            cat_counts[cat] = cat_counts.get(cat, 0) + int(cnt)

        ranked = sorted(cat_counts.items(), key=lambda kv: -kv[1])
        return [c for c, _ in ranked[: self.top_k]]
