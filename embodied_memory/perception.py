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
        # The OpenAI CLIP weights were trained with QuickGELU. open_clip's plain
        # "ViT-B-32" config uses standard GELU, so loading pretrained="openai"
        # onto it silently degrades the embeddings and COMPRESSES the cosine
        # range (open_clip emits a runtime "QuickGELU mismatch" warning). That
        # is why goal-vs-keyframe cosines were pinned ~0.226 with true matches
        # barely reaching 0.249. Use the matching "-quickgelu" model variant so
        # cosines reflect real CLIP similarity. See open_clip create_model docs.
        if pretrained == "openai" and not model_name.endswith("-quickgelu"):
            model_name = f"{model_name}-quickgelu"
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
# CLAP audio encoder (anomaly classifier only)
# ----------------------------------------------------------------------


class CLAPAudioEncoder:
    """``laion/clap-htsat-fused`` audio+text tower → 512-d joint embeddings.

    Used ONLY as the 3-way anomaly CLASSIFIER (cry / alarm / glass) in
    ``audio.classify_anomaly``; retrieval deliberately stays on the proven SBERT
    caption path (the flat CLAP/CLIP cross-modal cosine is what closed the
    embedding lever). Mirrors ``CLIPKeyframeEncoder``: lazy-loaded, device
    auto-pick (cuda → mps → cpu), encode_* L2-normalize.

    The model forward lives in ``_audio_features`` / ``_text_features`` (the
    single heavy seam) so the resample + normalize logic is unit-testable
    without loading the ~600 MB checkpoint.
    """

    TARGET_SR: int = 48000      # CLAP operates at 48 kHz
    EMBED_DIM: int = 512

    def __init__(self, model_name: str = "laion/clap-htsat-fused",
                 device: Optional[str] = None):
        self.model_name = model_name
        self._requested_device = device
        self._model = None
        self._processor = None
        self._device = None

    @property
    def embed_dim(self) -> int:
        return self.EMBED_DIM

    def _pick_device(self) -> str:
        if self._requested_device is not None:
            return self._requested_device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _lazy_load(self):
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import ClapModel, ClapProcessor
        except ImportError as e:
            raise RuntimeError(
                "transformers + torch are required for CLAPAudioEncoder "
                "(pip install transformers torch)."
            ) from e
        device = self._pick_device()
        model = ClapModel.from_pretrained(self.model_name).to(device).eval()
        processor = ClapProcessor.from_pretrained(self.model_name)
        self._model = model
        self._processor = processor
        self._device = device

    @staticmethod
    def _to_mono_48k(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Stereo→mono (channel mean) + resample to 48 kHz, float32 1-D."""
        w = np.asarray(waveform, dtype=np.float32)
        if w.ndim == 2:
            # (C, L) if the channel axis is small, else (L, C).
            w = w.mean(axis=0) if w.shape[0] <= w.shape[1] else w.mean(axis=1)
        w = w.reshape(-1).astype(np.float32)
        if int(sample_rate) != CLAPAudioEncoder.TARGET_SR:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(int(sample_rate), CLAPAudioEncoder.TARGET_SR)
            up = CLAPAudioEncoder.TARGET_SR // g
            down = int(sample_rate) // g
            w = resample_poly(w, up, down).astype(np.float32)
        return w

    def _audio_features(self, mono_48k: np.ndarray) -> np.ndarray:
        self._lazy_load()
        import torch
        inputs = self._processor(audios=[mono_48k], sampling_rate=self.TARGET_SR,
                                 return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self._model.get_audio_features(**inputs)
        return feats.squeeze(0).detach().cpu().float().numpy()

    def _text_features(self, text: str) -> np.ndarray:
        self._lazy_load()
        import torch
        inputs = self._processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self._model.get_text_features(**inputs)
        return feats.squeeze(0).detach().cpu().float().numpy()

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        return v / n if n > 0.0 else v

    def encode_audio(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Encode a waveform → L2-normalized 512-d float32 vector."""
        return self._normalize(self._audio_features(self._to_mono_48k(waveform, sample_rate)))

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a text prompt → L2-normalized 512-d float32 vector."""
        return self._normalize(self._text_features(text))


# ----------------------------------------------------------------------
# BLIP-2 ITM value scorer (VLFM-style semantic-frontier value signal)
# ----------------------------------------------------------------------


class Blip2ITMScorer:
    """``Salesforce/blip2-itm-vit-g`` image-text-MATCHING head → a scalar in [0,1].

    The semantic-frontier value map (``LTM_SEMANTIC_FRONTIER`` lever) needs a
    signal that DISCRIMINATES a goal-facing view from a wall on HM3D sim renders.
    The original CLIP ViT-B/32 dual-encoder cosine is FLAT here (the $0 gate
    measured goal-facing 0.2499 vs away 0.2294 → sep 0.020 < 0.05, the third
    independent CLIP-flatness measurement). BLIP-2's cross-attention ITM head
    co-encodes image+text and is far more discriminative — it is the exact model
    VLFM (ICRA-2024) used to reach HM3D ObjectNav SPL 0.304.

    ``score(rgb, text)`` → P(match) in [0,1] = softmax over the 2-logit ITM head,
    match-class column. Already in [0,1] (the existing clamp in the consumer is a
    defensive no-op). The 2B Qwen-VL captioner is unaffected — this is ONLY the
    frontier value signal.

    Mirrors ``CLAPAudioEncoder``: lazy-loaded (heavy imports inside ``_lazy_load``
    so the module imports without torch/transformers), device auto-pick
    (cuda → mps → cpu, overridable). The single heavy seam is ``_itm_logits``
    (processor + model forward, returns the (2,) logit vector); the pure
    ``score`` does the softmax + clamp and is unit-testable by monkeypatching
    ``_itm_logits`` with NO GPU.
    """

    def __init__(self, model_name: str = "Salesforce/blip2-itm-vit-g",
                 device: Optional[str] = None):
        self.model_name = model_name
        self._requested_device = device
        self._model = None
        self._processor = None
        self._device = None

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
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _lazy_load(self):
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoProcessor, Blip2ForImageTextRetrieval
        except ImportError as e:
            raise RuntimeError(
                "transformers (>=4.42) + torch are required for Blip2ITMScorer "
                "(pip install 'transformers>=4.42' torch). Blip2ForImageTextRetrieval "
                "landed in transformers 4.40+ and the use_image_text_matching_head "
                "API is stable from 4.42."
            ) from e
        import torch
        device = self._pick_device()
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        model = Blip2ForImageTextRetrieval.from_pretrained(
            self.model_name, torch_dtype=dtype).to(device).eval()
        processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = model
        self._processor = processor
        self._device = device

    def _itm_logits(self, rgb: np.ndarray, text: str) -> np.ndarray:
        """The single heavy seam: processor + ITM forward → (2,) logits
        [non-match, match]. Monkeypatched in tests so ``score`` runs GPU-free."""
        self._lazy_load()
        import torch
        from PIL import Image
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        img = Image.fromarray(rgb)
        inputs = self._processor(images=img, text=text, return_tensors="pt")
        # float16 on cuda/mps; the processor emits float32 pixel_values, so cast.
        cast_dtype = torch.float16 if self._device in ("cuda", "mps") else torch.float32
        moved = {}
        for k, v in inputs.items():
            if hasattr(v, "is_floating_point") and v.is_floating_point():
                moved[k] = v.to(self._device, cast_dtype)
            else:
                moved[k] = v.to(self._device)
        with torch.inference_mode():
            out = self._model(**moved, use_image_text_matching_head=True)
        # logits_per_image is (1, 2): col0 = non-match, col1 = match.
        logits = out.logits_per_image.float().reshape(-1).detach().cpu().numpy()
        return np.asarray(logits, dtype=np.float32).reshape(-1)

    @staticmethod
    def _match_prob(logits: np.ndarray) -> float:
        """softmax over the 2 ITM logits → the MATCH-class prob, clamped [0,1].

        The HF model card double-applies softmax (issue #38514); apply it ONCE
        here over the 2-logit vector and index the match column [1]. Defensive:
        a degenerate (1,) logit vector (ITC fallback) returns a sigmoid-like
        [0,1] read of that single score."""
        v = np.asarray(logits, dtype=np.float64).reshape(-1)
        if v.size == 0:
            return 0.0
        if v.size == 1:
            # ITC-style single cosine/logit: squash to [0,1] (sigmoid).
            p = 1.0 / (1.0 + np.exp(-float(v[0])))
            return float(max(0.0, min(1.0, p)))
        v = v - v.max()              # numerical-stable softmax
        e = np.exp(v)
        probs = e / e.sum()
        p = float(probs[1])          # match-class column
        return float(max(0.0, min(1.0, p)))

    def score(self, rgb: np.ndarray, text: str) -> float:
        """Image-text MATCH probability in [0,1] for ``rgb`` vs ``text``."""
        return self._match_prob(self._itm_logits(rgb, text))


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
