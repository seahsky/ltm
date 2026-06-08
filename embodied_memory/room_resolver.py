"""
Caption -> room-type resolver: the grounding primitive of the coarse-affordance
head (step 4).

The coarse LTM layer stores a POSITION-FREE ``category -> preferred_room_type``
prior (e.g. ``chair -> living_room``) that transfers across environments. To
ground it in a NEW scene the agent must tag its CURRENT observations with a
room-type. HM3D's semantic sensor returns zeros, so the only live room signal is
the rich Qwen-VL caption. This module maps a caption to one of a small room-type
taxonomy.

Resolution is EARLIEST-MENTION (not first-keyword in some list order): the
room-type whose keyword appears earliest in the caption wins — that is the
dominant subject / the room the agent is currently in. This fixes the salience
bug a first-match resolver has: "a cozy living room ... and a dining area" must
resolve to ``living_room`` (mentioned first), and "a hallway leading to a
bedroom" to ``hallway`` (the agent's location). Captions with no room word
ABSTAIN (return ``None``) — most ObjectNav frames are object close-ups, not room
shots, and a wrong room label is worse than no label.

Object words (bed, sofa, toilet) are deliberately NOT room indicators — they are
the search TARGETS, and a "bed" can appear in a caption that does not name the
bedroom; only an explicit ROOM word grounds a room-type.

Dependency-free (pure str ops) so it imports under bare python3 without the
``embodied_memory`` package's heavy deps.
"""

from __future__ import annotations

from typing import Dict, List, Optional


# Canonical room-types -> the room WORDS that indicate them (NOT object words).
# Order within a list does not matter; resolution is by earliest position in the
# caption across ALL keywords of ALL rooms.
ROOM_KEYWORDS: Dict[str, List[str]] = {
    "living_room": ["living room", "living-room", "livingroom", "lounge",
                    "sitting room", "family room", "great room"],
    "dining_room": ["dining room", "dining area", "dining-room", "diningroom",
                    "breakfast nook"],
    "bedroom": ["bedroom", "bed room", "bed-room"],
    "bathroom": ["bathroom", "bath room", "restroom", "rest room", "washroom",
                 "powder room", "lavatory"],
    "kitchen": ["kitchen", "kitchenette"],
    "hallway": ["hallway", "hall way", "corridor", "foyer", "entryway", "hall ",
                "passage", "stairwell", "staircase"],
}


# Static, hand-authored category -> preferred room-type prior — the cold-start
# coarse-affordance knowledge that transfers across scenes (the proposal's
# "general guidance in new environments"). Covers the HM3D-ObjectNav categories.
# This is the v1 prior; a learned per-(category, room) table (build_affordance_table
# fed a real room_resolver) can override it later (deferred Stage 5).
CATEGORY_ROOM_PRIOR: Dict[str, str] = {
    "chair": "living_room",
    "sofa": "living_room",
    "couch": "living_room",
    "tv_monitor": "living_room",
    "plant": "living_room",
    "bed": "bedroom",
    "toilet": "bathroom",
}


def preferred_room(category: Optional[str]) -> Optional[str]:
    """Return the affordant room-type for a goal ``category`` (static prior), or
    ``None`` if the category has no known room affordance."""
    if not category:
        return None
    return CATEGORY_ROOM_PRIOR.get(str(category).strip().lower())


def resolve_room(caption: Optional[str]) -> Optional[str]:
    """Return the room-type whose keyword appears EARLIEST in ``caption``, or
    ``None`` if the caption names no room.

    Earliest-mention picks the dominant / agent-current room (the first one
    described), so transit captions ("hallway leading to a bedroom") resolve to
    where the agent is.
    """
    if not caption:
        return None
    low = caption.lower()
    best_room: Optional[str] = None
    best_idx: Optional[int] = None
    for room, keywords in ROOM_KEYWORDS.items():
        for kw in keywords:
            idx = low.find(kw)
            if idx >= 0 and (best_idx is None or idx < best_idx):
                best_idx, best_room = idx, room
    return best_room


# ----------------------------------------------------------------------
# CLIP zero-shot ROOM classifier — the DENSE room-perception signal
# ----------------------------------------------------------------------
#
# coarse-1/coarse-2 (2026-06-09) showed the binding constraint for the coarse
# head is current-scene ROOM PERCEPTION, not the affordance prior: the Qwen-VL
# caption names the goal's affordant room too rarely (``resolve_room`` abstains on
# most ObjectNav close-up frames). HM3D ships per-OBJECT semantics with NO
# room-type regions, so GT-region grounding is ruled out. CLIP is already loaded
# in the loop (ViT-B/32), so a zero-shot room classifier on the keyframe RGB —
# cosine of the CLIP IMAGE embedding vs CLIP-text("a photo of a {room}") — is a
# no-download dense room signal. Scene-level room classification is a DIFFERENT,
# more tractable task than the per-instance discrimination CLIP failed at, so its
# earlier "flat ~0.226 cosine" failure (an instance problem) does not apply here.
#
# Keys MUST match ROOM_KEYWORDS (and thus CATEGORY_ROOM_PRIOR's values) so a
# CLIP-classified room and a caption-resolved room are interchangeable when
# compared against a goal category's preferred room.
ROOM_TEXT_PROMPTS: Dict[str, str] = {
    "living_room": "a photo of a living room",
    "dining_room": "a photo of a dining room",
    "bedroom": "a photo of a bedroom",
    "bathroom": "a photo of a bathroom",
    "kitchen": "a photo of a kitchen",
    "hallway": "a photo of a hallway, corridor, or entryway",
}


def build_room_text_embeddings(text_encode_fn) -> "Dict[str, object]":
    """Encode each room prompt ONCE into the joint CLIP space and L2-normalize.

    ``text_encode_fn`` is a CLIP text encoder (e.g. ``clip_encoder.encode_text``)
    mapping a prompt string -> a vector. Returns ``{room_type: unit np.float32}``.
    Call once per run and reuse — the prompts are fixed, so this is a one-time cost.
    """
    import numpy as np

    out: "Dict[str, object]" = {}
    for room, prompt in ROOM_TEXT_PROMPTS.items():
        v = np.asarray(text_encode_fn(prompt), dtype=np.float32)
        n = float(np.linalg.norm(v))
        out[room] = (v / n).astype(np.float32) if n > 1e-8 else v.astype(np.float32)
    return out


def _room_cosines(image_embedding, room_text_embeddings):
    """Sorted ``[(cosine, room), ...]`` (desc) of a CLIP IMAGE embedding against each
    room's CLIP-text embedding, or ``[]`` on degenerate/non-finite input. Inputs need
    not be pre-normalized (normalized defensively; a non-finite image -> ``[]`` so the
    abstain contract holds)."""
    import numpy as np

    if image_embedding is None or not room_text_embeddings:
        return []
    img = np.asarray(image_embedding, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(img)):       # NaN/inf must NOT slip to a garbage argmax
        return []
    n = float(np.linalg.norm(img))
    if n <= 1e-8:
        return []
    img = img / n

    sims = []
    for room, temb in room_text_embeddings.items():
        t = np.asarray(temb, dtype=np.float32).reshape(-1)
        tn = float(np.linalg.norm(t))
        if tn <= 1e-8 or t.shape != img.shape or not np.all(np.isfinite(t)):
            continue
        sims.append((float(np.dot(img, t / tn)), room))
    sims.sort(key=lambda s: -s[0])
    return sims


def room_clip_top_cos(image_embedding, room_text_embeddings):
    """RAW argmax ``(top_cosine, top_room)`` over the room prompts WITHOUT the abstain
    gate — the calibration / normalization-skew probe (SA-3). Returns
    ``(float('nan'), None)`` on degenerate input. With unit-norm inputs ``top_cosine``
    is bounded by 1.0; a value above 1.0 signals a double-normalization or dtype bug."""
    sims = _room_cosines(image_embedding, room_text_embeddings)
    if not sims:
        return float("nan"), None
    return sims[0]


def classify_room_clip(
    image_embedding,
    room_text_embeddings,
    *,
    min_cos: float = 0.25,
    margin: float = 0.02,
) -> Optional[str]:
    """Zero-shot room-type for a CLIP IMAGE embedding, or ``None`` (abstain).

    Computes cosine of ``image_embedding`` against each room's CLIP-text embedding
    (from :func:`build_room_text_embeddings`) and returns the argmax room — but
    only when it is CONFIDENT: the top cosine must clear ``min_cos`` AND beat the
    runner-up by ``margin``. Otherwise it abstains (``None``), exactly like
    :func:`resolve_room` — a wrong room tag is worse than none for grounding the
    affordance prior, and abstention preserves the coarse head's conservatism (the
    bias that beat every trained importance head in the instance-ambiguous space).

    The defaults (``min_cos=0.25``, ``margin=0.02``) are calibrated for the REAL CLIP
    ViT-B/32 image-text cosine scale (~0.18-0.30), NOT a synthetic 0..1 world — at
    real scale a lower floor makes the gate a no-op (the head would fire on nearly
    every frame). Tune per-scene via ``diagnose_room_clip_cosines.py``. Pure numpy;
    a non-finite image embedding abstains (the safety contract).
    """
    sims = _room_cosines(image_embedding, room_text_embeddings)
    if not sims:
        return None
    top_cos, top_room = sims[0]
    second = sims[1][0] if len(sims) > 1 else -1.0
    if top_cos < float(min_cos) or (top_cos - second) < float(margin):
        return None
    return top_room
