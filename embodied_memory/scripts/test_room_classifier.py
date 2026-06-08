"""
Sanity tests for the CLIP zero-shot ROOM classifier — the dense room-perception
signal for the coarse-affordance head (step 4).

coarse-1/coarse-2 proved the binding constraint is current-scene ROOM PERCEPTION,
not the affordance prior: the Qwen-VL caption names the goal's affordant room too
rarely (``resolve_room`` abstains on most ObjectNav frames, so the coarse head
almost never grounds). HM3D ships per-OBJECT semantics with no room-type regions,
so GT-region grounding is ruled out. CLIP (already loaded, ViT-B/32) gives a
no-download dense room signal: cosine of the keyframe's CLIP IMAGE embedding vs
CLIP-text("a photo of a {living room, bedroom, ...}"). Scene-level room
classification is a DIFFERENT, more tractable task than the instance discrimination
CLIP failed at.

``classify_room_clip`` keeps ``resolve_room``'s conservative abstain contract: it
returns None unless the top room cosine clears ``min_cos`` AND beats the runner-up
by ``margin`` — a wrong room tag is worse than none for grounding the prior.

Pure numpy + synthetic embeddings (no CLIP weights), so this runs fast and offline.

Invoke with::

    python embodied_memory/scripts/test_room_classifier.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

# room_resolver is import-light (numpy pulled lazily inside the CLIP fns); import it
# directly, NOT via the embodied_memory package whose __init__ pulls faiss.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../embodied_memory

from room_resolver import (  # noqa: E402
    ROOM_KEYWORDS,
    ROOM_TEXT_PROMPTS,
    build_room_text_embeddings,
    classify_room_clip,
    room_clip_top_cos,
)


# ----------------------------------------------------------------------
# helpers: synthetic CLIP space — one orthonormal basis vector per room, so a
# "perfect" image embedding for a room == that room's text embedding (cosine 1).
# ----------------------------------------------------------------------

_ROOMS = list(ROOM_TEXT_PROMPTS.keys())
_DIM = 16


def _basis(i: int) -> np.ndarray:
    v = np.zeros(_DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def _room_text_embeddings() -> dict:
    return {room: _basis(i) for i, room in enumerate(_ROOMS)}


def _mix(i: int, j: int, a: float) -> np.ndarray:
    """Unit vector that is `a` toward basis i and (1-a) toward basis j."""
    v = a * _basis(i) + (1.0 - a) * _basis(j)
    return (v / np.linalg.norm(v)).astype(np.float32)


# ----------------------------------------------------------------------
# taxonomy alignment — CLIP rooms MUST be the SAME 6 keys as the caption resolver
# (so a CLIP-tagged room and a caption-tagged room compare to preferred_room alike)
# ----------------------------------------------------------------------


def case_prompts_cover_exactly_the_taxonomy():
    assert set(ROOM_TEXT_PROMPTS.keys()) == set(ROOM_KEYWORDS.keys()), (
        set(ROOM_TEXT_PROMPTS.keys()) ^ set(ROOM_KEYWORDS.keys()))
    # each prompt is a non-empty natural-language phrase naming the room
    for room, prompt in ROOM_TEXT_PROMPTS.items():
        assert isinstance(prompt, str) and len(prompt) > 3, (room, prompt)
    print("  case prompts_cover_exactly_the_taxonomy: OK")


# ----------------------------------------------------------------------
# build_room_text_embeddings
# ----------------------------------------------------------------------


def case_build_encodes_and_normalizes_all_rooms():
    # a fake CLIP text encoder returns an UN-normalized basis vector per room
    def fake_encode(prompt: str) -> np.ndarray:
        room = next(r for r, p in ROOM_TEXT_PROMPTS.items() if p == prompt)
        return 7.0 * _basis(_ROOMS.index(room))  # deliberately not unit-norm

    embs = build_room_text_embeddings(fake_encode)
    assert set(embs.keys()) == set(_ROOMS)
    for room, v in embs.items():
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5, (room, np.linalg.norm(v))
        assert v.dtype == np.float32
    print("  case build_encodes_and_normalizes_all_rooms: OK")


# ----------------------------------------------------------------------
# classify_room_clip — argmax-with-abstain
# ----------------------------------------------------------------------


def case_classifies_clear_room_to_argmax():
    rte = _room_text_embeddings()
    for i, room in enumerate(_ROOMS):
        assert classify_room_clip(_basis(i), rte) == room, room
    print("  case classifies_clear_room_to_argmax: OK")


def case_returns_a_canonical_taxonomy_key():
    rte = _room_text_embeddings()
    out = classify_room_clip(_basis(0), rte)
    assert out in ROOM_KEYWORDS, out  # interchangeable with resolve_room output
    print("  case returns_a_canonical_taxonomy_key: OK")


def case_abstains_below_min_cos():
    rte = _room_text_embeddings()
    # an embedding orthogonal to every room basis -> all cosines 0 < min_cos
    orth = np.zeros(_DIM, dtype=np.float32)
    orth[_DIM - 1] = 1.0  # last dim unused by any room basis
    assert classify_room_clip(orth, rte, min_cos=0.20) is None
    print("  case abstains_below_min_cos: OK")


def case_abstains_when_ambiguous_margin():
    rte = _room_text_embeddings()
    # 50/50 between living_room(0) and bedroom(2): cos to each ~0.707, tie within margin
    amb = _mix(0, 2, 0.5)
    assert classify_room_clip(amb, rte, min_cos=0.10, margin=0.05) is None
    # but a decisive lean past the margin classifies
    lean = _mix(0, 2, 0.9)
    assert classify_room_clip(lean, rte, min_cos=0.10, margin=0.05) == "living_room"
    print("  case abstains_when_ambiguous_margin: OK")


def case_normalizes_unnormalized_image_embedding():
    rte = _room_text_embeddings()
    big = 42.0 * _basis(3)  # bathroom direction, large magnitude
    assert classify_room_clip(big, rte) == _ROOMS[3]
    print("  case normalizes_unnormalized_image_embedding: OK")


def case_abstains_on_degenerate_inputs():
    rte = _room_text_embeddings()
    assert classify_room_clip(None, rte) is None
    assert classify_room_clip(_basis(0), {}) is None
    assert classify_room_clip(_basis(0), None) is None
    assert classify_room_clip(np.zeros(_DIM, dtype=np.float32), rte) is None  # zero vec
    print("  case abstains_on_degenerate_inputs: OK")


def case_min_cos_and_margin_are_tunable():
    rte = _room_text_embeddings()
    weak = _mix(0, 5, 0.72)  # cos to living_room ~0.72/sqrt(...)~0.79
    # default-ish strict threshold accepts it; a very high min_cos rejects it
    assert classify_room_clip(weak, rte, min_cos=0.10, margin=0.01) == "living_room"
    assert classify_room_clip(weak, rte, min_cos=0.99, margin=0.01) is None
    print("  case min_cos_and_margin_are_tunable: OK")


def case_default_thresholds_are_conservative():
    # MF-1 (post-verification): in-code defaults must be calibrated for the REAL
    # CLIP ViT-B/32 image-text cosine scale (~0.18-0.30), NOT the synthetic 0..1
    # world. A frame whose top room cosine is 0.22 (a real "room-ish" frame) must
    # ABSTAIN under the defaults (min_cos>=0.25), but classify at an explicit 0.20.
    rte = _room_text_embeddings()
    # image = 0.22 toward living_room(0) + the rest along an UNUSED dim (no room) ->
    # cos to living_room = 0.22, cos to every other room = 0.
    img = np.zeros(_DIM, dtype=np.float32)
    img[0] = 0.22
    img[_DIM - 1] = float(np.sqrt(1.0 - 0.22 ** 2))  # last dim unused by any room
    assert classify_room_clip(img, rte) is None, "defaults must abstain at cos 0.22"
    assert classify_room_clip(img, rte, min_cos=0.20, margin=0.02) == "living_room"
    print("  case default_thresholds_are_conservative: OK")


def case_rejects_nonfinite_image_embedding():
    # the abstain contract is the head's whole safety story — a NaN/inf image
    # embedding must NOT slip through to a garbage argmax label.
    rte = _room_text_embeddings()
    nan_img = _basis(0).copy(); nan_img[1] = np.nan
    inf_img = _basis(0).copy(); inf_img[1] = np.inf
    assert classify_room_clip(nan_img, rte) is None
    assert classify_room_clip(inf_img, rte) is None
    print("  case rejects_nonfinite_image_embedding: OK")


# ----------------------------------------------------------------------
# room_clip_top_cos — RAW argmax cosine WITHOUT the abstain gate (SA-3 calibration
# + normalization-skew guard: a value > 1.0 means a double-norm/dtype bug).
# ----------------------------------------------------------------------


def case_top_cos_returns_raw_argmax():
    rte = _room_text_embeddings()
    for i, room in enumerate(_ROOMS):
        cos, top = room_clip_top_cos(_basis(i), rte)
        assert top == room, (room, top)
        assert abs(cos - 1.0) < 1e-5, cos
    # a sub-threshold frame still REPORTS its raw cosine (no gate)
    img = np.zeros(_DIM, dtype=np.float32)
    img[0] = 0.22
    img[_DIM - 1] = float(np.sqrt(1.0 - 0.22 ** 2))
    cos, top = room_clip_top_cos(img, rte)
    assert top == "living_room" and abs(cos - 0.22) < 1e-4, (cos, top)
    print("  case top_cos_returns_raw_argmax: OK")


def case_top_cos_degenerate_inputs():
    rte = _room_text_embeddings()
    for bad in (None, np.zeros(_DIM, dtype=np.float32)):
        cos, top = room_clip_top_cos(bad, rte)
        assert top is None and cos != cos, (cos, top)  # NaN cos, None room
    cos, top = room_clip_top_cos(_basis(0), {})
    assert top is None and cos != cos
    print("  case top_cos_degenerate_inputs: OK")


def case_top_cos_never_exceeds_one():
    # normalization-skew guard: with unit-norm inputs the cosine is bounded by 1.
    rte = _room_text_embeddings()
    cos, _ = room_clip_top_cos(42.0 * _basis(2), rte)  # large magnitude image
    assert cos <= 1.0 + 1e-6, cos
    print("  case top_cos_never_exceeds_one: OK")


def main() -> int:
    print("room_classifier (CLIP zero-shot) sanity tests")
    case_prompts_cover_exactly_the_taxonomy()
    case_build_encodes_and_normalizes_all_rooms()
    case_classifies_clear_room_to_argmax()
    case_returns_a_canonical_taxonomy_key()
    case_abstains_below_min_cos()
    case_abstains_when_ambiguous_margin()
    case_normalizes_unnormalized_image_embedding()
    case_abstains_on_degenerate_inputs()
    case_min_cos_and_margin_are_tunable()
    case_default_thresholds_are_conservative()
    case_rejects_nonfinite_image_embedding()
    case_top_cos_returns_raw_argmax()
    case_top_cos_degenerate_inputs()
    case_top_cos_never_exceeds_one()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
