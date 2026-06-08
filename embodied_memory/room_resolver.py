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
