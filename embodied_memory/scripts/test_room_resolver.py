"""
Sanity tests for ``room_resolver.resolve_room`` — the grounding primitive of the
coarse-affordance head (step 4).

The coarse layer stores a position-free ``category -> preferred_room_type`` prior;
to ground it in a NEW scene the agent must label its CURRENT observations with a
room-type. HM3D's semantic sensor returns zeros, so the only live room signal is
the rich Qwen-VL caption ("a cozy living room ...", "a spacious bedroom ..."). This
resolver maps a caption to one of a small room-type taxonomy by EARLIEST-mention
(the dominant / agent-current room), abstaining (None) when no room word appears.

Earliest-mention (not first-keyword-match) fixes the salience bug the design audit
flagged: "a cozy living room ... and a dining area" must resolve to living_room
(mentioned first), and "a hallway leading to a bedroom" to hallway (the agent's
location), not the trailing room.

Invoke with::

    python embodied_memory/scripts/test_room_resolver.py
"""

from __future__ import annotations

import os
import sys

# room_resolver is dependency-free; import it directly (NOT via the embodied_memory
# package, whose __init__ pulls faiss) so this runs under bare python3.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../embodied_memory

from room_resolver import resolve_room, preferred_room  # noqa: E402


def case_basic_room_types():
    assert resolve_room("A cozy living room with hardwood floors and a window.") == "living_room"
    assert resolve_room("A spacious bedroom with a bed, a ceiling fan, and a window.") == "bedroom"
    assert resolve_room("A bathroom with a white door, tiled walls, and a wooden floor.") == "bathroom"
    assert resolve_room("A kitchen scene with a stove and oven.") == "kitchen"
    assert resolve_room("A cozy dining room with a wooden table and chairs.") == "dining_room"
    assert resolve_room("A hallway with a coat rack and a door.") == "hallway"
    print("  case basic_room_types: OK")


def case_earliest_mention_wins_living_before_dining():
    # the SALIENCE bug: must be living_room (first mentioned), not dining_room
    c = "The scene depicts a cozy living room with a fireplace, a television, and a dining area."
    assert resolve_room(c) == "living_room", resolve_room(c)
    print("  case earliest_mention_wins_living_before_dining: OK")


def case_transit_frame_takes_agent_location():
    # "hallway leading to a bedroom": agent is IN the hallway (mentioned first)
    c = "The scene depicts a hallway leading to a bedroom with a bed and a window."
    assert resolve_room(c) == "hallway", resolve_room(c)
    print("  case transit_frame_takes_agent_location: OK")


def case_room_mentioned_late_still_resolves():
    c = "A wooden bookshelf filled with various books and decorative items stands in a cozy living room."
    assert resolve_room(c) == "living_room", resolve_room(c)
    print("  case room_mentioned_late_still_resolves: OK")


def case_abstains_when_no_room_word():
    # generic "a room" / object-only captions -> None (the abstain contract)
    assert resolve_room("A cluttered room with a wooden bookshelf, a pink chair, and a table.") is None
    assert resolve_room("A wooden chair tucked under a rustic table.") is None
    assert resolve_room("The image shows a closet filled with various clothing items.") is None
    assert resolve_room("") is None
    print("  case abstains_when_no_room_word: OK")


def case_synonyms_map_to_canonical():
    assert resolve_room("A small restroom with a white sink.") == "bathroom"
    assert resolve_room("A cozy lounge with a leather couch.") == "living_room"
    assert resolve_room("A long corridor with white walls.") == "hallway"
    print("  case synonyms_map_to_canonical: OK")


def case_object_words_are_not_room_words():
    # 'bed'/'sofa'/'toilet' are search TARGETS, not room indicators -> must not
    # be mistaken for bedroom/living_room/bathroom unless a ROOM word is present.
    assert resolve_room("A neatly made bed beneath a window.") is None
    assert resolve_room("A brown leather couch facing a fireplace.") is None
    assert resolve_room("A white toilet beside a pedestal sink.") is None
    print("  case object_words_are_not_room_words: OK")


def case_preferred_room_static_prior():
    assert preferred_room("chair") == "living_room"
    assert preferred_room("bed") == "bedroom"
    assert preferred_room("toilet") == "bathroom"
    assert preferred_room("sofa") == "living_room"
    assert preferred_room("CHAIR") == "living_room"      # case-insensitive
    assert preferred_room("unknown_category") is None     # abstain on unknown
    assert preferred_room(None) is None
    print("  case preferred_room_static_prior: OK")


def main() -> int:
    print("room_resolver sanity tests")
    case_basic_room_types()
    case_preferred_room_static_prior()
    case_earliest_mention_wins_living_before_dining()
    case_transit_frame_takes_agent_location()
    case_room_mentioned_late_still_resolves()
    case_abstains_when_no_room_word()
    case_synonyms_map_to_canonical()
    case_object_words_are_not_room_words()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
