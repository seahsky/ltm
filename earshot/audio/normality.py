"""Is that sound normal *here*? The room label seam, the prior, and the verdict.

ADR-0012 keeps ADR-0002's claim — whether a heard sound is anomalous depends on the
room it is heard in — and replaces its grounding. The label now comes from the
**Qwen2-VL-2B captioner the tree already carries**, behind a provider seam. CLIP does
not return: ticket 15 could not load it at all under ticket 13's torch pin, and it is
measured flat on HM3D sim renders across three independent measurements.

Why the claim is worth keeping at all: ADR-0004 relocated the *entire* discrimination
claim onto the room-normal distractor arm. Without scene-conditioned normality the CLAP
gate is decorative, any onset interrupts, and there is no discrimination claim anywhere
in the work.

**What is fixed here is the seam, not the implementation.** ``RoomLabeler`` is a
``Protocol`` (available from ``typing`` on the box's 3.9, so no backport), the smoke
runs ``NullRoomLabeler``, and ``CaptionerRoomLabeler`` ships live but unexercised —
R2's distractor arm is out of scope for this map.

**ADR-0002's $0 accuracy gate carries across the substitution and has NOT been run.**
The captioner has never been measured as a room classifier, so it clears the same bar
CLIP was held to — render frames at known room viewpoints, measure within-versus-between
separation, in the CapRL-gate pattern — before anything depends on the label. Until then
``CaptionerRoomLabeler`` is a seam with an implementation behind it, not a measured
component, and this docstring is where that is written down.

The abstain contract is what makes an unmeasured labeller safe to ship: no class, no
room, or a room the prior knows nothing about all return ``None``, and ``None`` means
the context-free verdict stands. A labeller that guesses would silently convert a
missing measurement into a decision.
"""

from __future__ import annotations

# `Protocol` and `runtime_checkable` are in `typing` from 3.8, so the box's 3.9 needs no
# backport and no `typing_extensions` dependency (ADR-0013 checked this explicitly).
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

__all__ = [
    "ROOM_PRIOR",
    "ROOM_KEYWORDS",
    "Captioner",
    "RoomLabeler",
    "NullRoomLabeler",
    "CaptionerRoomLabeler",
    "resolve_room_from_caption",
    "room_conditioned_anomaly",
    "is_anomalous_here",
    "known_rooms",
]


# Room type -> the set of sound classes that are NORMAL there (ADR-0002). The gate fires
# when the heard class is NOT in this set for the detected room.
#
# HAND-AUTHORED, AND THAT IS THE WEAKEST LINK IN THE ARM. HM3D ships per-object
# semantics with no room-type regions, so there is no room-type ground truth to derive
# this from and this table IS the ground truth for normality. A room absent from it
# carries no normality knowledge, so the verdict abstains rather than guessing — which
# is why the empty frozensets are written out: "bedroom knows of no normal sound" is a
# claim, and it is different from "bedroom is not in the table".
ROOM_PRIOR: Dict[str, FrozenSet[str]] = {
    "bathroom": frozenset({"running_water"}),
    "kitchen": frozenset({"running_water", "appliance_hum"}),
    "bedroom": frozenset(),
    "living_room": frozenset(),
    "dining_room": frozenset(),
    "hallway": frozenset(),
}

# Caption text -> room type. Carried from ``room_resolver.ROOM_KEYWORDS``; the keys are
# the same taxonomy as ``ROOM_PRIOR`` so a caption-resolved room and any future
# classifier's label are interchangeable. Earliest match in the caption wins, because a
# caption that opens "a bedroom with a door to the hallway" is about the bedroom.
ROOM_KEYWORDS: Dict[str, List[str]] = {
    "living_room": [
        "living room", "living-room", "livingroom", "lounge", "sitting room",
        "family room", "great room",
    ],
    "dining_room": ["dining room", "dining area", "dining-room", "diningroom", "breakfast nook"],
    "bedroom": ["bedroom", "bed room", "bed-room"],
    "bathroom": [
        "bathroom", "bath room", "restroom", "rest room", "washroom", "powder room",
        "lavatory",
    ],
    "kitchen": ["kitchen", "kitchenette"],
    "hallway": [
        "hallway", "hall way", "corridor", "foyer", "entryway", "hall ", "passage",
        "stairwell", "staircase",
    ],
}


@runtime_checkable
class Captioner(Protocol):
    """Anything that describes the current view in words.

    The structural interface of ``vlm.py``'s Qwen2-VL-2B connector, declared at the
    consumer rather than imported from it. ``audio/`` is allowed to import ``vlm``
    (ADR-0013's layer graph) and deliberately does not: the concrete captioner drags
    torch and transformers, and this layer's whole Mac surface depends on importing
    neither. The connector satisfies this by shape, and ticket 25 is where the two meet.
    """

    def caption(self, image: Any) -> str:
        ...  # pragma: no cover - protocol


@runtime_checkable
class RoomLabeler(Protocol):
    """The provider seam ADR-0012 fixes: an observation in, a room type or ``None`` out.

    ``None`` is not a failure — it is the abstain that keeps an unmeasured labeller from
    turning into a decision.
    """

    def label(self, observation: Any) -> Optional[str]:
        ...  # pragma: no cover - protocol


class NullRoomLabeler:
    """Always abstains. What the smoke runs.

    Not a stub standing in for missing code: with one sound that is the anomaly by
    construction (§4.3), there is nothing for a room label to condition, and a labeller
    that returned one would put an unmeasured component on the smoke's critical path.
    """

    def label(self, observation: Any) -> Optional[str]:
        del observation
        return None


class CaptionerRoomLabeler:
    """A room type from a caption of the current view (ADR-0012).

    A connector by composition — it holds the injected ``Captioner`` and owns no model
    itself, so this class is Mac-testable against a stub caption source while the thing
    it wraps is not.

    Abstains on an empty caption, on a captioner that raises, and on a caption naming no
    room. The exception path is an abstain rather than a propagate on one ground: the
    room label is an input to an *optional* refinement of the anomaly verdict, and a VLM
    hiccup at step 200 should not end an episode that is otherwise sound. Every other
    failure in this layer raises, so the asymmetry is deliberate and this sentence is
    where it is justified.
    """

    def __init__(self, captioner: Captioner, keywords: Optional[Dict[str, List[str]]] = None):
        self._captioner = captioner
        self._keywords = ROOM_KEYWORDS if keywords is None else keywords
        self.n_abstained = 0
        self.n_labelled = 0
        self.last_caption: Optional[str] = None

    def label(self, observation: Any) -> Optional[str]:
        try:
            caption = self._captioner.caption(observation)
        except Exception:  # noqa: BLE001 - see the class docstring
            self.last_caption = None
            self.n_abstained += 1
            return None
        self.last_caption = caption
        room = resolve_room_from_caption(caption, self._keywords)
        if room is None:
            self.n_abstained += 1
        else:
            self.n_labelled += 1
        return room


def resolve_room_from_caption(
    caption: Optional[str], keywords: Optional[Dict[str, List[str]]] = None
) -> Optional[str]:
    """Earliest-matching room keyword in a caption, or ``None``. Pure.

    Carried from ``room_resolver.resolve_room``. Kept as a free function because it is
    the only part of the labeller a test can pin exactly, and because the caption may
    come from somewhere other than the captioner (a recorded run, the audit record).
    """
    if not caption:
        return None
    lowered = str(caption).lower()
    best_index: Optional[int] = None
    best_room: Optional[str] = None
    for room, words in (ROOM_KEYWORDS if keywords is None else keywords).items():
        for word in words:
            index = lowered.find(word)
            if index >= 0 and (best_index is None or index < best_index):
                best_index, best_room = index, room
    return best_room


def room_conditioned_anomaly(
    sound_class: Optional[str],
    detected_room: Optional[str],
    room_prior: Optional[Dict[str, FrozenSet[str]]] = None,
) -> Optional[bool]:
    """Is ``sound_class`` unexpected in ``detected_room``? Pure. Carried verbatim.

    ``True`` unexpected here (anomalous), ``False`` expected here (normal, do not
    interrupt), ``None`` cannot condition — no class, no room, or a room the prior knows
    nothing about. No CLAP, no simulator, no captioner.
    """
    prior = ROOM_PRIOR if room_prior is None else room_prior
    if not sound_class or detected_room is None:
        return None
    if detected_room not in prior:
        return None
    return bool(sound_class not in prior[detected_room])


def is_anomalous_here(
    fired: bool,
    best_class: Optional[str],
    detected_room: Optional[str],
    room_prior: Optional[Dict[str, FrozenSet[str]]] = None,
) -> bool:
    """Compose the context-free CLAP verdict with the room, when the room is known.

    ``fired`` is ``clap.is_anomaly``'s verdict. When the room can be conditioned the
    room verdict **replaces** it, which is the behaviour the old ``is_anomaly`` had
    inline; when it abstains, the context-free verdict stands unchanged.

    Separated from ``clap.py`` so that each module holds one kind of evidence — cosines
    there, a hand-authored prior here — and so that "the room overrode the audio" is a
    line in the audit record rather than a branch buried in a scoring function.
    """
    verdict = room_conditioned_anomaly(best_class, detected_room, room_prior)
    return bool(fired) if verdict is None else bool(verdict)


def known_rooms() -> Sequence[str]:
    """The taxonomy, sorted. One place for a labeller to check its own output against."""
    return sorted(ROOM_PRIOR)
