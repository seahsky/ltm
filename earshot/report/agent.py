"""``AgentReport`` — the agent's testimony, and nothing else it knows.

Task spec §5.1, exactly nine fields, **identical in both localization arms**. That
identity is the whole point: it answers the "the sound is just a stopwatch, the
coordinate is handed to the agent" objection by making realizability checkable from
the schema instead of arguable from the source.

``source_xyz`` is gone. The oracle arm's privilege shows in its trajectory and in
``EpisodeAudit``, never in its testimony.

**The boundary is the type, not the controller** (ADR-0013). "The controller cannot see
ground truth" was not available as the rule — the oracle arm's controller holds
``source_xyz`` as its waypoint by construction — so the rule became: nothing privileged
can appear *here*, whatever the controller holds. ``tests/mac/test_report_boundary.py``
holds both halves: the field sets are disjoint, and this module imports nothing that
could supply a privileged value.

That import restriction is why this file reaches ``types`` and nothing else. Adding
``from ..sim import world`` to grab a geodesic would be a one-line convenience and a
red build.

Python 3.9: annotations are postponed and ``typing.get_type_hints()`` raises on a
``from __future__`` PEP 604 union, so every consumer reads ``__dataclass_fields__``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from ..types import Pose

__all__ = ["AgentReport", "SCHEMA_FIELDS"]

# Task spec §5.1, in order. Named here so ticket 26's smoke criterion 6 ("a report was
# emitted with §5.1's schema fully populated") has something to check *against* rather
# than reading the field list off the dataclass it is checking — which would assert only
# that the dataclass equals itself. `tests/mac/test_report_boundary.py` carries its own
# independent transcription of the same nine names for the same reason.
SCHEMA_FIELDS = (
    "primary_completed",
    "heard_at_step",
    "room",
    "anomaly_class",
    "stopped_at_pose",
    "visual_confirm_object",
    "investigate_aborted",
    "resumed",
    "n_benign_ignored",
)


@dataclass(frozen=True)
class AgentReport:
    """What the agent can say for itself. Structured, with no generated text.

    ADR-0008 dropped the LLM and nothing in §5.1 was ever prose, so this is a record
    rather than a narrative.

    Every field is agent-estimable:

    - ``heard_at_step`` is the onset the agent's own threshold fired at, not ``t_anom``.
    - ``room`` comes from the captioner (ADR-0012), not from the scene's annotations.
    - ``stopped_at_pose`` is the agent's own pose, which §5.1 substitutes for the
      source coordinate. Where it *actually* stopped relative to the source is
      ``EpisodeAudit.dist_at_stop``, computed by the analyst.
    - ``visual_confirm_object`` is ``None`` when the detector never confirmed, which
      §5.1 permits ("or absent") — so a null here is data, not an unpopulated field.
    """

    primary_completed: bool = False
    heard_at_step: Optional[int] = None
    room: Optional[str] = None
    anomaly_class: Optional[str] = None
    stopped_at_pose: Optional[Pose] = None
    visual_confirm_object: Optional[str] = None
    investigate_aborted: bool = False
    resumed: bool = False
    n_benign_ignored: int = 0

    def as_dict(self) -> Dict[str, Any]:
        """The serialised form — exactly §5.1's nine keys, always all nine.

        Written out rather than ``dataclasses.asdict`` so the key set is visible at the
        one place a reviewer reads, and so ``stopped_at_pose`` serialises through
        ``Pose.as_dict`` rather than through a generic recursion that would happen to
        agree today.
        """
        return {
            "primary_completed": bool(self.primary_completed),
            "heard_at_step": self.heard_at_step,
            "room": self.room,
            "anomaly_class": self.anomaly_class,
            "stopped_at_pose": (
                self.stopped_at_pose.as_dict() if self.stopped_at_pose is not None else None
            ),
            "visual_confirm_object": self.visual_confirm_object,
            "investigate_aborted": bool(self.investigate_aborted),
            "resumed": bool(self.resumed),
            "n_benign_ignored": int(self.n_benign_ignored),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentReport":
        """The inverse. Raises on an unknown key rather than dropping it.

        A tolerant reader is how a privileged field gets into a run directory and out
        of it again unnoticed — the artefact would carry ``source_xyz``, the type would
        not, and the disjointness test would still be green. So the strictness is part
        of the boundary rather than input hygiene.
        """
        unknown = sorted(set(data) - set(SCHEMA_FIELDS))
        if unknown:
            raise ValueError(
                "not §5.1's schema: unexpected key(s) {} in an agent report. The "
                "testimony is exactly {}".format(unknown, list(SCHEMA_FIELDS))
            )
        pose = data.get("stopped_at_pose")
        return cls(
            primary_completed=bool(data.get("primary_completed", False)),
            heard_at_step=data.get("heard_at_step"),
            room=data.get("room"),
            anomaly_class=data.get("anomaly_class"),
            stopped_at_pose=Pose.from_dict(pose) if pose is not None else None,
            visual_confirm_object=data.get("visual_confirm_object"),
            investigate_aborted=bool(data.get("investigate_aborted", False)),
            resumed=bool(data.get("resumed", False)),
            n_benign_ignored=int(data.get("n_benign_ignored", 0)),
        )


def missing_schema_keys(data: Mapping[str, Any]) -> tuple:
    """Which of §5.1's nine keys a serialised report is missing. Ticket 26's criterion 6.

    Deliberately about **keys**, not values. ``visual_confirm_object`` is legitimately
    absent when the detector never confirmed and ``heard_at_step`` is ``None`` on an
    episode that never heard the anomaly, so "fully populated" cannot mean "no nulls"
    without failing episodes that behaved correctly. What criterion 6 can check is that
    the schema itself arrived whole.
    """
    return tuple(name for name in SCHEMA_FIELDS if name not in data)
