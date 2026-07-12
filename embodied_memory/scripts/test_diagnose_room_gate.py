"""
TDD for the G0.1 scene-conditioning kill-switch: does the CLIP room classifier
reliably tell the two relevant rooms apart? (Gate for keyword #16 — a room-
conditioned anomaly gate is only trustworthy if the room verdict is reliable.)

Ground truth comes free from the object-category -> room map (CATEGORY_ROOM_PRIOR):
a frame at a toilet view_point is bathroom, at a bed view_point bedroom, etc. The
render/CLIP path is RACE integration; these two PURE functions carry the logic:

  - room_pair_accuracy(pairs, rooms): accuracy + abstain-rate + confusion over the
    frames whose TRUE room is in `rooms` (6-way pred; abstain counts as wrong).
  - room_gate_verdict(accuracy): GO / BORDERLINE / STOP.

Run: KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
       /opt/anaconda3/envs/ltm-embodied/bin/python \
       embodied_memory/scripts/test_diagnose_room_gate.py
"""
from __future__ import annotations

import sys

from embodied_memory.scripts.diagnose_room_clip_cosines import (
    build_room_pairs,
    gate_result,
    room_gate_verdict,
    room_pair_accuracy,
)


def case_pairwise_accuracy_counts_and_confusion():
    # 6 frames: 5 from the two rooms of interest, 1 from an out-of-scope room
    # (kitchen) that must be ignored; one in-scope abstain counts as wrong.
    pairs = [
        ("bathroom", "bathroom"),   # correct
        ("bathroom", "bedroom"),    # wrong — confused with bedroom
        ("bathroom", None),         # abstain -> wrong
        ("bedroom", "bedroom"),     # correct
        ("bedroom", "bedroom"),     # correct
        ("kitchen", "kitchen"),     # true not in {bathroom,bedroom} -> ignored
    ]
    r = room_pair_accuracy(pairs, {"bathroom", "bedroom"})
    assert r["n"] == 5, r                       # kitchen excluded
    assert r["n_correct"] == 3, r               # 3 of 5 correct
    assert r["n_abstain"] == 1, r
    assert abs(r["accuracy"] - 0.6) < 1e-9, r   # 3/5, abstain counted wrong
    assert r["confusion"][("bathroom", "bedroom")] == 1, r["confusion"]
    print("  case_pairwise_accuracy_counts_and_confusion: OK")


def case_gate_verdict_thresholds():
    # GREEN rule: the room classifier must separate the two rooms at >= ~0.75.
    assert room_gate_verdict(0.82) == "GO", room_gate_verdict(0.82)
    assert room_gate_verdict(0.75) == "GO", room_gate_verdict(0.75)   # boundary inclusive
    assert room_gate_verdict(0.68) == "BORDERLINE", room_gate_verdict(0.68)
    assert room_gate_verdict(0.60) == "BORDERLINE", room_gate_verdict(0.60)  # boundary inclusive
    assert room_gate_verdict(0.55) == "STOP", room_gate_verdict(0.55)
    print("  case_gate_verdict_thresholds: OK")


def case_build_room_pairs_maps_category_to_true_room():
    # GT room comes from CATEGORY_ROOM_PRIOR: toilet->bathroom, bed->bedroom,
    # chair->living_room; a category with no prior is dropped.
    pairs = build_room_pairs([
        ("toilet", "bathroom"), ("bed", "bedroom"), ("bed", "kitchen"),
        ("unknown_obj", "bathroom"),   # no room prior -> dropped
    ])
    assert ("bathroom", "bathroom") in pairs
    assert ("bedroom", "bedroom") in pairs
    assert ("bedroom", "kitchen") in pairs
    assert len(pairs) == 3, pairs
    print("  case_build_room_pairs_maps_category_to_true_room: OK")


def case_build_room_pairs_room_filter():
    pairs = build_room_pairs([("toilet", "bathroom"), ("bed", "bedroom")],
                             rooms={"bathroom"})
    assert pairs == [("bathroom", "bathroom")], pairs
    print("  case_build_room_pairs_room_filter: OK")


def case_gate_result_end_to_end():
    # perfect separation -> GO, accuracy 1.0
    v, acc = gate_result([("toilet", "bathroom"), ("toilet", "bathroom"),
                          ("bed", "bedroom"), ("bed", "bedroom")])
    assert v == "GO" and abs(acc["accuracy"] - 1.0) < 1e-9, (v, acc)
    # one confusion + one abstain -> below the GO bar
    v2, acc2 = gate_result([("toilet", "bedroom"), ("toilet", None),
                            ("bed", "bedroom"), ("bed", "bedroom")])
    assert acc2["accuracy"] == 0.5, acc2
    assert v2 in ("BORDERLINE", "STOP"), v2
    print("  case_gate_result_end_to_end: OK")


def main() -> int:
    print("running G0.1 room-gate tests…")
    case_pairwise_accuracy_counts_and_confusion()
    case_gate_verdict_thresholds()
    case_build_room_pairs_maps_category_to_true_room()
    case_build_room_pairs_room_filter()
    case_gate_result_end_to_end()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
