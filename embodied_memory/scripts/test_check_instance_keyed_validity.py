"""
TDD for check_instance_keyed_validity — the metric-validity gate (#1 Part B,
rank 5) that decides whether an instance-keyed multi-instance dataset actually
FORCES disambiguation BEFORE any paid matrix.

A multi-instance episode only tests instance discrimination if, from some warm
start, a DISTRACTOR instance is geodesically nearer than the cold-sighted TARGET
-- otherwise "go to the nearest same-category instance" succeeds without any
recall, and the harness reproduces the single-goal null. This gate classifies
each cell VALID / DEGENERATE / UNREACHABLE. The geodesic computation is RACE-only
(habitat pathfinder); the classification logic is pure stdlib and tested here on
injected distances.

    python embodied_memory/scripts/test_check_instance_keyed_validity.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import check_instance_keyed_validity as iv  # noqa: E402

INF = float("inf")


def case_classify_warm_start():
    # distractor nearer than target -> disambiguation is forced
    assert iv.classify_warm_start(5.0, [3.0, 9.0]) == "FORCES-DISAMBIGUATION"
    # target nearest -> go-to-nearest succeeds, no disambiguation
    assert iv.classify_warm_start(2.0, [3.0, 9.0]) == "TARGET-NEAREST"
    # target geodesic infinite/None -> unreachable warm start (NaN soft_SPL risk)
    assert iv.classify_warm_start(INF, [3.0]) == "UNREACHABLE"
    assert iv.classify_warm_start(None, [3.0]) == "UNREACHABLE"
    # no finite distractor -> target is trivially nearest
    assert iv.classify_warm_start(2.0, [INF, None]) == "TARGET-NEAREST"
    print("  case_classify_warm_start: OK")


def case_classify_cell_valid():
    recs = [{"geo_to_target": 5.0, "geo_to_distractors": [3.0]},   # forces
            {"geo_to_target": 2.0, "geo_to_distractors": [9.0]}]   # target-nearest
    v = iv.classify_cell(recs)
    assert v["verdict"] == "VALID", v            # >=1 forces -> VALID
    assert v["n_reachable"] == 2 and v["n_forces"] == 1, v
    print("  case_classify_cell_valid: OK")


def case_classify_cell_degenerate():
    # every reachable warm start has the target nearest -> go-to-nearest wins
    recs = [{"geo_to_target": 2.0, "geo_to_distractors": [9.0]},
            {"geo_to_target": 1.0, "geo_to_distractors": [4.0]}]
    assert iv.classify_cell(recs)["verdict"] == "DEGENERATE"
    print("  case_classify_cell_degenerate: OK")


def case_classify_cell_unreachable():
    recs = [{"geo_to_target": INF, "geo_to_distractors": [3.0]},
            {"geo_to_target": None, "geo_to_distractors": [4.0]}]
    assert iv.classify_cell(recs)["verdict"] == "UNREACHABLE"
    print("  case_classify_cell_unreachable: OK")


def case_scan_cells_green_red():
    cells = {("S", "chair"): [{"geo_to_target": 5.0, "geo_to_distractors": [3.0]}],
             ("S", "bed"): [{"geo_to_target": 2.0, "geo_to_distractors": [9.0]}]}
    rep = iv.scan_cells(cells)
    assert rep["green"] is True                  # chair is VALID
    assert rep["cells"][("S", "chair")]["verdict"] == "VALID"
    assert rep["cells"][("S", "bed")]["verdict"] == "DEGENERATE"
    # all-degenerate -> RED
    rep2 = iv.scan_cells({("S", "bed"): [{"geo_to_target": 2.0, "geo_to_distractors": [9.0]}]})
    assert rep2["green"] is False
    print("  case_scan_cells_green_red: OK")


def case_records_from_content_euclidean():
    # synthetic built instance-keyed content: one warm episode with labels.
    # Euclidean dist_fn (the local proxy) -> target nearer -> TARGET-NEAREST.
    content = {"episodes": [
        {"episode_id": "chair-cold-0", "object_category": "chair", "scene_id": "S",
         "start_position": [0, 0, 0], "info": {}},
        {"episode_id": "chair-warm-1", "object_category": "chair", "scene_id": "S",
         "start_position": [1, 0, 1],
         "info": {"instance_labels": {"target_object_id": "A", "target_center": [2, 0, 2],
                                      "distractor_centers": [[20, 0, 20]]}}},
    ]}
    cells = iv.records_from_content(content, iv.euclidean_xz)
    recs = cells[("S", "chair")]
    assert len(recs) == 1, cells                 # cold episode excluded
    assert abs(recs[0]["geo_to_target"] - math.hypot(1, 1)) < 1e-6, recs
    assert iv.classify_cell(recs)["verdict"] == "DEGENERATE"  # target nearer than distractor
    print("  case_records_from_content_euclidean: OK")


def main() -> int:
    print("check_instance_keyed_validity tests")
    case_classify_warm_start()
    case_classify_cell_valid()
    case_classify_cell_degenerate()
    case_classify_cell_unreachable()
    case_scan_cells_green_red()
    case_records_from_content_euclidean()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
