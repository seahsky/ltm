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
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import check_instance_keyed_validity as iv  # noqa: E402

INF = float("inf")


# Mock habitat-sim PathFinder/ShortestPath so the geodesic WIRING unit-tests
# without habitat-sim (the real load is RACE-only behind _load_pathfinder).
class _MockShortestPath:
    def __init__(self):
        self.requested_start = None
        self.requested_end = None
        self.geodesic_distance = INF


class _MockPathfinder:
    """snap_point = identity; find_path returns the Euclidean distance, except a
    point with x==999 (test sentinel) is 'unreachable' -> find_path False."""

    def snap_point(self, p):
        return p

    def find_path(self, sp):
        a, b = sp.requested_start, sp.requested_end
        if a is None or b is None or a[0] == 999 or b[0] == 999:
            sp.geodesic_distance = INF
            return False
        sp.geodesic_distance = math.dist(a, b)
        return True


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


def case_geodesic_dist_fn_wiring():
    # the geodesic dist_fn snaps -> ShortestPath -> find_path -> geodesic_distance,
    # returning None for None inputs and for unreachable (find_path False) pairs.
    df = iv.make_geodesic_dist_fn(_MockPathfinder(), _MockShortestPath)
    assert abs(df([0, 0, 0], [3, 0, 4]) - 5.0) < 1e-6
    assert df(None, [1, 0, 1]) is None
    assert df([999, 0, 0], [0, 0, 0]) is None          # unreachable -> None
    # and it composes with the classifier: a distractor nearer than target FORCES
    target, distractor = [10, 0, 0], [1, 0, 0]
    rec = {"geo_to_target": df([0, 0, 0], target),
           "geo_to_distractors": [df([0, 0, 0], distractor)]}
    assert iv.classify_warm_start(rec["geo_to_target"], rec["geo_to_distractors"]) \
        == "FORCES-DISAMBIGUATION"
    print("  case_geodesic_dist_fn_wiring: OK")


def case_find_navmesh_glob():
    with tempfile.TemporaryDirectory() as d:
        scene_dir = os.path.join(d, "00800-wcojb4TFT35")
        os.makedirs(scene_dir)
        nav = os.path.join(scene_dir, "wcojb4TFT35.basis.navmesh")
        open(nav, "w").close()
        assert iv._find_navmesh("wcojb4TFT35", [d]) == nav
        assert iv._find_navmesh("nonexistent", [d]) is None
        assert iv._find_navmesh("wcojb4TFT35", ["/no/such/dir"]) is None
    print("  case_find_navmesh_glob: OK")


def main() -> int:
    print("check_instance_keyed_validity tests")
    case_classify_warm_start()
    case_classify_cell_valid()
    case_classify_cell_degenerate()
    case_classify_cell_unreachable()
    case_scan_cells_green_red()
    case_records_from_content_euclidean()
    case_geodesic_dist_fn_wiring()
    case_find_navmesh_glob()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
