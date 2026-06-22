"""
TDD for check_multi_instance_feasible — the $0 precondition gate for the
multi-instance revisit harness (#1 Part A).

The harness needs episodes with >=2 REACHABLE same-category instances that are
far enough apart to be distinct goals (so instance disambiguation is actually
forced). This gate scans the HM3D val_mini content files and reports, per
(scene, category): how many instances carry view_points, the pairwise centroid
separation, and a FEASIBLE / CO-LOCATED / SINGLE verdict — so we learn for $0
whether the harness is buildable BEFORE any GPU. Pure stdlib (no sim/faiss).

    python embodied_memory/scripts/test_check_multi_instance_feasible.py
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import check_multi_instance_feasible as mf  # noqa: E402


def _vp(pos, iou=0.5):
    return {"agent_state": {"position": list(pos), "rotation": [0, 0, 0, 1]}, "iou": iou}


def _inst(object_id, vps):
    return {"object_id": object_id, "position": [0, 0, 0], "view_points": list(vps)}


def _content(scene, by_cat):
    # by_cat: {category: [instance, ...]} -> keyed as "<scene>.basis.glb_<cat>"
    gbc = {f"{scene}.basis.glb_{cat}": insts for cat, insts in by_cat.items()}
    return {"goals_by_category": gbc, "episodes": []}


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def case_category_instances_reads_vps():
    c = _content("S", {"chair": [
        _inst(1, [_vp((0, 0, 0)), _vp((2, 0, 0))]),
        _inst(2, []),                      # no view_points -> excluded
        _inst(3, [_vp((5, 0, 5))]),
    ]})
    got = mf.category_instances_vps(c, "chair")
    assert [oid for oid, _ in got] == [1, 3], got           # inst 2 dropped
    assert got[0][1] == [[0, 0, 0], [2, 0, 0]], got
    print("  case_category_instances_reads_vps: OK")


def case_centroid_and_min_sep():
    # inst1 centroid (1,0,0), inst3 centroid (5,0,5) -> xz sep sqrt(16+25)=6.403
    c = _content("S", {"chair": [
        _inst(1, [_vp((0, 0, 0)), _vp((2, 0, 0))]),
        _inst(3, [_vp((5, 0, 5))]),
    ]})
    cents = mf.instance_centroids(c, "chair")
    assert cents[0][1] == [1.0, 0.0, 0.0], cents
    seps = mf.pairwise_seps(cents)
    assert len(seps) == 1
    assert abs(seps[0]["sep"] - 6.4031) < 1e-3, seps
    print("  case_centroid_and_min_sep: OK")


def case_suffix_match_multitoken():
    c = _content("S", {"tv_monitor": [_inst(1, [_vp((0, 0, 0))]), _inst(2, [_vp((4, 0, 0))])]})
    got = mf.category_instances_vps(c, "tv_monitor")
    assert len(got) == 2, got
    assert mf.category_instances_vps(c, "sofa") == [], "absent category -> empty"
    print("  case_suffix_match_multitoken: OK")


# ----------------------------------------------------------------------
# verdicts
# ----------------------------------------------------------------------


def case_verdict_feasible():
    c = _content("S", {"bed": [_inst(1, [_vp((0, 0, 0))]), _inst(2, [_vp((4, 0, 4))])]})
    v = mf.cell_verdict(c, "bed", min_sep=1.5)
    assert v["verdict"] == "FEASIBLE", v
    assert v["n_vp_instances"] == 2, v
    assert v["max_sep"] > 1.5, v
    print("  case_verdict_feasible: OK")


def case_verdict_co_located():
    # two instances but centroids 0.05m apart -> CO-LOCATED (plant-like)
    c = _content("S", {"plant": [_inst(1, [_vp((0, 0, 0))]), _inst(2, [_vp((0.05, 0, 0))])]})
    v = mf.cell_verdict(c, "plant", min_sep=1.5)
    assert v["verdict"] == "CO-LOCATED", v
    print("  case_verdict_co_located: OK")


def case_verdict_single():
    c = _content("S", {"toilet": [_inst(1, [_vp((0, 0, 0))])]})
    v = mf.cell_verdict(c, "toilet", min_sep=1.5)
    assert v["verdict"] == "SINGLE", v
    # zero vp-instances also SINGLE (not a crash)
    c2 = _content("S", {"toilet": [_inst(1, [])]})
    assert mf.cell_verdict(c2, "toilet", 1.5)["verdict"] == "SINGLE"
    print("  case_verdict_single: OK")


def case_scan_and_recommend_exit():
    # one scene: bed FEASIBLE, plant CO-LOCATED, tv_monitor SINGLE
    c = _content("Z", {
        "bed": [_inst(1, [_vp((0, 0, 0))]), _inst(2, [_vp((5, 0, 5))])],
        "plant": [_inst(3, [_vp((0, 0, 0))]), _inst(4, [_vp((0.03, 0, 0))])],
        "tv_monitor": [_inst(5, [_vp((1, 0, 1))])],
    })
    report = mf.scan_contents({"Z": c}, ["bed", "plant", "tv_monitor"], min_sep=1.5)
    cells = {(r["scene"], r["category"]): r["verdict"] for r in report["cells"]}
    assert cells[("Z", "bed")] == "FEASIBLE"
    assert cells[("Z", "plant")] == "CO-LOCATED"
    assert cells[("Z", "tv_monitor")] == "SINGLE"
    assert report["green"] is True, report          # >=1 feasible -> GREEN
    assert "bed" in report["matrix_categories"], report
    # all-infeasible -> RED
    c2 = _content("Z", {"tv_monitor": [_inst(5, [_vp((1, 0, 1))])]})
    rep2 = mf.scan_contents({"Z": c2}, ["tv_monitor"], min_sep=1.5)
    assert rep2["green"] is False, rep2
    print("  case_scan_and_recommend_exit: OK")


def case_gz_roundtrip_loader():
    c = _content("S", {"bed": [_inst(1, [_vp((0, 0, 0))]), _inst(2, [_vp((5, 0, 5))])]})
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "S.json.gz")
        with gzip.open(p, "wt", encoding="utf-8") as f:
            json.dump(c, f)
        loaded = mf.load_content(p)
        v = mf.cell_verdict(loaded, "bed", 1.5)
        assert v["verdict"] == "FEASIBLE", v
    print("  case_gz_roundtrip_loader: OK")


def main() -> int:
    print("check_multi_instance_feasible tests")
    case_category_instances_reads_vps()
    case_centroid_and_min_sep()
    case_suffix_match_multitoken()
    case_verdict_feasible()
    case_verdict_co_located()
    case_verdict_single()
    case_scan_and_recommend_exit()
    case_gz_roundtrip_loader()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
