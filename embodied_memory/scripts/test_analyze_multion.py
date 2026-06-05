"""
Sanity tests for ``analyze_multion`` — the MultiON (sequential semantic
ObjectNav) ablation analyzer.

Exercises the pure analysis helpers on synthetic episodes (no Habitat /
model stack), mirroring ``test_analyze_revisit.py``:

  * per-setting Progress / PPL / success_multion aggregation,
  * paired S3-S1 bootstrap on Progress + PPL keyed (scene_id, episode_id),
  * the gap-by-sub-goal-index table (the "LTM compounds" signal),
  * recall-assisted-advance rate + advance step-cost split,
  * the ``analyze_ablation --multion`` dispatch.

Invoke with::

    python embodied_memory/scripts/test_analyze_multion.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyze_multion as am  # noqa: E402


def _found(idx, step, mem=False):
    return {"category": f"cat{idx}", "subgoal_idx": idx, "step_idx": step,
            "distance": 0.5, "memory_assisted": mem,
            "path_len_at_found": float(step) * 0.25}


def _ep(scene="S", eid="m0", k=3, found=(), ppl=None, progress=None,
        success=None, n_steps=100, n_mem_chosen=0):
    found = list(found)
    if progress is None:
        progress = len(found) / float(k)
    if success is None:
        success = len(found) == k
    return am.MultionEpisode(
        scene_id=scene, episode_id=eid,
        target_categories=[f"cat{i}" for i in range(k)],
        progress=progress, ppl=ppl, success_multion=success,
        n_steps=n_steps, path_len_taken=25.0, geodesic_optimal=9.0,
        subgoals_found=found, n_memory_chosen=n_mem_chosen,
        recall_assisted_advances=sum(1 for f in found if f["memory_assisted"]),
    )


def _run(setting, eps):
    return am.MultionRun(name=f"s{setting}", path=f"runs/s{setting}",
                         setting=setting, episodes=eps)


# ----------------------------------------------------------------------
# per-setting aggregation
# ----------------------------------------------------------------------


def case_summary_aggregates_progress_and_ppl():
    eps = [
        _ep(eid="a", found=[_found(0, 10), _found(1, 20), _found(2, 30)], ppl=0.6),
        _ep(eid="b", found=[_found(0, 15)], ppl=0.1),
        _ep(eid="c", found=[], ppl=None),  # ppl undefined -> excluded from mean
    ]
    s = am.summarize(eps)
    assert s["n"] == 3
    # progress: [1.0, 1/3, 0.0] -> mean 4/9
    assert abs(s["progress"] - (4.0 / 9.0)) < 1e-9, s["progress"]
    # ppl mean over DEFINED values only: (0.6 + 0.1)/2
    assert abs(s["ppl"] - 0.35) < 1e-9, s["ppl"]
    assert s["n_ppl"] == 2
    assert abs(s["success_multion"] - (1.0 / 3.0)) < 1e-9
    print("  case_summary_aggregates_progress_and_ppl: OK")


def case_summary_recall_assist_rate():
    eps = [
        _ep(eid="a", found=[_found(0, 10), _found(1, 20, mem=True)]),
        _ep(eid="b", found=[_found(0, 15, mem=True)]),
    ]
    s = am.summarize(eps)
    # 2 of 3 advances were memory-assisted
    assert abs(s["recall_assist_rate"] - (2.0 / 3.0)) < 1e-9, s
    print("  case_summary_recall_assist_rate: OK")


# ----------------------------------------------------------------------
# paired S3-S1 delta
# ----------------------------------------------------------------------


def case_paired_delta_progress():
    s1 = [_ep(eid="a", found=[_found(0, 10)]),
          _ep(eid="b", found=[])]
    s3 = [_ep(eid="a", found=[_found(0, 9), _found(1, 30), _found(2, 50)]),
          _ep(eid="b", found=[_found(0, 12), _found(1, 40)])]
    res = am.paired_delta(s1, s3, metric="progress", n_bootstrap=2000)
    assert res["n"] == 2, res["n"]
    # deltas [1.0 - 1/3, 2/3 - 0] = [2/3, 2/3] -> mean 2/3
    assert abs(res["mean"] - (2.0 / 3.0)) < 1e-9, res["mean"]
    assert res["p_le_zero"] < 0.05, res["p_le_zero"]
    print("  case_paired_delta_progress: OK")


def case_paired_delta_ppl_skips_undefined():
    s1 = [_ep(eid="a", ppl=0.1), _ep(eid="b", ppl=None)]
    s3 = [_ep(eid="a", ppl=0.5), _ep(eid="b", ppl=0.9)]
    res = am.paired_delta(s1, s3, metric="ppl", n_bootstrap=1000)
    # pair 'b' has an undefined S1 ppl -> excluded
    assert res["n"] == 1, res["n"]
    assert abs(res["mean"] - 0.4) < 1e-9, res["mean"]
    print("  case_paired_delta_ppl_skips_undefined: OK")


def case_paired_delta_no_scene_collision():
    s1 = [_ep(scene="A", eid="m0", found=[]),
          _ep(scene="B", eid="m0", found=[_found(0, 5)])]
    s3 = [_ep(scene="A", eid="m0", found=[_found(0, 5)]),
          _ep(scene="B", eid="m0", found=[_found(0, 5), _found(1, 9)])]
    res = am.paired_delta(s1, s3, metric="progress", n_bootstrap=500)
    assert res["n"] == 2, "must pair on (scene_id, episode_id)"
    print("  case_paired_delta_no_scene_collision: OK")


# ----------------------------------------------------------------------
# gap by sub-goal index
# ----------------------------------------------------------------------


def case_gap_by_subgoal_index():
    # S1 finds only sub-goal 0; S3 finds 0,1 (ep a) and 0,1,2 (ep b):
    # the S3-S1 gap must GROW with index (0, +1, +0.5).
    s1 = [_ep(eid="a", found=[_found(0, 10)]),
          _ep(eid="b", found=[_found(0, 12)])]
    s3 = [_ep(eid="a", found=[_found(0, 10), _found(1, 30)]),
          _ep(eid="b", found=[_found(0, 8), _found(1, 25), _found(2, 60)])]
    rows = am.gap_by_subgoal_index(s1, s3)
    assert len(rows) == 3, rows
    by_idx = {r["subgoal_idx"]: r for r in rows}
    assert by_idx[0]["rate_a"] == 1.0 and by_idx[0]["rate_b"] == 1.0
    assert abs(by_idx[0]["delta"] - 0.0) < 1e-9
    assert by_idx[1]["rate_a"] == 0.0 and by_idx[1]["rate_b"] == 1.0
    assert abs(by_idx[1]["delta"] - 1.0) < 1e-9
    assert abs(by_idx[2]["delta"] - 0.5) < 1e-9
    assert by_idx[2]["n"] == 2
    print("  case_gap_by_subgoal_index: OK")


def case_gap_verdict_micro3_shape_is_none():
    # The exact multion-micro3 bug: deltas [-0.5, 0, 0] are a SHRINKING gap,
    # but the old inline check (rows[-1] > rows[0]) printed "gap GROWS"
    # because 0 > -0.5.
    assert am.gap_growth_verdict(_rows([-0.5, 0.0, 0.0])) is None
    print("  case_gap_verdict_micro3_shape_is_none: OK")


def case_gap_verdict_monotone_positive_fires():
    msg = am.gap_growth_verdict(_rows([0.0, 0.3, 0.6]))
    assert msg is not None and "gap GROWS" in msg, msg
    print("  case_gap_verdict_monotone_positive_fires: OK")


def case_gap_verdict_non_monotonic_is_none():
    # A dip breaks the compounding story even if the last delta is positive.
    assert am.gap_growth_verdict(_rows([0.0, 0.5, 0.2])) is None
    print("  case_gap_verdict_non_monotonic_is_none: OK")


def case_gap_verdict_single_row_is_none():
    assert am.gap_growth_verdict(_rows([0.9])) is None
    assert am.gap_growth_verdict(_rows([])) is None
    print("  case_gap_verdict_single_row_is_none: OK")


def case_gap_verdict_flat_positive_fires():
    # Sustained positive gap counts: non-decreasing AND ends above zero.
    msg = am.gap_growth_verdict(_rows([0.4, 0.4]))
    assert msg is not None and "gap GROWS" in msg, msg
    print("  case_gap_verdict_flat_positive_fires: OK")


def _rows(deltas):
    """Minimal gap-table rows; the verdict only reads ``delta``."""
    return [{"subgoal_idx": i, "n": 2, "rate_a": 0.0, "rate_b": 0.0,
             "delta": float(d)} for i, d in enumerate(deltas)]


def case_advance_step_costs_split_by_memory():
    # step-cost of an advance = step_idx delta from the previous advance
    # (episode start for the first). Split by memory_assisted.
    eps = [_ep(eid="a", found=[_found(0, 10), _found(1, 20, mem=True),
                               _found(2, 50)])]
    costs = am.advance_step_costs(eps)
    assert sorted(costs["with_memory"]) == [10], costs
    assert sorted(costs["without_memory"]) == [10, 30], costs
    print("  case_advance_step_costs_split_by_memory: OK")


# ----------------------------------------------------------------------
# loader + report + dispatch
# ----------------------------------------------------------------------


def _write_run(d, setting, eps_raw):
    os.makedirs(d, exist_ok=True)
    for i, raw in enumerate(eps_raw):
        with open(os.path.join(d, f"episode_{i:03d}.json"), "w") as f:
            json.dump(raw, f)
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump({"ablation": {"setting": setting}, "episodes": []}, f)


def _raw_ep(eid, found, k=3, ppl=0.3, scene="S"):
    return {
        "scene_id": scene, "episode_id": eid, "episode_idx": 0,
        "target_category": "cat0", "is_multion": True,
        "target_categories": [f"cat{i}" for i in range(k)],
        "subgoals_found": found, "progress": len(found) / float(k),
        "success_multion": len(found) == k, "ppl": ppl, "spl_multion": 0.0,
        "path_len_taken": 25.0, "geodesic_optimal": 9.0,
        "geodesic_optimal_partial": False,
        "recall_assisted_advances": sum(1 for f in found if f["memory_assisted"]),
        "n_steps": 90, "soft_spl": 0.2, "spl": 0.0, "success": False,
        "n_memory_chosen": 1, "distance_to_goal": 3.0,
    }


def case_load_and_report():
    with tempfile.TemporaryDirectory() as tmp:
        d1, d3 = os.path.join(tmp, "m-s1"), os.path.join(tmp, "m-s3")
        _write_run(d1, 1, [_raw_ep("a", [_found(0, 10)], ppl=0.1)])
        _write_run(d3, 3, [_raw_ep("a", [_found(0, 9), _found(1, 30, mem=True),
                                         _found(2, 70)], ppl=0.6)])
        r1, r3 = am.load_multion_run(d1), am.load_multion_run(d3)
        assert r1.setting == 1 and r3.setting == 3
        assert len(r1.episodes) == 1 and len(r3.episodes) == 1
        assert r3.episodes[0].recall_assisted_advances == 1
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            am.print_report([r1, r3], n_bootstrap=500)
        out = buf.getvalue()
    assert "Progress" in out and "PPL" in out, out
    assert "gap by sub-goal index" in out.lower(), out
    assert "S3 - S1" in out, out
    assert "recall" in out.lower(), out
    print("  case_load_and_report: OK")


def case_non_multion_episodes_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "mixed-s1")
        raw_single = {"scene_id": "S", "episode_id": "x", "episode_idx": 0,
                      "target_category": "chair", "soft_spl": 0.1, "spl": 0.0,
                      "success": False, "n_steps": 50, "distance_to_goal": 4.0}
        _write_run(d, 1, [raw_single, _raw_ep("a", [_found(0, 10)])])
        run = am.load_multion_run(d)
    assert len(run.episodes) == 1, "single-goal episodes must be skipped"
    assert run.episodes[0].episode_id == "a"
    print("  case_non_multion_episodes_skipped: OK")


def case_analyze_ablation_multion_dispatch():
    import analyze_ablation as aa
    with tempfile.TemporaryDirectory() as tmp:
        d1, d3 = os.path.join(tmp, "m-s1"), os.path.join(tmp, "m-s3")
        _write_run(d1, 1, [_raw_ep("a", [_found(0, 10)], ppl=0.1)])
        _write_run(d3, 3, [_raw_ep("a", [_found(0, 9), _found(1, 30)], ppl=0.5)])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = aa.main(["--multion", d1, d3])
        out = buf.getvalue()
    assert rc == 0, rc
    assert "gap by sub-goal index" in out.lower(), out
    print("  case_analyze_ablation_multion_dispatch: OK")


def main() -> int:
    print("analyze_multion sanity tests")
    case_summary_aggregates_progress_and_ppl()
    case_summary_recall_assist_rate()
    case_paired_delta_progress()
    case_paired_delta_ppl_skips_undefined()
    case_paired_delta_no_scene_collision()
    case_gap_by_subgoal_index()
    case_gap_verdict_micro3_shape_is_none()
    case_gap_verdict_monotone_positive_fires()
    case_gap_verdict_non_monotonic_is_none()
    case_gap_verdict_single_row_is_none()
    case_gap_verdict_flat_positive_fires()
    case_advance_step_costs_split_by_memory()
    case_load_and_report()
    case_non_multion_episodes_skipped()
    case_analyze_ablation_multion_dispatch()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
