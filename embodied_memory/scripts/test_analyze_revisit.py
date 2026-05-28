"""
Sanity test for ``analyze_revisit`` — the Phase-A revisit (lifelong) analyzer.

The analyzer re-reads the existing G4 ablation runs, groups episodes by
``(scene_id, target_category)``, orders them by ``episode_idx`` to assign a
*visit order* (0 = first/"cold" sighting of the category in the scene, >=1 =
"warm" revisit), and asks whether the persisting LTM helps on warm revisits
(soft-SPL S3 vs S1) — the one regime where recalling a past sighting can pay
off. It touches no production code; this test exercises the pure analysis
helpers on synthetic episodes (no Habitat / model stack).

Invoke with::

    python embodied_memory/scripts/test_analyze_revisit.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyze_revisit as ar  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402


def _ep(scene, eid, cat, idx, soft=0.0, spl=0.0, success=False, n_steps=10,
        min_d2g=5.0, success_1m=False, n_mem_chosen=0, n_mem_cand=0, n_mem_dec=0):
    return ar.RevisitEpisode(
        scene_id=scene, episode_id=str(eid), target_category=cat, episode_idx=idx,
        soft_spl=soft, spl=spl, success=success, n_steps=n_steps,
        min_d2g=min_d2g, success_1m=success_1m,
        n_memory_chosen=n_mem_chosen, n_memory_candidates=n_mem_cand,
        n_memory_decisions=n_mem_dec,
    )


def _run(setting, eps):
    return ar.RevisitRun(name=f"s{setting}", path=f"runs/s{setting}",
                         setting=setting, episodes=eps)


# ----------------------------------------------------------------------
# assign_visit_order
# ----------------------------------------------------------------------


def case_visit_order_by_idx():
    eps = [
        _ep("S", "b", "chair", 6),
        _ep("S", "a", "chair", 0),
        _ep("S", "d", "chair", 11),
    ]
    ar.assign_visit_order(eps)
    by_id = {e.episode_id: e for e in eps}
    assert by_id["a"].visit_order == 0, by_id["a"].visit_order
    assert by_id["b"].visit_order == 1, by_id["b"].visit_order
    assert by_id["d"].visit_order == 2, by_id["d"].visit_order
    print("  case visit_order_by_idx (orders within a category by episode_idx): OK")


def case_visit_order_cold_warm_flags():
    eps = [_ep("S", "a", "chair", 0), _ep("S", "b", "chair", 6)]
    ar.assign_visit_order(eps)
    by_id = {e.episode_id: e for e in eps}
    assert by_id["a"].is_cold and not by_id["a"].is_warm
    assert by_id["b"].is_warm and not by_id["b"].is_cold
    print("  case visit_order_cold_warm_flags (first cold, rest warm): OK")


def case_visit_order_separates_categories_and_scenes():
    eps = [
        _ep("S", "a", "chair", 0), _ep("S", "b", "chair", 6),
        _ep("S", "c", "bed", 3), _ep("S", "e", "bed", 7),
        _ep("T", "f", "chair", 2),  # different scene -> its own cold
    ]
    ar.assign_visit_order(eps)
    by_id = {e.episode_id: e for e in eps}
    assert by_id["a"].visit_order == 0 and by_id["b"].visit_order == 1
    assert by_id["c"].visit_order == 0 and by_id["e"].visit_order == 1
    assert by_id["f"].visit_order == 0, "different scene must restart visit order"
    print("  case visit_order_separates_categories_and_scenes: OK")


def case_single_visit_has_no_warm():
    eps = [_ep("S", "a", "sofa", 5)]
    ar.assign_visit_order(eps)
    assert eps[0].is_cold and not eps[0].is_warm
    warm = [e for e in eps if e.is_warm]
    assert warm == [], "a category seen once contributes no warm visit"
    print("  case single_visit_has_no_warm: OK")


# ----------------------------------------------------------------------
# stratified summary
# ----------------------------------------------------------------------


def case_stratified_summary_splits_cold_warm():
    eps = [
        _ep("S", "a", "chair", 0, soft=0.0),
        _ep("S", "b", "chair", 6, soft=0.5),
        _ep("S", "d", "chair", 11, soft=0.9),
    ]
    ar.assign_visit_order(eps)
    summ = ar.stratified_summary(eps)
    assert summ["cold"]["n"] == 1
    assert summ["warm"]["n"] == 2
    assert abs(summ["cold"]["soft_spl"] - 0.0) < 1e-9
    assert abs(summ["warm"]["soft_spl"] - 0.7) < 1e-9, summ["warm"]["soft_spl"]
    print("  case stratified_summary_splits_cold_warm: OK")


def case_memory_fire_rate_on_warm():
    eps = [
        _ep("S", "a", "chair", 0, n_mem_chosen=0),   # cold
        _ep("S", "b", "chair", 6, n_mem_chosen=1),   # warm, fired
        _ep("S", "d", "chair", 11, n_mem_chosen=0),  # warm, no fire
        _ep("S", "g", "chair", 14, n_mem_chosen=2),  # warm, fired
    ]
    ar.assign_visit_order(eps)
    summ = ar.stratified_summary(eps)
    # 2 of 3 warm visits fired
    assert abs(summ["warm"]["memory_fire_rate"] - (2.0 / 3.0)) < 1e-9, summ["warm"]["memory_fire_rate"]
    assert abs(summ["cold"]["memory_fire_rate"] - 0.0) < 1e-9
    print("  case memory_fire_rate_on_warm: OK")


# ----------------------------------------------------------------------
# paired warm delta
# ----------------------------------------------------------------------


def case_warm_delta_pairs_only_warm_positive():
    s1 = [
        _ep("S", "a", "chair", 0, soft=0.1),   # cold (excluded)
        _ep("S", "b", "chair", 6, soft=0.2),   # warm
        _ep("S", "d", "chair", 11, soft=0.3),  # warm
    ]
    s3 = [
        _ep("S", "a", "chair", 0, soft=0.9),   # cold (excluded even if big)
        _ep("S", "b", "chair", 6, soft=0.6),   # warm
        _ep("S", "d", "chair", 11, soft=0.5),  # warm
    ]
    ar.assign_visit_order(s1)
    ar.assign_visit_order(s3)
    res = ar.paired_warm_delta(s1, s3, n_bootstrap=2000)
    assert res["n"] == 2, res["n"]
    # deltas [0.6-0.2, 0.5-0.3] = [0.4, 0.2] -> mean 0.3
    assert abs(res["mean"] - 0.3) < 1e-9, res["mean"]
    assert res["p_le_zero"] < 0.05, res["p_le_zero"]
    print("  case warm_delta_pairs_only_warm_positive: OK")


def case_warm_delta_negative():
    s1 = [_ep("S", "b", "chair", 6, soft=0.8), _ep("S", "d", "chair", 11, soft=0.7)]
    s3 = [_ep("S", "b", "chair", 6, soft=0.2), _ep("S", "d", "chair", 11, soft=0.3)]
    ar.assign_visit_order(s1)
    ar.assign_visit_order(s3)
    res = ar.paired_warm_delta(s1, s3, n_bootstrap=2000)
    assert res["mean"] < 0, res["mean"]
    assert res["p_le_zero"] > 0.5, res["p_le_zero"]
    print("  case warm_delta_negative: OK")


# ----------------------------------------------------------------------
# Gate A classification
# ----------------------------------------------------------------------


def case_classify_gate_c_rare_firing():
    g = ar.classify_gate_a(warm_fire_rate=0.05, warm_delta_mean=0.3, warm_delta_p=0.1)
    assert g == "c", g
    print("  case classify_gate_c_rare_firing: OK")


def case_classify_gate_a_fires_and_helps():
    g = ar.classify_gate_a(warm_fire_rate=0.5, warm_delta_mean=0.3, warm_delta_p=0.05)
    assert g == "a", g
    print("  case classify_gate_a_fires_and_helps: OK")


def case_classify_gate_b_fires_but_hurts():
    g = ar.classify_gate_a(warm_fire_rate=0.5, warm_delta_mean=-0.2, warm_delta_p=0.9)
    assert g == "b", g
    print("  case classify_gate_b_fires_but_hurts: OK")


# ----------------------------------------------------------------------
# loader
# ----------------------------------------------------------------------


def case_load_reads_episode_files():
    with tempfile.TemporaryDirectory() as d:
        ep0 = {
            "scene_id": "S", "episode_id": "5", "target_category": "chair",
            "episode_idx": 0, "soft_spl": 0.2, "spl": 0.0, "success": False,
            "n_steps": 9, "distance_to_goal": 3.0, "n_memory_chosen": 0,
            "n_memory_candidates": 1,
            "decisions": [{"chosen_source": "remembr"}, {"chosen_source": "stop"}],
        }
        ep1 = {
            "scene_id": "S", "episode_id": "8", "target_category": "chair",
            "episode_idx": 6, "soft_spl": 0.5, "spl": 0.0, "success": False,
            "n_steps": 20, "distance_to_goal": 0.5, "n_memory_chosen": 1,
            "n_memory_candidates": 2,
            "decisions": [{"chosen_source": "memory"}, {"chosen_source": "remembr"}],
        }
        with open(os.path.join(d, "episode_000.json"), "w") as f:
            json.dump(ep0, f)
        with open(os.path.join(d, "episode_001.json"), "w") as f:
            json.dump(ep1, f)
        with open(os.path.join(d, "summary.json"), "w") as f:
            json.dump({"ablation": {"setting": 3}, "episodes": []}, f)

        run = ar.load_revisit_run(d)
    assert run.setting == 3, run.setting
    assert len(run.episodes) == 2
    by_id = {e.episode_id: e for e in run.episodes}
    assert by_id["8"].target_category == "chair"
    assert by_id["8"].episode_idx == 6
    assert by_id["8"].n_memory_decisions == 1, by_id["8"].n_memory_decisions
    # distance_to_goal=0.5 -> min_d2g 0.5 -> success_1m True
    assert by_id["8"].success_1m is True
    assert by_id["0" if "0" in by_id else "5"].success_1m is False
    print("  case load_reads_episode_files (decisions/idx/d2g fallback): OK")


def case_load_infers_setting_from_name():
    with tempfile.TemporaryDirectory() as parent:
        d = os.path.join(parent, "revisit-smoke-chair-s1")
        os.makedirs(d)
        ep = {"scene_id": "S", "episode_id": "1", "target_category": "chair",
              "episode_idx": 0, "soft_spl": 0.1, "spl": 0.0, "success": False,
              "n_steps": 5, "distance_to_goal": 4.0}
        with open(os.path.join(d, "episode_000.json"), "w") as f:
            json.dump(ep, f)
        # summary without an ablation.setting -> fall back to dir name
        with open(os.path.join(d, "summary.json"), "w") as f:
            json.dump({"episodes": []}, f)
        run = ar.load_revisit_run(d)
    assert run.setting == 1, run.setting
    print("  case load_infers_setting_from_name: OK")


def case_s2_decomposition_reported():
    s1 = _run(1, [_ep("S", "a", "chair", 0, soft=0.1),
                  _ep("S", "b", "chair", 6, soft=0.2),
                  _ep("S", "d", "chair", 11, soft=0.3)])
    s2 = _run(2, [_ep("S", "a", "chair", 0, soft=0.1),
                  _ep("S", "b", "chair", 6, soft=0.25),
                  _ep("S", "d", "chair", 11, soft=0.35)])
    s3 = _run(3, [_ep("S", "a", "chair", 0, soft=0.9),
                  _ep("S", "b", "chair", 6, soft=0.6, n_mem_chosen=1),
                  _ep("S", "d", "chair", 11, soft=0.5, n_mem_chosen=1)])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ar.print_report([s1, s2, s3], n_bootstrap=500)
    out = buf.getvalue()
    assert "S2 - S1" in out, out
    assert "S3 - S2" in out, out
    assert "S3 - S1" in out, out
    print("  case s2_decomposition_reported: OK")


def _gate_helps_runs():
    # warm S3 > warm S1 and memory fires -> gate (a)
    s1 = _run(1, [_ep("S", "a", "chair", 0, soft=0.1),
                  _ep("S", "b", "chair", 6, soft=0.2),
                  _ep("S", "d", "chair", 11, soft=0.3)])
    s3 = _run(3, [_ep("S", "a", "chair", 0, soft=0.9),
                  _ep("S", "b", "chair", 6, soft=0.6, n_mem_chosen=1),
                  _ep("S", "d", "chair", 11, soft=0.5, n_mem_chosen=1)])
    s2 = _run(2, [_ep("S", "a", "chair", 0, soft=0.1),
                  _ep("S", "b", "chair", 6, soft=0.9),
                  _ep("S", "d", "chair", 11, soft=0.9)])
    return s1, s2, s3


def case_gate_unchanged_by_s2():
    s1, s2, s3 = _gate_helps_runs()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        gate_no_s2 = ar.print_report([s1, s3], n_bootstrap=500)
    s1b, s2b, s3b = _gate_helps_runs()
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        gate_s2 = ar.print_report([s1b, s2b, s3b], n_bootstrap=500)
    assert gate_no_s2 == "a", gate_no_s2
    assert gate_s2 == "a", gate_s2
    assert gate_no_s2 == gate_s2, (gate_no_s2, gate_s2)
    print("  case gate_unchanged_by_s2: OK")


def case_back_compat_no_s2_block():
    s1 = _run(1, [_ep("S", "a", "chair", 0, soft=0.1),
                  _ep("S", "b", "chair", 6, soft=0.2)])
    s3 = _run(3, [_ep("S", "a", "chair", 0, soft=0.9),
                  _ep("S", "b", "chair", 6, soft=0.6, n_mem_chosen=1)])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ar.print_report([s1, s3], n_bootstrap=500)
    out = buf.getvalue()
    assert "S2 - S1" not in out and "S3 - S2" not in out, out
    print("  case back_compat_no_s2_block: OK")


def case_warm_delta_multiscene_no_id_collision():
    # two scenes with the SAME episode_ids; pairing must key on
    # (scene_id, episode_id) so the scenes don't collide into one pair.
    s1 = [
        _ep("S", "chair-cold-0", "chair", 0, soft=0.1),
        _ep("S", "chair-warm-1", "chair", 1, soft=0.2),
        _ep("T", "chair-cold-0", "chair", 0, soft=0.1),
        _ep("T", "chair-warm-1", "chair", 1, soft=0.3),
    ]
    s3 = [
        _ep("S", "chair-cold-0", "chair", 0, soft=0.9),
        _ep("S", "chair-warm-1", "chair", 1, soft=0.6),
        _ep("T", "chair-cold-0", "chair", 0, soft=0.9),
        _ep("T", "chair-warm-1", "chair", 1, soft=0.8),
    ]
    ar.assign_visit_order(s1)
    ar.assign_visit_order(s3)
    res = ar.paired_warm_delta(s1, s3, n_bootstrap=1000)
    assert res["n"] == 2, res["n"]   # NOT collapsed to 1 despite shared ids
    # deltas [0.6-0.2, 0.8-0.3] = [0.4, 0.5] -> mean 0.45
    assert abs(res["mean"] - 0.45) < 1e-9, res["mean"]
    print("  case warm_delta_multiscene_no_id_collision: OK")


def case_binary_spl_block_printed_when_runs_have_spl():
    s1 = _run(1, [_ep("S", "a", "chair", 0, soft=0.1, spl=0.0),
                  _ep("S", "b", "chair", 6, soft=0.2, spl=0.0)])
    s3 = _run(3, [_ep("S", "a", "chair", 0, soft=0.9, spl=0.0),
                  _ep("S", "b", "chair", 6, soft=0.6, spl=0.4, n_mem_chosen=1)])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ar.print_report([s1, s3], n_bootstrap=500)
    out = buf.getvalue()
    # New binary-SPL block headers MUST be present
    assert "paired binary spl" in out.lower(), out
    assert "WARM binary S3 - S1" in out, out
    print("  case_binary_spl_block_printed_when_runs_have_spl: OK")


def main() -> int:
    print("Phase-A revisit analyzer sanity tests")
    case_visit_order_by_idx()
    case_visit_order_cold_warm_flags()
    case_visit_order_separates_categories_and_scenes()
    case_single_visit_has_no_warm()
    case_stratified_summary_splits_cold_warm()
    case_memory_fire_rate_on_warm()
    case_warm_delta_pairs_only_warm_positive()
    case_warm_delta_negative()
    case_classify_gate_c_rare_firing()
    case_classify_gate_a_fires_and_helps()
    case_classify_gate_b_fires_but_hurts()
    case_s2_decomposition_reported()
    case_gate_unchanged_by_s2()
    case_back_compat_no_s2_block()
    case_warm_delta_multiscene_no_id_collision()
    case_load_reads_episode_files()
    case_load_infers_setting_from_name()
    case_binary_spl_block_printed_when_runs_have_spl()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
