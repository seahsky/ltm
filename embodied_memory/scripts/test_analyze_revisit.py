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


def case_compare_runs_pairs_b_minus_a():
    # Two same-setting runs (e.g. heuristic vs trained R), paired by
    # (scene, episode_id). compare_runs returns B - A on warm + cold, soft + spl.
    a = [
        _ep("S", "a", "chair", 0, soft=0.10, spl=0.0),  # cold
        _ep("S", "b", "chair", 6, soft=0.20, spl=0.0),  # warm
        _ep("S", "d", "chair", 11, soft=0.30, spl=0.0),  # warm
    ]
    b = [
        _ep("S", "a", "chair", 0, soft=0.30, spl=0.0),  # cold
        _ep("S", "b", "chair", 6, soft=0.50, spl=1.0),  # warm
        _ep("S", "d", "chair", 11, soft=0.60, spl=0.0),  # warm
    ]
    ar.assign_visit_order(a)
    ar.assign_visit_order(b)
    res = ar.compare_runs(a, b, n_bootstrap=2000)
    # warm soft deltas [0.5-0.2, 0.6-0.3] = [0.3, 0.3] -> mean 0.3
    assert res["warm_soft"]["n"] == 2, res["warm_soft"]["n"]
    assert abs(res["warm_soft"]["mean"] - 0.3) < 1e-9, res["warm_soft"]["mean"]
    # cold soft delta [0.3-0.1] = 0.2
    assert abs(res["cold_soft"]["mean"] - 0.2) < 1e-9, res["cold_soft"]["mean"]
    # warm binary spl deltas [1-0, 0-0] = [1,0] -> mean 0.5
    assert abs(res["warm_spl"]["mean"] - 0.5) < 1e-9, res["warm_spl"]["mean"]
    print("  case compare_runs_pairs_b_minus_a: OK")


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
    # (scene_id, target_category, visit_order) — scene_id is in the key so the
    # two scenes don't collide into one pair.
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


def case_warm_delta_pairs_across_renumbered_episode_ids():
    # Habitat renumbers episode_id per run, so the SAME logical warm visit
    # (same scene, category, and rank-by-pinned-episode_idx) can carry a
    # DIFFERENT episode_id in S1 vs S3. Pairing MUST key on the renumbering-
    # invariant (scene, category, visit_order), NOT episode_id — else the pair
    # is silently dropped. This is the M3 stage-1 bug that vanished 3 of 7 warm
    # pairs (the strongest cell), biasing the headline.
    s1 = [
        _ep("S", "0", "bed", 0, soft=0.20),   # cold (id "0")
        _ep("S", "2", "bed", 1, soft=0.20),   # warm visit 1 (id "2")
    ]
    s3 = [
        _ep("S", "0", "bed", 0, soft=0.20),   # cold (id "0")
        _ep("S", "1", "bed", 1, soft=0.75),   # warm visit 1 — SAME visit, id "1" != "2"
    ]
    ar.assign_visit_order(s1)
    ar.assign_visit_order(s3)
    res = ar.paired_warm_delta(s1, s3, n_bootstrap=1000)
    assert res["n"] == 1, f"renumbered warm visit must still pair: n={res['n']}"
    assert abs(res["mean"] - 0.55) < 1e-9, res["mean"]
    assert res["n_dropped"] == 0, res["n_dropped"]
    print("  case warm_delta_pairs_across_renumbered_episode_ids: OK")


def case_warm_delta_reports_unpaired_count():
    # A genuine structural mismatch — a (scene,category) with a DIFFERENT number
    # of warm visits in S1 vs S3 (e.g. an episode crashed) — must be SURFACED
    # (n_dropped / n_s1 / n_s3), not silently dropped.
    s1 = [
        _ep("S", "a", "bed", 0, soft=0.1),   # cold
        _ep("S", "b", "bed", 1, soft=0.2),   # warm vo1
        _ep("S", "c", "bed", 2, soft=0.3),   # warm vo2 (no S3 counterpart)
    ]
    s3 = [
        _ep("S", "a", "bed", 0, soft=0.1),   # cold
        _ep("S", "b", "bed", 1, soft=0.8),   # warm vo1
    ]
    ar.assign_visit_order(s1)
    ar.assign_visit_order(s3)
    res = ar.paired_warm_delta(s1, s3, n_bootstrap=1000)
    assert res["n"] == 1, res["n"]             # only vo1 pairs
    assert res["n_dropped"] == 1, res["n_dropped"]  # vo2 unpaired -> reported
    assert res["n_s1"] == 2 and res["n_s3"] == 1, (res["n_s1"], res["n_s3"])
    print("  case warm_delta_reports_unpaired_count: OK")


def case_report_pairs_despite_renumbered_ids():
    # Report-level reproduction of the M3 stage-1 regression: a cell whose S3
    # run carries different episode_ids than its S1 run must STILL contribute
    # its warm pairs to the pooled WARM S3-S1 n (was 0 under the episode_id bug).
    s1 = _run(1, [_ep("A", "0", "bed", 0, soft=0.20),
                  _ep("A", "2", "bed", 1, soft=0.20),
                  _ep("A", "3", "bed", 2, soft=0.20)])
    s3 = _run(3, [_ep("A", "0", "bed", 0, soft=0.20),
                  _ep("A", "1", "bed", 1, soft=0.75, n_mem_chosen=1),
                  _ep("A", "5", "bed", 2, soft=0.70, n_mem_chosen=1)])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ar.print_report([s1, s3], n_bootstrap=500)
    out = buf.getvalue()
    import re
    m = re.search(r"WARM S3 - S1[^\n]*?n=(\d+)", out)
    assert m and int(m.group(1)) == 2, out   # both warm visits pair (was 0 under bug)
    print("  case_report_pairs_despite_renumbered_ids: OK")


def case_report_warns_on_unpaired_visits():
    # When warm visits are genuinely unpairable (different warm counts across
    # settings), the report must LOUDLY warn rather than silently shrink n.
    s1 = _run(1, [_ep("A", "a", "bed", 0, soft=0.1),
                  _ep("A", "b", "bed", 1, soft=0.2),
                  _ep("A", "c", "bed", 2, soft=0.3)])
    s3 = _run(3, [_ep("A", "a", "bed", 0, soft=0.1),
                  _ep("A", "b", "bed", 1, soft=0.8, n_mem_chosen=1)])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ar.print_report([s1, s3], n_bootstrap=500)
    out = buf.getvalue()
    assert "WARNING" in out and "unpaired" in out.lower(), out
    print("  case_report_warns_on_unpaired_visits: OK")


def _write_run_dir(root, name, cat, eids_softs, setting=3):
    # helper: write a run dir with episode_*.json (idx0=cold, idx>=1=warm) + summary
    d = os.path.join(root, name)
    os.makedirs(d)
    for i, (eid, soft) in enumerate(eids_softs):
        ep = {"scene_id": "S", "episode_id": str(eid), "target_category": cat,
              "episode_idx": i, "soft_spl": soft, "spl": 0.0, "success": False,
              "n_steps": 10, "distance_to_goal": 2.0}
        with open(os.path.join(d, f"episode_{i:03d}.json"), "w") as f:
            json.dump(ep, f)
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump({"ablation": {"setting": setting}, "episodes": []}, f)
    return d


def case_pool_dirs_merges_episodes():
    # pool_dirs concatenates episodes across several run dirs into ONE RevisitRun
    # (used to pool the per-cell temporal/baseline S3 dirs for a matrix-wide A/B).
    with tempfile.TemporaryDirectory() as root:
        d1 = _write_run_dir(root, "m3t-A-bed-s3", "bed", [(1, 0.3)])
        d2 = _write_run_dir(root, "m3t-A-chair-s3", "chair", [(2, 0.4)])
        pooled = ar.pool_dirs([d1, d2], "temporal")
    assert len(pooled.episodes) == 2, len(pooled.episodes)
    assert pooled.setting == 3, pooled.setting
    assert sorted(e.target_category for e in pooled.episodes) == ["bed", "chair"]
    print("  case_pool_dirs_merges_episodes: OK")


def case_compare_pooled_cli_routes():
    # main(--compare-a A... --compare-b B...) pools each side and prints the
    # paired B-A head-to-head (temporal-vs-baseline S3 across cells).
    with tempfile.TemporaryDirectory() as root:
        # each cell: cold (idx0) + warm (idx1); B (temporal) warm > A (baseline) warm
        a1 = _write_run_dir(root, "m3-bed-s3", "bed", [(1, 0.2), (11, 0.2)])
        a2 = _write_run_dir(root, "m3-chair-s3", "chair", [(1, 0.1), (12, 0.1)])
        b1 = _write_run_dir(root, "m3t-bed-s3", "bed", [(5, 0.2), (15, 0.6)])
        b2 = _write_run_dir(root, "m3t-chair-s3", "chair", [(6, 0.1), (16, 0.5)])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = ar.main(["--compare-a", a1, a2, "--compare-b", b1, b2])
        out = buf.getvalue()
    assert rc == 0, rc
    assert "compare" in out.lower(), out
    import re
    m = re.search(r"WARM[^\n]*?n=(\d+)", out)
    assert m and int(m.group(1)) == 2, out      # both cells' warm pairs contribute
    print("  case_compare_pooled_cli_routes: OK")


def case_compare_pooled_requires_both_groups():
    # --compare-a without --compare-b (or vice-versa) is a usage error, not a crash.
    try:
        ar.main(["--compare-a", "x"])
    except SystemExit:
        print("  case_compare_pooled_requires_both_groups: OK")
        return
    raise AssertionError("expected SystemExit (parser.error) when only one group given")


def case_paired_delta_drops_nonfinite_pair():
    # A navmesh-unreachable goal → Infinity geodesic → NaN soft_SPL. A single NaN
    # must NOT poison the paired mean: the bad pair is dropped, counted, and the
    # headline stays finite. (Lever-2 changed-world blocker fix; protects every
    # revisit analysis.)
    import math as _m
    s1 = [_ep("S", "a", "chair", 0, soft=0.1),
          _ep("S", "b", "chair", 5, soft=0.2),
          _ep("S", "c", "chair", 10, soft=0.3)]
    s3 = [_ep("S", "a", "chair", 0, soft=0.1),
          _ep("S", "b", "chair", 5, soft=0.6),
          _ep("S", "c", "chair", 10, soft=float("nan"))]
    ar.assign_visit_order(s1)
    ar.assign_visit_order(s3)
    res = ar.paired_warm_delta(s1, s3, n_bootstrap=1000)
    assert res["n"] == 1, res                      # only the finite warm pair
    assert res["n_nonfinite"] == 1, res            # the NaN pair was dropped + counted
    assert _m.isfinite(res["mean"]), res           # headline not poisoned
    assert abs(res["mean"] - 0.4) < 1e-6, res       # 0.6 - 0.2
    print("  case_paired_delta_drops_nonfinite_pair: OK")


def case_paired_delta_nonfinite_warning_printed():
    res = {"n": 1, "mean": 0.4, "lo": 0.4, "hi": 0.4, "p_le_zero": 0.0,
           "keys": [("S", "chair", 1)], "deltas": [0.4], "n_dropped": 0,
           "n_s1": 2, "n_s3": 2, "unpaired_keys": [], "n_nonfinite": 1}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ar._print_delta("WARM test", res)
    out = buf.getvalue().lower()
    assert "non-finite" in out or "nan" in out, out
    print("  case_paired_delta_nonfinite_warning_printed: OK")


def case_compare_verdict_negligible_negative_is_tie():
    # M4 floor artifact: a paired bootstrap can clamp the CI upper bound at
    # exactly 0 and report p(<=0)=1.000 for a delta of a few ten-thousandths.
    # Reported verbatim ("A beats B (p=0.000)") this reads as a real regression.
    # A |mean| below the practical-significance band must be called a TIE.
    wm = {"mean": -0.0005, "p_le_zero": 1.0}
    v = ar._compare_verdict(wm).lower()
    assert "tie" in v, v
    assert "a beats b" not in v, v
    print("  case_compare_verdict_negligible_negative_is_tie: OK")


def case_compare_verdict_real_negative_still_a_beats_b():
    # A genuine, above-band negative is still reported as A beating B.
    wm = {"mean": -0.05, "p_le_zero": 1.0}
    v = ar._compare_verdict(wm).lower()
    assert "a beats b" in v, v
    print("  case_compare_verdict_real_negative_still_a_beats_b: OK")


def case_compare_verdict_negligible_positive_is_tie():
    # Symmetric: a tiny positive floor delta is also a tie, not "B beats A".
    wm = {"mean": +0.0007, "p_le_zero": 0.02}
    v = ar._compare_verdict(wm).lower()
    assert "tie" in v, v
    assert "b beats a" not in v, v
    print("  case_compare_verdict_negligible_positive_is_tie: OK")


def case_compare_verdict_real_positive_b_beats_a():
    # A genuine, above-band, significant positive is still "B beats A".
    wm = {"mean": +0.171, "p_le_zero": 0.002}
    v = ar._compare_verdict(wm).lower()
    assert "b beats a" in v, v
    print("  case_compare_verdict_real_positive_b_beats_a: OK")


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


def case_pools_multiple_dirs_per_setting():
    # Matrix mode: two cells at setting 1 (distinct categories), one at setting 3.
    # pool_runs_by_setting must MERGE the two setting-1 dirs' episodes, not drop
    # one (the old by_setting[s]=r overwrote → only the last survived).
    s1a = _run(1, [_ep("A", 1, "bed", 0), _ep("A", 2, "bed", 1)])
    s1b = _run(1, [_ep("B", 1, "sofa", 0), _ep("B", 2, "sofa", 1)])
    s3 = _run(3, [_ep("A", 3, "bed", 0)])
    pooled = ar.pool_runs_by_setting([s1a, s1b, s3])
    assert set(pooled.keys()) == {1, 3}, pooled.keys()
    assert len(pooled[1].episodes) == 4, len(pooled[1].episodes)  # both cells pooled
    assert sorted({e.target_category for e in pooled[1].episodes}) == ["bed", "sofa"]
    assert len(pooled[3].episodes) == 1
    print("  case_pools_multiple_dirs_per_setting: OK")


def case_pool_single_dir_per_setting_unchanged():
    # Back-compat: the standard 3-run ablation (one dir per setting) is unchanged.
    s1 = _run(1, [_ep("A", 1, "bed", 0)])
    s3 = _run(3, [_ep("A", 2, "bed", 1)])
    pooled = ar.pool_runs_by_setting([s1, s3])
    assert set(pooled.keys()) == {1, 3}
    assert len(pooled[1].episodes) == 1 and len(pooled[3].episodes) == 1
    print("  case_pool_single_dir_per_setting_unchanged: OK")


def case_matrix_report_pools_cells_for_warm_delta():
    # End-to-end: two setting-1 cells + two setting-3 cells (distinct categories,
    # distinct scenes) → print_report must pair warm across BOTH cells, not just
    # the last-loaded one. Warm n should be >= 2 (one warm pair per cell).
    s1_bed = _run(1, [_ep("A", 1, "bed", 0, soft=0.1), _ep("A", 2, "bed", 1, soft=0.1)])
    s1_sofa = _run(1, [_ep("B", 1, "sofa", 0, soft=0.1), _ep("B", 2, "sofa", 1, soft=0.1)])
    s3_bed = _run(3, [_ep("A", 1, "bed", 0, soft=0.1), _ep("A", 2, "bed", 1, soft=0.9, n_mem_chosen=1)])
    s3_sofa = _run(3, [_ep("B", 1, "sofa", 0, soft=0.1), _ep("B", 2, "sofa", 1, soft=0.8, n_mem_chosen=1)])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ar.print_report([s1_bed, s1_sofa, s3_bed, s3_sofa], n_bootstrap=500)
    out = buf.getvalue()
    # Both warm pairs (bed + sofa) must contribute → WARM delta n=2, not n=1.
    assert "WARM S3 - S1" in out, out
    import re
    m = re.search(r"WARM S3 - S1[^\n]*?n=(\d+)", out)
    assert m and int(m.group(1)) == 2, out
    print("  case_matrix_report_pools_cells_for_warm_delta: OK")


def main() -> int:
    print("Phase-A revisit analyzer sanity tests")
    case_pools_multiple_dirs_per_setting()
    case_pool_single_dir_per_setting_unchanged()
    case_matrix_report_pools_cells_for_warm_delta()
    case_visit_order_by_idx()
    case_visit_order_cold_warm_flags()
    case_visit_order_separates_categories_and_scenes()
    case_single_visit_has_no_warm()
    case_stratified_summary_splits_cold_warm()
    case_memory_fire_rate_on_warm()
    case_warm_delta_pairs_only_warm_positive()
    case_compare_runs_pairs_b_minus_a()
    case_warm_delta_negative()
    case_classify_gate_c_rare_firing()
    case_classify_gate_a_fires_and_helps()
    case_classify_gate_b_fires_but_hurts()
    case_s2_decomposition_reported()
    case_gate_unchanged_by_s2()
    case_back_compat_no_s2_block()
    case_warm_delta_multiscene_no_id_collision()
    case_warm_delta_pairs_across_renumbered_episode_ids()
    case_warm_delta_reports_unpaired_count()
    case_report_pairs_despite_renumbered_ids()
    case_report_warns_on_unpaired_visits()
    case_pool_dirs_merges_episodes()
    case_compare_pooled_cli_routes()
    case_compare_pooled_requires_both_groups()
    case_paired_delta_drops_nonfinite_pair()
    case_paired_delta_nonfinite_warning_printed()
    case_compare_verdict_negligible_negative_is_tie()
    case_compare_verdict_real_negative_still_a_beats_b()
    case_compare_verdict_negligible_positive_is_tie()
    case_compare_verdict_real_positive_b_beats_a()
    case_load_reads_episode_files()
    case_load_infers_setting_from_name()
    case_binary_spl_block_printed_when_runs_have_spl()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
