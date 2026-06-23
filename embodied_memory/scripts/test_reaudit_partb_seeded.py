"""
test_reaudit_partb_seeded — TDD spec for the partb-seeded re-audit tool.

Covers the three open holes the adversarial review flagged on the seed-distractor
instance-keyed run (runs/partb-seeded-s{1,2,3}):
  (a) PER-CELL VALID / DEGENERATE / UNREACHABLE verdict (was missing from the
      run summary) — a VALID cell forces disambiguation, a DEGENERATE cell does
      not, so the +0.2085 must be read against k/m seeded cells VALID.
  (b) PER-CELL wrong-instance recall rate (+ the pooled aggregate ~35%).
  (c) DROP SENSITIVITY — which warm pairs were dropped for non-finite soft_spl /
      Inf distance (the n=21->17), and the paired warm S3-S1 delta WITH vs
      WITHOUT them.

All fixtures are SYNTHETIC (tiny in-memory run-dir summary/episode dicts + fake
instance_labels + fake positions); the pathfinder is MOCKED so the test runs
with no Habitat dependency.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import reaudit_partb_seeded as ra  # noqa: E402


# ----------------------------------------------------------------------
# synthetic fixtures
# ----------------------------------------------------------------------


def _ep(episode_idx, scene, cat, soft_spl, *, spl=0.0, min_d2g=0.4,
        target=None, distractors=None, mem_world_xys=None, seed_only=False):
    """A synthetic per-episode JSON dict (the `ep_log` shape the diagnostics read).

    ``mem_world_xys`` -> one decision with that many memory candidates (xz points).
    ``target`` / ``distractors`` -> instance_labels (None => single-goal, skipped
    by the wrong-instance scorer).
    """
    decisions = []
    if mem_world_xys:
        decisions = [{
            "step_idx": 5,
            "chosen_source": "memory",
            "candidates": [
                {"id": f"m{i}", "world_xy": xy, "source": "memory", "raw_score": 0.5}
                for i, xy in enumerate(mem_world_xys)
            ],
        }]
    ep = {
        "episode_idx": episode_idx,
        "episode_id": str(episode_idx),
        "scene_id": scene,
        "target_category": cat,
        "soft_spl": soft_spl,
        "spl": spl,
        "success": bool(spl > 0),
        "n_steps": 50,
        "min_distance_to_goal": min_d2g,
        "success_1m": min_d2g < 1.0,
        "n_memory_candidates": len(mem_world_xys or []),
        "n_memory_chosen": 1 if mem_world_xys else 0,
        "seed_only": seed_only,
        "decisions": decisions,
    }
    if target is not None:
        ep["instance_labels"] = {
            "target_object_id": "obj_target",
            "target_center": target,
            "distractor_centers": distractors or [],
        }
    return ep


def _write_run_dir(tmp_path, name, episodes):
    """Materialize a run dir with episode_*.json + a summary.json (so both the
    analyze_revisit loader and the _load_run_dirs loader see it)."""
    d = tmp_path / name
    d.mkdir()
    for i, ep in enumerate(episodes):
        (d / f"episode_{i:03d}.json").write_text(json.dumps(ep))
    setting = int(name.rsplit("-s", 1)[-1])
    summary = {"ablation": {"setting": setting}, "episodes": episodes,
               "n_episodes_completed": len(episodes)}
    (d / "summary.json").write_text(json.dumps(summary))
    return str(d)


def _write_content(tmp_path, scene, episodes_with_labels):
    """Write a built instance-keyed content/<scene>.json.gz with `-warm-` ids and
    info.instance_labels (the shape check_instance_keyed_validity reads)."""
    cdir = tmp_path / "content"
    cdir.mkdir(exist_ok=True)
    eps = []
    for e in episodes_with_labels:
        eps.append({
            "episode_id": e["episode_id"],
            "scene_id": scene,
            "object_category": e["object_category"],
            "start_position": e["start_position"],
            "info": {"instance_labels": e["instance_labels"]},
        })
    path = cdir / f"{scene}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"episodes": eps}, f)
    return str(path)


# ----------------------------------------------------------------------
# (a) per-cell VALID / DEGENERATE — Euclidean proxy (no pathfinder)
# ----------------------------------------------------------------------


def test_valid_cell_distractor_nearer():
    """A warm start where a distractor is xz-nearer than the target => VALID."""
    # start at origin; target far (10,_,0), distractor near (1,_,0) => FORCES.
    content = {"episodes": [{
        "episode_id": "chair-warm-0",
        "scene_id": "sceneA",
        "object_category": "chair",
        "start_position": [0.0, 0.0, 0.0],
        "info": {"instance_labels": {
            "target_center": [10.0, 0.0, 0.0],
            "distractor_centers": [[1.0, 0.0, 0.0]],
        }},
    }]}
    rep = ra.per_cell_validity_from_contents([content], use_pathfinder=False)
    v = rep["cells"][("sceneA", "chair")]
    assert v["verdict"] == "VALID"
    assert v["n_forces"] == 1
    assert rep["green"] is True


def test_degenerate_cell_target_nearest():
    """Every warm start has the target nearest => DEGENERATE (eval never forces)."""
    content = {"episodes": [{
        "episode_id": "bed-warm-0",
        "scene_id": "sceneA",
        "object_category": "bed",
        "start_position": [0.0, 0.0, 0.0],
        "info": {"instance_labels": {
            "target_center": [1.0, 0.0, 0.0],         # target nearest
            "distractor_centers": [[10.0, 0.0, 0.0]],  # distractor far
        }},
    }]}
    rep = ra.per_cell_validity_from_contents([content], use_pathfinder=False)
    v = rep["cells"][("sceneA", "bed")]
    assert v["verdict"] == "DEGENERATE"
    assert v["n_forces"] == 0
    assert rep["green"] is False


def test_mixed_cells_overall_green_and_unreachable():
    """A scan over multiple cells: VALID present => overall GREEN; an all-None
    cell => UNREACHABLE."""
    content = {"episodes": [
        {"episode_id": "chair-warm-0", "scene_id": "sA", "object_category": "chair",
         "start_position": [0.0, 0.0, 0.0],
         "info": {"instance_labels": {"target_center": [9.0, 0.0, 0.0],
                                      "distractor_centers": [[1.0, 0.0, 0.0]]}}},
        {"episode_id": "bed-warm-0", "scene_id": "sA", "object_category": "bed",
         "start_position": None,  # unreachable: no start => dist None
         "info": {"instance_labels": {"target_center": [1.0, 0.0, 0.0],
                                      "distractor_centers": [[9.0, 0.0, 0.0]]}}},
    ]}
    rep = ra.per_cell_validity_from_contents([content], use_pathfinder=False)
    assert rep["cells"][("sA", "chair")]["verdict"] == "VALID"
    assert rep["cells"][("sA", "bed")]["verdict"] == "UNREACHABLE"
    assert rep["green"] is True


# ----------------------------------------------------------------------
# (a') pathfinder path is MOCKED — geodesic flips a Euclidean verdict
# ----------------------------------------------------------------------


def test_pathfinder_geodesic_mocked(monkeypatch, tmp_path):
    """With --use-pathfinder, distances come from the (mocked) navmesh. We mock the
    pathfinder so a wall makes the Euclidean-near distractor geodesically FAR =>
    the cell flips VALID(euclid) -> DEGENERATE(geodesic). No Habitat import."""
    content_path = _write_content(tmp_path, "sceneG", [{
        "episode_id": "chair-warm-0",
        "object_category": "chair",
        "start_position": [0.0, 0.0, 0.0],
        "instance_labels": {"target_center": [10.0, 0.0, 0.0],
                            "distractor_centers": [[1.0, 0.0, 0.0]]},
    }])

    # Euclidean: distractor (1m) < target (10m) => VALID.
    rep_e = ra.per_cell_validity_from_contents(
        [ra._read_content(content_path)], use_pathfinder=False)
    assert rep_e["cells"][("sceneG", "chair")]["verdict"] == "VALID"

    # Mock geodesic: target 5m, distractor 99m (wall) => TARGET-NEAREST => DEGENERATE.
    def fake_geo(a, b):
        if a is None or b is None:
            return None
        bx = float(b[0])
        return 5.0 if abs(bx - 10.0) < 1e-6 else 99.0

    monkeypatch.setattr(ra, "_build_geodesic_dist_fns",
                        lambda paths, roots: {"sceneG": fake_geo})
    rep_g = ra.run_validity(content_globs=[content_path], use_pathfinder=True,
                            navmesh_root=None)
    assert rep_g["cells"][("sceneG", "chair")]["verdict"] == "DEGENERATE"
    assert rep_g["approx"] is False


def test_pathfinder_missing_degrades_to_euclidean(monkeypatch, tmp_path):
    """If the pathfinder cannot be built (no navmesh / no habitat), the tool must
    NOT crash — it degrades to the Euclidean proxy and flags approx=True."""
    content_path = _write_content(tmp_path, "sceneG", [{
        "episode_id": "chair-warm-0",
        "object_category": "chair",
        "start_position": [0.0, 0.0, 0.0],
        "instance_labels": {"target_center": [10.0, 0.0, 0.0],
                            "distractor_centers": [[1.0, 0.0, 0.0]]},
    }])
    monkeypatch.setattr(ra, "_build_geodesic_dist_fns", lambda paths, roots: {})
    rep = ra.run_validity(content_globs=[content_path], use_pathfinder=True,
                          navmesh_root=None)
    # falls back to Euclidean => VALID, and loudly marks itself approximate.
    assert rep["cells"][("sceneG", "chair")]["verdict"] == "VALID"
    assert rep["approx"] is True


# ----------------------------------------------------------------------
# (b) wrong-instance recall rate — per cell + aggregate
# ----------------------------------------------------------------------


def test_wrong_instance_rate_per_cell_and_aggregate():
    target = [10.0, 0.0, 0.0]
    distractor = [0.0, 0.0, 0.0]
    # sceneA/chair: one warm ep, recalled candidate sits ON the distractor => wrong.
    chair_warm = _ep(1, "sceneA", "chair", 0.30, target=target,
                     distractors=[distractor], mem_world_xys=[[0.1, 0.0]])
    # sceneA/bed: one warm ep, recalled candidate sits ON the target => right.
    bed_warm = _ep(1, "sceneA", "bed", 0.30,
                   target=target, distractors=[distractor],
                   mem_world_xys=[[10.0, 0.0]])
    episodes = [chair_warm, bed_warm]

    per_cell, agg = ra.wrong_instance_by_cell(episodes)

    assert per_cell[("sceneA", "chair")] == {"fires": 1, "wrong": 1, "rate": 1.0}
    assert per_cell[("sceneA", "bed")] == {"fires": 1, "wrong": 0, "rate": 0.0}
    # aggregate reproduces the pooled diagnose_goal_anchored_recall number.
    assert agg == {"fires": 2, "wrong": 1, "rate": 0.5}


def test_wrong_instance_rate_aggregate_matches_upstream():
    """The aggregate must equal diagnose_goal_anchored_recall.wrong_instance_recall_rate
    (we reuse its logic, not reimplement)."""
    import diagnose_goal_anchored_recall as dg
    target, distractor = [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    episodes = [
        _ep(1, "sA", "chair", 0.3, target=target, distractors=[distractor],
            mem_world_xys=[[0.0, 0.0]]),                 # wrong
        _ep(2, "sA", "chair", 0.3, target=target, distractors=[distractor],
            mem_world_xys=[[10.0, 0.0]]),                # right
        _ep(3, "sB", "bed", 0.3, target=target, distractors=[distractor],
            mem_world_xys=[[9.5, 0.0]]),                 # right
    ]
    _, agg = ra.wrong_instance_by_cell(episodes)
    upstream = dg.wrong_instance_recall_rate(episodes)
    assert agg["fires"] == upstream["fires"]
    assert agg["wrong"] == upstream["wrong"]
    assert math.isclose(agg["rate"], upstream["rate"])


def test_wrong_instance_silent_for_single_goal():
    """No instance_labels (single-goal) => no cells, aggregate None (silent)."""
    episodes = [_ep(1, "sA", "chair", 0.3, mem_world_xys=[[0.0, 0.0]])]
    per_cell, agg = ra.wrong_instance_by_cell(episodes)
    assert per_cell == {}
    assert agg is None


# ----------------------------------------------------------------------
# (c) drop sensitivity — identify Inf/NaN-dropped warm pairs + delta with/without
# ----------------------------------------------------------------------


def _build_s1_s3(tmp_path):
    """3 warm pairs in sceneA/chair (visit_order 1,2,3) + the cold (visit_order 0).
    One warm pair (visit_order 2) has Inf soft_spl in S1 (navmesh-unreachable) =>
    must be DROPPED; the other two are finite.
       finite deltas: (0.50-0.10)=+0.40, (0.60-0.20)=+0.40 => mean +0.40 kept.
       dropped pair contributes a (huge) value only when forced in.
    """
    inf = float("inf")
    s1 = [
        _ep(0, "sceneA", "chair", 0.05),                  # cold
        _ep(1, "sceneA", "chair", 0.10),                  # warm v1 finite
        _ep(2, "sceneA", "chair", inf, min_d2g=inf),      # warm v2 -> DROP
        _ep(3, "sceneA", "chair", 0.20),                  # warm v3 finite
    ]
    s3 = [
        _ep(0, "sceneA", "chair", 0.05),
        _ep(1, "sceneA", "chair", 0.50),
        _ep(2, "sceneA", "chair", 0.90),                  # S3 finite, S1 inf -> still DROP
        _ep(3, "sceneA", "chair", 0.60),
    ]
    s1_dir = _write_run_dir(tmp_path, "partb-seeded-s1", s1)
    s3_dir = _write_run_dir(tmp_path, "partb-seeded-s3", s3)
    return s1_dir, s3_dir


def test_drop_identifies_nonfinite_pair(tmp_path):
    s1_dir, s3_dir = _build_s1_s3(tmp_path)
    report = ra.drop_sensitivity(s1_dir, s3_dir, metric="soft_spl")

    # exactly one dropped warm pair, the visit_order-2 chair.
    assert report["n_kept"] == 2
    assert report["n_dropped_nonfinite"] == 1
    dropped = report["dropped"]
    assert len(dropped) == 1
    drow = dropped[0]
    assert drow["scene_id"] == "sceneA"
    assert drow["target_category"] == "chair"
    assert drow["visit_order"] == 2
    # the reason names which side was non-finite.
    assert "s1" in drow["why"].lower() or "inf" in drow["why"].lower()


def test_drop_delta_with_vs_without(tmp_path):
    s1_dir, s3_dir = _build_s1_s3(tmp_path)
    report = ra.drop_sensitivity(s1_dir, s3_dir, metric="soft_spl")

    # WITHOUT the drop (the headline path): two finite +0.40 deltas => +0.40.
    assert math.isclose(report["mean_without_drops"], 0.40, abs_tol=1e-9)
    assert report["n_kept"] == 2

    # WITH the drop forced in: the inf pair makes the included delta non-finite,
    # so the "with" mean is reported as non-finite (cannot be averaged) — the tool
    # states this rather than silently dropping.
    assert not math.isfinite(report["mean_with_drops"])
    assert report["n_with_drops"] == 3


def test_drop_no_nonfinite_pairs(tmp_path):
    """All-finite warm pairs => zero drops, with==without."""
    s1 = [_ep(0, "sA", "chair", 0.05), _ep(1, "sA", "chair", 0.10),
          _ep(2, "sA", "chair", 0.20)]
    s3 = [_ep(0, "sA", "chair", 0.05), _ep(1, "sA", "chair", 0.50),
          _ep(2, "sA", "chair", 0.60)]
    s1_dir = _write_run_dir(tmp_path, "x-s1", s1)
    s3_dir = _write_run_dir(tmp_path, "x-s3", s3)
    report = ra.drop_sensitivity(s1_dir, s3_dir, metric="soft_spl")
    assert report["n_dropped_nonfinite"] == 0
    assert report["dropped"] == []
    assert math.isclose(report["mean_with_drops"], report["mean_without_drops"])


# ----------------------------------------------------------------------
# overall verdict line
# ----------------------------------------------------------------------


def test_overall_verdict_genuine_when_a_cell_valid():
    cells = {("sA", "chair"): {"verdict": "VALID"},
             ("sA", "bed"): {"verdict": "DEGENERATE"}}
    line = ra.overall_verdict_line(cells)
    assert "GENUINE" in line
    assert "1/2" in line


def test_overall_verdict_degenerate_when_none_valid():
    cells = {("sA", "chair"): {"verdict": "DEGENERATE"},
             ("sA", "bed"): {"verdict": "UNREACHABLE"}}
    line = ra.overall_verdict_line(cells)
    assert "did not force" in line.lower() or "DEGENERATE" in line
    assert "0/2" in line


# ----------------------------------------------------------------------
# end-to-end smoke through the CLI main (Euclidean, no pathfinder)
# ----------------------------------------------------------------------


def test_main_smoke_euclidean(tmp_path, capsys):
    # run dirs (S1/S3) with one valid wrong-instance warm episode each + a drop.
    s1_dir, s3_dir = _build_s1_s3(tmp_path)
    # add instance_labels to the warm S1/S3 chair episodes so wrong-instance fires,
    # and a content file so validity has a VALID cell.
    content_path = _write_content(tmp_path, "sceneA", [{
        "episode_id": "chair-warm-0",
        "object_category": "chair",
        "start_position": [0.0, 0.0, 0.0],
        "instance_labels": {"target_center": [10.0, 0.0, 0.0],
                            "distractor_centers": [[1.0, 0.0, 0.0]]},
    }])
    rc = ra.main([
        "--run-dirs", s1_dir, s3_dir,
        "--episodes", content_path,
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "PER-CELL VALIDITY" in out
    assert "DROP SENSITIVITY" in out
    assert "READOUT" in out
    # the validity verdict line is present.
    assert "VALID" in out
