"""
Sanity tests for the ``--revisit`` dispatch added to ``analyze_ablation.main()``.

Tests that:
1. Importing ``analyze_ablation`` does NOT eagerly pull in ``analyze_revisit``
   (lazy-import contract).
2. ``--revisit`` flag routes to the visit-order report (revisit markers present;
   phase-2 gate marker absent).
3. No flag gives the standard phase-2 gate (phase-2 gate marker present; revisit
   marker absent).
4. ``--revisit`` with three settings (S1, S2, S3) emits the S2 decomposition
   block.

Invoke with::

    python embodied_memory/scripts/test_analyze_ablation.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyze_ablation  # noqa: E402


# ----------------------------------------------------------------------
# fixture helpers
# ----------------------------------------------------------------------


def _write_episode(directory: str, index: int, ep_dict: dict) -> None:
    fname = f"episode_{index:03d}.json"
    with open(os.path.join(directory, fname), "w") as f:
        json.dump(ep_dict, f)


def _make_run_dir(parent: str, setting: int, episodes: list) -> str:
    """Create a run subdir for the given setting with per-episode JSON files.

    The directory is named ``abl-sN`` (so ``load_revisit_run`` can also infer
    the setting from the dir name as a fallback).  ``summary.json`` is written
    with ``{"ablation": {"setting": N}, "episodes": []}``.

    Each episode dict must carry the fields consumed by both
    ``analyze_ablation.load_run`` and ``analyze_revisit.load_revisit_run``:
      scene_id, episode_id (str), target_category, episode_idx (int),
      soft_spl, spl, success, n_steps, distance_to_goal,
      n_memory_chosen, n_memory_candidates, decisions (list).
    """
    run_dir = os.path.join(parent, f"abl-s{setting}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump({"ablation": {"setting": setting}, "episodes": []}, f)
    for idx, ep in enumerate(episodes):
        _write_episode(run_dir, idx, ep)
    return run_dir


def _cold_ep(scene="sc1", episode_id="ep0", soft_spl=0.05, n_memory_chosen=0):
    """Return a cold-visit episode dict (episode_idx=0, first of its category)."""
    return {
        "scene_id": scene,
        "episode_id": episode_id,
        "target_category": "chair",
        "episode_idx": 0,
        "soft_spl": soft_spl,
        "spl": 0.0,
        "success": False,
        "n_steps": 20,
        "distance_to_goal": 3.5,
        "n_memory_chosen": n_memory_chosen,
        "n_memory_candidates": 0,
        "decisions": [],
    }


def _warm_ep(scene="sc1", episode_id="ep6", soft_spl=0.3, n_memory_chosen=1):
    """Return a warm-visit episode dict (episode_idx=6, revisit of same category)."""
    return {
        "scene_id": scene,
        "episode_id": episode_id,
        "target_category": "chair",
        "episode_idx": 6,
        "soft_spl": soft_spl,
        "spl": 0.0,
        "success": False,
        "n_steps": 15,
        "distance_to_goal": 1.5,
        "n_memory_chosen": n_memory_chosen,
        "n_memory_candidates": 2,
        "decisions": [{"chosen_source": "memory"}] if n_memory_chosen > 0 else [],
    }


def _make_s1_s3_dirs(parent: str):
    s1_eps = [_cold_ep(soft_spl=0.05), _warm_ep(soft_spl=0.10, n_memory_chosen=0)]
    s3_eps = [_cold_ep(soft_spl=0.05), _warm_ep(soft_spl=0.40, n_memory_chosen=1)]
    s1_dir = _make_run_dir(parent, 1, s1_eps)
    s3_dir = _make_run_dir(parent, 3, s3_eps)
    return s1_dir, s3_dir


def _make_s1_s2_s3_dirs(parent: str):
    s1_eps = [_cold_ep(soft_spl=0.05), _warm_ep(soft_spl=0.10, n_memory_chosen=0)]
    s2_eps = [_cold_ep(soft_spl=0.05), _warm_ep(soft_spl=0.10, n_memory_chosen=0)]
    s3_eps = [_cold_ep(soft_spl=0.05), _warm_ep(soft_spl=0.40, n_memory_chosen=1)]
    s1_dir = _make_run_dir(parent, 1, s1_eps)
    s2_dir = _make_run_dir(parent, 2, s2_eps)
    s3_dir = _make_run_dir(parent, 3, s3_eps)
    return s1_dir, s2_dir, s3_dir


# ----------------------------------------------------------------------
# cases
# ----------------------------------------------------------------------


def case_lazy_import_contract():
    """analyze_revisit must NOT be imported as a side-effect of importing
    analyze_ablation.  Only analyze_ablation is imported at module top; the
    revisit module must stay out of sys.modules until --revisit is used."""
    assert "analyze_revisit" not in sys.modules, (
        "analyze_revisit was imported eagerly by analyze_ablation — "
        "this breaks the lazy-import contract."
    )
    print("  case_lazy_import_contract: OK")


def case_revisit_flag_runs_visit_order_report():
    """--revisit routes to analyze_revisit.print_report (revisit markers present,
    phase-2 gate marker absent)."""
    with tempfile.TemporaryDirectory() as parent:
        s1_dir, s3_dir = _make_s1_s3_dirs(parent)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = analyze_ablation.main(["--revisit", "--bootstrap", "200", s1_dir, s3_dir])
        out = buf.getvalue()

    assert rc == 0, f"main() returned {rc!r}, expected 0"

    # revisit markers must be present
    assert "visit groups" in out, f"'visit groups' not in output:\n{out}"
    assert "cold vs warm" in out, f"'cold vs warm' not in output:\n{out}"
    assert "WARM S3 - S1" in out, f"'WARM S3 - S1' not in output:\n{out}"
    assert "Gate A verdict" in out, f"'Gate A verdict' not in output:\n{out}"

    # phase-2 gate marker must be ABSENT
    assert "phase 2 gate" not in out, (
        f"'phase 2 gate' should NOT appear in revisit output:\n{out}"
    )

    print("  case_revisit_flag_runs_visit_order_report: OK")


def case_no_flag_runs_phase2_gate():
    """Without --revisit, main() runs the standard phase-2 gate (phase-2 marker
    present, Gate-A verdict absent)."""
    with tempfile.TemporaryDirectory() as parent:
        s1_dir, s3_dir = _make_s1_s3_dirs(parent)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = analyze_ablation.main(["--bootstrap", "200", s1_dir, s3_dir])
        out = buf.getvalue()

    assert rc == 0, f"main() returned {rc!r}, expected 0"

    # phase-2 gate marker must be present
    assert "phase 2 gate" in out, f"'phase 2 gate' not in output:\n{out}"

    # revisit-specific marker must be ABSENT
    assert "Gate A verdict" not in out, (
        f"'Gate A verdict' should NOT appear in standard output:\n{out}"
    )

    print("  case_no_flag_runs_phase2_gate: OK")


def case_revisit_with_s2_decomposition():
    """--revisit with S1, S2, S3 emits the S2 decomposition lines."""
    with tempfile.TemporaryDirectory() as parent:
        s1_dir, s2_dir, s3_dir = _make_s1_s2_s3_dirs(parent)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = analyze_ablation.main(
                ["--revisit", "--bootstrap", "200", s1_dir, s2_dir, s3_dir]
            )
        out = buf.getvalue()

    assert rc == 0, f"main() returned {rc!r}, expected 0"
    assert "WARM S2 - S1" in out, f"'WARM S2 - S1' not in output:\n{out}"
    assert "WARM S3 - S2" in out, f"'WARM S3 - S2' not in output:\n{out}"

    print("  case_revisit_with_s2_decomposition: OK")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------


def main() -> int:
    print("analyze_ablation --revisit dispatch sanity tests")
    # NOTE: case_lazy_import_contract MUST run first — before any --revisit call
    # pulls analyze_revisit into sys.modules.
    case_lazy_import_contract()
    case_revisit_flag_runs_visit_order_report()
    case_no_flag_runs_phase2_gate()
    case_revisit_with_s2_decomposition()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
