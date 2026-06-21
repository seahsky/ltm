"""
TDD for ``check_seed_pose`` — the renumbering-invariant pose verifier that the
non-LOS Tier-3 gate uses to HARD-assert the captioned episode IS the cold seed.

Background (2026-06-21): habitat overwrites ``episode_id`` with ``str(load_index)``
and its iterator default ``shuffle=True`` meant a 1-episode caption run could grab a
random WARM pose instead of the cold seed at index 0. The old driver only WARNed on
``"cold-0" not in episode_id`` — which (a) can never be true after renumbering and
(b) is not a geometry check anyway. This helper instead compares the captioned
``start_position`` (now in summary.json) against the seed ``start_position`` read
straight from the gate-built content file — pure, stdlib-only, sim-free.

Invoke with::

    python embodied_memory/scripts/test_check_seed_pose.py
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import check_seed_pose as csp  # noqa: E402


SEED_XYZ = [0.7821, -0.0051, -5.1784]
WARM_XYZ = [2.3737, -0.0051, -6.8049]


def _write_content(path, episodes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"episodes": episodes}, f)


def _seed_content():
    return [
        {"episode_id": "chair-glass_break-cold-0", "object_category": "chair",
         "start_position": list(SEED_XYZ),
         "info": {"anomaly_class": "glass_break"}},
        {"episode_id": "chair-glass_break-warm-1", "object_category": "chair",
         "start_position": list(WARM_XYZ),
         "info": {"anomaly_class": "glass_break"}},
    ]


# ---- poses_match -----------------------------------------------------------

def case_poses_match_exact():
    assert csp.poses_match(SEED_XYZ, list(SEED_XYZ), eps=0.05) is True
    print("  case poses_match_exact: OK")


def case_poses_match_within_eps():
    near = [SEED_XYZ[0] + 0.01, SEED_XYZ[1] - 0.01, SEED_XYZ[2] + 0.02]
    assert csp.poses_match(SEED_XYZ, near, eps=0.05) is True
    print("  case poses_match_within_eps: OK")


def case_poses_mismatch_outside_eps():
    # The actual failure mode: a warm pose ~2 m away must NOT match.
    assert csp.poses_match(SEED_XYZ, WARM_XYZ, eps=0.05) is False
    print("  case poses_mismatch_outside_eps: OK")


def case_poses_match_none_is_false():
    assert csp.poses_match(None, SEED_XYZ, eps=0.05) is False
    assert csp.poses_match(SEED_XYZ, None, eps=0.05) is False
    assert csp.poses_match([1.0, 2.0], SEED_XYZ, eps=0.05) is False  # wrong dim
    print("  case poses_match_none_is_false: OK")


# ---- read_seed_start_position ---------------------------------------------

def case_reads_seed_from_content():
    with tempfile.TemporaryDirectory() as d:
        cp = os.path.join(d, "content", "wcojb4TFT35.json.gz")
        _write_content(cp, _seed_content())
        got = csp.read_seed_start_position(cp)
        assert csp.poses_match(got, SEED_XYZ, eps=1e-6), got
        print("  case reads_seed_from_content: OK")


def case_reads_seed_filtered_by_class():
    # Two cold seeds for different classes — the class filter selects the right one.
    eps = [
        {"episode_id": "chair-alarm-cold-0", "start_position": [9.0, 0.0, 9.0],
         "info": {"anomaly_class": "alarm"}},
        {"episode_id": "chair-glass_break-cold-0", "start_position": list(SEED_XYZ),
         "info": {"anomaly_class": "glass_break"}},
    ]
    with tempfile.TemporaryDirectory() as d:
        cp = os.path.join(d, "content", "s.json.gz")
        _write_content(cp, eps)
        got = csp.read_seed_start_position(cp, anomaly_class="glass_break")
        assert csp.poses_match(got, SEED_XYZ, eps=1e-6), got
        print("  case reads_seed_filtered_by_class: OK")


def case_no_seed_raises():
    eps = [{"episode_id": "chair-glass_break-warm-1", "start_position": list(WARM_XYZ),
            "info": {"anomaly_class": "glass_break"}}]
    with tempfile.TemporaryDirectory() as d:
        cp = os.path.join(d, "content", "s.json.gz")
        _write_content(cp, eps)
        try:
            csp.read_seed_start_position(cp)
        except csp.NoSeedError:
            print("  case no_seed_raises: OK")
            return
        raise AssertionError("expected NoSeedError when no *-cold-0 episode present")


# ---- resolve_content_path (top -> content) --------------------------------

def case_resolve_content_path_from_top():
    with tempfile.TemporaryDirectory() as d:
        top = os.path.join(d, "audiogoal.json.gz")
        cp = os.path.join(d, "content", "wcojb4TFT35.json.gz")
        _write_content(cp, _seed_content())
        with gzip.open(top, "wt", encoding="utf-8") as f:
            json.dump({"episodes": []}, f)
        got = csp.resolve_content_path(top, "wcojb4TFT35")
        assert os.path.abspath(got) == os.path.abspath(cp), got
        print("  case resolve_content_path_from_top: OK")


def case_resolve_content_path_passthrough():
    # If already a content file (…/content/<scene>.json.gz), return as-is.
    with tempfile.TemporaryDirectory() as d:
        cp = os.path.join(d, "content", "wcojb4TFT35.json.gz")
        _write_content(cp, _seed_content())
        got = csp.resolve_content_path(cp, "wcojb4TFT35")
        assert os.path.abspath(got) == os.path.abspath(cp), got
        print("  case resolve_content_path_passthrough: OK")


# ---- CLI exit codes (the contract the driver depends on) -------------------

def case_cli_seed_match_exit0():
    with tempfile.TemporaryDirectory() as d:
        cp = os.path.join(d, "content", "wcojb4TFT35.json.gz")
        _write_content(cp, _seed_content())
        rc = csp.main(["--content", cp, "--captioned-xyz",
                       "%s,%s,%s" % tuple(SEED_XYZ), "--eps", "0.05"])
        assert rc == 0, rc
        print("  case cli_seed_match_exit0: OK")


def case_cli_wrong_pose_exit1():
    with tempfile.TemporaryDirectory() as d:
        cp = os.path.join(d, "content", "wcojb4TFT35.json.gz")
        _write_content(cp, _seed_content())
        # captioned the WARM pose — the exact production failure → exit 1.
        rc = csp.main(["--content", cp, "--captioned-xyz",
                       "%s,%s,%s" % tuple(WARM_XYZ), "--eps", "0.05"])
        assert rc == 1, rc
        print("  case cli_wrong_pose_exit1: OK")


def case_cli_missing_captioned_field_exit2():
    # summary.json with a row that has NO start_position → cannot verify → exit 2.
    with tempfile.TemporaryDirectory() as d:
        cp = os.path.join(d, "content", "s.json.gz")
        _write_content(cp, _seed_content())
        sm = os.path.join(d, "summary.json")
        with open(sm, "w", encoding="utf-8") as f:
            json.dump({"episodes": [{"episode_id": "2"}]}, f)
        rc = csp.main(["--content", cp, "--summary", sm])
        assert rc == 2, rc
        print("  case cli_missing_captioned_field_exit2: OK")


def case_cli_check_seed_exists_exit0():
    with tempfile.TemporaryDirectory() as d:
        cp = os.path.join(d, "content", "wcojb4TFT35.json.gz")
        _write_content(cp, _seed_content())
        rc = csp.main(["--content", cp, "--check-seed-exists"])
        assert rc == 0, rc
        print("  case cli_check_seed_exists_exit0: OK")


def case_cli_check_seed_exists_missing_exit2():
    eps = [{"episode_id": "chair-glass_break-warm-1", "start_position": list(WARM_XYZ),
            "info": {"anomaly_class": "glass_break"}}]
    with tempfile.TemporaryDirectory() as d:
        cp = os.path.join(d, "content", "s.json.gz")
        _write_content(cp, eps)
        rc = csp.main(["--content", cp, "--check-seed-exists"])
        assert rc == 2, rc
        print("  case cli_check_seed_exists_missing_exit2: OK")


def case_cli_summary_match_exit0():
    with tempfile.TemporaryDirectory() as d:
        cp = os.path.join(d, "content", "s.json.gz")
        _write_content(cp, _seed_content())
        sm = os.path.join(d, "summary.json")
        with open(sm, "w", encoding="utf-8") as f:
            json.dump({"episodes": [{"episode_id": "2",
                                     "start_position": list(SEED_XYZ)}]}, f)
        rc = csp.main(["--content", cp, "--summary", sm, "--eps", "0.05"])
        assert rc == 0, rc
        print("  case cli_summary_match_exit0: OK")


def main() -> int:
    print("check_seed_pose TDD")
    case_poses_match_exact()
    case_poses_match_within_eps()
    case_poses_mismatch_outside_eps()
    case_poses_match_none_is_false()
    case_reads_seed_from_content()
    case_reads_seed_filtered_by_class()
    case_no_seed_raises()
    case_resolve_content_path_from_top()
    case_resolve_content_path_passthrough()
    case_cli_seed_match_exit0()
    case_cli_wrong_pose_exit1()
    case_cli_missing_captioned_field_exit2()
    case_cli_check_seed_exists_exit0()
    case_cli_check_seed_exists_missing_exit2()
    case_cli_summary_match_exit0()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
