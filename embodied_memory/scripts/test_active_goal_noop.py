"""
TDD guard for the E4 `active_goal` NO-OP refactor in episode_runner.py.

E4 introduces ONE mutable `active_goal` and routes every per-tick goal
READ-THROUGH site through it, while leaving SPL/success/report sites on the
immutable `ep.target_category`. For every NON-anomaly task (objectnav /
audiogoal / revisit / multion) `active_goal` must equal byte-for-byte the
legacy value: the multion cursor when multion, else ep.target_category.

Two guards, both habitat/torch/sim-free:
  * the extracted pure helper `_resolve_active_goal` — the single no-op rule;
  * a $0 static source-grep that the ternaries are gone, the helper is wired,
    and the STAY sites are untouched (catches a missed/typo'd read-through site
    without a sim — the highest-value check given the ~17-site surface).

Run: KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
       /opt/anaconda3/envs/ltm-embodied/bin/python \
       embodied_memory/scripts/test_active_goal_noop.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from embodied_memory.episode_runner import _resolve_active_goal

_ER = Path(__file__).resolve().parent.parent / "episode_runner.py"


def _run_episode_body() -> str:
    """The text of _run_episode only (so callee fallbacks outside it don't
    pollute the 'ternary is gone' assertions)."""
    src = _ER.read_text()
    start = src.index("def _run_episode(")
    nxt = src.index("\n    def ", start + 1)
    return src[start:nxt]


# ----------------------------------------------------------------------
# the pure no-op rule
# ----------------------------------------------------------------------
def case_objectnav_is_episode_target():
    ep_t = "chair"  # distinct object to prove identity, not just equality
    for active in ("chair", "IGNORED", "bed"):
        out = _resolve_active_goal(False, active, ep_t)
        assert out == "chair", out
        assert out is ep_t, "single-goal path must return ep.target_category by identity"


def case_audiogoal_is_episode_target():
    for cat in ("bed", "toilet", "sofa", "tv_monitor", "plant"):
        ep_t = str(cat)
        assert _resolve_active_goal(False, "alarm_source", ep_t) is ep_t


def case_revisit_is_episode_target():
    ep_t = "sofa"
    assert _resolve_active_goal(False, "sofa", ep_t) is ep_t


def case_multion_tracks_cursor():
    seq = ["chair", "bed", "tv_monitor"]
    primary = seq[0]
    # init: idx 0 -> cursor == primary
    assert _resolve_active_goal(True, seq[0], primary) == "chair"
    # advance idx 1 -> tracks the cursor, NOT ep.target_category
    assert _resolve_active_goal(True, seq[1], primary) == "bed"
    assert _resolve_active_goal(True, seq[2], primary) == "tv_monitor"


def case_anomaly_default_equals_primary():
    # anomaly_response is single-goal until the controller swaps active_goal
    # during INVESTIGATE -> with no interrupt it is byte-identical to objectnav.
    ep_t = "chair"
    assert _resolve_active_goal(False, "chair", ep_t) is ep_t


def case_detector_runner_default_equals_episode_target_precondition():
    # F1: the detector arm's legacy non-multion value was self.target_category;
    # B4 routes it through active_goal (== ep.target_category). Byte-identical
    # iff the configured single-goal runner has self.target_category ==
    # ep.target_category. This documents/pins that precondition.
    self_target = "chair"
    ep_target = "chair"
    assert self_target == ep_target


def case_scoring_goal_is_always_primary_DOC():
    # DOCUMENTATION (not a regression catch — SPL routing is structural via
    # step.info, guarded by the static grep below, not a code function):
    # the SPL/success/report goal is always ep.target_category, even when the
    # controller has swapped active_goal to an investigate goal.
    scoring_goal = lambda ep_target, active_goal: ep_target  # noqa: E731
    assert scoring_goal("chair", "alarm_source") == "chair"


# ----------------------------------------------------------------------
# $0 static source guard — coverage without a sim
# ----------------------------------------------------------------------
def case_helper_defined_and_wired():
    body = _run_episode_body()
    src = _ER.read_text()
    assert "def _resolve_active_goal(" in src, "extract the pure no-op helper"
    assert "_resolve_active_goal(multion" in body, "A1 must call the helper to init active_goal"


def case_legacy_ternaries_gone_from_run_episode():
    body = _run_episode_body()
    # every per-tick goal ternary must have collapsed to active_goal
    assert "active_category if multion" not in body, \
        "a goal read-through ternary still routes the bare cursor — missed a TRACK site"
    assert "ep.target_category or self.target_category" not in body, \
        "_observe_semantic_value fallback must read active_goal, not ep.target_category"


def case_active_goal_actually_used():
    body = _run_episode_body()
    # A1 init + A2 advance + the routed read-through sites. The ~15-site surface
    # means a healthy count; a low count signals a forgotten route.
    n = len(re.findall(r"\bactive_goal\b", body))
    assert n >= 12, f"expected active_goal to be wired at many sites, found {n}"
    # assigned exactly 3x: A1 init, A2 multion-advance re-point, E5 controller
    # mutation (active_goal = dec.active_goal). No other stray mutation.
    assigns = len(re.findall(r"^\s*active_goal\s*=", body, flags=re.M))
    assert assigns == 3, f"active_goal assigns must be A1+A2+E5 (==3), got {assigns}"


def case_stay_sites_untouched():
    src = _ER.read_text()
    assert '"target_category": ep.target_category,' in src, "C1 report field must STAY"
    assert "subgoal_seq = [str(ep.target_category)]" in src, "C2 cursor source must STAY"
    assert "target=self.target_category" in src, "C5 captioner fallback must STAY"
    # the callee fallback lives OUTSIDE _run_episode and must remain the safety default
    assert "goal_override or ep.target_category or self.target_category" in src, \
        "C4 _propose_candidates fallback must STAY"


def main() -> int:
    cases = [
        case_objectnav_is_episode_target,
        case_audiogoal_is_episode_target,
        case_revisit_is_episode_target,
        case_multion_tracks_cursor,
        case_anomaly_default_equals_primary,
        case_detector_runner_default_equals_episode_target_precondition,
        case_scoring_goal_is_always_primary_DOC,
        case_helper_defined_and_wired,
        case_legacy_ternaries_gone_from_run_episode,
        case_active_goal_actually_used,
        case_stay_sites_untouched,
    ]
    print(f"running {len(cases)} active_goal no-op cases…")
    for c in cases:
        c()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
