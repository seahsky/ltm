"""
diagnose_audio_doa_calib — the $0 pre-flight gate for the audio-DOA rerank head
(``LTM_AUDIO_DOA``, stage S2). Forks the diagnose-first discipline of
``diagnose_sbert_cosines.py``: MEASURE on logged episodes whether the proposed
audio→geometry disambiguation head *can* help at all BEFORE spending a RACE
matrix building it.

The head boosts the same-category memory candidate whose stored bearing agrees
in lateral sign with the heard ILD direction. Two verifier-flagged blockers must
hold for that to ever work; this script measures all three signals and emits one
RECOMMEND verdict:

  * RECALL-GAP   — the CORRECT instance is rarely in the recalled candidate set
                   (a read-side reorderer cannot resurrect an absent instance).
  * FRAME-BROKEN — the heard lateral sign does not agree (under either sign
                   convention) with the bearing to the GT source better than
                   chance — the RIR grid stores no render yaw, so lateral_sign
                   (render frame) and the candidate bearing (rotating agent
                   frame) are mis-aligned → re-render with recorded yaw (S1b).
  * CO-LINEAR    — correct and wrong same-category instances sit on the SAME
                   lateral side, so a left/right DOA cue cannot separate them.
  * GO           — all three pass: build the S2 head.
  * INSUFFICIENT-DATA — the logs lack the instrumented fields
                   (audio_lateral_sign / source_position / target_position);
                   re-run a small batch with the S0 instrumentation on.

It also reports WHICH sign convention is consistent (agreeA: heard==right(rel);
agreeB: heard==-right(rel)) — empirically pinning the convention the verifiers
flagged as fatal-if-assumed (frontier_planner +rel→TURN_LEFT vs audio +1→right).

GT source_position / target_position are used ONLY here as OFFLINE labels — never
at runtime (the live head uses only agent-estimable last_lateral / last_energy).

Run (after a batch with S0 instrumentation has produced episode_*.json)::

    python3 embodied_memory/scripts/diagnose_audio_doa_calib.py runs/m3-* runs/audiodoa-*
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# ---- thresholds (documented; the RECOMMEND verdict gates on these) ----------
MIN_PRESENCE = 0.50   # correct instance must be recalled in >= this fraction of fires
MIN_FRAME = 0.60      # heard-sign vs source-bearing agreement must beat ~chance
MIN_SEP = 0.30        # correct/wrong on opposite sides in >= this fraction of eligible
NEAR_M = 1.5          # a memory candidate within this of the GT goal = the correct instance
DRIFT_SPLIT_DEG = 45.0  # low- vs high-yaw-drift split for the frame-decay report


# ---- pure geometry helpers (unit-tested) ------------------------------------
def _norm_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def bearing_agent_frame(agent_pos, agent_yaw: float, target_xz) -> float:
    """Relative bearing (rad) of ``target_xz`` (x, z) from the agent, in the SAME
    convention as ``memory_bridge`` FrontierCandidate.bearing_rad:
    ``world_bearing = atan2(dx, dz)`` then ``rel = world_bearing - agent_yaw``.
    ``agent_pos`` is the (x, y, z) agent position."""
    dx = float(target_xz[0]) - float(agent_pos[0])
    dz = float(target_xz[1]) - float(agent_pos[2])
    return _norm_angle(math.atan2(dx, dz) - float(agent_yaw))


def right_sign_from_bearing(rel: float, eps: float = 1e-3) -> int:
    """+1 if the point is on the agent's RIGHT (matching ``audio.lateral_sign``'s
    +1 = right), -1 if LEFT, 0 if ~ahead/behind. In this codebase
    ``frontier_planner._bearing_to_action`` maps +rel → TURN_LEFT, so a positive
    ``sin(rel)`` is LEFT and RIGHT is ``sin(rel) < 0``. (Reported under BOTH
    hypotheses by the frame metric so the convention is measured, not assumed.)"""
    s = math.sin(rel)
    if abs(s) < eps:
        return 0
    return 1 if s < 0 else -1


def _memory_cands(decision: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [c for c in decision.get("candidates", []) if c.get("source") == "memory"]


def decision_pose(decision: Dict[str, Any], steps_by_idx: Dict[int, Dict[str, Any]]):
    """(agent_pos, agent_yaw, audio_lateral_sign) at a decision, PREFERRING the
    decision-level fields (logged at the audio-processed decision step) and
    falling back to the joined keyframe step for back-compat with older logs.
    Older logs without decision-level fields only join when the decision happens
    to land on a (sparse) keyframe step."""
    s = steps_by_idx.get(decision.get("step_idx")) or {}
    pos = decision.get("agent_pos", s.get("agent_pos"))
    yaw = decision.get("agent_yaw", s.get("agent_yaw"))
    sign = decision.get("audio_lateral_sign", s.get("audio_lateral_sign"))
    return pos, yaw, sign


def _dist_xz(world_xy, target_xz) -> float:
    return math.hypot(float(world_xy[0]) - float(target_xz[0]),
                      float(world_xy[1]) - float(target_xz[1]))


def correct_present(decision: Dict[str, Any], target_xz, near_m: float = NEAR_M) -> bool:
    """Is a memory candidate within ``near_m`` of the GT goal present in the
    recalled set? (The 'correct instance present' precondition.)"""
    return any(_dist_xz(c["world_xy"], target_xz) <= near_m for c in _memory_cands(decision))


def opposite_side_present(decision, agent_pos, agent_yaw, target_xz,
                          near_m: float = NEAR_M) -> bool:
    """True iff the recalled memory set has a CORRECT candidate (near the GT goal)
    AND a WRONG one (far) on OPPOSITE agent-relative lateral sides — the geometry
    a left/right DOA cue can exploit. Same-side/co-linear → False."""
    correct_sides, wrong_sides = [], []
    for c in _memory_cands(decision):
        side = right_sign_from_bearing(bearing_agent_frame(agent_pos, agent_yaw, c["world_xy"]))
        if side == 0:
            continue
        if _dist_xz(c["world_xy"], target_xz) <= near_m:
            correct_sides.append(side)
        else:
            wrong_sides.append(side)
    return any(cs != ws for cs in correct_sides for ws in wrong_sides)


# ---- per-episode + aggregate analysis ---------------------------------------
def analyze_episode(ep: Dict[str, Any], near_m: float = NEAR_M) -> Dict[str, Any]:
    """Counters for one episode_*.json dict (see episode_runner ep_log)."""
    steps_by_idx = {s["step_idx"]: s for s in ep.get("steps", [])}
    src = ep.get("source_position")           # (x,y,z) GT, offline label only
    tgt = ep.get("target_position")           # (x,y,z) GT goal, offline label only
    src_xz = (src[0], src[2]) if src else None
    tgt_xz = (tgt[0], tgt[2]) if tgt else None
    start_yaw = ep["steps"][0]["agent_yaw"] if ep.get("steps") else None

    a = {"fire_decisions": 0, "correct_present": 0,
         "sep_eligible": 0, "sep_opposite": 0,
         "frame_steps": 0, "agreeA": 0, "agreeB": 0,
         "frame_lowdrift": 0, "agree_lowdrift": 0,
         "frame_highdrift": 0, "agree_highdrift": 0,
         "has_src": src is not None, "has_tgt": tgt is not None,
         "has_audio_sign": False}

    for d in ep.get("decisions", []):
        mc = _memory_cands(d)
        if not mc:
            continue
        a["fire_decisions"] += 1
        pos, yaw, sign = decision_pose(d, steps_by_idx)
        if tgt_xz is not None and correct_present(d, tgt_xz, near_m):
            a["correct_present"] += 1
        if tgt_xz is not None and pos is not None and yaw is not None:
            a["sep_eligible"] += 1
            if opposite_side_present(d, pos, yaw, tgt_xz, near_m):
                a["sep_opposite"] += 1
        if src_xz is not None and pos is not None and yaw is not None and sign not in (None, 0):
            a["has_audio_sign"] = True
            rel = bearing_agent_frame(pos, yaw, src_xz)
            rs = right_sign_from_bearing(rel)
            if rs != 0:
                heard = int(sign > 0) - int(sign < 0)
                a["frame_steps"] += 1
                agree = int(heard == rs)
                a["agreeA"] += agree
                a["agreeB"] += int(heard == -rs)
                if start_yaw is not None:
                    drift = abs(math.degrees(_norm_angle(yaw - start_yaw)))
                    if drift < DRIFT_SPLIT_DEG:
                        a["frame_lowdrift"] += 1
                        a["agree_lowdrift"] += agree
                    else:
                        a["frame_highdrift"] += 1
                        a["agree_highdrift"] += agree
    return a


def aggregate(episodes: List[Dict[str, Any]], near_m: float = NEAR_M) -> Dict[str, Any]:
    keys = ["fire_decisions", "correct_present", "sep_eligible", "sep_opposite",
            "frame_steps", "agreeA", "agreeB", "frame_lowdrift", "agree_lowdrift",
            "frame_highdrift", "agree_highdrift"]
    agg = {k: 0 for k in keys}
    agg.update({"has_src_any": False, "has_tgt_any": False, "has_audio_sign_any": False,
                "n_episodes": 0})
    for ep in episodes:
        a = analyze_episode(ep, near_m)
        for k in keys:
            agg[k] += a[k]
        agg["has_src_any"] = agg["has_src_any"] or a["has_src"]
        agg["has_tgt_any"] = agg["has_tgt_any"] or a["has_tgt"]
        agg["has_audio_sign_any"] = agg["has_audio_sign_any"] or a["has_audio_sign"]
        agg["n_episodes"] += 1
    return agg


def recommend(agg: Dict[str, Any], *, min_presence: float = MIN_PRESENCE,
              min_frame: float = MIN_FRAME, min_sep: float = MIN_SEP
              ) -> Tuple[str, str]:
    """Map aggregate counters → (VERDICT, reason). Priority: data-availability →
    recall-gap → frame → co-linear → GO."""
    if agg["fire_decisions"] == 0:
        return "INSUFFICIENT-DATA", "no memory-firing decisions in the logs"
    if not agg["has_tgt_any"] or not agg["has_src_any"]:
        return ("INSUFFICIENT-DATA",
                "source_position/target_position absent — re-run a batch with the S0 instrumentation")
    presence = agg["correct_present"] / agg["fire_decisions"]
    if presence < min_presence:
        return ("RECALL-GAP",
                f"correct instance recalled in only {presence:.0%} of fires (< {min_presence:.0%}) "
                f"— a read-side reorderer cannot resurrect an absent instance")
    if not agg["has_audio_sign_any"]:
        return ("INSUFFICIENT-DATA",
                "audio_lateral_sign absent — re-run a batch with the S0 instrumentation")
    frameA = agg["agreeA"] / max(agg["frame_steps"], 1)
    frameB = agg["agreeB"] / max(agg["frame_steps"], 1)
    frame = max(frameA, frameB)
    sign_hyp = "heard==right(rel)" if frameA >= frameB else "heard==-right(rel) (INVERTED)"
    if frame < min_frame:
        return ("FRAME-BROKEN",
                f"heard-sign agreement {frame:.0%} ~ chance (< {min_frame:.0%}) under both "
                f"conventions — lateral_sign (render frame) vs candidate bearing (agent frame) "
                f"mis-aligned; re-render the RIR grid with recorded listener yaw (S1b)")
    sep = agg["sep_opposite"] / max(agg["sep_eligible"], 1)
    if sep < min_sep:
        return ("CO-LINEAR",
                f"correct/wrong instances on opposite sides in only {sep:.0%} of eligible fires "
                f"(< {min_sep:.0%}) — a left/right DOA cue cannot separate same-side instances")
    return ("GO",
            f"presence {presence:.0%}, frame-agree {frame:.0%} ({sign_hyp}), separation {sep:.0%} "
            f"— build the S2 audio-DOA head with this sign convention")


def _load_run_dirs(patterns: List[str]) -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    for pat in patterns:
        for run in sorted(glob.glob(pat)):
            for ep_path in sorted(glob.glob(os.path.join(run, "episode_*.json"))):
                if ep_path.endswith("_error.json"):
                    continue
                try:
                    with open(ep_path, "r", encoding="utf-8") as f:
                        episodes.append(json.load(f))
                except (OSError, json.JSONDecodeError):
                    continue
    return episodes


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: diagnose_audio_doa_calib.py <run_dir_glob> [run_dir_glob ...]")
        return 2
    episodes = _load_run_dirs(argv)
    if not episodes:
        print(f"[audio-doa-calib] no episode_*.json under {argv}")
        return 2
    agg = aggregate(episodes)
    verdict, reason = recommend(agg)

    print(f"[audio-doa-calib] {agg['n_episodes']} episodes, "
          f"{agg['fire_decisions']} memory-firing decisions")
    print(f"  instrumented fields: source_position={agg['has_src_any']} "
          f"target_position={agg['has_tgt_any']} audio_lateral_sign={agg['has_audio_sign_any']}")
    if agg["fire_decisions"]:
        print(f"  recall presence : {agg['correct_present']}/{agg['fire_decisions']} "
              f"= {agg['correct_present'] / agg['fire_decisions']:.0%}")
    if agg["frame_steps"]:
        print(f"  frame agreement : A(heard==right) {agg['agreeA']}/{agg['frame_steps']} "
              f"= {agg['agreeA'] / agg['frame_steps']:.0%}  |  "
              f"B(inverted) {agg['agreeB']}/{agg['frame_steps']} "
              f"= {agg['agreeB'] / agg['frame_steps']:.0%}")
        if agg["frame_lowdrift"] and agg["frame_highdrift"]:
            print(f"    yaw-drift decay: <{DRIFT_SPLIT_DEG:.0f}deg "
                  f"{agg['agree_lowdrift'] / agg['frame_lowdrift']:.0%}  |  "
                  f">={DRIFT_SPLIT_DEG:.0f}deg "
                  f"{agg['agree_highdrift'] / agg['frame_highdrift']:.0%}")
    if agg["sep_eligible"]:
        print(f"  lateral separation: {agg['sep_opposite']}/{agg['sep_eligible']} "
              f"= {agg['sep_opposite'] / agg['sep_eligible']:.0%}")
    print(f"\nRECOMMEND: {verdict} — {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
