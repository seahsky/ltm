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
                   EVEN at a generous radius (genuinely absent; a read-side
                   reorderer cannot resurrect it).
  * PRESENCE-OFFSET — recalled candidates ARE near the goal but sit just outside
                   the tight object-CENTER radius because a stored candidate is a
                   navigable VIEW_POINT (the agent's caption-time pose), offset
                   ~1.5-2m from the object center for large objects. This is a
                   reference-frame artifact, NOT an absent instance — re-measure
                   to the nearest goal view_point. A query/rerank fix is NOT
                   excluded by this. (correct_present at NEAR_M is anchored to the
                   OBJECT CENTER, whereas the success metric distance_to_goal is
                   geodesic-to-VIEW_POINT — the two disagree by exactly this offset.)
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
NEAR_M = 1.5          # a memory candidate within this of the GT goal CENTER = "present" (object-center anchor)
DRIFT_SPLIT_DEG = 45.0  # low- vs high-yaw-drift split for the frame-decay report
# Presence SWEEP: a recalled candidate is a VIEW_POINT (caption-time agent pose),
# offset ~1.5-2m from the OBJECT CENTER for large objects (bed/sofa). Measuring
# presence to the object center at a single 1.5m radius therefore mislabels a
# CORRECT view-pose recall as "absent". We report presence at several radii so a
# rise with radius reveals the offset artifact, and gate the RECALL-GAP verdict on
# the LARGEST radius (genuine absence = sparse even there).
PRESENCE_RADII = (1.5, 2.5, 3.5)
RADIUS_KEYS = {1.5: "cp_15", 2.5: "cp_25", 3.5: "cp_35"}
_GATE_RADIUS = 3.5    # presence at this radius gates RECALL-GAP vs PRESENCE-OFFSET


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

    a = {k: 0 for k in (
        "fire_decisions", "correct_present", "cp_15", "cp_25", "cp_35",
        "sep_eligible", "sep_opp_agent", "sep_opp_world",
        "frame_steps_agent", "frame_steps_world", "agreeA", "agreeB", "agreeWA", "agreeWB",
        "frame_lowdrift", "agree_lowdrift", "frame_highdrift", "agree_highdrift")}
    a.update({"has_src": src is not None, "has_tgt": tgt is not None, "has_audio_sign": False})

    for d in ep.get("decisions", []):
        if not _memory_cands(d):
            continue
        a["fire_decisions"] += 1
        pos, yaw, sign = decision_pose(d, steps_by_idx)
        if tgt_xz is not None and correct_present(d, tgt_xz, near_m):
            a["correct_present"] += 1
        # Presence SWEEP: min candidate distance to the OBJECT CENTER, counted at
        # several radii so a rise with radius exposes the view_point->center offset
        # artifact (a correct view-pose recall sits ~1.5-2m out for large objects).
        if tgt_xz is not None:
            cands = _memory_cands(d)
            if cands:
                mind = min(_dist_xz(c["world_xy"], tgt_xz) for c in cands)
                for r, k in RADIUS_KEYS.items():
                    if mind <= r:
                        a[k] += 1
        if tgt_xz is not None and pos is not None and yaw is not None:
            a["sep_eligible"] += 1
            # separation in the agent frame AND the world frame (render is identity-
            # oriented, render_rir_grid.py sets only st.position) so the audio cue is
            # a WORLD-frame left/right — see the frame agreement below.
            if opposite_side_present(d, pos, yaw, tgt_xz, near_m):
                a["sep_opp_agent"] += 1
            if opposite_side_present(d, pos, 0.0, tgt_xz, near_m):
                a["sep_opp_world"] += 1
        if src_xz is not None and pos is not None and yaw is not None and sign not in (None, 0):
            a["has_audio_sign"] = True
            heard = int(sign > 0) - int(sign < 0)
            # The RIR grid is rendered with the listener at IDENTITY orientation
            # (render_rir_grid.py:243 sets st.position only) → lateral_sign lives in
            # the WORLD frame, not the agent's rotating frame. Test BOTH so the
            # verdict tells "wrong comparison frame (free fix → use world bearing)"
            # apart from "RIR can't localize (needs an S1b re-render)".
            rs_agent = right_sign_from_bearing(bearing_agent_frame(pos, yaw, src_xz))
            rs_world = right_sign_from_bearing(bearing_agent_frame(pos, 0.0, src_xz))
            if rs_agent != 0:
                a["frame_steps_agent"] += 1
                aa = int(heard == rs_agent)
                a["agreeA"] += aa
                a["agreeB"] += int(heard == -rs_agent)
                if start_yaw is not None:
                    drift = abs(math.degrees(_norm_angle(yaw - start_yaw)))
                    if drift < DRIFT_SPLIT_DEG:
                        a["frame_lowdrift"] += 1
                        a["agree_lowdrift"] += aa
                    else:
                        a["frame_highdrift"] += 1
                        a["agree_highdrift"] += aa
            if rs_world != 0:
                a["frame_steps_world"] += 1
                a["agreeWA"] += int(heard == rs_world)
                a["agreeWB"] += int(heard == -rs_world)
    return a


def aggregate(episodes: List[Dict[str, Any]], near_m: float = NEAR_M) -> Dict[str, Any]:
    keys = ["fire_decisions", "correct_present", "cp_15", "cp_25", "cp_35",
            "sep_eligible", "sep_opp_agent", "sep_opp_world",
            "frame_steps_agent", "frame_steps_world", "agreeA", "agreeB", "agreeWA", "agreeWB",
            "frame_lowdrift", "agree_lowdrift", "frame_highdrift", "agree_highdrift"]
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
    # Presence SWEEP gate: a recalled candidate is a VIEW_POINT offset from the
    # OBJECT CENTER, so a low presence at the tight NEAR_M radius does NOT prove
    # the instance is absent. Only declare RECALL-GAP when presence is low EVEN at
    # the generous _GATE_RADIUS; otherwise the candidates are present-but-offset
    # (a reference-frame artifact) -> PRESENCE-OFFSET, and a query/rerank fix is
    # NOT excluded.
    fd = agg["fire_decisions"]
    presence_gate = agg.get("cp_35", 0) / fd if fd else 0.0
    if presence < min_presence:
        if presence_gate >= min_presence:
            return ("PRESENCE-OFFSET",
                    f"presence at {NEAR_M}m-to-OBJECT-CENTER is only {presence:.0%}, but rises to "
                    f"{presence_gate:.0%} at {_GATE_RADIUS}m — the recalled candidates are PRESENT "
                    f"but OFFSET (they are view_points ~1.5-2m off the object center), NOT absent. "
                    f"This is a reference-frame artifact (the success metric distance_to_goal is "
                    f"geodesic-to-view_point, a different anchor). Re-measure presence to the nearest "
                    f"goal view_point before concluding a recall gap; a query/rerank fix is NOT "
                    f"excluded by this diagnostic.")
        return ("RECALL-GAP",
                f"correct instance recalled in only {presence:.0%} of fires at {NEAR_M}m AND only "
                f"{presence_gate:.0%} even at {_GATE_RADIUS}m (< {min_presence:.0%}) — genuinely sparse "
                f"in the recalled set (a read-side reorderer cannot resurrect an absent instance)")
    if not agg["has_audio_sign_any"]:
        return ("INSUFFICIENT-DATA",
                "audio_lateral_sign absent — re-run a batch with the S0 instrumentation")
    if agg["frame_steps_agent"] == 0 and agg["frame_steps_world"] == 0:
        return ("INSUFFICIENT-DATA",
                "source always abeam/behind — no lateral side to test the heard sign against")
    na = max(agg["frame_steps_agent"], 1)
    nw = max(agg["frame_steps_world"], 1)
    # Test BOTH frames (agent / world=identity-render) × BOTH sign conventions;
    # the audio cue is a WORLD-frame left/right, so the world frame is expected
    # to win unless a future S1b re-render moves it to the agent frame.
    combos = [
        ("agent", "heard==right(agent-bearing)", agg["agreeA"] / na),
        ("agent", "heard==-right(agent-bearing) [INVERTED]", agg["agreeB"] / na),
        ("world", "heard==right(world-bearing)", agg["agreeWA"] / nw),
        ("world", "heard==-right(world-bearing) [INVERTED]", agg["agreeWB"] / nw),
    ]
    best_frame, best_sign, frame = max(combos, key=lambda c: c[2])
    if frame < min_frame:
        return ("FRAME-BROKEN",
                f"heard-sign agreement {frame:.0%} ~ chance (< {min_frame:.0%}) under BOTH the world "
                f"(identity-render) and agent frame x both signs — the offline RIR cannot supply a "
                f"usable source bearing here; re-render with recorded listener yaw + re-certify "
                f"lateral-sign under varied yaw (S1b), else the head is a structural honest-negative")
    sep_count = agg["sep_opp_world"] if best_frame == "world" else agg["sep_opp_agent"]
    sep = sep_count / max(agg["sep_eligible"], 1)
    if sep < min_sep:
        return ("CO-LINEAR",
                f"frame OK ({best_frame} frame, {best_sign}, {frame:.0%}) but correct/wrong instances "
                f"on opposite sides in only {sep:.0%} of fires (< {min_sep:.0%}) — a left/right DOA cue "
                f"cannot separate same-side instances")
    return ("GO",
            f"presence {presence:.0%}, frame-agree {frame:.0%} in the {best_frame.upper()} frame "
            f"({best_sign}), separation {sep:.0%} — build the S2 head using the {best_frame} frame "
            f"and this sign convention")


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
    if agg["fire_decisions"] and agg["has_tgt_any"]:
        fd = agg["fire_decisions"]
        print(f"  recall presence : {agg['correct_present']}/{fd} "
              f"= {agg['correct_present'] / fd:.0%}   (<= {NEAR_M}m to OBJECT CENTER)")
        sweep = "  presence sweep  : " + "  ".join(
            f"<={r}m {agg.get(k, 0)}/{fd}={agg.get(k, 0) / fd:.0%}" for r, k in RADIUS_KEYS.items())
        print(sweep)
        print("    NOTE: a recalled candidate is a VIEW_POINT (caption-time pose), offset ~1.5-2m from")
        print("    the OBJECT CENTER for large objects -> presence rising with radius = OFFSET ARTIFACT,")
        print("    not an absent instance. Success metric (succ@1m) is geodesic-to-view_point (other anchor).")
        print("    Decisive re-measure: anchor presence to the nearest goal view_point.")
    elif agg["fire_decisions"]:
        # No GT labels in these logs: the presence/sweep counters are 0 by absence
        # of a reference point, NOT because candidates are far — printing "0%" here
        # would be misleading. Say so explicitly.
        print(f"  recall presence : n/a ({agg['fire_decisions']} fires, but no source/target "
              f"GT labels in these logs — presence is unmeasurable)")
        print("    run on S0-instrumented logs (source_position/target_position present), e.g.")
        print("    the query-expansion run dirs: diagnose_audio_doa_calib.py runs/m3q-*")
    if agg["frame_steps_agent"]:
        na = agg["frame_steps_agent"]
        print(f"  frame agree (AGENT, n={na}): A {agg['agreeA']}/{na}={agg['agreeA']/na:.0%}  |  "
              f"B(inv) {agg['agreeB']}/{na}={agg['agreeB']/na:.0%}")
        if agg["frame_lowdrift"] and agg["frame_highdrift"]:
            print(f"    agent yaw-drift decay: <{DRIFT_SPLIT_DEG:.0f}deg "
                  f"{agg['agree_lowdrift'] / agg['frame_lowdrift']:.0%}  |  "
                  f">={DRIFT_SPLIT_DEG:.0f}deg "
                  f"{agg['agree_highdrift'] / agg['frame_highdrift']:.0%}")
    if agg["frame_steps_world"]:
        nw = agg["frame_steps_world"]
        print(f"  frame agree (WORLD, n={nw}): A {agg['agreeWA']}/{nw}={agg['agreeWA']/nw:.0%}  |  "
              f"B(inv) {agg['agreeWB']}/{nw}={agg['agreeWB']/nw:.0%}  "
              f"<- audio is a world-frame cue (identity render)")
    if agg["sep_eligible"]:
        print(f"  lateral separation: agent {agg['sep_opp_agent']}/{agg['sep_eligible']}"
              f"={agg['sep_opp_agent']/agg['sep_eligible']:.0%}  |  "
              f"world {agg['sep_opp_world']}/{agg['sep_eligible']}"
              f"={agg['sep_opp_world']/agg['sep_eligible']:.0%}")
    print(f"\nRECOMMEND: {verdict} — {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
