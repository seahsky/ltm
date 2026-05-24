#!/usr/bin/env python3
"""
Verify a smoke-gate run (Run-4 movement gate, or Run-5 oracle gate).

Two gate flavours, selected by ``--backbone`` or auto-detected from the run's
``summary.json`` (``ablation.backbone``):

**Movement gate** (``frontier`` / ``remembr`` backbones) — the Run-4 table:

  - crash-free        (JSON parses, top-level keys present)
  - n_steps           > 50
  - path_traveled     ≥ 4 m
  - n_frontier_chosen ≥ 1

  ``dist_to_goal`` and ``n_stop_signals`` are info only. The per-step JSON
  doesn't carry ``distance_to_goal`` (the serializer drops ``step.info``), so
  the plan's "dist_to_goal < starting − 2 m" check is reported as a derived
  ``displacement`` (start→end straight-line) without a hard pass/fail —
  ``path_traveled ≥ 4 m`` is the canonical movement gate in its place.

**Oracle gate** (``oracle`` backbone — Run-5 diagnostic) — the
ShortestPathFollower bypasses per-step keyframe logging, so ``path_traveled``
and ``n_frontier_chosen`` are always 0 and meaningless here. The gate is
instead "did a perfect planner reach the goal?":

  - crash-free
  - reached_goal      (success, or dist_to_goal < 1.0 m)

  PASS → env is navigable, the pipeline is the bottleneck. FAIL on every
  episode → env/episode/action-space is broken (unreachable spawn, wrong goal
  coords, action-space mismatch) and no planner/perception fix matters.

Usage::

    python embodied_memory/scripts/verify_smoke_gate.py [run_dir] [--backbone ...]

Defaults ``run_dir`` to ``runs/remembr-smoke-frontier``. Exits 0 if the gate
passes, 1 otherwise — safe to chain after the smoke command::

    python -m embodied_memory.run_hm3d_pol --mode live --backbone oracle ... \\
        && python embodied_memory/scripts/verify_smoke_gate.py runs/oracle-smoke-...
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple


# Oracle reached-goal radius — matches the ShortestPathFollower goal_radius and
# ObjectNav success distance used by the oracle backbone.
ORACLE_GOAL_RADIUS_M = 1.0


GATES: List[Tuple[str, str]] = [
    # (key, label) — order matters for output stability.
    ("crash_free", "crash-free"),
    ("n_steps", "n_steps > 50"),
    ("path_traveled", "path_traveled ≥ 4 m"),
    ("n_frontier_chosen", "n_frontier_chosen ≥ 1"),
]


def _path_length(steps: List[Dict[str, Any]]) -> float:
    """Sum of straight-line segments between consecutive agent_pos
    samples. Habitat positions are (x, y, z); ground-plane is (x, z)."""
    if not steps:
        return 0.0
    pts = []
    for s in steps:
        ap = s.get("agent_pos")
        if not ap or len(ap) < 3:
            continue
        pts.append((float(ap[0]), float(ap[2])))
    return sum(
        math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        for i in range(len(pts) - 1)
    )


def _displacement(steps: List[Dict[str, Any]]) -> float:
    """Straight-line distance from the agent's first to last logged xz."""
    pts = []
    for s in steps:
        ap = s.get("agent_pos")
        if not ap or len(ap) < 3:
            continue
        pts.append((float(ap[0]), float(ap[2])))
    if len(pts) < 2:
        return 0.0
    return math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])


def _decision_metrics(decisions: List[Dict[str, Any]], n_steps: int) -> Dict[str, Any]:
    """Quantify replan-thrash from the per-decision log (Run-6 diagnostic).

    Run-6 fixed the start-wall freeze (agent now moves all N steps) but path
    stayed low with ~0 net progress. Hypothesis: the runner re-plans almost
    every step and the rerank flips the steering target each time, so the
    controller turns to chase a moving target instead of translating. Each
    decision logs the step it fired on, the raw planner top-1 id, the chosen
    id, and the chosen world_xy — enough to test that directly:

      - n_decisions / decision_rate: how often the runner re-planned. With
        ``decision_period=10`` an un-stuck agent re-plans ~n_steps/10 times; a
        rate near 1.0 means ``is_decision_step()`` is True nearly every step
        (``_is_stuck()`` firing, since turning-in-place is <0.1 m of travel).
      - median inter-decision gap (steps): intended gap is decision_period
        (default 10). A median of 1 confirms stuck-triggered replan every step.
      - disagree_frac: chosen != raw planner top-1 (rerank overrode the pick).
      - distinct_target_frac: unique chosen targets (0.25 m-rounded) / decisions.
      - mean_target_jump_m: mean straight-line move of the steering target
        between consecutive decisions. Large + erratic == target flipping.
    """
    n = len(decisions)
    out: Dict[str, Any] = {
        "n_decisions": n,
        "decision_rate": (n / n_steps) if n_steps else 0.0,
        "median_gap": float("nan"),
        "disagree_frac": float("nan"),
        "distinct_target_frac": float("nan"),
        "mean_target_jump_m": float("nan"),
    }
    if n == 0:
        return out
    steps = [int(d.get("step_idx", 0)) for d in decisions]
    gaps = [steps[i + 1] - steps[i] for i in range(len(steps) - 1)]
    if gaps:
        out["median_gap"] = float(statistics.median(gaps))
    disagree = sum(
        1 for d in decisions
        if int(d.get("chosen_id", -1)) != int(d.get("raw_top1_id", -2))
    )
    out["disagree_frac"] = disagree / n
    targets = []
    for d in decisions:
        xy = d.get("chosen_world_xy")
        if xy and len(xy) >= 2:
            targets.append((float(xy[0]), float(xy[1])))
    if targets:
        rounded = {(round(x / 0.25), round(y / 0.25)) for x, y in targets}
        out["distinct_target_frac"] = len(rounded) / len(targets)
        jumps = [
            math.hypot(targets[i + 1][0] - targets[i][0],
                       targets[i + 1][1] - targets[i][1])
            for i in range(len(targets) - 1)
        ]
        if jumps:
            out["mean_target_jump_m"] = sum(jumps) / len(jumps)
    return out


def _detect_backbone(run_dir: str) -> Optional[str]:
    """Read the run's backbone from ``summary.json`` (``ablation.backbone``),
    so the verifier can auto-select the oracle vs movement gate. Returns None
    if summary.json is absent/unreadable or carries no backbone."""
    path = os.path.join(run_dir, "summary.json")
    try:
        with open(path) as f:
            summary = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    bb = (summary.get("ablation") or {}).get("backbone")
    return str(bb) if bb else None


def _evaluate(ep: Dict[str, Any]) -> Dict[str, Any]:
    steps = ep.get("steps") or []
    decisions = ep.get("decisions") or []
    path_traveled = _path_length(steps)
    displacement = _displacement(steps)

    n_steps = int(ep.get("n_steps", 0))
    n_frontier_chosen = int(ep.get("n_frontier_chosen", 0))
    n_frontier_total = sum(
        int(d.get("n_frontier_candidates", 0)) for d in decisions
    )

    # Oracle reached-goal: success, or final distance_to_goal within the radius.
    d2g = ep.get("distance_to_goal")
    reached_goal = bool(ep.get("success", False)) or (
        isinstance(d2g, (int, float)) and float(d2g) < ORACLE_GOAL_RADIUS_M
    )

    return {
        # movement-gate keys
        "crash_free": True,  # if we got here, JSON parsed and keys read
        "n_steps_val": n_steps,
        "n_steps": n_steps > 50,
        "path_traveled_val": path_traveled,
        "path_traveled": path_traveled >= 4.0,
        "n_frontier_chosen_val": n_frontier_chosen,
        "n_frontier_chosen": n_frontier_chosen >= 1,
        # oracle-gate keys
        "reached_goal": reached_goal,
        "oracle_no_goal_val": bool(ep.get("oracle_no_goal", False)),
        # grid census (Run-5 densification instrumentation)
        "grid_free_val": int(ep.get("grid_cells_free", 0)),
        "grid_occupied_val": int(ep.get("grid_cells_occupied", 0)),
        "grid_frontier_val": int(ep.get("grid_frontier_cells", 0)),
        # info-only
        "displacement_val": displacement,
        "dist_to_goal_val": d2g,
        "n_stop_signals_val": int(ep.get("n_stop_signals", 0)),
        "success_val": bool(ep.get("success", False)),
        "soft_spl_val": float(ep.get("soft_spl", 0.0)),
        "spl_val": float(ep.get("spl", 0.0)),
        "n_frontier_total_val": n_frontier_total,
        "n_decisions_val": len(decisions),
        "decision_metrics": _decision_metrics(decisions, n_steps),
        # controller census (Run-6 instrumentation; 0 on older runs)
        "action_forward_val": int(ep.get("action_forward", 0)),
        "action_turn_val": int(ep.get("action_turn", 0)),
        "action_stop_val": int(ep.get("action_stop", 0)),
        "astar_path_val": int(ep.get("astar_path", 0)),
        "astar_reach_fb_val": int(ep.get("astar_reachable_fallback", 0)),
        "astar_fallback_val": int(ep.get("astar_fallback", 0)),
        "collision_escape_val": int(ep.get("collision_escape", 0)),
        "replan_scheduled_val": int(ep.get("replan_scheduled", 0)),
        "replan_forced_val": int(ep.get("replan_forced", 0)),
        "replan_stuck_val": int(ep.get("replan_stuck", 0)),
        # ReMEmbR backbone certification (Phase-2). None on non-remembr runs.
        "remembr_stub_mode_val": ep.get("remembr_stub_mode"),
        "remembr_sample_caption_val": ep.get("remembr_sample_caption"),
        "remembr_stop_event_val": ep.get("remembr_stop_event"),
        # metadata
        "scene_id": ep.get("scene_id"),
        "target_category": ep.get("target_category"),
        "episode_id": ep.get("episode_id"),
    }


def _fmt_passfail(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _print_report(run_dir: str, r: Dict[str, Any]) -> bool:
    print(f"=== Run-4 smoke gate verify ===")
    print(f"  run_dir:        {run_dir}")
    print(f"  scene_id:       {r.get('scene_id')}")
    print(f"  episode_id:     {r.get('episode_id')}")
    print(f"  target:         {r.get('target_category')}")
    print()
    print("  --- gating ---")
    print(f"  [{_fmt_passfail(r['crash_free'])}] crash-free            "
          f"(JSON parsed, n_decisions={r['n_decisions_val']})")
    print(f"  [{_fmt_passfail(r['n_steps'])}] n_steps > 50           "
          f"(actual: {r['n_steps_val']})")
    print(f"  [{_fmt_passfail(r['path_traveled'])}] path_traveled ≥ 4 m   "
          f"(actual: {r['path_traveled_val']:.2f} m)")
    print(f"  [{_fmt_passfail(r['n_frontier_chosen'])}] n_frontier_chosen ≥ 1  "
          f"(actual: {r['n_frontier_chosen_val']}, "
          f"of {r['n_frontier_total_val']} proposed)")
    print()
    print("  --- info-only ---")
    d2g = r['dist_to_goal_val']
    d2g_s = f"{d2g:.2f} m" if isinstance(d2g, (int, float)) else str(d2g)
    print(f"  displacement (start→end):  {r['displacement_val']:.2f} m")
    print(f"  dist_to_goal (final):      {d2g_s}")
    print(f"  grid free/occupied/front:  {r['grid_free_val']} / "
          f"{r['grid_occupied_val']} / {r['grid_frontier_val']}")
    print(f"  n_stop_signals:            {r['n_stop_signals_val']}")
    print(f"  success / spl / soft_spl:  {r['success_val']} / "
          f"{r['spl_val']:.3f} / {r['soft_spl_val']:.3f}")
    print()

    gates_passed = all(r[k] for k, _ in GATES)
    print(f"  gate: {_fmt_passfail(gates_passed)}")

    _print_thrash_block([(f"episode_{r.get('episode_id', '?')}", r)])
    _print_controller_block([(f"episode_{r.get('episode_id', '?')}", r)])
    _print_remembr_block([(f"episode_{r.get('episode_id', '?')}", r)])

    if not gates_passed:
        print()
        print("  Diagnostic hints (from Run-4 plan):")
        if not r["n_steps"]:
            print("    - n_steps low → controller-stall regressed; check that "
                  "frontier_planner.py still has the collision-escape "
                  "(commit 117028d) and force-replan (commit 6265870) patches.")
        if not r["path_traveled"]:
            print("    - path_traveled low → agent spinning at start. Confirm "
                  "n_frontier_chosen > 0; if 0, frontier candidates aren't "
                  "entering the pool (merge logic broken or all de-duped).")
        if not r["n_frontier_chosen"]:
            print("    - n_frontier_chosen = 0 → merge logic wrong, all "
                  "frontier picks de-duped against LLM, or rerank scoring "
                  "floor never picks frontier. Inspect decisions[*]"
                  ".n_frontier_candidates in the JSON.")
    return gates_passed


def _oracle_read(reached: bool) -> List[str]:
    """The decision-tree read for the oracle gate (Run-5 plan)."""
    if reached:
        return [
            "  Read: env is navigable with a perfect planner → the pipeline is",
            "        the bottleneck, not the environment. Proceed to the",
            "        densified-grid `remembr` smoke (escape check).",
        ]
    return [
        "  Read: oracle did NOT reach the goal → env/episode/action-space is",
        "        likely broken (unreachable spawn pocket, wrong goal coords, or",
        "        discrete action-space mismatch). No planner/perception fix",
        "        matters until this is resolved — pivot to env debugging.",
    ]


def _print_report_oracle(run_dir: str, r: Dict[str, Any]) -> bool:
    print("=== oracle smoke verify ===")
    print(f"  run_dir:        {run_dir}")
    print(f"  scene_id:       {r.get('scene_id')}")
    print(f"  episode_id:     {r.get('episode_id')}")
    print(f"  target:         {r.get('target_category')}")
    print()
    print("  --- gating (oracle: perfect-planner reachability) ---")
    print(f"  [{_fmt_passfail(r['crash_free'])}] crash-free")
    d2g = r["dist_to_goal_val"]
    d2g_s = f"{d2g:.2f} m" if isinstance(d2g, (int, float)) else str(d2g)
    print(f"  [{_fmt_passfail(r['reached_goal'])}] reached_goal "
          f"(success or dist_to_goal < {ORACLE_GOAL_RADIUS_M:.1f} m)")
    print(f"        success={r['success_val']}  dist_to_goal={d2g_s}  "
          f"n_steps={r['n_steps_val']}  spl={r['spl_val']:.3f}")
    print(f"        grid: free={r['grid_free_val']}  occupied={r['grid_occupied_val']}  "
          f"frontier={r['grid_frontier_val']}")
    if r["oracle_no_goal_val"]:
        print("  WARNING: oracle_no_goal=True — episode had no target_position; "
              "agent STOPed at step 0. Data issue, not a navigation result.")
    print()
    gate = r["crash_free"] and r["reached_goal"]
    print(f"  gate: {_fmt_passfail(gate)}")
    print()
    for line in _oracle_read(r["reached_goal"]):
        print(line)
    return gate


def _print_multi_summary_oracle(
    run_dir: str, reports: List[Tuple[str, Dict[str, Any]]]
) -> bool:
    print("=== oracle smoke verify (multi-episode) ===")
    print(f"  run_dir: {run_dir}")
    print(f"  episodes: {len(reports)}")
    print()
    hdr = (f"  {'ep':>3} {'scene':<16} {'tgt':<9} {'succ':>5} {'d2g_m':>6} "
           f"{'steps':>5} {'spl':>6} {'g_free':>6} {'g_occ':>6} {'g_front':>7} "
           f"{'no_goal':>7}  reached")
    print(hdr)
    any_reached = False
    for ep_path, r in reports:
        ep_idx = os.path.basename(ep_path).replace("episode_", "").replace(".json", "")
        d2g = r["dist_to_goal_val"]
        d2g_s = f"{d2g:6.2f}" if isinstance(d2g, (int, float)) else "   n/a"
        reached = r["reached_goal"]
        any_reached = any_reached or reached
        scene = (r.get("scene_id") or "?")[:16]
        tgt = (r.get("target_category") or "?")[:9]
        ng = "yes" if r["oracle_no_goal_val"] else "no"
        print(
            f"  {ep_idx:>3} {scene:<16} {tgt:<9} {str(r['success_val']):>5} "
            f"{d2g_s} {r['n_steps_val']:>5d} {r['spl_val']:>6.3f} "
            f"{r['grid_free_val']:>6d} {r['grid_occupied_val']:>6d} "
            f"{r['grid_frontier_val']:>7d} {ng:>7}  {_fmt_passfail(reached)}"
        )
    print()
    print("  --- roll-up gate ---")
    print(f"  navigable (any episode reached goal): {_fmt_passfail(any_reached)}")
    print(f"  gate: {_fmt_passfail(any_reached)}")
    print()
    for line in _oracle_read(any_reached):
        print(line)
    return any_reached


def _load_episode(path: str):
    if not os.path.isfile(path):
        return None, f"not found: {path}"
    try:
        with open(path) as f:
            return json.load(f), None
    except json.JSONDecodeError as e:
        return None, f"parse failed: {e}"


def _print_thrash_block(reports: List[Tuple[str, Dict[str, Any]]]) -> None:
    """Replan/target-stability diagnostic (Run-6). Reads the per-decision log
    to show *why* path stays low despite continuous motion: how often the
    runner re-plans and whether the steering target flips each time."""
    print()
    print("  --- decision-thrash (Run-6: replan / target stability) ---")
    print(f"  {'ep':>3} {'decis':>6} {'rate':>5} {'medgap':>6} "
          f"{'disagree':>8} {'distinct':>8} {'jump_m':>7}")
    for ep_path, r in reports:
        ep_idx = os.path.basename(ep_path).replace("episode_", "").replace(".json", "")
        m = r["decision_metrics"]
        print(
            f"  {ep_idx:>3} {m['n_decisions']:>6d} {m['decision_rate']:>5.2f} "
            f"{m['median_gap']:>6.1f} {m['disagree_frac'] * 100:>7.0f}% "
            f"{m['distinct_target_frac'] * 100:>7.0f}% {m['mean_target_jump_m']:>7.2f}"
        )
    print()
    print("  read: medgap≈1 (intended decision_period=10) → is_decision_step() is")
    print("        True ~every step; _is_stuck() reads turning-in-place (<0.1 m of")
    print("        xy travel) as stuck → replan. High disagree/distinct/jump_m →")
    print("        the rerank flips the steering target each decision, so the ±15°")
    print("        controller turns to chase a moving target instead of translating.")
    print("        Fix: rotation-aware _is_stuck() and/or commit to current_candidate.")


def _print_controller_block(reports: List[Tuple[str, Dict[str, Any]]]) -> None:
    """Controller census (Run-6 instrumentation): action mix, A* path-vs-
    fallback, collision-escape, and the replan-trigger breakdown. This is what
    separates the candidate root causes of a stable-target-but-stuck episode:
      - repl_force ≫ repl_stuck + many astar_fb  → A* no-path force-replan loop
        (chosen target unreachable on the grid; fix = reachable candidates).
      - coll_esc high                            → geometry stall the grid
        misses (grid-vs-navmesh mismatch; fix = navmesh/clearance, not replan).
      - repl_stuck ≫ repl_force, astar_ok high   → turning-in-place reads as
        stuck (fix = rotation-aware _is_stuck).
      - fwd ≪ turn always                         → ±15° deadband vs 30° turn
        oscillation (fix = controller smoothing / commitment).
    All-zero columns mean the run predates this instrumentation."""
    if not any(
        r["action_turn_val"] or r["astar_path_val"] or r["astar_fallback_val"]
        or r["astar_reach_fb_val"] or r["replan_forced_val"] or r["replan_stuck_val"]
        for _, r in reports
    ):
        print()
        print("  --- controller census --- (run predates Run-6 instrumentation; "
              "re-run the smoke to populate)")
        return
    print()
    print("  --- controller census (Run-6: action mix / A* outcome / replan trigger) ---")
    print(f"  {'ep':>3} | {'fwd':>4} {'turn':>4} {'stop':>4} | "
          f"{'astar_ok':>8} {'reach_fb':>8} {'line_fb':>7} {'coll_esc':>8} | "
          f"{'sched':>5} {'force':>5} {'stuck':>5}")
    for ep_path, r in reports:
        ep_idx = os.path.basename(ep_path).replace("episode_", "").replace(".json", "")
        print(
            f"  {ep_idx:>3} | {r['action_forward_val']:>4d} {r['action_turn_val']:>4d} "
            f"{r['action_stop_val']:>4d} | {r['astar_path_val']:>8d} "
            f"{r['astar_reach_fb_val']:>8d} {r['astar_fallback_val']:>7d} "
            f"{r['collision_escape_val']:>8d} | "
            f"{r['replan_scheduled_val']:>5d} {r['replan_forced_val']:>5d} "
            f"{r['replan_stuck_val']:>5d}"
        )
    print()
    print("  read (Run-6.1): astar_ok + reach_fb should dominate and line_fb→~0;")
    print("        reach_fb = chosen frontier unreachable, routed to nearest reachable")
    print("        cell (collision-aware). line_fb high → still boxed in (de-noise /")
    print("        relax inflation). force should drop as reach_fb does not force-replan.")
    print("        fwd≫turn and path↑ = the wedge is broken.")


def _print_remembr_block(reports: List[Tuple[str, Dict[str, Any]]]) -> None:
    """ReMEmbR backbone certification (Phase-2). Reports whether the real
    Qwen weights loaded (``remembr_stub_mode: false``) and shows a sample
    caption so a stub run ("stub-caption step=N") can never be mistaken for a
    real one ("a bedroom with a bed and ..."). Silent stub fallback hid for the
    whole project once — this block makes it impossible to miss in a smoke."""
    vals = [r.get("remembr_stub_mode_val") for _, r in reports]
    if all(v is None for v in vals):
        return  # not a remembr run (oracle/frontier) — nothing to certify
    print()
    print("  --- ReMEmbR backbone (Phase-2 certification) ---")
    any_stub = False
    for ep_path, r in reports:
        ep_idx = os.path.basename(ep_path).replace("episode_", "").replace(".json", "")
        stub = r.get("remembr_stub_mode_val")
        cap = r.get("remembr_sample_caption_val")
        status = "REAL" if stub is False else ("STUB" if stub else "n/a")
        if stub:
            any_stub = True
        cap_s = (cap[:72] + "…") if isinstance(cap, str) and len(cap) > 73 else cap
        print(f"  {ep_idx:>3}  backbone={status:<4}  sample_caption: {cap_s!r}")
        se = r.get("remembr_stop_event_val")
        if isinstance(se, dict):
            mc = se.get("matched_caption")
            mc_s = (mc[:56] + "…") if isinstance(mc, str) and len(mc) > 57 else mc
            cos = se.get("stop_cos")
            dist = se.get("stop_dist_m")
            cos_s = f"{cos:.3f}" if isinstance(cos, (int, float)) else cos
            dist_s = f"{dist:.2f}m" if isinstance(dist, (int, float)) else dist
            print(f"       STOP@step{se.get('step')}: cos={cos_s} dist={dist_s} "
                  f"matched={mc_s!r}")
    print()
    if any_stub:
        print("  *** WARNING: at least one episode ran in STUB mode — NOT real")
        print("      ReMEmbR. Captions are placeholders; do not report this as a")
        print("      paper-faithful result. Check accelerate / weights / VRAM.")
    else:
        print("  backbone REAL on all episodes (stub_mode=false). Paper-faithful.")


def _print_multi_summary(run_dir: str, reports: List[Tuple[str, Dict[str, Any]]]) -> bool:
    """Multi-episode summary table + roll-up gate.

    Per-episode gating mirrors single-episode mode. Overall gate semantics:
      - Architectural pass: total n_frontier_chosen across all episodes >= 1.
      - Navigation pass:    at least 1 episode satisfies n_steps > 50 AND
                            path_traveled >= 4 m.
      - Overall PASS if both architectural and navigation pass.
    """
    print(f"=== Run-4 smoke gate verify (multi-episode) ===")
    print(f"  run_dir: {run_dir}")
    print(f"  episodes: {len(reports)}")
    print()
    hdr = (f"  {'ep':>3} {'scene':<18} {'tgt':<10} {'steps':>5} {'path_m':>7} "
           f"{'d2g_m':>7} {'fchosen':>7} {'g_free':>6} {'g_front':>7} "
           f"{'crash':>5}  arch  nav")
    print(hdr)
    arch_total = 0
    nav_any = False
    for ep_path, r in reports:
        ep_idx = os.path.basename(ep_path).replace("episode_", "").replace(".json", "")
        n_steps = r["n_steps_val"]
        path_v = r["path_traveled_val"]
        d2g = r["dist_to_goal_val"]
        d2g_s = f"{d2g:6.2f}" if isinstance(d2g, (int, float)) else "   n/a"
        n_fc = r["n_frontier_chosen_val"]
        n_stop = r["n_stop_signals_val"]
        crash = "OK" if r["crash_free"] else "FAIL"
        arch_ep = r["n_frontier_chosen"]
        nav_ep = r["n_steps"] and r["path_traveled"]
        arch_total += n_fc
        if nav_ep:
            nav_any = True
        scene = (r.get("scene_id") or "?")[:18]
        tgt = (r.get("target_category") or "?")[:10]
        print(
            f"  {ep_idx:>3} {scene:<18} {tgt:<10} {n_steps:>5d} {path_v:>7.2f} "
            f"{d2g_s} {n_fc:>7d} {r['grid_free_val']:>6d} "
            f"{r['grid_frontier_val']:>7d} {crash:>5}  "
            f"{_fmt_passfail(arch_ep)} {_fmt_passfail(nav_ep)}"
        )
    print()
    print("  --- roll-up gate ---")
    arch_pass = arch_total >= 1
    print(f"  architectural (any-episode n_frontier_chosen >= 1): "
          f"{_fmt_passfail(arch_pass)} (total chosen across run: {arch_total})")
    print(f"  navigation (any-episode n_steps>50 AND path>=4m):   "
          f"{_fmt_passfail(nav_any)}")
    overall = arch_pass and nav_any
    print(f"  gate: {_fmt_passfail(overall)}")

    _print_thrash_block(reports)
    _print_controller_block(reports)
    _print_remembr_block(reports)

    if not overall:
        print()
        print("  Diagnostic hints:")
        if not arch_pass:
            print("    - architectural fail: no frontier candidates chosen in any "
                  "episode. Merge logic broken or all picks de-duped against LLM. "
                  "Inspect a decisions[*] dump (chosen_source distribution).")
        if not nav_any:
            print("    - navigation fail: no episode satisfies n_steps>50 AND "
                  "path>=4m. If architectural passes, this is downstream of "
                  "(a) false STOP firing early — bump REMEMBR_STOP_MIN_STEP, "
                  "(b) scene corner-stall — try a different scene/episode, or "
                  "(c) LLM degeneracy — stub firing in place of real ANSWERs.")
    return overall


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "run_dir",
        nargs="?",
        default="runs/remembr-smoke-frontier",
        help="smoke run directory (default: runs/remembr-smoke-frontier)",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="single episode index (default: scan all episode_*.json in run_dir)",
    )
    parser.add_argument(
        "--backbone",
        choices=["frontier", "remembr", "oracle"],
        default=None,
        help="Gate flavour. Default: auto-detect from summary.json "
             "(ablation.backbone). 'oracle' switches to the reached-goal gate; "
             "everything else uses the n_steps/path/frontier movement gate.",
    )
    args = parser.parse_args(argv)

    backbone = args.backbone or _detect_backbone(args.run_dir)
    is_oracle = backbone == "oracle"

    if args.episode is not None:
        ep_path = os.path.join(args.run_dir, f"episode_{args.episode:03d}.json")
        ep, err = _load_episode(ep_path)
        if ep is None:
            print(f"ERROR: {err}")
            return 2
        report = _evaluate(ep)
        printer = _print_report_oracle if is_oracle else _print_report
        ok = printer(args.run_dir, report)
        return 0 if ok else 1

    # multi-episode: scan all
    ep_paths = sorted(glob.glob(os.path.join(args.run_dir, "episode_*.json")))
    ep_paths = [p for p in ep_paths if "_error" not in os.path.basename(p)]
    if not ep_paths:
        print(f"ERROR: no episode_*.json files in {args.run_dir}")
        return 2

    reports: List[Tuple[str, Dict[str, Any]]] = []
    for p in ep_paths:
        ep, err = _load_episode(p)
        if ep is None:
            print(f"WARN: skipping {p}: {err}")
            continue
        reports.append((p, _evaluate(ep)))

    if len(reports) == 1:
        # Single-episode dir — defer to the single-episode report.
        printer = _print_report_oracle if is_oracle else _print_report
        ok = printer(args.run_dir, reports[0][1])
        return 0 if ok else 1

    summarizer = _print_multi_summary_oracle if is_oracle else _print_multi_summary
    ok = summarizer(args.run_dir, reports)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
