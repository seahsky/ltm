"""Diagnose why ReMEmbR never triggers STOP across a run-dir.

Reads every episode_*.json under one or more run dirs and reports:
  1. Action histogram (was `stop` (action=0) ever emitted? how often?)
  2. Per-target STOP rate
  3. Near-goal episodes: for each ep, the closest distance reached and whether
     STOP was ever proposed after that point
  4. Caption content around the closest point — does the captioner ever mention
     the target category text when the agent is near the goal?
  5. Memory-source decisions: how often did a `memory`-sourced candidate get
     chosen, and did those land closer to the goal than `planner` candidates?

Usage:
    python embodied_memory/scripts/diagnose_stop.py runs/abl-s3-remembr
    python embodied_memory/scripts/diagnose_stop.py runs/abl-s{1,2,3}-remembr
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter, defaultdict

ACTION_NAMES = {0: "stop", 1: "fwd", 2: "left", 3: "right", 4: "up", 5: "down"}
SUCCESS_RADIUS_M = 1.0  # HM3D ObjectNav success threshold
NEAR_RADIUS_M = 2.0     # "should have stopped" threshold


def load_episode(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def episode_files(run_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(run_dir, "episode_*.json")))


def diagnose_run(run_dir: str) -> None:
    files = episode_files(run_dir)
    if not files:
        print(f"  (no episode_*.json under {run_dir})")
        return

    print(f"\n=== {run_dir} ({len(files)} episodes) ===")

    action_hist: Counter = Counter()
    stop_by_target: dict[str, list[bool]] = defaultdict(list)
    near_misses: list[dict] = []
    captions_near_goal: list[dict] = []
    chosen_source_hist: Counter = Counter()
    src_to_final_dist: dict[str, list[float]] = defaultdict(list)
    stop_emitted_anywhere = 0

    for fp in files:
        ep = load_episode(fp)
        target = ep.get("target_category", "?")
        final_dist = float(ep.get("distance_to_goal", float("nan")))
        steps = ep.get("steps", []) or []
        decisions = ep.get("decisions", []) or []

        # --- action histogram ----------------------------------------------
        ep_stops = 0
        for s in steps:
            a = s.get("action")
            if a is None:
                continue
            action_hist[a] += 1
            if a == 0:
                ep_stops += 1
        stop_by_target[target].append(ep_stops > 0)
        if ep_stops:
            stop_emitted_anywhere += 1

        # --- closest reach & did we stop near goal? ------------------------
        # No per-step distance in JSON, but agent_pos is logged.
        # Estimate closeness via final_dist + a rough lower bound:
        # if any step had nearly the same xy as the goal, we'd know. We don't
        # have the goal xy logged per-episode, but we have final_dist at the
        # last step. Use final_dist as a conservative "did we approach?"
        # signal; flag near-miss when final_dist < NEAR_RADIUS_M.
        if final_dist < NEAR_RADIUS_M:
            # Was STOP ever emitted in the last 25 steps?
            tail = steps[-25:]
            tail_actions = [s.get("action") for s in tail if s.get("action") is not None]
            stopped_late = 0 in tail_actions
            near_misses.append({
                "file": os.path.basename(fp),
                "scene": ep.get("scene_id"),
                "target": target,
                "final_dist": final_dist,
                "n_steps": ep.get("n_steps"),
                "success": ep.get("success"),
                "stop_in_last_25_steps": stopped_late,
                "tail_action_hist": dict(Counter(ACTION_NAMES.get(a, str(a)) for a in tail_actions)),
            })
            # Sample captions from the tail — did captioner mention target?
            tail_caps = [s.get("caption", "") for s in tail]
            mentions = [c for c in tail_caps if target.lower() in c.lower()]
            captions_near_goal.append({
                "file": os.path.basename(fp),
                "target": target,
                "final_dist": final_dist,
                "n_tail_captions": len(tail_caps),
                "n_caption_mentions_target": len(mentions),
                "example_caption": tail_caps[-1][:200] if tail_caps else "",
                "example_mention": mentions[0][:200] if mentions else "",
            })

        # --- chosen_source distribution ------------------------------------
        for dec in decisions:
            src = dec.get("chosen_source", "?")
            chosen_source_hist[src] += 1
        # final-distance per chosen-source signal: which source did the LAST
        # decision use? does memory-sourced final picks correlate with shorter
        # final_dist?
        if decisions:
            last_src = decisions[-1].get("chosen_source", "?")
            src_to_final_dist[last_src].append(final_dist)

    # ---- report -----------------------------------------------------------
    total_actions = sum(action_hist.values()) or 1
    print(f"  action histogram (n={total_actions}):")
    for a in sorted(action_hist):
        n = action_hist[a]
        print(f"    {ACTION_NAMES.get(a, str(a)):<6} {n:>6}  ({100*n/total_actions:5.1f}%)")

    print(f"  episodes that emitted ANY stop: {stop_emitted_anywhere}/{len(files)}")
    print(f"  per-target STOP-ever rate:")
    for t in sorted(stop_by_target):
        bs = stop_by_target[t]
        rate = sum(bs) / len(bs)
        print(f"    {t:<14}  {sum(bs):>2}/{len(bs):<2}  ({rate:5.1%})")

    print(f"  chosen_source histogram (n={sum(chosen_source_hist.values())}):")
    for src, n in chosen_source_hist.most_common():
        print(f"    {src:<10} {n:>6}")
    if src_to_final_dist:
        print(f"  mean final_dist by last-step chosen_source:")
        for src, vals in src_to_final_dist.items():
            if vals:
                print(f"    {src:<10} n={len(vals):<3} mean={sum(vals)/len(vals):.2f}m  min={min(vals):.2f}m")

    print(f"  near-miss episodes (final_dist < {NEAR_RADIUS_M}m): {len(near_misses)}")
    for nm in sorted(near_misses, key=lambda x: x["final_dist"])[:10]:
        flag = "STOP-IN-TAIL" if nm["stop_in_last_25_steps"] else "NO-STOP-IN-TAIL"
        print(f"    {nm['file']}  {nm['scene']}/{nm['target']:<12}  d={nm['final_dist']:.2f}m  {flag}  tail={nm['tail_action_hist']}")

    if captions_near_goal:
        n_mentioned = sum(1 for c in captions_near_goal if c["n_caption_mentions_target"] > 0)
        print(f"  near-goal caption check: {n_mentioned}/{len(captions_near_goal)} eps had ≥1 caption mentioning target")
        for c in sorted(captions_near_goal, key=lambda x: x["final_dist"])[:5]:
            print(f"    {c['file']}  target='{c['target']}'  d={c['final_dist']:.2f}m  mentions={c['n_caption_mentions_target']}/{c['n_tail_captions']}")
            print(f"       last caption: {c['example_caption']!r}")
            if c["example_mention"]:
                print(f"       mention cap:  {c['example_mention']!r}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for d in sys.argv[1:]:
        diagnose_run(d)


if __name__ == "__main__":
    main()
