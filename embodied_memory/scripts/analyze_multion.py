"""
MultiON (sequential semantic ObjectNav) ablation analysis.

Single-goal ObjectNav structurally under-tests the LTM (Run 7: at most one
useful recall per episode). The multion eval chains K categories per episode
(``make_multion_smoke.py`` + the ``episode_runner`` sub-goal cursor), so the
LTM's recall value can compound across sub-goals. This analyzer mines the
multion episode logs, read-only:

  * per-setting Progress (k_found/K), PPL (Progress weighted by path length),
    success_multion, recall-assisted-advance rate;
  * paired S3-S1 bootstrap deltas on Progress + PPL, keyed
    ``(scene_id, episode_id)`` (S2 rows added when a setting-2 run is given);
  * the **gap-by-sub-goal-index table** — does the S3-S1 found-rate gap GROW
    with sub-goal index? Later sub-goals benefit from more accumulated memory,
    so a growing gap is the cleanest "LTM compounds" signal;
  * advance step-cost split by whether a memory candidate was chosen while
    the sub-goal was active (the direct payoff measure).

Reuses ``analyze_ablation``'s bootstrap; touches no production code.

Usage (or via the front door: ``analyze_ablation.py --multion …``)::

    python embodied_memory/scripts/analyze_multion.py \
        runs/multion-s1 runs/multion-s2 runs/multion-s3
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# analyze_ablation lives next to this file and is import-clean (stdlib only).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_ablation import (  # noqa: E402
    _one_sided_p_le_zero,
    paired_bootstrap_mean_diff,
)
from analyze_revisit import _infer_setting  # noqa: E402


# ----------------------------------------------------------------------
# data model
# ----------------------------------------------------------------------


@dataclass
class MultionEpisode:
    scene_id: str
    episode_id: str
    target_categories: List[str]
    progress: float
    ppl: Optional[float]          # None when L_opt was fully unreachable
    success_multion: bool
    n_steps: int
    path_len_taken: float
    geodesic_optimal: float
    subgoals_found: List[Dict[str, Any]] = field(default_factory=list)
    n_memory_chosen: int = 0
    recall_assisted_advances: int = 0

    @property
    def k(self) -> int:
        return len(self.target_categories)

    def found_flag(self, idx: int) -> bool:
        return any(int(s.get("subgoal_idx", -1)) == idx
                   for s in self.subgoals_found)


@dataclass
class MultionRun:
    name: str
    path: str
    setting: Optional[int]
    episodes: List[MultionEpisode] = field(default_factory=list)


# ----------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------


def _raw_to_episode(raw: Dict[str, Any]) -> Optional[MultionEpisode]:
    if not raw.get("is_multion"):
        return None
    ppl = raw.get("ppl")
    return MultionEpisode(
        scene_id=str(raw.get("scene_id")),
        episode_id=str(raw.get("episode_id")),
        target_categories=[str(c) for c in (raw.get("target_categories") or [])],
        progress=float(raw.get("progress", 0.0)),
        ppl=(float(ppl) if ppl is not None else None),
        success_multion=bool(raw.get("success_multion", False)),
        n_steps=int(raw.get("n_steps", 0)),
        path_len_taken=float(raw.get("path_len_taken", 0.0)),
        geodesic_optimal=float(raw.get("geodesic_optimal", 0.0)),
        subgoals_found=list(raw.get("subgoals_found") or []),
        n_memory_chosen=int(raw.get("n_memory_chosen", 0) or 0),
        recall_assisted_advances=int(raw.get("recall_assisted_advances", 0) or 0),
    )


def load_multion_run(path: str) -> MultionRun:
    """Load a run dir, keeping only multion episodes (``is_multion`` true)."""
    summary: Dict[str, Any] = {}
    summary_path = os.path.join(path, "summary.json")
    if os.path.isfile(summary_path):
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except (OSError, json.JSONDecodeError):
            summary = {}

    episodes: List[MultionEpisode] = []
    for ep_path in sorted(glob.glob(os.path.join(path, "episode_*.json"))):
        if ep_path.endswith("_error.json"):
            continue
        try:
            with open(ep_path) as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        ep = _raw_to_episode(raw)
        if ep is not None:
            episodes.append(ep)

    return MultionRun(
        name=os.path.basename(os.path.normpath(path)) or path,
        path=path,
        setting=_infer_setting(path, summary),
        episodes=episodes,
    )


# ----------------------------------------------------------------------
# aggregation
# ----------------------------------------------------------------------


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def summarize(episodes: List[MultionEpisode]) -> Dict[str, float]:
    """Per-setting multion summary. PPL averages over episodes where it is
    DEFINED (``n_ppl`` reports how many) — an undefined PPL means every L_opt
    leg was unreachable, which is a dataset problem, not a policy score."""
    ppls = [e.ppl for e in episodes if e.ppl is not None]
    n_advances = sum(len(e.subgoals_found) for e in episodes)
    n_assisted = sum(e.recall_assisted_advances for e in episodes)
    return {
        "n": len(episodes),
        "progress": _mean([e.progress for e in episodes]),
        "ppl": _mean(ppls),
        "n_ppl": len(ppls),
        "success_multion": _mean(
            [1.0 if e.success_multion else 0.0 for e in episodes]),
        "n_steps": _mean([float(e.n_steps) for e in episodes]),
        "path_len": _mean([e.path_len_taken for e in episodes]),
        "recall_assist_rate": (n_assisted / n_advances) if n_advances else 0.0,
        "n_advances": n_advances,
    }


def paired_delta(
    a_eps: List[MultionEpisode],
    b_eps: List[MultionEpisode],
    metric: str = "progress",
    n_bootstrap: int = 5000,
) -> Dict[str, Any]:
    """Paired B-A delta on ``metric``, keyed (scene_id, episode_id). Pairs
    where either side has the metric undefined (None) are excluded."""
    a_by = {(e.scene_id, e.episode_id): e for e in a_eps}
    b_by = {(e.scene_id, e.episode_id): e for e in b_eps}
    keys = sorted(set(a_by) & set(b_by))
    deltas: List[float] = []
    used = []
    for k in keys:
        va, vb = getattr(a_by[k], metric), getattr(b_by[k], metric)
        if va is None or vb is None:
            continue
        deltas.append(float(vb) - float(va))
        used.append(k)
    mean, lo, hi = paired_bootstrap_mean_diff(deltas, n_resamples=n_bootstrap, ci=0.9)
    p = _one_sided_p_le_zero(deltas, n_bootstrap)
    return {"n": len(deltas), "mean": mean, "lo": lo, "hi": hi,
            "p_le_zero": p, "keys": used, "deltas": deltas}


def gap_by_subgoal_index(
    a_eps: List[MultionEpisode],
    b_eps: List[MultionEpisode],
) -> List[Dict[str, Any]]:
    """Per sub-goal index i: found-rate in each run + the paired B-A delta of
    the found flag over common (scene_id, episode_id) pairs. A delta that
    GROWS with i is the "LTM compounds across sub-goals" signal."""
    a_by = {(e.scene_id, e.episode_id): e for e in a_eps}
    b_by = {(e.scene_id, e.episode_id): e for e in b_eps}
    keys = sorted(set(a_by) & set(b_by))
    if not keys:
        return []
    max_k = max(max((a_by[k].k for k in keys), default=0),
                max((b_by[k].k for k in keys), default=0))
    rows: List[Dict[str, Any]] = []
    for idx in range(max_k):
        # only episodes whose chain actually has an idx-th sub-goal count
        sub = [k for k in keys if a_by[k].k > idx and b_by[k].k > idx]
        if not sub:
            continue
        fa = [1.0 if a_by[k].found_flag(idx) else 0.0 for k in sub]
        fb = [1.0 if b_by[k].found_flag(idx) else 0.0 for k in sub]
        rows.append({
            "subgoal_idx": idx,
            "n": len(sub),
            "rate_a": _mean(fa),
            "rate_b": _mean(fb),
            "delta": _mean([b - a for a, b in zip(fa, fb)]),
        })
    return rows


def advance_step_costs(episodes: List[MultionEpisode]) -> Dict[str, List[float]]:
    """Step-cost of each advance (step_idx delta from the previous advance;
    episode start for the first), split by whether a memory candidate was
    chosen while that sub-goal was active — the direct payoff measure."""
    out: Dict[str, List[float]] = {"with_memory": [], "without_memory": []}
    for e in episodes:
        found = sorted(e.subgoals_found, key=lambda s: int(s.get("subgoal_idx", 0)))
        prev_step = 0
        for s in found:
            cost = float(int(s.get("step_idx", 0)) - prev_step)
            prev_step = int(s.get("step_idx", 0))
            bucket = "with_memory" if s.get("memory_assisted") else "without_memory"
            out[bucket].append(cost)
    return out


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------


def _print_delta(label: str, res: Dict[str, Any]) -> None:
    print(f"  {label}: n={res['n']:d}  mean={res['mean']:+.4f}  "
          f"90% CI=[{res['lo']:+.4f}, {res['hi']:+.4f}]  "
          f"one-sided p(<=0)={res['p_le_zero']:.3f}")


def print_report(runs: List[MultionRun], n_bootstrap: int) -> None:
    print("=== runs ===")
    for r in runs:
        print(f"  {r.name}: setting={r.setting} n_multion_episodes={len(r.episodes)}")
    print()

    print("=== per-setting multion summary ===")
    print(f"  {'run':<22} {'n':>3} {'Progress':>9} {'PPL':>7} {'(n_ppl)':>7} "
          f"{'success':>8} {'steps':>7} {'path_m':>7} {'recall_assist':>13}")
    for r in runs:
        s = summarize(r.episodes)
        print(f"  {r.name:<22} {s['n']:>3} {s['progress']:>9.3f} "
              f"{s['ppl']:>7.3f} {s['n_ppl']:>7d} {s['success_multion']:>8.3f} "
              f"{s['n_steps']:>7.1f} {s['path_len']:>7.1f} "
              f"{s['recall_assist_rate']:>13.3f}")
    print()

    by_setting: Dict[int, MultionRun] = {}
    for r in runs:
        if r.setting in (1, 2, 3):
            by_setting[int(r.setting)] = r
    if 1 not in by_setting or 3 not in by_setting:
        print("(skip paired deltas + gap table: need both setting 1 and 3 runs.)")
        return

    s1, s3 = by_setting[1], by_setting[3]
    s2 = by_setting.get(2)

    print("=== paired deltas (S3 - S1), bootstrap, 90% CI ===")
    _print_delta("Progress S3 - S1 (PRIMARY)",
                 paired_delta(s1.episodes, s3.episodes, "progress", n_bootstrap))
    _print_delta("PPL      S3 - S1 (headline; efficiency-weighted)",
                 paired_delta(s1.episodes, s3.episodes, "ppl", n_bootstrap))
    if s2 is not None:
        _print_delta("Progress S2 - S1 (STM-only)",
                     paired_delta(s1.episodes, s2.episodes, "progress", n_bootstrap))
        _print_delta("Progress S3 - S2 (LTM-specific)",
                     paired_delta(s2.episodes, s3.episodes, "progress", n_bootstrap))
        _print_delta("PPL      S2 - S1 (STM-only)",
                     paired_delta(s1.episodes, s2.episodes, "ppl", n_bootstrap))
        _print_delta("PPL      S3 - S2 (LTM-specific)",
                     paired_delta(s2.episodes, s3.episodes, "ppl", n_bootstrap))
    print()

    print("=== gap by sub-goal index (S3 - S1 found-rate; growth = LTM compounds) ===")
    rows = gap_by_subgoal_index(s1.episodes, s3.episodes)
    print(f"  {'idx':>3} {'n':>3} {'S1 rate':>8} {'S3 rate':>8} {'delta':>8}")
    for row in rows:
        print(f"  {row['subgoal_idx']:>3} {row['n']:>3} {row['rate_a']:>8.3f} "
              f"{row['rate_b']:>8.3f} {row['delta']:>+8.3f}")
    if len(rows) >= 2 and rows[-1]["delta"] > rows[0]["delta"]:
        print("  -> gap GROWS with sub-goal index (compounding signal).")
    print()

    print("=== advance step-cost (S3): recall-assisted vs not ===")
    costs = advance_step_costs(s3.episodes)
    wm, wo = costs["with_memory"], costs["without_memory"]
    print(f"  with memory pick:    n={len(wm):<3d} mean_steps={_mean(wm):.1f}")
    print(f"  without memory pick: n={len(wo):<3d} mean_steps={_mean(wo):.1f}")
    print()


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="MultiON ablation analysis")
    parser.add_argument("run_dirs", nargs="+",
                        help="Run directories (>=2; need S1 and S3).")
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args(argv)

    if len(args.run_dirs) < 2:
        parser.error("at least two run directories are required")

    runs = [load_multion_run(p) for p in args.run_dirs]
    print_report(runs, args.bootstrap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
