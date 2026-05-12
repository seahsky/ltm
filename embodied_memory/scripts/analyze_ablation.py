"""
Paired-episode analysis for the Phase-1 ablation.

Reads two or more run directories (one per ablation setting), joins their
per-episode summaries on (scene_id, episode_id), and prints paired SPL /
success / steps deltas with paired-bootstrap 95% CIs.

Each run directory must contain a ``summary.json`` written by
``EpisodeRunner.run`` (which embeds a per-episode list under the ``episodes``
key). Older runs that predate that key are read by scanning
``episode_*.json`` files in the directory.

Usage:
    python embodied_memory/scripts/analyze_ablation.py runs/abl-s1 runs/abl-s2 runs/abl-s3

Output is a plain-text table on stdout — no plotting deps.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import random


# ----------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------


@dataclass
class EpisodeRow:
    scene_id: str
    episode_id: str
    success: bool
    spl: float
    n_steps: int
    rerank_disagreements: int
    retrieval_hits: int


@dataclass
class RunData:
    name: str          # display label (basename of the run dir)
    path: str
    setting: Optional[int]
    ablation: Dict[str, Any]
    episodes: Dict[Tuple[str, str], EpisodeRow]

    @property
    def n(self) -> int:
        return len(self.episodes)


def _coerce_episode(raw: Dict[str, Any]) -> Optional[EpisodeRow]:
    scene = raw.get("scene_id")
    ep_id = raw.get("episode_id")
    if scene is None or ep_id is None:
        return None
    return EpisodeRow(
        scene_id=str(scene),
        episode_id=str(ep_id),
        success=bool(raw.get("success", False)),
        spl=float(raw.get("spl", 1.0 if raw.get("success") else 0.0)),
        n_steps=int(raw.get("n_steps", 0)),
        rerank_disagreements=int(raw.get("rerank_disagreements", 0)),
        retrieval_hits=int(raw.get("retrieval_hits", 0)),
    )


def load_run(path: str) -> RunData:
    summary_path = os.path.join(path, "summary.json")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"missing summary.json in {path}")
    with open(summary_path) as f:
        summary = json.load(f)

    episodes: Dict[Tuple[str, str], EpisodeRow] = {}

    raw_eps = summary.get("episodes") or []
    for raw in raw_eps:
        row = _coerce_episode(raw)
        if row is not None:
            episodes[(row.scene_id, row.episode_id)] = row

    # Fallback: walk per-episode JSON files for older runs that didn't embed
    # an `episodes` array in summary.json.
    if not episodes:
        for ep_path in sorted(glob.glob(os.path.join(path, "episode_*.json"))):
            try:
                with open(ep_path) as f:
                    raw = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            row = _coerce_episode(raw)
            if row is not None:
                episodes[(row.scene_id, row.episode_id)] = row

    ablation = summary.get("ablation") or {}
    setting = ablation.get("setting") if isinstance(ablation, dict) else None
    return RunData(
        name=os.path.basename(os.path.normpath(path)) or path,
        path=path,
        setting=setting,
        ablation=ablation if isinstance(ablation, dict) else {},
        episodes=episodes,
    )


# ----------------------------------------------------------------------
# bootstrap
# ----------------------------------------------------------------------


def paired_bootstrap_mean_diff(
    deltas: List[float],
    n_resamples: int = 5000,
    ci: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Return (mean_delta, lo, hi) of a paired bootstrap on the per-episode
    delta list. Uses the percentile bootstrap on resampled means.

    For ``n < 2``, returns the trivial value with NaN bounds.
    """
    n = len(deltas)
    if n == 0:
        nan = float("nan")
        return nan, nan, nan
    mean = sum(deltas) / n
    if n < 2:
        return mean, float("nan"), float("nan")
    rng = random.Random(seed)
    means: List[float] = []
    for _ in range(n_resamples):
        s = 0.0
        for _i in range(n):
            s += deltas[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    alpha = (1.0 - ci) / 2.0
    lo = means[int(alpha * n_resamples)]
    hi = means[int((1.0 - alpha) * n_resamples) - 1]
    return mean, lo, hi


# ----------------------------------------------------------------------
# pairing + reporting
# ----------------------------------------------------------------------


METRICS = [
    ("spl", "SPL", lambda r: r.spl, "%+0.4f"),
    ("success", "success", lambda r: 1.0 if r.success else 0.0, "%+0.4f"),
    ("n_steps", "steps", lambda r: float(r.n_steps), "%+0.2f"),
]


def paired_episode_keys(runs: List[RunData]) -> List[Tuple[str, str]]:
    """Intersection of (scene, episode) keys across all runs."""
    if not runs:
        return []
    keys = set(runs[0].episodes.keys())
    for r in runs[1:]:
        keys &= set(r.episodes.keys())
    return sorted(keys)


def per_run_means(run: RunData, keys: List[Tuple[str, str]]) -> Dict[str, float]:
    if not keys:
        return {}
    out: Dict[str, float] = {}
    for slug, _, getter, _ in METRICS:
        vals = [getter(run.episodes[k]) for k in keys]
        out[slug] = sum(vals) / len(vals)
    out["n_paired"] = float(len(keys))
    out["n_total"] = float(run.n)
    return out


def print_per_setting_summary(runs: List[RunData], paired_keys: List[Tuple[str, str]]):
    print("=== per-setting aggregate (over paired episodes) ===")
    print(f"{'run':<25} {'setting':>7} {'n_paired':>8} {'n_total':>8} "
          f"{'mean_SPL':>10} {'success':>9} {'mean_steps':>11} "
          f"{'rerank_dis':>11} {'retr_hits':>10}")
    for r in runs:
        m = per_run_means(r, paired_keys)
        rerank_dis_total = sum(r.episodes[k].rerank_disagreements for k in paired_keys) if paired_keys else 0
        retr_hits_total = sum(r.episodes[k].retrieval_hits for k in paired_keys) if paired_keys else 0
        print(
            f"{r.name:<25} {str(r.setting):>7} {int(m.get('n_paired', 0)):>8d} "
            f"{int(m.get('n_total', 0)):>8d} "
            f"{m.get('spl', float('nan')):>10.4f} "
            f"{m.get('success', float('nan')):>9.4f} "
            f"{m.get('n_steps', float('nan')):>11.2f} "
            f"{rerank_dis_total:>11d} {retr_hits_total:>10d}"
        )
    print()


def print_pairwise_deltas(
    runs: List[RunData],
    paired_keys: List[Tuple[str, str]],
    n_bootstrap: int,
    ci: float,
):
    print(f"=== paired deltas (b - a), bootstrap n={n_bootstrap}, CI={int(ci*100)}% ===")
    print(f"{'comparison (b - a)':<32} {'metric':<8} {'mean':>10} {'lo':>10} {'hi':>10} {'n':>5}")
    for i in range(len(runs)):
        for j in range(len(runs)):
            if i == j:
                continue
            a, b = runs[i], runs[j]
            label = f"{b.name} - {a.name}"
            for slug, _, getter, _fmt in METRICS:
                deltas = [getter(b.episodes[k]) - getter(a.episodes[k]) for k in paired_keys]
                mean, lo, hi = paired_bootstrap_mean_diff(deltas, n_resamples=n_bootstrap, ci=ci)
                print(f"{label:<32} {slug:<8} {mean:>10.4f} {lo:>10.4f} {hi:>10.4f} {len(deltas):>5d}")
    print()


def print_phase1_gate(runs: List[RunData], paired_keys: List[Tuple[str, str]], n_bootstrap: int):
    """Phase 1 → Phase 2 gate: setting 3 mean SPL > setting 1 mean SPL with
    paired-bootstrap p < 0.1 (one-sided)."""
    by_setting: Dict[int, RunData] = {}
    for r in runs:
        if r.setting in (1, 2, 3):
            by_setting[int(r.setting)] = r
    if 1 not in by_setting or 3 not in by_setting:
        print("(skip gate: need both setting 1 and setting 3 runs to evaluate.)")
        return
    s1, s3 = by_setting[1], by_setting[3]
    if not paired_keys:
        print("(skip gate: no paired episodes between settings.)")
        return
    deltas = [s3.episodes[k].spl - s1.episodes[k].spl for k in paired_keys]
    mean, lo, hi = paired_bootstrap_mean_diff(deltas, n_resamples=n_bootstrap, ci=0.9)
    # one-sided p ≈ fraction of resamples ≤ 0
    rng = random.Random(0)
    n = len(deltas)
    le_zero = 0
    for _ in range(n_bootstrap):
        s = 0.0
        for _i in range(n):
            s += deltas[rng.randrange(n)]
        if s <= 0:
            le_zero += 1
    p_one_sided = le_zero / n_bootstrap if n_bootstrap > 0 else float("nan")
    pass_gate = (mean > 0.0) and (p_one_sided < 0.1)
    print("=== phase 1 → phase 2 gate ===")
    print(f"  setting 3 - setting 1 SPL delta: mean={mean:+.4f}  90% CI=[{lo:+.4f}, {hi:+.4f}]  one-sided p={p_one_sided:.3f}")
    print(f"  gate: {'PASS' if pass_gate else 'FAIL'} (require mean>0 and p<0.1)")
    if not pass_gate:
        print("  → audit the LTM stack before integrating ReMEmbR; the stand-in backbone")
        print("    isn't producing a measurable memory signal.")
    print()


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Paired ablation analysis")
    parser.add_argument("run_dirs", nargs="+", help="Two or more run directories.")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--ci", type=float, default=0.95)
    args = parser.parse_args(argv)

    if len(args.run_dirs) < 2:
        parser.error("at least two run directories are required")

    runs = [load_run(p) for p in args.run_dirs]

    print("=== runs ===")
    for r in runs:
        print(f"  {r.name}: setting={r.setting} ablation={r.ablation} n_episodes={r.n}")
    print()

    paired_keys = paired_episode_keys(runs)
    if not paired_keys:
        print("WARNING: no (scene_id, episode_id) keys appear in every run — paired analysis is empty.")
        print("Per-run totals follow but pairwise deltas will be NaN.")
    print_per_setting_summary(runs, paired_keys)
    print_pairwise_deltas(runs, paired_keys, args.bootstrap, args.ci)
    print_phase1_gate(runs, paired_keys, args.bootstrap)

    return 0


if __name__ == "__main__":
    sys.exit(main())
