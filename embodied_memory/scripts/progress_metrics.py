"""
Live metric readout for a run IN PROGRESS, from the per-episode JSONs on disk.

`analyze_ablation.py` is the end-of-run analyzer and needs `summary.json`, which
the runner only writes when the arm finishes (episode_runner.py:1143). A run that
has been going for days has no summary yet, so this reads the per-episode logs
(episode_runner.py:978) and reports the partial numbers.

Reports CONTEXT.md's four headline metrics plus Cost:

  Cost              steps-to-complete and wall-clock per episode
  Benchmark SPL     native binary `spl`, STOP within 0.1 m — the ring VLFM's
                    0.304 and VLingNav's 0.429 are measured at
  soft-SPL          graded path efficiency, the primary science metric
  Find-SR           primary find-task completion; 1.0 m primary, 0.1 m diagnostic
  Anomaly-response SR   controller ran the full loop (investigated AND resumed)

Two honesty rules from CONTEXT.md are enforced in the output rather than left to
the reader:

  * `success_1m` is a STOP-INDEPENDENT closest-approach diagnostic and is NEVER
    a success rate. It is printed, labelled, and never folded into SR.
  * Anomaly-response SR here does NOT verify onset provenance (whether the
    interrupt fired on the anomaly or on something before `t_anom`). That check
    lives in the `[audio] onset @step` log line, not in the episode JSON, and its
    absence is what invalidated the n=64 census (ADR-0003/0004). The caveat is
    printed alongside the number.

Run:   PYTHONPATH=. python embodied_memory/scripts/progress_metrics.py runs/r1v1-s1
Tests: PYTHONPATH=. python embodied_memory/scripts/test_progress_metrics.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

EpisodeLog = Dict[str, Any]


def _mean(values: Sequence[float]) -> Optional[float]:
    """Arithmetic mean, or None for an empty sequence. Pure."""
    return (sum(values) / len(values)) if values else None


def _median(values: Sequence[float]) -> Optional[float]:
    """Median, or None for an empty sequence. Pure."""
    if not values:
        return None
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0


def _numbers(episodes: Sequence[EpisodeLog], key: str) -> List[float]:
    """Every finite numeric value of ``key``, skipping missing/None/bool. Pure."""
    out: List[float] = []
    for ep in episodes:
        v = ep.get(key)
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def _rate(episodes: Sequence[EpisodeLog], key: str) -> Optional[float]:
    """Fraction of episodes whose ``key`` is truthy, over episodes that HAVE the
    key. None when no episode carries it. Pure."""
    present = [ep for ep in episodes if ep.get(key) is not None]
    if not present:
        return None
    return sum(1 for ep in present if ep.get(key)) / len(present)


def _reports(episodes: Sequence[EpisodeLog]) -> List[EpisodeLog]:
    """The anomaly_report sub-objects, for episodes that carry one. Pure."""
    out: List[EpisodeLog] = []
    for ep in episodes:
        rep = ep.get("anomaly_report")
        if isinstance(rep, dict):
            out.append(rep)
    return out


@dataclass(frozen=True)
class RunMetrics:
    """Partial-run aggregate. Every metric is Optional: a field is None when no
    completed episode carries it (e.g. the controller fields on an objectnav
    run), never 0.0, so 'absent' is never read as 'zero'."""

    n_episodes: int
    n_scenes: int
    last_scene: Optional[str]

    # Cost
    mean_steps: Optional[float]
    median_steps: Optional[float]
    mean_wallclock_min: Optional[float]
    total_wallclock_h: Optional[float]

    # Benchmark SPL (0.1 m ring) — cross-quotable to VLFM / VLingNav
    benchmark_spl: Optional[float]
    sr_01m: Optional[float]

    # soft-SPL — primary science metric
    soft_spl: Optional[float]

    # Reach diagnostic — STOP-independent, NOT a success rate
    reach_1m: Optional[float]
    mean_min_d2g: Optional[float]

    # Mission
    n_controller_episodes: int
    find_sr_1m: Optional[float]
    find_sr_01m: Optional[float]
    anomaly_response_sr: Optional[float]
    investigated_rate: Optional[float]
    resumed_rate: Optional[float]
    aborted_rate: Optional[float]
    n_benign_ignored: int


def aggregate(episodes: Sequence[EpisodeLog]) -> RunMetrics:
    """Aggregate completed per-episode logs into :class:`RunMetrics`.

    Pure: reads the sequence, mutates nothing, does no I/O.

    Anomaly-response SR is ``investigated AND resumed`` — CONTEXT.md's full loop
    (onset -> investigated -> resumed). ``investigate_aborted`` episodes are
    counted in the denominator, since giving up the detour on the step budget is
    a controller failure, not a missing measurement.
    """
    finished = sorted(_numbers(episodes, "finished_at"))
    wall_min: Optional[float] = None
    wall_h: Optional[float] = None
    if len(finished) >= 2:
        span = finished[-1] - finished[0]
        wall_min = span / 60.0 / (len(finished) - 1)
        wall_h = span / 3600.0

    scenes = [ep.get("scene_id") for ep in episodes if ep.get("scene_id")]
    reports = _reports(episodes)

    anomaly_sr: Optional[float] = None
    if reports:
        anomaly_sr = sum(
            1 for r in reports if r.get("investigated") and r.get("resumed")
        ) / len(reports)

    return RunMetrics(
        n_episodes=len(episodes),
        n_scenes=len(set(scenes)),
        last_scene=(scenes[-1] if scenes else None),
        mean_steps=_mean(_numbers(episodes, "n_steps")),
        median_steps=_median(_numbers(episodes, "n_steps")),
        mean_wallclock_min=wall_min,
        total_wallclock_h=wall_h,
        benchmark_spl=_mean(_numbers(episodes, "spl")),
        sr_01m=_rate(episodes, "success"),
        soft_spl=_mean(_numbers(episodes, "soft_spl")),
        reach_1m=_rate(episodes, "success_1m"),
        mean_min_d2g=_mean(_numbers(episodes, "min_distance_to_goal")),
        n_controller_episodes=len(reports),
        find_sr_1m=_rate(reports, "primary_completed_1m"),
        find_sr_01m=_rate(reports, "primary_completed"),
        anomaly_response_sr=anomaly_sr,
        investigated_rate=_rate(reports, "investigated"),
        resumed_rate=_rate(reports, "resumed"),
        aborted_rate=_rate(reports, "investigate_aborted"),
        n_benign_ignored=int(sum(_numbers(reports, "n_benign_ignored"))),
    )


def _fmt(value: Optional[float], digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def format_metrics(m: RunMetrics, label: str) -> str:
    """Render :class:`RunMetrics` as the operator-facing block. Pure."""
    lines: List[str] = []
    lines.append(f"--- metrics so far: {label} (n={m.n_episodes} completed, "
                 f"{m.n_scenes} scene(s), last={m.last_scene or '?'})")
    if m.n_episodes == 0:
        lines.append("    (no completed episodes yet)")
        return "\n".join(lines)

    lines.append(
        f"  COST            steps mean {_fmt(m.mean_steps, 1)} / median "
        f"{_fmt(m.median_steps, 0)}  |  wall-clock {_fmt(m.mean_wallclock_min, 2)} min/ep, "
        f"{_fmt(m.total_wallclock_h, 1)} h so far")
    lines.append(
        f"  BENCHMARK SPL   {_fmt(m.benchmark_spl)}   SR@0.1m {_fmt(m.sr_01m)}"
        f"      [VLFM 0.304 | VLingNav 0.429, same 0.1 m ring]")
    lines.append(f"  soft-SPL        {_fmt(m.soft_spl)}")

    if m.n_controller_episodes:
        lines.append(
            f"  FIND-SR         @1.0m {_fmt(m.find_sr_1m)}   @0.1m {_fmt(m.find_sr_01m)}")
        lines.append(
            f"  ANOMALY-RESP SR {_fmt(m.anomaly_response_sr)}   "
            f"(investigated {_fmt(m.investigated_rate)} | resumed {_fmt(m.resumed_rate)} | "
            f"aborted {_fmt(m.aborted_rate)} | benign ignored {m.n_benign_ignored})")
        lines.append(
            "                  CAVEAT: onset PROVENANCE is not checked here (it is not in the"
            " episode JSON).")
        lines.append(
            "                  An onset before t_anom cannot be the anomaly — grep the log for"
            " '[audio] onset @step'.")
    else:
        lines.append("  FIND-SR         n/a — no episode carries an anomaly_report "
                     "(objectnav run: use SR@0.1m above)")
        lines.append("  ANOMALY-RESP SR n/a — controller not enabled on this run")

    lines.append(
        f"  reach@1m        {_fmt(m.reach_1m)}   min_d2g mean {_fmt(m.mean_min_d2g, 2)} m"
        f"   <- STOP-INDEPENDENT diagnostic, NOT a success rate")
    return "\n".join(lines)


def load_episodes(run_dir: str) -> List[EpisodeLog]:
    """Read completed per-episode JSONs from ``run_dir`` in episode order.

    Skips ``episode_*_error.json`` (crashed episodes carry no metrics) and any
    file that fails to parse — a run in progress can be caught mid-write.
    """
    out: List[EpisodeLog] = []
    for path in sorted(glob.glob(os.path.join(run_dir, "episode_*.json"))):
        if path.endswith("_error.json"):
            continue
        try:
            with open(path, "r") as fh:
                payload = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict):
            out.append(payload)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: progress_metrics.py <run_dir> [run_dir ...]", file=sys.stderr)
        return 2
    for run_dir in args:
        if not os.path.isdir(run_dir):
            print(f"--- metrics so far: {run_dir} (missing)")
            continue
        print(format_metrics(aggregate(load_episodes(run_dir)), run_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
