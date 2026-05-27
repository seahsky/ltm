"""
Visit-order ("lifelong / revisit") analysis for the LTM-embodied ablation.

Phase-2 closed with an honest negative: the hierarchical LTM is net-neutral on
the HM3D ``val_mini`` ObjectNav ablation (soft-SPL S3-S1 ~= -0.009, n.s.). The
suspected cause is *structural*, not mechanical: ObjectNav is single-goal-per-
episode and ``val_mini`` goals barely recur, so the LTM's one real value —
recalling a past sighting of the goal — almost never applies.

The LTM, however, already persists across episodes within a run
(``EmbodiedMemoryBridge`` is built once; the FAISS layers are never cleared;
recall is scene-filtered). So the existing G4 runs *already contain* warm
revisits — e.g. chair/bed recur 4x in ``wcojb4TFT35``, plant 5x in
``TEEsavR23oF``. They were just never analysed by visit order.

This script mines that, read-only. For each run it groups episodes by
``(scene_id, target_category)``, orders them by ``episode_idx``, and assigns a
*visit order*: 0 = first occurrence ("cold" — LTM holds no prior sighting of
this category in this scene), >=1 = "warm" revisit (LTM may hold a sighting
from an earlier same-category visit). It then reports, stratified by cold vs
warm, the soft-SPL / reach diagnostics for each setting and the paired
warm-visit soft-SPL delta S3-S1 (bootstrap CI + one-sided p) — the regime
where memory can actually pay off — plus whether memory fired on warm visits.

It reuses ``analyze_ablation``'s loaders + bootstrap and touches no production
code.

Gate A classification (printed verdict):
  (a) memory fires on warm visits AND warm-S3 > warm-S1   -> green light Phase C
  (b) memory fires on warm visits but does NOT help/hurts -> diagnose first
  (c) memory rarely/never fires on warm visits            -> cold-visit confound;
                                                             run Phase B controlled

Usage:
    python embodied_memory/scripts/analyze_revisit.py \
        runs/abl-s1-qwen runs/abl-s2-qwen runs/abl-s3-qwen

Output is a plain-text report on stdout — no plotting deps.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# analyze_ablation lives next to this file and is import-clean (stdlib only),
# so adding the script dir to sys.path lets us reuse its loaders + bootstrap
# without triggering the embodied_memory package __init__ (which pulls faiss).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_ablation import (  # noqa: E402
    _coerce_episode,
    _one_sided_p_le_zero,
    paired_bootstrap_mean_diff,
)


# ----------------------------------------------------------------------
# data model
# ----------------------------------------------------------------------


@dataclass
class RevisitEpisode:
    scene_id: str
    episode_id: str
    target_category: str
    episode_idx: int
    soft_spl: float
    spl: float
    success: bool
    n_steps: int
    min_d2g: float
    success_1m: bool
    n_memory_chosen: int = 0
    n_memory_candidates: int = 0
    # count of decisions whose chosen_source == "memory" (per-episode trace)
    n_memory_decisions: int = 0
    # assigned by assign_visit_order: 0 = cold (first sighting), >=1 = warm
    visit_order: int = -1

    @property
    def is_cold(self) -> bool:
        return self.visit_order == 0

    @property
    def is_warm(self) -> bool:
        return self.visit_order >= 1

    @property
    def memory_fired(self) -> bool:
        return self.n_memory_chosen > 0


@dataclass
class RevisitRun:
    name: str
    path: str
    setting: Optional[int]
    episodes: List[RevisitEpisode] = field(default_factory=list)


# ----------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------


_SETTING_RE = re.compile(r"s([123])\b|s([123])(?:[-_]|$)")


def _infer_setting(path: str, summary: Dict[str, Any]) -> Optional[int]:
    ablation = summary.get("ablation")
    if isinstance(ablation, dict):
        s = ablation.get("setting")
        if s in (1, 2, 3):
            return int(s)
    # fall back to a `-s<N>` token in the directory name
    # (e.g. abl-s3-qwen, revisit-smoke-chair-s1)
    base = os.path.basename(os.path.normpath(path))
    m = re.search(r"[-_]s([123])(?:[-_]|$)", base)
    if m:
        return int(m.group(1))
    return None


def _raw_to_episode(raw: Dict[str, Any]) -> Optional[RevisitEpisode]:
    base = _coerce_episode(raw)
    if base is None:
        return None
    cat = raw.get("target_category") or raw.get("object_category") or "?"
    idx_raw = raw.get("episode_idx")
    if idx_raw is None:
        idx_raw = raw.get("episode_index")
    try:
        idx = int(idx_raw) if idx_raw is not None else 0
    except (TypeError, ValueError):
        idx = 0
    decisions = raw.get("decisions") or []
    n_mem_dec = sum(
        1 for d in decisions if isinstance(d, dict) and d.get("chosen_source") == "memory"
    )
    return RevisitEpisode(
        scene_id=base.scene_id,
        episode_id=base.episode_id,
        target_category=str(cat),
        episode_idx=idx,
        soft_spl=base.soft_spl,
        spl=base.spl,
        success=base.success,
        n_steps=base.n_steps,
        min_d2g=base.min_distance_to_goal,
        success_1m=base.success_1m,
        n_memory_chosen=base.n_memory_chosen,
        n_memory_candidates=int(raw.get("n_memory_candidates", 0) or 0),
        n_memory_decisions=n_mem_dec,
    )


def load_revisit_run(path: str) -> RevisitRun:
    """Load a run dir into a ``RevisitRun``.

    Prefers per-episode ``episode_*.json`` files (they carry ``decisions[]``,
    ``target_category`` and ``episode_idx``); falls back to the ``episodes``
    array embedded in ``summary.json`` for runs without per-episode files.
    """
    summary_path = os.path.join(path, "summary.json")
    summary: Dict[str, Any] = {}
    if os.path.isfile(summary_path):
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except (OSError, json.JSONDecodeError):
            summary = {}

    episodes: List[RevisitEpisode] = []
    files = sorted(glob.glob(os.path.join(path, "episode_*.json")))
    if files:
        for ep_path in files:
            try:
                with open(ep_path) as f:
                    raw = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            ep = _raw_to_episode(raw)
            if ep is not None:
                episodes.append(ep)
    else:
        for raw in summary.get("episodes") or []:
            ep = _raw_to_episode(raw)
            if ep is not None:
                episodes.append(ep)

    return RevisitRun(
        name=os.path.basename(os.path.normpath(path)) or path,
        path=path,
        setting=_infer_setting(path, summary),
        episodes=episodes,
    )


# ----------------------------------------------------------------------
# visit-order assignment + stratification
# ----------------------------------------------------------------------


def assign_visit_order(episodes: List[RevisitEpisode]) -> List[RevisitEpisode]:
    """Assign ``visit_order`` per ``(scene_id, target_category)`` group.

    Within each group, episodes are ordered by ``episode_idx`` (ties broken by
    ``episode_id``); the earliest gets visit_order 0 ("cold"), the rest 1, 2,
    ... ("warm"). Mutates in place and returns the list for convenience.
    """
    groups: Dict[Tuple[str, str], List[RevisitEpisode]] = {}
    for e in episodes:
        groups.setdefault((e.scene_id, e.target_category), []).append(e)
    for grp in groups.values():
        grp.sort(key=lambda e: (e.episode_idx, e.episode_id))
        for order, e in enumerate(grp):
            e.visit_order = order
    return episodes


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def _strata_block(eps: List[RevisitEpisode]) -> Dict[str, float]:
    return {
        "n": len(eps),
        "soft_spl": _mean([e.soft_spl for e in eps]),
        "spl": _mean([e.spl for e in eps]),
        "min_d2g": _mean([e.min_d2g for e in eps]),
        "success_1m": _mean([1.0 if e.success_1m else 0.0 for e in eps]),
        "n_steps": _mean([float(e.n_steps) for e in eps]),
        "memory_fire_rate": _mean([1.0 if e.memory_fired else 0.0 for e in eps]),
        "n_mem_chosen": sum(e.n_memory_chosen for e in eps),
    }


def stratified_summary(episodes: List[RevisitEpisode]) -> Dict[str, Dict[str, float]]:
    """Split episodes into cold (visit_order 0) and warm (>=1) and report means.

    ``assign_visit_order`` must have been called first.
    """
    cold = [e for e in episodes if e.is_cold]
    warm = [e for e in episodes if e.is_warm]
    return {"cold": _strata_block(cold), "warm": _strata_block(warm)}


# ----------------------------------------------------------------------
# paired warm-visit delta
# ----------------------------------------------------------------------


def paired_warm_delta(
    s1: List[RevisitEpisode],
    s3: List[RevisitEpisode],
    n_bootstrap: int = 5000,
    metric: str = "soft_spl",
) -> Dict[str, Any]:
    """Paired S3-S1 delta on the metric over episodes that are warm in BOTH runs.

    Pairs on ``(scene_id, episode_id)``. ``assign_visit_order`` must have been
    called on both lists. Returns mean / 90% CI / one-sided p(<=0) / n.
    """
    s1_by = {(e.scene_id, e.episode_id): e for e in s1 if e.is_warm}
    s3_by = {(e.scene_id, e.episode_id): e for e in s3 if e.is_warm}
    keys = sorted(set(s1_by) & set(s3_by))
    deltas = [getattr(s3_by[k], metric) - getattr(s1_by[k], metric) for k in keys]
    mean, lo, hi = paired_bootstrap_mean_diff(deltas, n_resamples=n_bootstrap, ci=0.9)
    p = _one_sided_p_le_zero(deltas, n_bootstrap)
    return {"n": len(keys), "mean": mean, "lo": lo, "hi": hi, "p_le_zero": p,
            "keys": keys, "deltas": deltas}


def paired_cold_delta(
    s1: List[RevisitEpisode],
    s3: List[RevisitEpisode],
    n_bootstrap: int = 5000,
    metric: str = "soft_spl",
) -> Dict[str, Any]:
    """Control: paired S3-S1 delta over episodes that are cold in BOTH runs.

    Memory should be inert on cold visits (no prior same-category sighting), so
    this is the expected-near-zero control for the warm delta.
    """
    s1_by = {(e.scene_id, e.episode_id): e for e in s1 if e.is_cold}
    s3_by = {(e.scene_id, e.episode_id): e for e in s3 if e.is_cold}
    keys = sorted(set(s1_by) & set(s3_by))
    deltas = [getattr(s3_by[k], metric) - getattr(s1_by[k], metric) for k in keys]
    mean, lo, hi = paired_bootstrap_mean_diff(deltas, n_resamples=n_bootstrap, ci=0.9)
    p = _one_sided_p_le_zero(deltas, n_bootstrap)
    return {"n": len(keys), "mean": mean, "lo": lo, "hi": hi, "p_le_zero": p,
            "keys": keys, "deltas": deltas}


# ----------------------------------------------------------------------
# Gate A classification
# ----------------------------------------------------------------------


def classify_gate_a(
    warm_fire_rate: float,
    warm_delta_mean: float,
    warm_delta_p: float,
    fire_threshold: float = 0.25,
) -> str:
    """Classify the Phase-A outcome into (a) / (b) / (c).

    (c) memory rarely fires on warm visits (fire_rate < threshold) -> the cold
        visit likely never saw the goal, so warm memory was empty: a confound,
        not a refutation. Resolve with Phase B (controlled starts).
    (a) memory fires AND warm-S3 > warm-S1 (delta mean > 0) -> green light.
    (b) memory fires but does not help (delta mean <= 0) -> deeper issue;
        diagnose before building.

    ``warm_delta_p`` is reported for confidence but does not change the
    a/b/c branch (a positive-but-n.s. delta is still "fires and helps,
    direction-positive" — Phase C exists to get a powered number).
    """
    if warm_fire_rate < fire_threshold:
        return "c"
    if warm_delta_mean > 0.0:
        return "a"
    return "b"


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------


def _fmt_block(label: str, b: Dict[str, float]) -> str:
    return (
        f"  {label:<6} n={int(b['n']):<3d} "
        f"soft_SPL={b['soft_spl']:+.4f}  SPL={b['spl']:+.4f}  "
        f"succ@1m={b['success_1m']:.3f}  min_d2g={b['min_d2g']:.3f}m  "
        f"steps={b['n_steps']:.1f}  mem_fire_rate={b['memory_fire_rate']:.3f}  "
        f"mem_chosen={int(b['n_mem_chosen'])}"
    )


def print_visit_distribution(run: RevisitRun) -> None:
    groups: Dict[Tuple[str, str], List[int]] = {}
    for e in run.episodes:
        groups.setdefault((e.scene_id, e.target_category), []).append(e.visit_order)
    n_cold = sum(1 for e in run.episodes if e.is_cold)
    n_warm = sum(1 for e in run.episodes if e.is_warm)
    print(f"  visit groups (scene, category): {len(groups)}  "
          f"cold={n_cold}  warm={n_warm}")


def _print_delta(label: str, res: Dict[str, Any]) -> None:
    """Print one paired-delta block uniformly (used for S3-S1, S2-S1, S3-S2,
    and the cold control)."""
    print(f"  {label}: n={res['n']:d}  mean={res['mean']:+.4f}  "
          f"90% CI=[{res['lo']:+.4f}, {res['hi']:+.4f}]  "
          f"one-sided p(<=0)={res['p_le_zero']:.3f}")


def print_report(runs: List[RevisitRun], n_bootstrap: int) -> str:
    """Print the full Phase-A report and return the Gate A classification."""
    for r in runs:
        assign_visit_order(r.episodes)

    print("=== runs ===")
    for r in runs:
        print(f"  {r.name}: setting={r.setting} n_episodes={len(r.episodes)}")
        print_visit_distribution(r)
    print()

    print("=== cold vs warm stratified means (per setting) ===")
    for r in runs:
        summ = stratified_summary(r.episodes)
        print(f"[{r.name}  setting={r.setting}]")
        print(_fmt_block("cold", summ["cold"]))
        print(_fmt_block("warm", summ["warm"]))
    print()

    by_setting: Dict[int, RevisitRun] = {}
    for r in runs:
        if r.setting in (1, 2, 3):
            by_setting[int(r.setting)] = r

    if 1 not in by_setting or 3 not in by_setting:
        print("(skip warm delta + Gate A: need both setting 1 and setting 3 runs.)")
        return "skip"

    s1, s3 = by_setting[1], by_setting[3]
    s2 = by_setting.get(2)

    warm = paired_warm_delta(s1.episodes, s3.episodes, n_bootstrap=n_bootstrap)
    cold = paired_cold_delta(s1.episodes, s3.episodes, n_bootstrap=n_bootstrap)

    print("=== paired soft-SPL delta, bootstrap, 90% CI ===")
    _print_delta("WARM S3 - S1 (full vs memory-off; PRIMARY gate)", warm)
    if s2 is not None:
        warm_s2_s1 = paired_warm_delta(s1.episodes, s2.episodes, n_bootstrap=n_bootstrap)
        warm_s3_s2 = paired_warm_delta(s2.episodes, s3.episodes, n_bootstrap=n_bootstrap)
        _print_delta("WARM S2 - S1 (STM-only effect; module 1)", warm_s2_s1)
        _print_delta("WARM S3 - S2 (LTM-specific: consolidation+LTM+rerank)", warm_s3_s2)
    _print_delta("COLD S3 - S1 (control, expect ~0)", cold)
    print()

    # memory firing on warm visits in S3 (the full system)
    s3_warm = [e for e in s3.episodes if e.is_warm]
    warm_fire_rate = _mean([1.0 if e.memory_fired else 0.0 for e in s3_warm]) if s3_warm else 0.0
    n_warm_fired = sum(1 for e in s3_warm if e.memory_fired)
    print("=== memory firing on warm visits (S3, full system) ===")
    print(f"  warm visits: {len(s3_warm)}   fired (n_memory_chosen>0): {n_warm_fired}   "
          f"fire_rate={warm_fire_rate:.3f}")
    print("  NOTE: source-episode attribution (was the recalled waypoint from an")
    print("  earlier *same-category* episode?) is NOT recoverable from these runs —")
    print("  the serialized decisions[] trace drops the candidate's ltm_episode_id.")
    print("  Phase C adds that field; here we report only that memory fired.")
    print()

    gate = classify_gate_a(warm_fire_rate, warm["mean"], warm["p_le_zero"])
    verdicts = {
        "a": "memory FIRES on warm visits AND warm-S3 > warm-S1 "
             "-> GREEN LIGHT: build the controlled revisit eval (Phase C).",
        "b": "memory FIRES on warm visits but does NOT help (delta <= 0) "
             "-> diagnose (wrong-instance recall / detour cost) with "
             "inspect_memory_rerank.py before building.",
        "c": "memory RARELY FIRES on warm visits "
             "-> likely the cold visit never saw the goal (empty warm memory). "
             "Run Phase B with controlled starts that guarantee a cold sighting.",
    }
    print("=== Gate A verdict ===")
    print(f"  outcome: ({gate})  {verdicts.get(gate, '')}")
    if gate == "a" and warm["p_le_zero"] >= 0.1:
        print("  (direction positive but not yet significant at p<0.1 — expected on "
              "this small, interleaved sample; Phase C powers it up.)")
    print()
    return gate


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Visit-order (revisit) ablation analysis")
    parser.add_argument("run_dirs", nargs="+", help="Run directories (>=2; need S1 and S3).")
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args(argv)

    if len(args.run_dirs) < 2:
        parser.error("at least two run directories are required")

    runs = [load_revisit_run(p) for p in args.run_dirs]
    print_report(runs, args.bootstrap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
