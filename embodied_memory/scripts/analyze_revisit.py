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
import math
import os
import random
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
    # seed-distractors build: a seed-only episode exists ONLY to caption +
    # consolidate a same-category distractor into the LTM (a retrieval-level
    # disambiguation setup). It is NOT a cold/warm visit — drop it BEFORE
    # assign_visit_order so visit_order stays cold(target)=0 / warm(>=1) and the
    # paired n is identical to a non-seeded build (the seed only changes the LTM
    # the warm visit recalls against, never the scored/paired set). The flag rides
    # episode.info -> metadata -> ep_log (set by make_revisit_smoke; Habitat
    # renumbers episode_id, so the flag, not an id substring, is the carrier).
    if raw.get("seed_only"):
        return None
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


def _visit_key(e: RevisitEpisode) -> Tuple[str, str, int]:
    """Renumbering-invariant identity of a visit: ``(scene, category, visit_order)``.

    NOT ``episode_id`` — Habitat renumbers ``episode_id`` per run, so the SAME
    logical visit (same scene, category, and rank-by-pinned-``episode_idx``) can
    carry different ids across settings. Keying on ``episode_id`` silently drops
    such a pair (the M3 stage-1 regression: 3 of 7 warm pairs vanished, the
    strongest cell, skewing the headline). ``visit_order`` is derived from the
    pinned ``episode_idx`` ordering, so it pairs the same physical episode across
    settings regardless of the id label.
    """
    return (e.scene_id, e.target_category, e.visit_order)


def _paired_delta(
    s1: List[RevisitEpisode],
    s3: List[RevisitEpisode],
    select,
    n_bootstrap: int,
    metric: str,
) -> Dict[str, Any]:
    """Paired S3-S1 delta over episodes matching ``select`` (is_warm / is_cold)
    in BOTH runs, keyed on ``_visit_key``.

    Surfaces ``n_dropped`` (visits present in exactly one setting — a genuine
    structural mismatch the caller MUST report, e.g. a crashed episode) plus the
    per-setting counts and the unpaired keys, so a partial pairing can never be
    mistaken for a complete one. ``assign_visit_order`` must have been called on
    both lists. Returns mean / 90% CI / one-sided p(<=0) / n.
    """
    s1_by = {_visit_key(e): e for e in s1 if select(e)}
    s3_by = {_visit_key(e): e for e in s3 if select(e)}
    paired = sorted(set(s1_by) & set(s3_by))
    unpaired = sorted(set(s1_by) ^ set(s3_by))
    # Drop pairs whose metric is non-finite in EITHER setting. A navmesh-
    # unreachable goal gives Infinity geodesic → NaN soft_SPL; a single NaN would
    # otherwise poison the entire paired mean (sum/n propagates NaN). Surfaced as
    # n_nonfinite + a loud warning so a silently-shrunken pairing can't masquerade
    # as the full set.
    keys = [k for k in paired
            if math.isfinite(getattr(s1_by[k], metric))
            and math.isfinite(getattr(s3_by[k], metric))]
    n_nonfinite = len(paired) - len(keys)
    deltas = [getattr(s3_by[k], metric) - getattr(s1_by[k], metric) for k in keys]
    mean, lo, hi = paired_bootstrap_mean_diff(deltas, n_resamples=n_bootstrap, ci=0.9)
    p = _one_sided_p_le_zero(deltas, n_bootstrap)
    return {"n": len(keys), "mean": mean, "lo": lo, "hi": hi, "p_le_zero": p,
            "keys": keys, "deltas": deltas,
            "n_dropped": len(unpaired), "n_s1": len(s1_by), "n_s3": len(s3_by),
            "unpaired_keys": unpaired, "n_nonfinite": n_nonfinite}


def paired_warm_delta(
    s1: List[RevisitEpisode],
    s3: List[RevisitEpisode],
    n_bootstrap: int = 5000,
    metric: str = "soft_spl",
) -> Dict[str, Any]:
    """Paired S3-S1 delta on the metric over episodes that are warm in BOTH runs.

    Pairs on ``(scene_id, target_category, visit_order)`` (see ``_visit_key``).
    """
    return _paired_delta(s1, s3, lambda e: e.is_warm, n_bootstrap, metric)


def paired_cold_delta(
    s1: List[RevisitEpisode],
    s3: List[RevisitEpisode],
    n_bootstrap: int = 5000,
    metric: str = "soft_spl",
) -> Dict[str, Any]:
    """Control: paired S3-S1 delta over episodes that are cold in BOTH runs.

    Memory should be inert on cold visits (no prior same-category sighting), so
    this is the expected-near-zero control for the warm delta. Same keying as
    ``paired_warm_delta``.
    """
    return _paired_delta(s1, s3, lambda e: e.is_cold, n_bootstrap, metric)


# ----------------------------------------------------------------------
# direct two-run comparison (e.g. trained-R vs heuristic-R, same setting)
# ----------------------------------------------------------------------


def compare_runs(
    a_eps: List[RevisitEpisode],
    b_eps: List[RevisitEpisode],
    n_bootstrap: int = 5000,
) -> Dict[str, Any]:
    """Paired B - A deltas between two runs of the SAME setting.

    Both runs must cover the SAME episodes (same dataset); pairs on
    ``(scene_id, target_category, visit_order)``. Used to compare two variants of Setting 3 (e.g.
    a trained importance head vs the heuristic) head-to-head, instead of each
    vs the memory-off S1. ``assign_visit_order`` must have been called on both.
    Returns warm/cold deltas on soft-SPL and binary SPL.
    """
    return {
        "warm_soft": paired_warm_delta(a_eps, b_eps, n_bootstrap, metric="soft_spl"),
        "cold_soft": paired_cold_delta(a_eps, b_eps, n_bootstrap, metric="soft_spl"),
        "warm_spl": paired_warm_delta(a_eps, b_eps, n_bootstrap, metric="spl"),
        "cold_spl": paired_cold_delta(a_eps, b_eps, n_bootstrap, metric="spl"),
    }


# Practical-significance floor for the compare verdict, on the soft-SPL [0,1]
# scale. The paired bootstrap can clamp a CI bound at exactly 0 and report
# p(<=0)≈0.000 for a delta of a few ten-thousandths (a floor artifact, not a
# real win/loss) — reported verbatim that misreads as a significant regression.
# The thesis-relevant effects (+0.10..+0.24) are >20x this; a |Δ| below it is a
# tie regardless of the bootstrap sign. (M4's warm Δ=-0.0005 is well below.)
_VERDICT_TIE_BAND = 0.005


def _compare_verdict(wm: Dict, tie_band: float = _VERDICT_TIE_BAND) -> str:
    """Render the warm soft-SPL B−A verdict, treating sub-band |Δ| as a tie."""
    mean = wm["mean"]
    p_le = wm["p_le_zero"]
    if abs(mean) < tie_band:
        return (f"no meaningful warm soft-SPL difference (Δ={mean:+.4f}, "
                f"|Δ|<{tie_band:g} — statistical tie at the floor).")
    if mean > 0 and p_le < 0.1:
        return f"B beats A on warm soft-SPL (p={p_le:.3f})."
    if mean > 0:
        return (f"B higher on warm soft-SPL but not significant "
                f"(p={p_le:.3f}) — directional only.")
    if mean < 0 and (1.0 - p_le) < 0.1:
        return f"A beats B on warm soft-SPL (p={1.0 - p_le:.3f})."
    return "no significant warm soft-SPL difference (statistical tie)."


def print_compare(run_a: RevisitRun, run_b: RevisitRun, n_bootstrap: int) -> None:
    """Report the head-to-head paired delta (B - A) between two same-setting runs."""
    assign_visit_order(run_a.episodes)
    assign_visit_order(run_b.episodes)

    print("=== compare (B - A; same episodes, paired by scene/category/visit) ===")
    print(f"  A = {run_a.name}  (setting={run_a.setting})")
    print(f"  B = {run_b.name}  (setting={run_b.setting})")
    print()

    print("=== warm/cold stratified means ===")
    for tag, r in (("A", run_a), ("B", run_b)):
        summ = stratified_summary(r.episodes)
        print(f"[{tag}: {r.name}]")
        print(_fmt_block("cold", summ["cold"]))
        print(_fmt_block("warm", summ["warm"]))
    print()

    res = compare_runs(run_a.episodes, run_b.episodes, n_bootstrap=n_bootstrap)
    print("=== paired soft-SPL delta B - A, bootstrap, 90% CI ===")
    _print_delta(f"WARM  {run_b.name} - {run_a.name}", res["warm_soft"])
    _print_delta(f"COLD  {run_b.name} - {run_a.name}", res["cold_soft"])
    print()
    print("=== paired binary SPL delta B - A, bootstrap, 90% CI ===")
    _print_delta(f"WARM  {run_b.name} - {run_a.name}", res["warm_spl"])
    _print_delta(f"COLD  {run_b.name} - {run_a.name}", res["cold_spl"])
    print()
    print(f"=== verdict ===\n  {_compare_verdict(res['warm_soft'])}\n")


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
    and the cold control), then a LOUD warning if any visit was unpairable.

    A non-zero ``n_dropped`` means a visit existed in exactly one setting (a
    genuine structural mismatch, e.g. a crashed episode) — the delta is over a
    SUBSET, so we must never let it pass for a complete headline silently."""
    print(f"  {label}: n={res['n']:d}  mean={res['mean']:+.4f}  "
          f"90% CI=[{res['lo']:+.4f}, {res['hi']:+.4f}]  "
          f"one-sided p(<=0)={res['p_le_zero']:.3f}")
    if res.get("n_dropped", 0) > 0:
        print(f"  ⚠️  WARNING [{label}]: {res['n_dropped']} visit(s) unpaired and "
              f"EXCLUDED (S1 had {res['n_s1']}, S3 had {res['n_s3']}, paired {res['n']}) "
              f"— this delta is over a SUBSET, NOT a clean headline. "
              f"Unpaired (scene,category,visit_order): {res['unpaired_keys']}")
    if res.get("n_nonfinite", 0) > 0:
        print(f"  ⚠️  WARNING [{label}]: {res['n_nonfinite']} paired visit(s) had a "
              f"NON-FINITE metric (NaN/Inf — e.g. a navmesh-unreachable goal) and were "
              f"DROPPED so the mean stays finite. The delta is over the remaining n={res['n']}.")


def pool_runs_by_setting(runs: List[RevisitRun]) -> Dict[int, "RevisitRun"]:
    """Bucket runs by setting (1/2/3), POOLING episodes when several run dirs
    share a setting.

    The standard ablation passes one dir per setting. The AudioGoal/revisit
    MATRIX passes one dir per (scene, anomaly_class) CELL × setting (e.g. 6
    cells × {S1,S2,S3} = 18 dirs), so each setting has multiple dirs that must
    be MERGED — not overwritten (the old ``by_setting[s] = r`` silently kept
    only the last cell). Pairing downstream groups by ``(scene_id,
    target_category)``, so pooling cells is correct as long as cells use
    DISTINCT categories (one category per class). One dir per setting is
    unchanged (single name preserved, single episode list)."""
    pooled_eps: Dict[int, List[RevisitEpisode]] = {}
    pooled_names: Dict[int, List[str]] = {}
    for r in runs:
        if r.setting not in (1, 2, 3):
            continue
        s = int(r.setting)
        pooled_eps.setdefault(s, []).extend(r.episodes)
        pooled_names.setdefault(s, []).append(r.name)
    out: Dict[int, RevisitRun] = {}
    for s, eps in pooled_eps.items():
        names = pooled_names[s]
        name = names[0] if len(names) == 1 else f"{len(names)}cells-pooled"
        out[s] = RevisitRun(name=name, path="", setting=s, episodes=eps)
    return out


def pool_dirs(paths: List[str], label: str) -> "RevisitRun":
    """Load several run dirs and POOL their episodes into one ``RevisitRun``.

    Used by the pooled head-to-head compare (``--compare-a`` / ``--compare-b``):
    the per-cell S3 dirs of one arm (e.g. temporal-context-on) are pooled into a
    single run, then compared paired against the other arm's pooled run. Pairing
    downstream keys on ``(scene_id, target_category, visit_order)``, so pooling
    distinct-category cells is correct (no key collisions). Setting is taken from
    the first dir (both arms are normally S3)."""
    eps: List[RevisitEpisode] = []
    setting: Optional[int] = None
    for p in paths:
        r = load_revisit_run(p)
        if setting is None:
            setting = r.setting
        eps.extend(r.episodes)
    return RevisitRun(name=label, path="", setting=setting, episodes=eps)


# ----------------------------------------------------------------------
# rliable-style robust statistics + power accounting (opt-in --power)
# ----------------------------------------------------------------------
#
# In the spirit of Agarwal et al., "Deep RL at the Edge of the Statistical
# Precipice" (rliable, NeurIPS 2021): report robust interval estimates rather
# than bare point means on a small sample. Two of rliable's four ideas transfer
# to our PAIRED single-task setting and are computed here on the per-pair delta
# list that ``_paired_delta`` already returns:
#   * IQM       — interquartile mean (mean of the central 50% of the deltas),
#                 insensitive to a few outlier pairs.
#   * P(improve)— probability of improvement P(S3>S1): fraction of warm pairs
#                 with a positive delta (ties 0.5). The single-task PAIRED
#                 specialization of rliable's across-task P(X>Y).
# Stratified-across-task bootstrap CIs and performance profiles do NOT apply
# (one task; the paired pairing IS the right stratification) — we say so rather
# than fake them. All stdlib (the analyzer deliberately avoids the faiss-pulling
# package __init__); no rliable/numpy dependency.

_Z_05 = 1.6449   # one-sided z at alpha=0.05
_Z_20 = 0.8416   # z at beta=0.20 (i.e. 80% power)


def _iqm(vals: List[float]) -> float:
    """Interquartile mean: mean of the central 50% of ``vals``.

    Drops ``n//4`` from each end of the sorted list. For ``n < 4`` the quartile
    is 0, so this degenerates to the plain mean (nothing to trim) — annotate
    that at the call site so the IQM column is not mistaken for independent
    signal at tiny n.
    """
    n = len(vals)
    if n == 0:
        return float("nan")
    s = sorted(vals)
    lo = n // 4
    hi = n - lo
    central = s[lo:hi] if hi > lo else s
    return sum(central) / len(central)


def _prob_improvement(deltas: List[float]) -> float:
    """P(S3>S1): fraction of pairs with delta>0 (ties counted 0.5)."""
    n = len(deltas)
    if n == 0:
        return float("nan")
    return sum(1.0 if d > 0 else 0.5 if d == 0 else 0.0 for d in deltas) / n


def _bootstrap_stat(deltas, stat_fn, n_resamples: int = 5000,
                    ci: float = 0.9, seed: int = 0):
    """``(point, lo, hi)`` of a percentile bootstrap of ``stat_fn`` over
    resampled deltas. Mirrors ``analyze_ablation.paired_bootstrap_mean_diff``
    exactly (same seeded RNG, same percentile-index arithmetic) so the new CIs
    are consistent with the existing mean CIs. ``point`` is ``stat_fn`` on the
    full sample; ``lo/hi`` are NaN for ``n < 2``.
    """
    n = len(deltas)
    if n == 0:
        nan = float("nan")
        return nan, nan, nan
    point = stat_fn(deltas)
    if n < 2:
        return point, float("nan"), float("nan")
    rng = random.Random(seed)
    stats: List[float] = []
    for _ in range(n_resamples):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        stats.append(stat_fn(sample))
    stats.sort()
    alpha = (1.0 - ci) / 2.0
    lo = stats[int(alpha * n_resamples)]
    hi = stats[int((1.0 - alpha) * n_resamples) - 1]
    return point, lo, hi


def _sample_sd(vals: List[float]) -> float:
    n = len(vals)
    if n < 2:
        return float("nan")
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return math.sqrt(var)


def _power_note(deltas: List[float]) -> Dict[str, Any]:
    """Normal-approximation power accounting on the paired-delta sample.

    Returns the realized sample SD, the minimum effect detectable at one-sided
    p<0.05 (``mde_05 = z_.05 * sd / sqrt(n)``), the effect detectable with 80%
    power (``mdes_80 = (z_.05 + z_.20) * sd / sqrt(n)``), and whether the
    observed mean clears ``mdes_80``. This is a PLANNING HEURISTIC — the
    reported p-values come from the paired bootstrap, not this approximation,
    and the SD is itself estimated from the same small sample.
    """
    n = len(deltas)
    mean = sum(deltas) / n if n else float("nan")
    sd = _sample_sd(deltas)
    if n < 2 or not math.isfinite(sd):
        return {"n": n, "mean": mean, "sd": sd,
                "mde_05": float("nan"), "mdes_80": float("nan"),
                "adequately_powered": False}
    rootn = math.sqrt(n)
    mde_05 = _Z_05 * sd / rootn
    mdes_80 = (_Z_05 + _Z_20) * sd / rootn
    # Zero variance (every pair identical, e.g. an inert cold control at exactly
    # 0.000) is degenerate, not "powered" — the normal approximation says nothing.
    powered = bool(sd > 0.0 and math.isfinite(mean) and abs(mean) >= mdes_80)
    return {"n": n, "mean": mean, "sd": sd, "mde_05": mde_05, "mdes_80": mdes_80,
            "degenerate": sd == 0.0, "adequately_powered": powered}


def _fmt_ci(point: float, lo: float, hi: float) -> str:
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return f"{point:+.4f} [CI n/a, n<2]"
    return f"{point:+.4f} [90% CI {lo:+.4f}, {hi:+.4f}]"


def print_power_block(warm: Dict[str, Any], cold: Dict[str, Any],
                      n_bootstrap: int) -> None:
    """rliable-style robustness + power readout on the warm/cold soft-SPL
    deltas. Opt-in (``analyze_revisit --power``); prints nothing on the default
    path so existing output stays byte-identical.
    """
    print("=== rliable-style robustness + power (soft-SPL S3-S1) ===")
    for tag, res in (("WARM", warm), ("COLD", cold)):
        deltas = list(res.get("deltas", []))
        n = len(deltas)
        if n == 0:
            print(f"  [{tag}] no paired deltas")
            continue
        iqm_pt, iqm_lo, iqm_hi = _bootstrap_stat(deltas, _iqm, n_resamples=n_bootstrap)
        pi_pt, pi_lo, pi_hi = _bootstrap_stat(deltas, _prob_improvement,
                                              n_resamples=n_bootstrap)
        note = _power_note(deltas)
        mean = res.get("mean", note["mean"])
        print(f"  [{tag}] n={n}  mean={mean:+.4f}")
        iqm_tail = "  (IQM==mean for n<4)" if n < 4 else ""
        print(f"    IQM        = {_fmt_ci(iqm_pt, iqm_lo, iqm_hi)}{iqm_tail}")
        print(f"    P(improve) = {_fmt_ci(pi_pt, pi_lo, pi_hi)}   (P(S3>S1), ties=0.5)")
        if note.get("degenerate"):
            print("    power: zero variance (all paired deltas identical) — "
                  "MDES undefined")
        elif math.isfinite(note["mdes_80"]):
            verdict = ("adequately powered" if note["adequately_powered"]
                       else "UNDERPOWERED for an effect this size")
            rel = ">=" if note["adequately_powered"] else "<"
            print(f"    power: sd={note['sd']:.4f}  MDE(p<.05)={note['mde_05']:.4f}  "
                  f"MDES(80%)={note['mdes_80']:.4f}  -> |mean| {rel} MDES ({verdict})")
        else:
            print("    power: n<2 — cannot estimate SD/MDES")
    print("  NOTE: IQM + P(improve) are rliable-style robust stats on the paired")
    print("  delta sample; performance profiles / across-task stratified CIs do")
    print("  NOT apply (single task). MDES is a normal-approx planning heuristic;")
    print("  the reported p-values come from the paired bootstrap, not this approx.")
    print()


def print_report(runs: List[RevisitRun], n_bootstrap: int,
                 power: bool = False) -> str:
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

    # Pool dirs per setting (matrix mode merges multiple (scene,class) cells per
    # setting; standard ablation has one dir per setting → unchanged).
    by_setting = pool_runs_by_setting(runs)

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

    # rliable-style robustness + power (opt-in --power; default path unchanged).
    if power:
        print_power_block(warm, cold, n_bootstrap)

    # --- paired binary SPL block (precision-bound metric) ---
    warm_b = paired_warm_delta(s1.episodes, s3.episodes,
                               n_bootstrap=n_bootstrap, metric="spl")
    cold_b = paired_cold_delta(s1.episodes, s3.episodes,
                               n_bootstrap=n_bootstrap, metric="spl")
    print("=== paired binary SPL delta, bootstrap, 90% CI ===")
    _print_delta("WARM binary S3 - S1 (full vs memory-off; binary precision)", warm_b)
    if s2 is not None:
        warm_b_s2_s1 = paired_warm_delta(s1.episodes, s2.episodes,
                                         n_bootstrap=n_bootstrap, metric="spl")
        warm_b_s3_s2 = paired_warm_delta(s2.episodes, s3.episodes,
                                         n_bootstrap=n_bootstrap, metric="spl")
        _print_delta("WARM binary S2 - S1", warm_b_s2_s1)
        _print_delta("WARM binary S3 - S2", warm_b_s3_s2)
    _print_delta("COLD binary S3 - S1 (control, expect ~0)", cold_b)
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
    parser.add_argument("run_dirs", nargs="*", help="Run directories (>=2; need S1 and S3).")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--power", action="store_true",
                        help="Append an rliable-style robustness + power block "
                             "(IQM, probability-of-improvement P(S3>S1), and an "
                             "MDES/power note) to the S1/S2/S3 report. Opt-in; the "
                             "default output is byte-identical without it.")
    parser.add_argument("--compare", action="store_true",
                        help="Head-to-head paired B - A delta between exactly two "
                             "same-setting runs (e.g. heuristic-R vs trained-R S3), "
                             "instead of the S1/S2/S3 Gate-A report.")
    parser.add_argument("--compare-a", nargs="+", metavar="DIR",
                        help="POOLED head-to-head: baseline run dirs (group A). Each "
                             "arm's per-cell S3 dirs are pooled into one run, then "
                             "compared paired B-A (e.g. temporal-context S3 vs baseline "
                             "S3 across the matrix). Pair with --compare-b.")
    parser.add_argument("--compare-b", nargs="+", metavar="DIR",
                        help="POOLED head-to-head: variant run dirs (group B).")
    args = parser.parse_args(argv)

    # Pooled head-to-head (matrix-wide temporal-vs-baseline A/B).
    if args.compare_a or args.compare_b:
        if not (args.compare_a and args.compare_b):
            parser.error("--compare-a and --compare-b must be given together")
        run_a = pool_dirs(args.compare_a, f"A:{len(args.compare_a)}cells-pooled")
        run_b = pool_dirs(args.compare_b, f"B:{len(args.compare_b)}cells-pooled")
        print_compare(run_a, run_b, args.bootstrap)
        return 0

    if len(args.run_dirs) < 2:
        parser.error("at least two run directories are required")

    if args.compare:
        if len(args.run_dirs) != 2:
            parser.error("--compare needs exactly two run directories (A B)")
        run_a, run_b = (load_revisit_run(p) for p in args.run_dirs)
        print_compare(run_a, run_b, args.bootstrap)
        return 0

    runs = [load_revisit_run(p) for p in args.run_dirs]
    print_report(runs, args.bootstrap, power=args.power)
    return 0


if __name__ == "__main__":
    sys.exit(main())
