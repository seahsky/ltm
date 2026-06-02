"""
diagnose_pipeline.py — component-level diagnostics for the embodied-memory
pipeline, mined entirely from the episode_*.json logs the runner already
writes (no Habitat, no models, no GPU).

Answers three questions per run, decomposing "why is success low" into
observe → store/retrieve → (navigate/terminate is the oracle ladder, separate):

  #1  Target observation rate
      Did the agent's VLM captions ever mention the goal object during the
      episode? If the target never appears in any keyframe caption, memory
      cannot store anything useful — exploration coverage is the bottleneck.

  #2  Retrieval relevance (no-instrumentation audit)
      For every memory-source candidate at a decision, match its world_xy to
      the nearest keyframe across the whole run and read THAT keyframe's
      caption. If retrieved memories' nearest captions are hallways/kitchens
      rather than the target, retrieval is pulling the wrong locations. Also
      reports the memory-candidate cosine (raw_score) distribution.

  #3  Trajectory dump
      Emits per-episode (step_idx, x, z, caption) so trajectories can be
      plotted offline. (GT goal overlay needs a one-line log add — separate.)

Usage::

    python embodied_memory/scripts/diagnose_pipeline.py runs/detector-c9-s3-nodet [more_run_dirs...]
    python embodied_memory/scripts/diagnose_pipeline.py --trajectories runs/detector-c9-s3-nodet

Cold/warm split mirrors the revisit analyzer: group by (scene_id,
target_category), order by episode_idx, first visit = cold, rest = warm.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# Goal-object keyword synonyms (word-boundary matched). Conservative on purpose:
# 'sofa'/'couch' are a SEPARATE HM3D category from 'chair', so they are NOT
# chair synonyms — counting them would mask the wrong-instance problem.
_SYNONYMS: Dict[str, List[str]] = {
    "chair": ["chair", "chairs", "armchair", "armchairs", "stool", "stools"],
    "bed": ["bed", "beds", "mattress", "mattresses"],
    "toilet": ["toilet", "toilets"],
    "plant": ["plant", "plants", "houseplant", "potted plant"],
    "tv_monitor": ["tv", "television", "monitor", "screen"],
    "sofa": ["sofa", "sofas", "couch", "couches"],
}


def caption_mentions(caption: str, category: str) -> bool:
    """True iff ``caption`` mentions the goal object as a whole word.

    Word-boundary matched so 'bedroom' does not count as a 'bed' and
    'chairman' does not count as a 'chair'. Falls back to the bare category
    token when no synonym list exists.
    """
    if not caption:
        return False
    words = _SYNONYMS.get(category, [category])
    text = caption.lower()
    for w in words:
        if re.search(r"\b" + re.escape(w.lower()) + r"\b", text):
            return True
    return False


def classify_visits(episodes: List[Dict[str, Any]]) -> List[bool]:
    """Return per-episode ``is_cold`` flags. Within each (scene_id,
    target_category) group, the lowest episode_idx is the cold (first) visit;
    every later visit is warm. Mirrors analyze_revisit's visit ordering."""
    first_seen: Dict[Tuple[str, str], int] = {}
    for ep in episodes:
        key = (str(ep.get("scene_id")), str(ep.get("target_category")))
        idx = int(ep.get("episode_idx", 0))
        if key not in first_seen or idx < first_seen[key]:
            first_seen[key] = idx
    cold: List[bool] = []
    for ep in episodes:
        key = (str(ep.get("scene_id")), str(ep.get("target_category")))
        cold.append(int(ep.get("episode_idx", 0)) == first_seen[key])
    return cold


def episode_observation(
    steps: List[Dict[str, Any]], category: str
) -> Tuple[bool, float, int]:
    """Return (observed, fraction_of_keyframes_mentioning, n_keyframes) for one
    episode's keyframe ``steps`` (each carrying a ``caption``)."""
    caps = [s.get("caption", "") for s in steps]
    n = len(caps)
    hits = sum(1 for c in caps if caption_mentions(c, category))
    return (hits > 0, (hits / n if n else 0.0), n)


def nearest_caption(
    world_xy, index: List[Tuple[Tuple[float, float], str]]
) -> Optional[str]:
    """Caption of the keyframe whose (x, z) is closest to ``world_xy``."""
    if not index:
        return None
    wx, wz = float(world_xy[0]), float(world_xy[1])
    best_cap, best_d = None, float("inf")
    for (kx, kz), cap in index:
        d = (wx - kx) ** 2 + (wz - kz) ** 2
        if d < best_d:
            best_d, best_cap = d, cap
    return best_cap


def memory_candidate_audit(
    decisions: List[Dict[str, Any]],
    index: List[Tuple[Tuple[float, float], str]],
    category: str,
) -> Dict[str, Any]:
    """Audit memory-source candidates across an episode's decisions.

    Returns counts, the cosine (raw_score) distribution, and ``on_target_rate``
    = fraction of memory candidates whose nearest keyframe caption mentions the
    goal object (the no-instrumentation proxy for "is retrieval relevant?").
    """
    cosines: List[float] = []
    on_target = 0
    n_mem = 0
    for dec in decisions:
        for c in dec.get("candidates", []):
            if c.get("source") != "memory":
                continue
            n_mem += 1
            cosines.append(float(c.get("raw_score", 0.0)))
            cap = nearest_caption(c.get("world_xy", [0, 0]), index)
            if cap and caption_mentions(cap, category):
                on_target += 1
    cosines_sorted = sorted(cosines)

    def _median(xs: List[float]) -> float:
        if not xs:
            return float("nan")
        m = len(xs) // 2
        return xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])

    return {
        "n_memory": n_mem,
        "on_target_rate": (on_target / n_mem if n_mem else 0.0),
        "cos_min": (cosines_sorted[0] if cosines else float("nan")),
        "cos_med": _median(cosines_sorted),
        "cos_max": (cosines_sorted[-1] if cosines else float("nan")),
    }


# ----------------------------------------------------------------------
# file I/O + reporting (not unit-tested; pure helpers above are)
# ----------------------------------------------------------------------


def _load_episodes(run_dir: str) -> List[Dict[str, Any]]:
    eps = []
    for path in sorted(glob.glob(os.path.join(run_dir, "episode_*.json"))):
        if path.endswith("_error.json"):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                eps.append(json.load(f))
        except Exception as e:  # pragma: no cover - defensive
            print(f"  WARN: could not read {path}: {e}")
    return eps


def _keyframe_index(
    episodes: List[Dict[str, Any]]
) -> List[Tuple[Tuple[float, float], str]]:
    """All (x, z), caption pairs across every episode's keyframes in the run.
    Memory persists across episodes within a scene/category, so the source of a
    retrieved waypoint may live in an earlier episode — index the whole run."""
    index = []
    for ep in episodes:
        for s in ep.get("steps", []):
            pos = s.get("agent_pos")
            if pos and len(pos) >= 3:
                index.append(((float(pos[0]), float(pos[2])), s.get("caption", "")))
    return index


def _fmt(x: float) -> str:
    return "nan" if (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def report_run(run_dir: str, dump_trajectories: bool = False) -> None:
    episodes = _load_episodes(run_dir)
    if not episodes:
        print(f"\n=== {run_dir}: no episode_*.json found ===")
        return
    cold_flags = classify_visits(episodes)
    index = _keyframe_index(episodes)

    print(f"\n=== {os.path.basename(run_dir.rstrip('/'))}  "
          f"({len(episodes)} episodes, {len(index)} keyframes) ===")
    print(f"  {'idx':>3} {'scene':<13} {'cat':<6} {'visit':<5} "
          f"{'obs':<4} {'obs_frac':>8} {'succ@1m':>7} {'succ.1m':>7} "
          f"{'min_d2g':>7} {'mem':>4} {'ontgt':>6} {'cos_med':>7}")
    agg: Dict[str, Dict[str, list]] = {
        "cold": {"obs": [], "succ1": [], "succ01": [], "ontgt": []},
        "warm": {"obs": [], "succ1": [], "succ01": [], "ontgt": []},
    }
    for ep, is_cold in zip(episodes, cold_flags):
        cat = str(ep.get("target_category"))
        observed, frac, _ = episode_observation(ep.get("steps", []), cat)
        audit = memory_candidate_audit(ep.get("decisions", []), index, cat)
        bucket = "cold" if is_cold else "warm"
        succ01 = bool(ep.get("success", False))
        succ1 = bool(ep.get("success_1m", False))
        agg[bucket]["obs"].append(1.0 if observed else 0.0)
        agg[bucket]["succ1"].append(1.0 if succ1 else 0.0)
        agg[bucket]["succ01"].append(1.0 if succ01 else 0.0)
        if audit["n_memory"]:
            agg[bucket]["ontgt"].append(audit["on_target_rate"])
        print(f"  {int(ep.get('episode_idx', 0)):>3} "
              f"{str(ep.get('scene_id'))[:13]:<13} {cat:<6} "
              f"{'cold' if is_cold else 'warm':<5} "
              f"{'Y' if observed else '.':<4} {frac:>8.2f} "
              f"{'Y' if succ1 else '.':>7} {'Y' if succ01 else '.':>7} "
              f"{float(ep.get('min_distance_to_goal', float('nan'))):>7.2f} "
              f"{audit['n_memory']:>4} {_fmt(audit['on_target_rate']):>6} "
              f"{_fmt(audit['cos_med']):>7}")

    def _mean(xs: list) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    print("  " + "-" * 92)
    for bucket in ("cold", "warm"):
        a = agg[bucket]
        if not a["obs"]:
            continue
        print(f"  {bucket:<5} n={len(a['obs']):<2}  "
              f"observation_rate={_fmt(_mean(a['obs']))}  "
              f"succ@1m={_fmt(_mean(a['succ1']))}  "
              f"succ@0.1m={_fmt(_mean(a['succ01']))}  "
              f"retrieval_on_target={_fmt(_mean(a['ontgt']))}")

    if dump_trajectories:
        out = os.path.join(run_dir, "trajectories.json")
        traj = {
            str(ep.get("episode_idx")): [
                [int(s.get("step_idx", 0)),
                 float(s["agent_pos"][0]), float(s["agent_pos"][2]),
                 s.get("caption", "")]
                for s in ep.get("steps", []) if s.get("agent_pos")
            ]
            for ep in episodes
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(traj, f)
        print(f"  trajectories -> {out}")


def main(argv: List[str]) -> int:
    dump = "--trajectories" in argv
    run_dirs = [a for a in argv if not a.startswith("--")]
    if not run_dirs:
        print(__doc__)
        return 2
    for rd in run_dirs:
        report_run(rd, dump_trajectories=dump)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
