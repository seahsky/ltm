"""
analyze_lifelong_ab — summarize the lifelong cross-visit write-ON vs write-OFF A/B
(Step 2, oracle-source upper bound) directly from each arm's ``summary.json``.

Per cell we have two run dirs built on the SAME dataset: arm A (write-OFF) and arm
B (write-ON, ``LTM_AUDIO_WRITE=1``). Episodes split into the SEED (the visit-1
episode that fires the anomaly → ``n_audio_onset_fired > 0``) and the RECALL
episodes (silent visit-2 → ``n_audio_onset_fired == 0``). The decisive readouts:

  * seed ``n_audio_writes`` (arm B)  — did the audio→LTM write finally FIRE?
  * recall ``n_audio_event_recalled`` (arm B) — was the written waypoint recalled
    from a distance (not deduped)?
  * recall soft-SPL / success@1m, paired B−A — did the oracle write HELP, or is it
    redundant with the seed's own visual sighting (B−A ≈ 0)?

The pure helpers are unit-tested in ``test_analyze_lifelong_ab.py``; only ``main``
touches the filesystem.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple


def split_seed_recall(episodes: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(seed, recall): seed = episodes whose anomaly fired (onset>0), recall = the
    silent visit-2 episodes (onset==0)."""
    def _onset(e: Dict[str, Any]) -> int:
        try:
            return int(e.get("n_audio_onset_fired", 0) or 0)
        except (TypeError, ValueError):
            return 0
    seed = [e for e in episodes if _onset(e) > 0]
    recall = [e for e in episodes if _onset(e) == 0]
    return seed, recall


def _f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _mean(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def arm_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Per-arm rollup from a run's full summary dict."""
    eps = summary.get("episodes") or []
    seed, recall = split_seed_recall(eps)
    return {
        "n_seed": len(seed),
        "n_recall": len(recall),
        "seed_writes": sum(int(_f(e.get("n_audio_writes"))) for e in seed),
        "seed_skip_reason": (seed[0].get("audio_write_skip_reason") if seed else None),
        "recall_recalled": sum(int(_f(e.get("n_audio_event_recalled"))) for e in recall),
        "recall_soft_spl": _mean([_f(e.get("soft_spl")) for e in recall]),
        "recall_succ1m": _mean([1.0 if e.get("success_1m") else 0.0 for e in recall]),
    }


def paired_recall_delta(a_eps: List[Dict[str, Any]], b_eps: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    """Pair recall episodes by run order (A and B share the dataset) → per-pair
    B−A soft-SPL and success@1m lists."""
    _, ar = split_seed_recall(a_eps)
    _, br = split_seed_recall(b_eps)
    ar = sorted(ar, key=lambda e: int(_f(e.get("episode_idx"))))
    br = sorted(br, key=lambda e: int(_f(e.get("episode_idx"))))
    n = min(len(ar), len(br))
    dsoft = [_f(br[i].get("soft_spl")) - _f(ar[i].get("soft_spl")) for i in range(n)]
    dsucc = [(1.0 if br[i].get("success_1m") else 0.0) - (1.0 if ar[i].get("success_1m") else 0.0)
             for i in range(n)]
    return dsoft, dsucc


def redundancy_verdict(seed_writes: int, recall_recalled: int, dsoft: List[float],
                       *, tie_band: float = 0.02) -> str:
    """A blunt one-line verdict for the oracle-source A/B."""
    if seed_writes <= 0:
        return "NO-WRITE — the seed never wrote (chase audio_write_skip_reason)"
    if recall_recalled <= 0:
        return "WRITE-NOT-RECALLED — wrote but never recalled in visit-2 (deduped / out-competed)"
    m = _mean(dsoft) if dsoft else float("nan")
    if math.isnan(m):
        return "NO-PAIRS — write+recall confirmed but no paired recall episodes"
    if m > tie_band:
        return f"HELPS — oracle write lifts recall soft-SPL B−A={m:+.3f} (upper bound; vision-redundancy controlled by A/B)"
    if m < -tie_band:
        return f"HURTS — oracle write lowers recall soft-SPL B−A={m:+.3f} (over-fire / wrong-instance?)"
    return f"REDUNDANT — write fires + recalled but B−A={m:+.3f} ≈ 0 (seed's visual sighting already suffices → needs a non-LOS seed)"


def _load_summary(run_dir: str) -> Optional[Dict[str, Any]]:
    p = os.path.join(run_dir, "summary.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Lifelong write-ON vs write-OFF A/B summary.")
    ap.add_argument("--a", nargs="+", required=True, help="arm-A (write-OFF) run dirs, one per cell")
    ap.add_argument("--b", nargs="+", required=True, help="arm-B (write-ON) run dirs, parallel to --a")
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))

    if len(args.a) != len(args.b):
        print(f"[lifelong-ab] FATAL: {len(args.a)} A dirs != {len(args.b)} B dirs")
        return 2

    print("\n==================== LIFELONG A/B (write-ON vs write-OFF, oracle source) ====================")
    print(f"  {'cell (B dir)':38.38s} {'seedW':>5} {'rcl':>4} {'A_sspl':>7} {'B_sspl':>7} {'dB-A':>7} {'verdict'}")
    print("  " + "-" * 104)
    pooled_dsoft: List[float] = []
    pooled_dsucc: List[float] = []
    n_cells = 0
    for adir, bdir in zip(args.a, args.b):
        asum, bsum = _load_summary(adir), _load_summary(bdir)
        if asum is None or bsum is None:
            print(f"  {os.path.basename(bdir):38.38s} MISSING summary.json "
                  f"(A={'ok' if asum else 'NONE'} B={'ok' if bsum else 'NONE'}) — skipped")
            continue
        n_cells += 1
        a, b = arm_summary(asum), arm_summary(bsum)
        dsoft, dsucc = paired_recall_delta(asum.get("episodes") or [], bsum.get("episodes") or [])
        pooled_dsoft += dsoft
        pooled_dsucc += dsucc
        v = redundancy_verdict(b["seed_writes"], b["recall_recalled"], dsoft)
        print(f"  {os.path.basename(bdir):38.38s} {b['seed_writes']:>5} {b['recall_recalled']:>4} "
              f"{a['recall_soft_spl']:>7.3f} {b['recall_soft_spl']:>7.3f} {_mean(dsoft):>7.3f}  {v}")
        if b["seed_writes"] <= 0:
            print(f"      seed skip_reason (B) = {b['seed_skip_reason']!r}")

    print("  " + "-" * 104)
    if pooled_dsoft:
        m = _mean(pooled_dsoft)
        pos = sum(1 for d in pooled_dsoft if d > 0)
        print(f"  POOLED recall pairs n={len(pooled_dsoft)} over {n_cells} cell(s): "
              f"soft-SPL B−A mean={m:+.4f} (pos {pos}/{len(pooled_dsoft)}), "
              f"succ@1m B−A mean={_mean(pooled_dsucc):+.4f}")
        print(f"  POOLED verdict: {redundancy_verdict(1, 1, pooled_dsoft)}")
    else:
        print("  POOLED: no paired recall episodes across any cell")
    print("============================================================================================\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
