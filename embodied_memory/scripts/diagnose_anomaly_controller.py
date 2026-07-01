#!/usr/bin/env python
"""
diagnose_anomaly_controller — confirm the interrupt-resume state machine ran in
a ``--task anomaly_response`` run.

Onset firing (diagnose_audio_onset) only proves the TRIGGER. This reads the
``anomaly_report`` the E5-S7 hook writes into each ``<run>/episode_NNN.json``
(investigated / resumed / primary_completed / primary_completed_1m /
investigate_aborted / n_benign_ignored) and prints a per-episode table + rates
+ a parseable ``CONTROLLER_VERDICT=`` line:

  NO_ANOMALY_REPORT — no report in the logs (not an anomaly_response run / hook
                      didn't fire)
  NO_INTERRUPT      — reports present but the agent never left SEARCH (onset
                      never fired / no source cue)
  CONTROLLER_RAN    — >=1 episode ran onset -> INVESTIGATE -> CHECK -> RESUME
                      (investigated AND resumed) = the machine ran end-to-end
  PARTIAL           — interrupts happened but no clean investigate+resume
                      (aborted on the detour budget / ended mid-investigate)

Cross-references ``<run>/summary.json`` for per-episode ``n_audio_onset_fired``
when available. Pure JSON logic — no sim, no GPU.

Run:
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python \
    embodied_memory/scripts/diagnose_anomaly_controller.py runs/0c-anom-i-alarm-s3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------------------------------------------------
# pure helpers (unit-tested)
# ----------------------------------------------------------------------
def episode_row(ep_log: Dict[str, Any], name: str) -> Dict[str, Any]:
    """One row from a loaded episode_NNN.json (ep_log). ``anomaly_report`` fields
    are None when the report is absent (non-anomaly run / hook off)."""
    rep = ep_log.get("anomaly_report") or {}
    return {
        "name": name,
        "episode_id": str(ep_log.get("episode_id", ep_log.get("episode_idx", name))),
        "n_steps": ep_log.get("n_steps"),
        "success_1m": ep_log.get("success_1m"),
        "soft_spl": ep_log.get("soft_spl"),
        "replan_stuck": ep_log.get("replan_stuck"),
        "has_report": bool(rep),
        "investigated": rep.get("investigated"),
        "resumed": rep.get("resumed"),
        "investigate_aborted": rep.get("investigate_aborted"),
        "primary_completed": rep.get("primary_completed"),
        "primary_completed_1m": rep.get("primary_completed_1m"),
        "n_benign_ignored": rep.get("n_benign_ignored"),
        "anomaly_class": rep.get("anomaly_class"),
    }


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    reps = [r for r in rows if r.get("has_report")]

    def _rate(key: str) -> Optional[float]:
        vals = [bool(r[key]) for r in reps if r.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None

    return {
        "n_episodes": len(rows),
        "n_with_report": len(reps),
        "n_investigated": sum(1 for r in reps if r.get("investigated")),
        "n_resumed": sum(1 for r in reps if r.get("resumed")),
        "n_aborted": sum(1 for r in reps if r.get("investigate_aborted")),
        "n_full": sum(1 for r in reps if r.get("investigated") and r.get("resumed")),
        "investigate_rate": _rate("investigated"),
        "resume_rate": _rate("resumed"),
        "primary_completed_rate": _rate("primary_completed"),
        "primary_completed_1m_rate": _rate("primary_completed_1m"),
    }


def verdict(agg: Dict[str, Any]) -> Tuple[str, str]:
    if agg["n_with_report"] == 0:
        return ("NO_ANOMALY_REPORT",
                "No anomaly_report in the episode logs — not an anomaly_response "
                "run, or the E5-S7 report hook didn't fire.")
    if agg["n_investigated"] == 0 and agg["n_resumed"] == 0:
        return ("NO_INTERRUPT",
                "Reports present but the agent NEVER left SEARCH (onset never "
                "fired / no source cue). The controller was inert — check "
                "diagnose_audio_onset (gate/energy).")
    if agg["n_full"] > 0:
        return ("CONTROLLER_RAN",
                f"{agg['n_full']} episode(s) ran onset -> INVESTIGATE -> CHECK -> "
                "RESUME (investigated AND resumed) — the interrupt-resume machine "
                "ran end-to-end.")
    return ("PARTIAL",
            "Interrupts happened but no clean investigate+resume (aborted on the "
            "detour budget, or the episode ended mid-INVESTIGATE). Inspect "
            "investigate_aborted / n_steps.")


# ----------------------------------------------------------------------
# I/O + report
# ----------------------------------------------------------------------
def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "Y" if v else "."
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="a finished anomaly_response out-dir (episode_*.json)")
    args = ap.parse_args(argv)
    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        print(f"FATAL: run dir not found: {run_dir}")
        return 2

    # per-episode onset cross-ref from summary.json
    summary = _load_json(os.path.join(run_dir, "summary.json")) or {}
    onset_by_id: Dict[str, int] = {}
    for ep in (summary.get("episodes") or []):
        for k in (ep.get("episode_id"), ep.get("episode_idx")):
            if k is not None:
                onset_by_id[str(k)] = int(ep.get("n_audio_onset_fired", 0) or 0)

    ep_files = [f for f in sorted(glob.glob(os.path.join(run_dir, "episode_*.json")))
                if not f.endswith("_error.json")]

    rows = []
    for f in ep_files:
        ep = _load_json(f)
        if ep is None:
            continue
        rows.append(episode_row(ep, os.path.basename(f)))

    print(f"=== diagnose_anomaly_controller: {run_dir} ===")
    print(f"  episodes: {len(rows)}")
    print()
    hdr = (f"  {'episode':<8} {'steps':>5} {'onset':>5} {'inves':>5} {'resum':>5} "
           f"{'abort':>5} {'compl':>5} {'compl1m':>7} {'benign':>6} {'succ1m':>6} {'class':>7}")
    print(hdr)
    for r in rows:
        onset = onset_by_id.get(r["episode_id"], onset_by_id.get(
            r["name"].replace("episode_", "").replace(".json", "").lstrip("0") or "0"))
        print(f"  {r['episode_id']:<8} {_fmt(r['n_steps']):>5} {_fmt(onset):>5} "
              f"{_fmt(r['investigated']):>5} {_fmt(r['resumed']):>5} "
              f"{_fmt(r['investigate_aborted']):>5} {_fmt(r['primary_completed']):>5} "
              f"{_fmt(r['primary_completed_1m']):>7} {_fmt(r['n_benign_ignored']):>6} "
              f"{_fmt(r['success_1m']):>6} {_fmt(r['anomaly_class']):>7}")

    agg = aggregate(rows)

    def _pct(x):
        return "n/a" if x is None else f"{100.0 * x:.0f}%"

    print()
    print(f"  with_report={agg['n_with_report']}/{agg['n_episodes']}  "
          f"investigated={agg['n_investigated']} (rate {_pct(agg['investigate_rate'])})  "
          f"resumed={agg['n_resumed']} (rate {_pct(agg['resume_rate'])})  "
          f"aborted={agg['n_aborted']}  full(inv+res)={agg['n_full']}")
    print(f"  primary_completed@0.1m rate={_pct(agg['primary_completed_rate'])}  "
          f"@1.0m rate={_pct(agg['primary_completed_1m_rate'])}")
    v, rec = verdict(agg)
    print(f"CONTROLLER_VERDICT={v}")
    print(f"  -> {rec}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
