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
        # ADR-0002 scene-conditioning ground truth (present only on two-rooms
        # episodes; None otherwise → discrimination_rates ignores it).
        "expected_interrupt": rep.get("expected_interrupt"),
    }


def onset_provenance(row: Dict[str, Any]) -> str:
    """WHAT fired this episode's interrupt: ``ANOMALY`` / ``FALSE_FIRE`` /
    ``NO_ONSET`` / ``UNKNOWN``.

    An onset BEFORE ``t_anom`` cannot be the anomaly — the anomaly is not playing
    yet, so something else (the background bed) tripped the threshold. This is the
    only signal that distinguishes a working interrupt from a vacuum cleaner:
    ``n_audio_onset_fired`` counts onsets rather than causes, and
    ``n_audio_gate_rejected == 0`` means the gate ACCEPTED the first
    over-threshold tick (onset is one-shot), not that it had nothing to reject.

    Missing fields read ``UNKNOWN``, never ``ANOMALY``: archived summaries predate
    these fields, and the entire point is refusing to certify what we cannot see.
    """
    if not int(row.get("n_audio_onset_fired") or 0):
        return "NO_ONSET"
    step = row.get("audio_onset_step")
    t_anom = row.get("audio_t_anom")
    if step is None or t_anom is None:
        return "UNKNOWN"
    return "ANOMALY" if int(step) >= int(t_anom) else "FALSE_FIRE"


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    reps = [r for r in rows if r.get("has_report")]
    prov = [onset_provenance(r) for r in rows]

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
        # Onset provenance — WHAT fired the interrupts (see onset_provenance).
        "n_onset_anomaly": prov.count("ANOMALY"),
        "n_onset_false_fire": prov.count("FALSE_FIRE"),
        "n_onset_unknown": prov.count("UNKNOWN"),
        "n_onset_none": prov.count("NO_ONSET"),
    }


def discrimination_rates(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """ADR-0002 scene-conditioning discrimination (P3.3).

    Over the two-rooms rows (those carrying an ``expected_interrupt`` label), report
    the FALSE-interrupt rate (room-NORMAL episodes the agent wrongly interrupted;
    want 0) and the CORRECT-interrupt rate (room-ANOMALOUS episodes it rightly
    interrupted; want 1), plus the 2×2 confusion matrix. A row "interrupted" iff it
    left SEARCH — ``investigated`` OR ``investigate_aborted``. Rows without a label
    (non-two-rooms episodes) are ignored, so a normal anomaly-response run yields
    ``n_normal = n_anomalous = 0`` and ``None`` rates."""
    labeled = [r for r in rows if r.get("expected_interrupt") is not None]

    def _interrupted(r: Dict[str, Any]) -> bool:
        return bool(r.get("investigated")) or bool(r.get("investigate_aborted"))

    normal = [r for r in labeled if r["expected_interrupt"] is False]
    anom = [r for r in labeled if r["expected_interrupt"] is True]
    n_normal, n_anom = len(normal), len(anom)
    n_false = sum(1 for r in normal if _interrupted(r))
    n_correct = sum(1 for r in anom if _interrupted(r))
    confusion: Dict[Tuple[bool, bool], int] = {}
    for r in labeled:
        confusion[(bool(r["expected_interrupt"]), _interrupted(r))] = \
            confusion.get((bool(r["expected_interrupt"]), _interrupted(r)), 0) + 1
    return {
        "n_normal": n_normal,
        "n_anomalous": n_anom,
        "n_false_interrupt": n_false,
        "n_correct_interrupt": n_correct,
        "false_interrupt_rate": (n_false / n_normal) if n_normal else None,
        "correct_interrupt_rate": (n_correct / n_anom) if n_anom else None,
        "confusion": confusion,
    }


def discrimination_verdict(rates: Dict[str, Any], *, go_correct: float = 0.75,
                           go_false: float = 0.25) -> str:
    """GO / BORDERLINE / STOP / NO_DATA for the discrimination A/B. GO when the
    room-conditioned gate interrupts room-anomalous sounds reliably (correct-rate
    >= ``go_correct``) AND rarely false-fires on room-normal ones (false-rate <=
    ``go_false``). NO_DATA when no two-rooms episodes were run."""
    cr = rates.get("correct_interrupt_rate")
    fr = rates.get("false_interrupt_rate")
    if cr is None or fr is None:
        return "NO_DATA"
    if cr >= go_correct and fr <= go_false:
        return "GO"
    if cr < 0.5 or fr > 0.5:
        return "STOP"
    return "BORDERLINE"


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
    # Provenance FIRST: a loop is only a loop if the ANOMALY started it. When every
    # interrupt predates t_anom the machine ran on a false trigger (the background
    # bed), and certifying that is exactly how runs/anomresp-bed-s{1,3} reported
    # CONTROLLER_RAN while the alarm never fired anything. Requires at least one
    # observed false fire, so archived runs (all UNKNOWN) keep the old behaviour.
    if agg.get("n_onset_false_fire", 0) > 0 and agg.get("n_onset_anomaly", 0) == 0:
        return ("FALSE_FIRE",
                f"{agg['n_onset_false_fire']} interrupt(s) fired BEFORE t_anom and "
                "none after — the anomaly was not playing yet, so something else "
                "(the background bed) tripped the onset. The controller may have "
                "run, but it ran on a false trigger: this run does not measure "
                "anomaly response. Check bg_gain vs onset_rms (ADR-0004) and "
                "`grep '[audio] onset' the log.")
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


def _rows_for_dir(run_dir: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Load one run dir's episode rows + its per-episode onset cross-ref.

    The controller's own ``anomaly_report`` (in episode_*.json) says what the
    machine DID; only ``summary.episodes`` says what fired it (onset_step vs
    t_anom). Merge the provenance in, keyed by episode id, so ``aggregate`` can
    refuse to certify a run whose interrupts predate the sound. Archived summaries
    lack the fields → the keys stay absent → onset_provenance reads UNKNOWN.
    """
    summary = _load_json(os.path.join(run_dir, "summary.json")) or {}
    onset_by_id: Dict[str, int] = {}
    prov_by_id: Dict[str, Dict[str, Any]] = {}
    for ep in (summary.get("episodes") or []):
        prov = {k: ep[k] for k in ("n_audio_onset_fired", "audio_onset_step",
                                   "audio_t_anom") if k in ep}
        for k in (ep.get("episode_id"), ep.get("episode_idx")):
            if k is not None:
                onset_by_id[str(k)] = int(ep.get("n_audio_onset_fired", 0) or 0)
                prov_by_id[str(k)] = prov
    ep_files = [f for f in sorted(glob.glob(os.path.join(run_dir, "episode_*.json")))
                if not f.endswith("_error.json")]
    rows = []
    for f in ep_files:
        ep = _load_json(f)
        if ep is not None:
            row = episode_row(ep, os.path.basename(f))
            row.update(prov_by_id.get(row["episode_id"], {}))
            rows.append(row)
    return rows, onset_by_id


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100.0 * x:.0f}%"


def _print_dir_table(run_dir: str, rows: List[Dict[str, Any]], onset_by_id: Dict[str, int]) -> None:
    print(f"=== diagnose_anomaly_controller: {run_dir} ===")
    print(f"  episodes: {len(rows)}")
    print()
    # step@ / t_anom / fired: the provenance columns. Reading them side by side is
    # the whole check — step@ < t_anom means the anomaly wasn't playing yet.
    print(f"  {'episode':<8} {'steps':>5} {'onset':>5} {'step@':>5} {'t_anom':>6} "
          f"{'fired':>10} {'inves':>5} {'resum':>5} {'abort':>5} {'compl1m':>7} "
          f"{'benign':>6} {'succ1m':>6} {'class':>7}")
    for r in rows:
        onset = onset_by_id.get(r["episode_id"], onset_by_id.get(
            r["name"].replace("episode_", "").replace(".json", "").lstrip("0") or "0"))
        prov = onset_provenance(r)
        print(f"  {r['episode_id']:<8} {_fmt(r['n_steps']):>5} {_fmt(onset):>5} "
              f"{_fmt(r.get('audio_onset_step')):>5} {_fmt(r.get('audio_t_anom')):>6} "
              f"{prov:>10} "
              f"{_fmt(r['investigated']):>5} {_fmt(r['resumed']):>5} "
              f"{_fmt(r['investigate_aborted']):>5} "
              f"{_fmt(r['primary_completed_1m']):>7} {_fmt(r['n_benign_ignored']):>6} "
              f"{_fmt(r['success_1m']):>6} {_fmt(r['anomaly_class']):>7}")


def _print_agg(agg: Dict[str, Any], label: str = "") -> None:
    pre = f"  [{label}] " if label else "  "
    print(f"{pre}with_report={agg['n_with_report']}/{agg['n_episodes']}  "
          f"investigated={agg['n_investigated']} (rate {_pct(agg['investigate_rate'])})  "
          f"resumed={agg['n_resumed']} (rate {_pct(agg['resume_rate'])})  "
          f"aborted={agg['n_aborted']}  full(inv+res)={agg['n_full']}")
    print(f"  primary_completed@0.1m rate={_pct(agg['primary_completed_rate'])}  "
          f"@1.0m rate={_pct(agg['primary_completed_1m_rate'])}")
    print(f"  onset provenance: anomaly={agg.get('n_onset_anomaly', 0)}  "
          f"FALSE_FIRE={agg.get('n_onset_false_fire', 0)}  "
          f"none={agg.get('n_onset_none', 0)}  unknown={agg.get('n_onset_unknown', 0)}"
          "   [FALSE_FIRE = onset fired BEFORE t_anom => not the anomaly]")


def _print_discrimination(rows: List[Dict[str, Any]], label: str = "") -> None:
    """ADR-0002 discrimination confusion matrix + verdict (P3.3). Prints nothing
    when no two-rooms episodes were run (keeps ordinary-run output unchanged)."""
    rates = discrimination_rates(rows)
    if not (rates["n_normal"] or rates["n_anomalous"]):
        return
    pre = f"  [{label}] " if label else "  "
    print(f"{pre}scene-conditioning discrimination (two-rooms):")
    print(f"    room-NORMAL   n={rates['n_normal']:<3} false-interrupt "
          f"{rates['n_false_interrupt']}/{rates['n_normal']} "
          f"(rate {_pct(rates['false_interrupt_rate'])})  [want 0%]")
    print(f"    room-ANOMALOUS n={rates['n_anomalous']:<3} correct-interrupt "
          f"{rates['n_correct_interrupt']}/{rates['n_anomalous']} "
          f"(rate {_pct(rates['correct_interrupt_rate'])})  [want 100%]")
    print(f"    DISCRIMINATION_VERDICT={discrimination_verdict(rates)}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="+",
                    help="one or more finished anomaly_response out-dirs (episode_*.json). "
                         "Multiple dirs => a POOLED controller verdict across all of them "
                         "(the matrix systems headline).")
    args = ap.parse_args(argv)
    run_dirs = [d for d in args.run_dir if os.path.isdir(d)]
    missing = [d for d in args.run_dir if not os.path.isdir(d)]
    for d in missing:
        print(f"WARN: run dir not found (skipped): {d}")
    if not run_dirs:
        print("FATAL: no valid run dir given")
        return 2

    all_rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        rows, onset_by_id = _rows_for_dir(run_dir)
        _print_dir_table(run_dir, rows, onset_by_id)
        agg = aggregate(rows)
        print()
        _print_agg(agg)
        _print_discrimination(rows)
        v, rec = verdict(agg)
        print(f"CONTROLLER_VERDICT={v}")
        print(f"  -> {rec}")
        print()
        all_rows.extend(rows)

    # Pooled verdict across every dir = the multi-cell matrix systems headline.
    if len(run_dirs) > 1:
        print("=" * 60)
        print(f"POOLED across {len(run_dirs)} run dirs ({len(all_rows)} episodes)")
        pooled = aggregate(all_rows)
        _print_agg(pooled, label="pooled")
        _print_discrimination(all_rows, label="pooled")
        pv, prec = verdict(pooled)
        print(f"POOLED_CONTROLLER_VERDICT={pv}")
        print(f"  -> {prec}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
