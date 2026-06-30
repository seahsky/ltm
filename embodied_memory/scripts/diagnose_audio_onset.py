#!/usr/bin/env python
"""
diagnose_audio_onset — the $0 onset-blocker diagnostic for a --task
anomaly_response (or audiogoal) run.

When ``n_audio_onset_fired == 0`` the interrupt-resume controller never enters
INVESTIGATE, so it is decisive to know WHY. Two mutually-exclusive causes, and
they need opposite fixes:

  * ENERGY_TOO_LOW   — the rendered per-step ``audio_energy`` never clears the
    ``onset_rms`` threshold (the agent is never audibly close enough, or the
    onset threshold is calibrated too high). Fix: pin a lower ``--onset-rms``.
  * GATE_SUPPRESSING — energy DOES clear ``onset_rms`` yet onset never fires,
    so the forced open-set CLAP anomaly-gate (``audio.is_anomaly``, ON for
    anomaly_response) is REJECTING the onset (it was calibrated on clean clips;
    the RIR-convolved binaural clip can read as "normal"). Fix: relax / disable
    the gate, or recalibrate its delta on convolved audio.

The script reads a finished run dir (``<run>/summary.json`` for the per-episode
``n_audio_onset_fired`` totals + ``<run>/episode_*.json`` for the per-step
``audio_energy``), parses the calibrated ``onset_rms`` from the run log
(``<run>.log`` — ``RECOMMEND_ONSET_RMS=`` / ``onset_rms ... = X``) unless given
explicitly, and prints a per-episode table + a parseable ``ONSET_VERDICT=`` line.

Pure JSON/log parsing — no sim, no GPU, no torch.

Run:
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python \
    embodied_memory/scripts/diagnose_audio_onset.py runs/0c-anom-early-alarm-s3 \
    [--onset-rms 0.05] [--log runs/0c-anom-early-alarm-s3.log]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------------------------------------------------
# pure helpers (unit-tested)
# ----------------------------------------------------------------------
def episode_energy_stats(ep_log: Dict[str, Any]) -> Dict[str, float]:
    """Per-episode audio-energy summary from a loaded episode_NNN.json dict.

    Counts only steps that carry a non-None ``audio_energy`` toward the audible
    tally; missing/None entries are ignored (a step before t_anom renders
    silence -> audio_energy None)."""
    steps = ep_log.get("steps") or []
    energies = [float(s["audio_energy"]) for s in steps
                if isinstance(s, dict) and s.get("audio_energy") is not None]
    n_audible = sum(1 for e in energies if e > 0.0)
    max_e = max(energies) if energies else 0.0
    mean_e = (sum(energies) / len(energies)) if energies else 0.0
    return {
        "n_steps": len(steps),
        "n_audible": n_audible,
        "max_energy": max_e,
        "mean_energy": mean_e,
    }


def find_onset_rms(text: str) -> Optional[float]:
    """Parse the onset_rms actually used from the tee'd run/calib log. Matches
    both ``RECOMMEND_ONSET_RMS=<v>`` (diagnose_onset_calib) and the driver's
    ``onset_rms ... = <v>`` (calibrated/pinned) lines; returns the LAST value
    (the one actually applied), or None when absent."""
    pat = re.compile(r"(?:RECOMMEND_ONSET_RMS=|onset_rms[^=\n]*=\s*)([0-9]*\.?[0-9]+)")
    matches = pat.findall(text or "")
    return float(matches[-1]) if matches else None


def classify_onset_blocker(
    n_onset_fired: int, max_energy: float, onset_rms: Optional[float]
) -> Tuple[str, str]:
    """Decide why onset did/didn't fire. Returns (verdict, recommendation)."""
    if n_onset_fired > 0:
        return ("ONSET_FIRES",
                "Onset fired — the controller can enter INVESTIGATE. No blocker.")
    if onset_rms is None:
        return ("UNKNOWN_THRESHOLD",
                "onset_rms not found in the log; pass --onset-rms <v> (the "
                "calibrated value) so the energy-vs-threshold verdict can be made.")
    if max_energy >= onset_rms:
        return ("GATE_SUPPRESSING",
                f"Rendered audio_energy reaches {max_energy:.4f} >= onset_rms "
                f"{onset_rms:.4f}, yet onset never fired => the forced CLAP "
                "anomaly-gate (is_anomaly) is REJECTING the onset on the "
                "RIR-convolved clip. Fix: relax/disable the gate for the smoke "
                "(or recalibrate its delta on convolved audio).")
    return ("ENERGY_TOO_LOW",
            f"Rendered audio_energy peaks at {max_energy:.4f} < onset_rms "
            f"{onset_rms:.4f} => the agent is never audibly close enough / the "
            "threshold is too high. Fix: pin a lower --onset-rms (below the peak).")


# ----------------------------------------------------------------------
# I/O + report
# ----------------------------------------------------------------------
def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_log_text(run_dir: str, log_arg: Optional[str]) -> Tuple[str, Optional[str]]:
    """Return (text, path_used). Default log = '<run_dir>.log' (the driver tees
    there), then '<run_dir>/run.log'."""
    candidates = []
    if log_arg:
        candidates.append(log_arg)
    candidates.append(run_dir.rstrip("/") + ".log")
    candidates.append(os.path.join(run_dir, "run.log"))
    for p in candidates:
        if os.path.isfile(p):
            try:
                return open(p, "r", encoding="utf-8", errors="replace").read(), p
            except Exception:
                continue
    return "", None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="a finished run out-dir (has summary.json + episode_*.json)")
    ap.add_argument("--onset-rms", type=float, default=None,
                    help="override the onset threshold (else parsed from the log)")
    ap.add_argument("--log", type=str, default=None,
                    help="path to the tee'd run/calib log (default <run_dir>.log)")
    args = ap.parse_args(argv)

    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        print(f"FATAL: run dir not found: {run_dir}")
        return 2

    # onset_rms: explicit > parsed-from-log > None
    onset_rms = args.onset_rms
    log_path = None
    if onset_rms is None:
        text, log_path = _read_log_text(run_dir, args.log)
        onset_rms = find_onset_rms(text)

    # per-episode n_audio_onset_fired from summary.json
    summary = _load_json(os.path.join(run_dir, "summary.json")) or {}
    onset_by_ep = {}
    for ep in (summary.get("episodes") or []):
        onset_by_ep[str(ep.get("episode_id", ep.get("episode_idx")))] = \
            int(ep.get("n_audio_onset_fired", 0) or 0)

    ep_files = sorted(glob.glob(os.path.join(run_dir, "episode_*.json")))
    ep_files = [f for f in ep_files if not f.endswith("_error.json")]

    print(f"=== diagnose_audio_onset: {run_dir} ===")
    print(f"  onset_rms = {onset_rms if onset_rms is not None else 'UNKNOWN'}"
          f"{'  (from ' + log_path + ')' if (log_path and args.onset_rms is None) else ''}")
    if onset_rms is None and args.onset_rms is None:
        print("  (no onset_rms in the log — pass --onset-rms to get the energy-vs-threshold verdict)")
    print(f"  episodes with per-step logs: {len(ep_files)}")
    print()
    print(f"  {'episode':<22} {'steps':>6} {'audible':>8} {'maxE':>9} {'meanE':>9} {'onset_fired':>12}")

    global_max = 0.0
    total_onset = 0
    for f in ep_files:
        ep = _load_json(f)
        if ep is None:
            continue
        s = episode_energy_stats(ep)
        global_max = max(global_max, s["max_energy"])
        name = os.path.basename(f)
        # match summary by episode_id/idx in the ep_log
        eid = str(ep.get("episode_id", ep.get("episode_idx", "")))
        fired = onset_by_ep.get(eid, 0)
        total_onset += fired
        flag = ""
        if onset_rms is not None and s["max_energy"] >= onset_rms and fired == 0:
            flag = "  <- energy clears bar, no onset (gate?)"
        print(f"  {name:<22} {s['n_steps']:>6} {s['n_audible']:>8} "
              f"{s['max_energy']:>9.4f} {s['mean_energy']:>9.4f} {fired:>12}{flag}")

    # fall back to summary total if no per-ep mapping
    if total_onset == 0:
        total_onset = sum(onset_by_ep.values())

    verdict, rec = classify_onset_blocker(total_onset, global_max, onset_rms)
    print()
    print(f"  global max audio_energy = {global_max:.4f}   total onset_fired = {total_onset}")
    print(f"ONSET_VERDICT={verdict}")
    print(f"  -> {rec}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
