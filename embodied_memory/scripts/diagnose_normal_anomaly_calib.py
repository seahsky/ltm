"""
diagnose_normal_anomaly_calib — the $0 go/no-go gate for Step 1 (the open-set
CLAP normal-vs-anomaly gate, ``audio.is_anomaly``).

Before wiring the gate into a (paid) ablation we must know CLAP can even separate
"anomaly" (baby cry / alarm / glass break) from "normal background" (footsteps /
coughing / knock / vacuum) on our clips. This scores each staged clip with
``audio.is_anomaly`` (delta=tau=0 so it just reads the raw margin = best-anomaly
cosine − best-normal cosine), prints the per-clip table, and computes the best
separating threshold (Youden-J) + EER.

GATE RULE (the decision this script exists to make):
  * GO        — clean separation (perfect, or EER <= 0.15): build the runtime
                gate with the recommended delta.
  * BORDERLINE— EER in (0.15, 0.30]: usable but calibrate carefully / add clips.
  * STOP      — EER > 0.30: CLAP can't separate these on our renders → the whole
                Step-1 arc is a category-discrimination ceiling (honest $0 negative).

Stage the clips first (needs internet, e.g. on RACE)::

    python embodied_memory/scripts/fetch_anomaly_clips.py --include-benign
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python \
        embodied_memory/scripts/diagnose_normal_anomaly_calib.py

The pure helpers (``best_threshold`` / ``verdict``) are unit-tested offline in
``test_diagnose_normal_anomaly_calib.py``; only ``main`` needs CLAP.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ----------------------------------------------------------------------
# pure helpers (no audio / CLAP import — unit-testable offline)
# ----------------------------------------------------------------------


def best_threshold(
    anom_margins: Sequence[float], benign_margins: Sequence[float]
) -> Dict[str, Any]:
    """Pick the margin threshold that best separates anomaly from benign.

    Sweeps every candidate cut (each observed margin + the midpoints between
    adjacent margins) and reports the Youden-J optimum (maximizes TPR − FPR;
    ties → the lower threshold) plus the equal-error rate (EER). ``perfect`` is
    True iff some cut gives TPR=1, FPR=0 (the groups don't overlap). Returns
    ``{"ok": False, ...}`` if either group is empty.
    """
    anom = [float(x) for x in anom_margins]
    benign = [float(x) for x in benign_margins]
    if not anom or not benign:
        return {"ok": False, "reason": "need at least one clip in each group"}

    cuts: List[float] = sorted(set(anom + benign))
    sweep: List[float] = []
    for i, v in enumerate(cuts):
        sweep.append(v)
        if i + 1 < len(cuts):
            sweep.append((v + cuts[i + 1]) / 2.0)
    # also a cut strictly below the minimum so TPR can reach 1.0
    sweep.append(min(cuts) - 1e-6)

    best: Optional[Dict[str, float]] = None
    eer = 1.0
    perfect = False
    for t in sorted(set(sweep)):
        tpr = sum(1 for x in anom if x >= t) / len(anom)
        fpr = sum(1 for x in benign if x >= t) / len(benign)
        fnr = 1.0 - tpr
        j = tpr - fpr
        if best is None or j > best["youden_j"] or (j == best["youden_j"] and t < best["delta"]):
            best = {"delta": float(t), "youden_j": float(j), "tpr": float(tpr), "fpr": float(fpr)}
        eer = min(eer, max(fpr, fnr))
        if tpr >= 1.0 and fpr <= 0.0:
            perfect = True

    assert best is not None
    # When the groups are cleanly separable, recommend the safe midpoint between
    # the worst benign and the worst anomaly; else fall back to the Youden cut.
    if perfect:
        rec = (max(benign) + min(anom)) / 2.0
    else:
        rec = best["delta"]
    return {
        "ok": True,
        "perfect": bool(perfect),
        "eer": float(eer),
        "recommend_delta": float(rec),
        "youden": best,
        "n_anom": len(anom),
        "n_benign": len(benign),
        "anom_margin_min": float(min(anom)),
        "benign_margin_max": float(max(benign)),
    }


def verdict(stats: Dict[str, Any]) -> Tuple[str, str]:
    """GO / BORDERLINE / STOP from ``best_threshold`` output."""
    if not stats.get("ok"):
        return "STOP", stats.get("reason", "insufficient data")
    eer = float(stats["eer"])
    if stats.get("perfect") or eer <= 0.15:
        return "GO", f"clean separation (perfect={stats.get('perfect')}, EER={eer:.2f})"
    if eer <= 0.30:
        return "BORDERLINE", f"usable but noisy (EER={eer:.2f}) — add clips / calibrate"
    return "STOP", f"CLAP cannot separate these (EER={eer:.2f}) — category ceiling"


# ----------------------------------------------------------------------
# clip loading + scoring (needs scipy; audio/CLAP imported lazily in main)
# ----------------------------------------------------------------------


def load_wav(path: str):
    """Return ``(mono float32 in [-1,1], sample_rate)`` for a wav file."""
    import numpy as np
    from scipy.io import wavfile

    sr, data = wavfile.read(path)
    x = np.asarray(data, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(np.asarray(data).dtype, np.integer):
        x = x / float(np.iinfo(np.asarray(data).dtype).max)
    return x, int(sr)


def _clip_paths(d: str) -> List[str]:
    return sorted(glob.glob(os.path.join(d, "*.wav")))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Step-1 normal-vs-anomaly CLAP calibration ($0 gate).")
    ap.add_argument("--anomaly-dir", default="data/anomaly_audio")
    ap.add_argument("--benign-dir", default="data/benign_audio")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))

    anom_paths = _clip_paths(args.anomaly_dir)
    benign_paths = _clip_paths(args.benign_dir)
    if not anom_paths or not benign_paths:
        print(f"[calib] need clips in BOTH dirs. anomaly={len(anom_paths)} "
              f"({args.anomaly_dir}), benign={len(benign_paths)} ({args.benign_dir}).")
        print("[calib] stage them: python embodied_memory/scripts/fetch_anomaly_clips.py --include-benign")
        return 2

    from embodied_memory import audio
    from embodied_memory.perception import CLAPAudioEncoder

    print(f"[calib] loading CLAP on {args.device} …")
    enc = CLAPAudioEncoder(device=args.device)

    def _score(paths):
        rows = []
        for p in paths:
            wav, sr = load_wav(p)
            _, best_class, s = audio.is_anomaly(wav, sr, enc)
            rows.append((os.path.basename(p), best_class, s["s_anom"], s["s_norm"], s["margin"]))
        return rows

    print("\n  clip                  best_class    s_anom   s_norm   margin")
    print("  " + "-" * 62)
    anom_rows = _score(anom_paths)
    benign_rows = _score(benign_paths)
    for name, cls, sa, sn, m in anom_rows:
        print(f"  [A] {name:18.18s} {cls:11.11s} {sa:7.3f}  {sn:7.3f}  {m:+7.3f}")
    for name, cls, sa, sn, m in benign_rows:
        print(f"  [N] {name:18.18s} {cls:11.11s} {sa:7.3f}  {sn:7.3f}  {m:+7.3f}")

    stats = best_threshold([r[4] for r in anom_rows], [r[4] for r in benign_rows])
    v, why = verdict(stats)
    print("\n  " + "-" * 62)
    if stats.get("ok"):
        print(f"  anomaly margins: min {stats['anom_margin_min']:+.3f} | "
              f"benign margins: max {stats['benign_margin_max']:+.3f}")
        print(f"  Youden delta={stats['youden']['delta']:+.3f} "
              f"(TPR {stats['youden']['tpr']:.2f}, FPR {stats['youden']['fpr']:.2f}), "
              f"EER={stats['eer']:.2f}, perfect={stats['perfect']}")
        print(f"  RECOMMEND_DELTA={stats['recommend_delta']:.3f}")
    print(f"  VERDICT: {v} — {why}")
    return 0 if v != "STOP" else 1


if __name__ == "__main__":
    raise SystemExit(main())
