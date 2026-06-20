"""
fetch_anomaly_clips — stage real ESC-50 anomaly recordings for the AudioGoal task.

Downloads ONE representative clip per locked anomaly class from the ESC-50 dataset
(Piczak 2015, CC BY-NC) into ``data/anomaly_audio/<class>.wav``, which
``run_hm3d_pol`` auto-resolves via ``audio_task.resolve_anomaly_clip`` (no
``--anomaly-clip`` needed). The driver also threads the same file into the onset
calibration so the energy scale matches. Without these files the runner falls
back to the deterministic synthetic burst, so this is a one-time staging step.

ESC-50 categories used (real recordings of the exact classes):
    baby_cry -> crying_baby, alarm -> clock_alarm, glass_break -> glass_breaking.

Run on a host with internet (e.g. RACE)::

    python embodied_memory/scripts/fetch_anomaly_clips.py
    # pick a different clip per class:  --index 1   (0..39)
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import urllib.request
from typing import Any, Dict, List, Optional

ESC50_BASE = "https://github.com/karoldvl/ESC-50/raw/master"
# locked AudioGoal class -> ESC-50 category (audio.ANOMALY_CLASSES)
CLASS_TO_ESC50: Dict[str, str] = {
    "baby_cry": "crying_baby",
    "alarm": "clock_alarm",
    "glass_break": "glass_breaking",
}

# Benign/NORMAL household sounds -> ESC-50 category, for the Step-1
# normal-vs-anomaly calibration (diagnose_normal_anomaly_calib.py). These are
# the "routine, do not respond" negatives the gate must REJECT. Staged into a
# separate dir (data/benign_audio/) so they never collide with the anomaly clips
# the runner auto-resolves.
BENIGN_TO_ESC50: Dict[str, str] = {
    "footsteps": "footsteps",
    "coughing": "coughing",
    "knock": "door_wood_knock",
    "vacuum": "vacuum_cleaner",
}


def select_clip(rows: List[Dict[str, Any]], esc_category: str, index: int = 0) -> Optional[str]:
    """Pure: pick the ``index``-th ESC-50 filename for ``esc_category`` from the
    parsed meta rows (sorted for determinism; index wraps). ``None`` if the
    category has no rows."""
    files = sorted(str(r.get("filename")) for r in rows if r.get("category") == esc_category)
    if not files:
        return None
    return files[index % len(files)]


def _fetch(url: str, timeout: int = 120) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (fixed ESC-50 host)
        return r.read()


def _stage(rows, class_map: Dict[str, str], classes, out_dir: str, index: int) -> int:
    """Download one clip per ``classes`` (looked up in ``class_map``) into
    ``out_dir`` as ``<class>.wav``. Returns how many staged OK."""
    os.makedirs(out_dir, exist_ok=True)
    ok = 0
    for cls in classes:
        esc = class_map.get(cls)
        if esc is None:
            print(f"  WARN: unknown class {cls!r} (known: {list(class_map)})")
            continue
        fname = select_clip(rows, esc, index)
        if fname is None:
            print(f"  WARN: no ESC-50 clip for category {esc!r}")
            continue
        data = _fetch(f"{ESC50_BASE}/audio/{fname}")
        dst = os.path.join(out_dir, f"{cls}.wav")
        with open(dst, "wb") as f:
            f.write(data)
        try:
            from scipy.io import wavfile
            sr, _ = wavfile.read(dst)
            print(f"  {cls} <- ESC-50 {esc}/{fname} ({len(data)} B, {sr} Hz) -> {dst}")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  WARN: {dst} not a readable wav ({e})")
    return ok


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Stage ESC-50 anomaly clips for AudioGoal.")
    ap.add_argument("--out-dir", default="data/anomaly_audio")
    ap.add_argument("--index", type=int, default=0, help="which clip per class (0..39)")
    ap.add_argument("--classes", nargs="*", default=list(CLASS_TO_ESC50),
                    help="subset of baby_cry/alarm/glass_break")
    ap.add_argument("--include-benign", action="store_true",
                    help="also stage benign/NORMAL negatives (footsteps/coughing/knock/"
                         "vacuum) into --benign-out-dir, for the Step-1 calibration.")
    ap.add_argument("--benign-out-dir", default="data/benign_audio")
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))

    print(f"[fetch-anomaly] ESC-50 meta -> {ESC50_BASE}/meta/esc50.csv")
    rows = list(csv.DictReader(io.StringIO(_fetch(f"{ESC50_BASE}/meta/esc50.csv").decode())))

    ok = _stage(rows, CLASS_TO_ESC50, args.classes, args.out_dir, args.index)
    print(f"[fetch-anomaly] anomaly: {ok}/{len(args.classes)} staged into {args.out_dir}")

    bok = 0
    if args.include_benign:
        bclasses = list(BENIGN_TO_ESC50)
        bok = _stage(rows, BENIGN_TO_ESC50, bclasses, args.benign_out_dir, args.index)
        print(f"[fetch-anomaly] benign: {bok}/{len(bclasses)} staged into {args.benign_out_dir}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
