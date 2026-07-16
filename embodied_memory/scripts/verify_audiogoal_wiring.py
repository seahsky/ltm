"""
verify_audiogoal_wiring — assert the M1 audio wiring fired in a run log.

Reads the stdout log of a ``--task audiogoal`` run and asserts the once-per-
episode ``[audio] onset @step N class=X target=Y energy=E`` line appeared with
sane values: onset at or after ``--t-anom``, a real energy, a valid detected
class, and an INTERNALLY-CONSISTENT override (``target == CLASS_TO_OBJECT[class]``
— i.e. the audio→object→retrieval mapping fired correctly). This proves the
WIRING; it does NOT assume which class a given clip yields (CLAP's accuracy on
real FSD50K clips is an M3 metric). Pass ``--expect-class`` only when the clip's
true class is known (e.g. a real FSD50K baby_cry .wav). Exits non-zero (RED) on
any failure.

    python embodied_memory/scripts/verify_audiogoal_wiring.py \
        --log runs/audiogoal-m1/frontier.log --run runs/audiogoal-m1/frontier --t-anom 30
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import re
import sys


def _class_object_map():
    """Load CLASS_TO_OBJECT / ANOMALY_CLASSES from audio.py WITHOUT importing the
    faiss-pulling package __init__ (audio.py is numpy-only at import)."""
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "_vaudio", os.path.join(here, os.pardir, "audio.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod   # register before exec so audio.AugmentSpec (@dataclass) resolves its fields
    spec.loader.exec_module(mod)
    return mod.CLASS_TO_OBJECT, mod.ANOMALY_CLASSES


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, help="stdout log of the audiogoal run")
    ap.add_argument("--run", default=None, help="run dir with episode_*.json (best-effort)")
    ap.add_argument("--t-anom", type=int, default=30)
    ap.add_argument("--expect-class", default=None,
                    help="assert the DETECTED class equals this (only for a "
                         "known-class real clip; omit for the synthetic burst)")
    args = ap.parse_args()
    class_to_object, anomaly_classes = _class_object_map()

    if not os.path.isfile(args.log):
        print(f"RED: log not found: {args.log}")
        return 2
    text = open(args.log, "r", errors="replace").read()

    m = re.search(
        r"\[audio\] onset @step (\d+) class=(\S+) target=(\S+) energy=([0-9.]+)", text)
    if not m:
        print("RED: no '[audio] onset' line — onset never fired / audio not wired")
        return 1
    step, cls, target, energy = int(m.group(1)), m.group(2), m.group(3), float(m.group(4))
    print(f"  onset: step={step} class={cls} target={target} energy={energy:.3f}")

    fails = []
    if step < args.t_anom:
        fails.append(f"onset step {step} < t_anom {args.t_anom} (rendered before silence ended)")
    if energy <= 0.0:
        fails.append(f"onset energy {energy} not positive (audio not heard)")
    if cls not in anomaly_classes:
        fails.append(f"detected class {cls!r} not a valid anomaly class {anomaly_classes}")
    else:
        # WIRING check: the override target must be the object the detected class
        # maps to (audio → class → CLASS_TO_OBJECT → retrieval target).
        expected = class_to_object.get(cls)
        if target != expected:
            fails.append(f"override target={target!r} != CLASS_TO_OBJECT[{cls!r}]={expected!r} "
                         f"(audio→object mapping broken)")
    if args.expect_class and cls != args.expect_class:
        fails.append(f"detected class {cls!r} != --expect-class {args.expect_class!r}")

    # best-effort: confirm the episode JSON carries post-onset audio energy
    if args.run and os.path.isdir(args.run):
        eps = sorted(glob.glob(os.path.join(args.run, "episode_*.json")))
        n_audio_steps = 0
        for ep_path in eps:
            try:
                data = json.load(open(ep_path))
            except Exception:
                continue
            for s in (data.get("steps") or data.get("decisions") or []):
                info = s.get("info", s) if isinstance(s, dict) else {}
                if isinstance(info, dict) and float(info.get("audio_energy", 0.0)) > 0.0:
                    n_audio_steps += 1
        print(f"  episode-JSON steps with audio_energy>0: {n_audio_steps} "
              f"(informational; schema-dependent)")

    if fails:
        for f in fails:
            print(f"RED: {f}")
        return 1
    print(f"GREEN: audio wiring fired — onset@{step} (>= t_anom {args.t_anom}), "
          f"class={cls}, retrieval target overridden to {target!r}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
