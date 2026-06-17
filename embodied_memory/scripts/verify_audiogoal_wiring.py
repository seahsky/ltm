"""
verify_audiogoal_wiring — assert the M1 audio wiring fired in a run log.

Reads the stdout log of a ``--task audiogoal`` run and asserts the once-per-
episode ``[audio] onset @step N class=X target=Y energy=E`` line appeared with
sane values: onset at or after ``--t-anom``, a real energy, and the expected
class→object override (so the retrieval target was overridden to the audio-
inferred object). Best-effort scans episode JSONs in ``--run`` for the audio
diagnostics too. Exits non-zero (RED) on any failure.

    python embodied_memory/scripts/verify_audiogoal_wiring.py \
        --log runs/audiogoal-m1/frontier.log --run runs/audiogoal-m1/frontier \
        --t-anom 30 --expect-object crib
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, help="stdout log of the audiogoal run")
    ap.add_argument("--run", default=None, help="run dir with episode_*.json (best-effort)")
    ap.add_argument("--t-anom", type=int, default=30)
    ap.add_argument("--expect-object", default="crib", help="expected class→object override")
    args = ap.parse_args()

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
    if cls in ("None", "none", ""):
        fails.append("anomaly_class not set (CLAP classify did not run)")
    if target != args.expect_object:
        fails.append(f"retrieval override target={target!r} != expected {args.expect_object!r}")

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
