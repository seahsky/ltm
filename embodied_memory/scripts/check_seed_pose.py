"""
check_seed_pose — renumbering-invariant verifier that a captioned episode IS the
cold (non-LOS) SEED, by comparing 3D start positions rather than episode ids.

WHY (2026-06-21): habitat overwrites each episode's ``episode_id`` with
``str(load_index)`` and its episode-iterator default is ``shuffle=True``. So a
1-step caption run can grab a random WARM pose; the old Tier-3 driver only WARNed
on ``"cold-0" not in episode_id`` — a check that can NEVER be true after renumbering
and isn't a geometry assertion anyway. This module reads the SEED's authored
``start_position`` straight from the gate-built content file (the dataset is the
ground truth) and compares it, within an epsilon, to the captioned pose recorded in
``summary.json`` (``episodes[i].start_position``, added to the runner for this gate).

Pure + stdlib-only (gzip + json + math) — no faiss/torch/habitat — so it runs in any
environment and unit-tests without the sim. The CLI exits 0 on a match (the captioned
pose IS the seed) and non-zero on a mismatch / missing data, so the Tier-3 driver can
HARD-ABORT RED-INVALID instead of WARNing.

CLI::

    python embodied_memory/scripts/check_seed_pose.py \
        --content runs/nonlos-gate/content/wcojb4TFT35.json.gz \
        --captioned-xyz 0.7821,-0.0051,-5.1784 \
        [--anomaly-class glass_break] [--eps 0.05]

  or resolve the content file from a top-level --episodes-path + --scene::

    python embodied_memory/scripts/check_seed_pose.py \
        --episodes-path runs/nonlos-gate/audiogoal.json.gz --scene wcojb4TFT35 \
        --summary runs/nonlos-tier3/summary.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
from typing import List, Optional, Sequence


class NoSeedError(ValueError):
    """No ``*-cold-0`` SEED episode found in the content file."""


def poses_match(a: Optional[Sequence[float]], b: Optional[Sequence[float]],
                eps: float = 0.05) -> bool:
    """True iff a and b are both 3-vectors within Euclidean distance ``eps`` (m).

    Conservative: any None / wrong-length / non-numeric input returns False (so a
    missing summary field can never silently pass the gate)."""
    if a is None or b is None:
        return False
    try:
        av = [float(x) for x in a]
        bv = [float(x) for x in b]
    except (TypeError, ValueError):
        return False
    if len(av) != 3 or len(bv) != 3:
        return False
    d = math.sqrt(sum((p - q) ** 2 for p, q in zip(av, bv)))
    return d <= float(eps)


def _load_gz(path: str) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def resolve_content_path(episodes_path: str, scene: str) -> str:
    """Map a top-level ``--episodes-path`` (…/<name>.json.gz, empty episodes) to its
    per-scene content file ``<dir>/content/<scene>.json.gz`` (where the built
    episodes live), mirroring habitat's relative content resolution. If
    ``episodes_path`` is already a ``…/content/<scene>.json.gz``, return it as-is."""
    d = os.path.dirname(os.path.abspath(episodes_path))
    if os.path.basename(d) == "content":
        return episodes_path
    return os.path.join(d, "content", f"{scene}.json.gz")


def read_seed_start_position(content_path: str,
                             anomaly_class: Optional[str] = None) -> List[float]:
    """Return the ``start_position`` of the cold SEED episode (id contains
    ``-cold-`` — the builder writes ``{cat}-{class}-cold-0``). When ``anomaly_class``
    is given, require the episode's ``info.anomaly_class`` to match (disambiguates a
    multi-class content file). Raises ``NoSeedError`` if none is found."""
    content = _load_gz(content_path)
    eps = content.get("episodes") or []
    for ep in eps:
        eid = str(ep.get("episode_id", ""))
        if "-cold-" not in eid:
            continue
        if anomaly_class is not None:
            info = ep.get("info") or {}
            if str(info.get("anomaly_class")) != str(anomaly_class):
                continue
        sp = ep.get("start_position")
        if sp is None:
            raise NoSeedError(
                f"SEED episode {eid!r} in {content_path} has no start_position")
        return [float(v) for v in sp]
    raise NoSeedError(
        f"no '*-cold-*' SEED episode in {content_path}"
        + (f" for anomaly_class={anomaly_class!r}" if anomaly_class else ""))


def _captioned_xyz_from_summary(summary_path: str) -> Optional[List[float]]:
    """Pull episodes[0].start_position from a run summary.json (the captioned pose
    — the 1-step caption run writes exactly one episode row)."""
    d = _load_gz(summary_path) if summary_path.endswith(".gz") else json.load(
        open(summary_path, encoding="utf-8"))
    rows = d.get("episodes") or []
    if not rows:
        return None
    sp = rows[0].get("start_position")
    if sp is None:
        return None
    try:
        return [float(v) for v in sp]
    except (TypeError, ValueError):
        return None


def _parse_xyz(s: str) -> List[float]:
    parts = [p for p in s.replace(" ", "").split(",") if p != ""]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"--captioned-xyz must be 'x,y,z', got {s!r}")
    return [float(p) for p in parts]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--content", help="…/content/<scene>.json.gz with the built episodes")
    g.add_argument("--episodes-path", help="top-level …/<name>.json.gz (needs --scene)")
    ap.add_argument("--scene", help="bare scene name (with --episodes-path)")
    ap.add_argument("--anomaly-class", default=None,
                    help="disambiguate the seed in a multi-class content file")
    cap = ap.add_mutually_exclusive_group(required=True)
    cap.add_argument("--captioned-xyz", type=_parse_xyz,
                     help="the captioned pose 'x,y,z' (e.g. read off a HUD)")
    cap.add_argument("--summary",
                     help="run summary.json; reads episodes[0].start_position")
    cap.add_argument("--check-seed-exists", action="store_true",
                     help="preflight: just confirm a resolvable *-cold-* SEED with a "
                          "start_position exists (no captioned pose needed); exit 0/2")
    ap.add_argument("--eps", type=float, default=0.05, help="match radius in metres")
    args = ap.parse_args(argv)

    if args.episodes_path is not None:
        if not args.scene:
            ap.error("--episodes-path requires --scene")
        content_path = resolve_content_path(args.episodes_path, args.scene)
    else:
        content_path = args.content
    if not os.path.isfile(content_path):
        print(f"RED-INVALID: content file not found: {content_path}", file=sys.stderr)
        return 2

    try:
        seed_xyz = read_seed_start_position(content_path, args.anomaly_class)
    except NoSeedError as ex:
        print(f"RED-INVALID: {ex}", file=sys.stderr)
        return 2

    if args.check_seed_exists:
        sx, sy, sz = seed_xyz
        print(f"SEED-EXISTS: {sx:.4f},{sy:.4f},{sz:.4f}  (from {content_path})")
        return 0

    if args.captioned_xyz is not None:
        captioned = args.captioned_xyz
    else:
        captioned = _captioned_xyz_from_summary(args.summary)

    if captioned is None:
        print("RED-INVALID: captioned start_position is missing — the summary lacks "
              "episodes[0].start_position (re-run with the start_position-emitting "
              "runner) so the seed cannot be verified.", file=sys.stderr)
        return 2

    sx, sy, sz = seed_xyz
    cx, cy, cz = captioned
    d = math.sqrt((sx - cx) ** 2 + (sy - cy) ** 2 + (sz - cz) ** 2)
    print(f"  seed  start_position = {sx:.4f},{sy:.4f},{sz:.4f}  (from {content_path})")
    print(f"  captioned pose       = {cx:.4f},{cy:.4f},{cz:.4f}")
    print(f"  distance             = {d:.4f} m   (eps={args.eps} m)")
    if poses_match(seed_xyz, captioned, eps=args.eps):
        print("SEED-MATCH: the captioned pose IS the non-LOS seed.")
        return 0
    print("RED-INVALID: captioned pose is NOT the non-LOS seed — habitat captioned a "
          "DIFFERENT episode (e.g. a shuffled warm far-start). The Tier-3 verdict is "
          "untrustworthy. Caption index 0 deterministically (shuffle must be OFF) and "
          "re-run.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
