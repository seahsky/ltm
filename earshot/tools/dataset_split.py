"""The 5:3:2 split: development, verification, test. Deterministic, and split by the right unit.

    python -m earshot.tools.dataset_split                    # scenes and recordings
    python -m earshot.tools.dataset_split --ratio 5,3,2
    python -m earshot.tools.dataset_split --out runs/split.json

Read-only apart from `--out`, no GPU, instant.

**Not train/val/test, because nothing here is trained.** ADR-0018's claim against SAVN-CE is
that CLAP is a frozen open-set encoder: a 22nd class costs one more prompt, against the 14 days
on 4xA800 their ACCDDOA retrain costs. There is no training set to hold out from. The three
blocks are roles, not phases:

- **development** -- where decisions get made. The vocabulary, the prune bars, the prompts.
- **verification** -- does a decision made on development survive data it did not see.
- **test** -- touched ONCE, for the number that gets reported.

The ratio is the caller's; the naming is what keeps the blocks from being misused.

**Split by SCENE, never by episode.** Episodes inside a scene share a room, a source and a
renderer, so a random episode-level split puts the test block in rooms development already
tuned on. `funnel_diff`'s scene test already disagreed with an episode-level McNemar once for
exactly this reason.

**Two units, two orderings, and the difference is deliberate.**

- *Scenes* are ordered by `sha256` of the label. Scene IDs carry no meaning, so alphabetical
  order is arbitrary, and hashing makes the assignment stable when a scene is ADDED: a new
  label lands in its own slot instead of reshuffling every other one.
- *Recordings* are split into CONTIGUOUS index blocks, because `clap_gate.sh --clip-start`
  takes a start offset. A hashed scatter would be unusable by the tool that consumes it.

**The collision this prints below is superseded.** ADR-0018's 2026-09-01 amendment records
that the seen/unseen axis is realized by FILTERING one built store (``without_scene``), not
by holding scenes out for the matrix: an episode is ``(scene, class, anchor instance,
recording)``, and all four cells of the matrix run on the identical episode set. There is no
second 10/10 seen/unseen scene split competing with this tool's 5/3/2 ratio (10/6/4 of
twenty val scenes) -- the printed warning below predates that decision and describes a
design ADR-0018 does not use. See
``docs/adr/0018-two-memories-and-the-generalization-matrix.md``, the amendment dated
2026-09-01.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from earshot.tools.power import sign_test_threshold

__all__ = ["ROLES", "split_counts", "split_scenes", "split_recordings", "main"]

# Ordered, and the order is the ratio's order. Renaming these is a decision, not a tidy-up.
ROLES: Tuple[str, str, str] = ("development", "verification", "test")

DEFAULT_RATIO: Tuple[int, int, int] = (5, 3, 2)

# provenance: ESC-50 (Piczak 2015) ships exactly this many recordings per class.
ESC50_RECORDINGS_PER_CLASS = 40


@dataclass(frozen=True)
class Block:
    """One role's share of a unit, and enough to act on it."""

    role: str
    members: Tuple[str, ...]
    span: Optional[Tuple[int, int]] = None  # inclusive index range, recordings only

    @property
    def n(self) -> int:
        return len(self.members)


def split_counts(n: int, ratio: Sequence[int] = DEFAULT_RATIO) -> Tuple[int, ...]:
    """Exact integer counts summing to `n`, by largest remainder.

    Largest remainder rather than repeated flooring: flooring loses up to two units on a
    3-way split, and a silently dropped scene is a denominator nobody counted.
    """
    if n < 0:
        raise ValueError("n must not be negative, got {}".format(n))
    parts = [int(part) for part in ratio]
    if len(parts) != len(ROLES):
        raise ValueError(
            "ratio must have {} parts, one per role {}, got {}".format(
                len(ROLES), ROLES, parts
            )
        )
    if any(part < 0 for part in parts) or sum(parts) == 0:
        raise ValueError("ratio parts must be non-negative and not all zero, got {}".format(parts))

    total = sum(parts)
    exact = [n * part / total for part in parts]
    floors = [int(value) for value in exact]
    remainder = n - sum(floors)
    # Ties break on the earlier role, so the split is a function of (n, ratio) alone.
    order = sorted(range(len(parts)), key=lambda i: (-(exact[i] - floors[i]), i))
    for i in order[:remainder]:
        floors[i] += 1
    return tuple(floors)


def _stable_order(labels: Sequence[str]) -> List[str]:
    """Labels ordered by `sha256` of the label. Stable under insertion, unlike a shuffle."""
    return sorted(labels, key=lambda label: hashlib.sha256(label.encode("utf-8")).hexdigest())


def split_scenes(
    labels: Sequence[str], ratio: Sequence[int] = DEFAULT_RATIO
) -> Tuple[Block, ...]:
    """Assign scene labels to the three roles. Deterministic, and stable when a scene is added."""
    unique = sorted(set(labels))
    if len(unique) != len(labels):
        raise ValueError(
            "duplicate scene label(s): {}".format(
                sorted({label for label in labels if list(labels).count(label) > 1})
            )
        )
    ordered = _stable_order(unique)
    counts = split_counts(len(ordered), ratio)
    blocks: List[Block] = []
    cursor = 0
    for role, count in zip(ROLES, counts):
        blocks.append(Block(role=role, members=tuple(sorted(ordered[cursor : cursor + count]))))
        cursor += count
    return tuple(blocks)


def split_recordings(
    n_recordings: int = ESC50_RECORDINGS_PER_CLASS, ratio: Sequence[int] = DEFAULT_RATIO
) -> Tuple[Block, ...]:
    """Assign recording INDICES to the three roles as contiguous blocks.

    Contiguous because `clap_gate.sh --clip-start` consumes a start offset. `members` carries
    the indices as strings so a `Block` is one type; `span` is what a caller actually passes.
    """
    if n_recordings < 1:
        raise ValueError("n_recordings must be at least 1, got {}".format(n_recordings))
    counts = split_counts(n_recordings, ratio)
    blocks: List[Block] = []
    cursor = 0
    for role, count in zip(ROLES, counts):
        indices = tuple(str(i) for i in range(cursor, cursor + count))
        span = (cursor, cursor + count - 1) if count else None
        blocks.append(Block(role=role, members=indices, span=span))
        cursor += count
    return tuple(blocks)


def _scene_labels(split: str, root: str) -> List[str]:
    from earshot.task.episodes import available_scenes, find_split_dir

    return list(available_scenes(find_split_dir(split, root=root)))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Split scenes and ESC-50 recordings into development / verification / test."
    )
    parser.add_argument("--split", default="val", help="HM3D split to draw scenes from")
    parser.add_argument("--root", default=".")
    parser.add_argument("--ratio", default="5,3,2")
    parser.add_argument("--n-recordings", type=int, default=ESC50_RECORDINGS_PER_CLASS)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(None if argv is None else list(argv))

    try:
        ratio = tuple(int(part) for part in args.ratio.split(","))
    except ValueError:
        print("FATAL: --ratio must be comma-separated integers, got {!r}".format(args.ratio),
              file=sys.stderr)
        return 2

    print("")
    print("=== roles, not phases ===")
    print("  development   decisions are made here: vocabulary, prune bars, prompts")
    print("  verification  does a decision made on development survive unseen data")
    print("  test          touched ONCE, for the number that gets reported")
    print("")
    print("  Nothing in this design is trained. CLAP is frozen and open-set, which is the")
    print("  claim against SAVN-CE's 14-day retrain, so there is no training set.")

    try:
        labels = _scene_labels(args.split, args.root)
    except Exception as exc:  # EpisodeDataError, OSError
        labels = []
        print("")
        print("  NO SCENES: {}".format(exc))

    payload: Dict[str, object] = {
        "ratio": list(ratio),
        "roles": list(ROLES),
        "split": args.split,
    }

    if labels:
        scene_blocks = split_scenes(labels, ratio)
        payload["scenes"] = {b.role: list(b.members) for b in scene_blocks}
        print("")
        print("=== scenes ({} in {}) ===".format(len(labels), args.split))
        print("  ordered by sha256 of the label, so adding a scene does not reshuffle the rest")
        print("")
        for block in scene_blocks:
            threshold = sign_test_threshold(block.n) if block.n else None
            note = (
                "sign test IMPOSSIBLE at any outcome"
                if threshold is None
                else "sign test needs {} of {} to agree".format(threshold, block.n)
            )
            print("  {:14s} {:3d}  {}".format(block.role, block.n, note))
            print("    {}".format(", ".join(block.members) or "none"))

        print("")
        # This block used to print a COLLISION warning telling the operator the matrix
        # needed its own 10/10 seen/unseen scene split. ADR-0018's 2026-09-01 amendment
        # decided otherwise, and a tool that keeps printing a superseded instruction is
        # how a decision gets re-litigated at the terminal by someone who never reads the
        # ADR. The note stays (the split above is still a real constraint on THIS tool's
        # axis) but it now says what was decided instead of what was open.
        print("  NOTE — this is no longer a collision. ADR-0018 (amended 2026-09-01)")
        print("     realizes the matrix's seen/unseen axis by FILTERING one built store")
        print("     (`memory.store.without_scene`), not by holding scenes out, so all")
        print(
            "     four cells run on the identical episode set and the {} split above".format(
                ":".join(str(part) for part in ratio)
            )
        )
        print(
            "     is the only claim on these {} scenes.".format(len(labels))
        )

    record_blocks = split_recordings(args.n_recordings, ratio)
    payload["recordings"] = {
        b.role: {"n": b.n, "first": b.span[0] if b.span else None,
                 "last": b.span[1] if b.span else None}
        for b in record_blocks
    }
    print("")
    print("=== ESC-50 recordings per class ({} available) ===".format(args.n_recordings))
    print("  contiguous blocks, because clap_gate.sh --clip-start takes a start offset")
    print("")
    for block in record_blocks:
        if block.span is None:
            print("  {:14s}   0  empty".format(block.role))
            continue
        first, last = block.span
        print(
            "  {:14s} {:3d}  clips {}..{}   --clip-start {} --n-per-class {}".format(
                block.role, block.n, first, last, first, block.n
            )
        )

    print("")
    print("  ALREADY SPENT: clapgate-2 staged clips 0..7 and clapheld-1 staged 8..15, both")
    print("  inside the development block. The bank of record was chosen there, so")
    print("  verification and test are still clean. Do not re-stage from 0 for a final number.")
    print("")

    if args.out:
        directory = os.path.dirname(args.out)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as sink:
            json.dump(payload, sink, indent=2, sort_keys=True)
        print("  written: {}".format(args.out))
        print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
