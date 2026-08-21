"""The bank of record: the classes that cleared the bar on EVERY disjoint recording set.

    python -m earshot.tools.bank_intersect runs/clapgate-2 runs/clapheld-1

Read-only apart from the one file it writes, no GPU, seconds.

**Why an intersection rather than a single run's prune.** `clapgate-2` (ESC-50 clips 0-7) and
`clapheld-1` (clips 8-15) agree on the aggregate to 0.013 anchor top-1 and on the open-set EER
to three decimals, but they do NOT agree per class. `water_drops` scored 0.998 anchor recall on
one set and 0.449 on the other; `mouse_click` went 0.308 to 0.789. Eight recordings do not pin
a class, so each run's prune picks a different twelfth class and neither one is trustworthy
alone.

A class in the intersection cleared the bar on two recording sets that share no audio. That is
a held-out validation per class, obtained for free, and it is the strongest claim available
without staging more recordings.

**What this file is NOT.** Scoring either input run against this bank is not an unbiased
measurement: the bank was derived using both of them. The evidence for the bank is the
side-by-side table this prints, where every kept class clears the bar in every column
independently. A fresh unbiased number needs a THIRD recording set (`--clip-start 16`).

**Disjointness is checked, not assumed.** Each run's `provenance.txt` carries `clip_start` and
`n_per_class`; overlapping ranges make the whole argument vacuous, so they raise here.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Optional, Sequence, Tuple

from earshot.audio.separation import prune, summarise
from earshot.audio.vocabulary import CANDIDATE_VOCABULARY, ROOM_OF_ANCHOR
from earshot.tools.anchor_report import load_rows

__all__ = ["load_provenance", "clip_range", "assert_disjoint", "main"]


def load_provenance(run_dir: pathlib.Path) -> Dict[str, str]:
    """`provenance.txt` as a dict. Raises if absent: a run that cannot say what it staged
    cannot take part in a disjointness argument."""
    path = run_dir / "provenance.txt"
    if not path.is_file():
        raise ValueError(
            "{} has no provenance.txt, so its recording range is unknown and it cannot be "
            "claimed disjoint from anything".format(run_dir)
        )
    out: Dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            for field in line.strip().split():
                if "=" in field:
                    key, _, value = field.partition("=")
                    out[key] = value
    return out


def clip_range(provenance: Dict[str, str]) -> Tuple[int, int]:
    """`(first, last)` ESC-50 recording index this run staged, inclusive.

    `clip_start` is absent from any run made before the flag existed. Those runs staged from 0
    by construction, so it defaults to 0 rather than raising: the default is a fact about the
    old code path, not a guess.
    """
    start = int(provenance.get("clip_start", 0))
    count = int(provenance.get("n_per_class", 8))
    if count < 1:
        raise ValueError("n_per_class={} is not a recording count".format(count))
    return start, start + count - 1


def assert_disjoint(ranges: Sequence[Tuple[str, Tuple[int, int]]]) -> None:
    """Raise if any two runs staged an overlapping recording range."""
    for index, (name_a, (low_a, high_a)) in enumerate(ranges):
        for name_b, (low_b, high_b) in ranges[index + 1 :]:
            if low_a <= high_b and low_b <= high_a:
                raise ValueError(
                    "{} staged recordings {}..{} and {} staged {}..{}; they OVERLAP, so a "
                    "class clearing the bar in both is not two independent observations and "
                    "the intersection argument is vacuous".format(
                        name_a, low_a, high_a, name_b, low_b, high_b
                    )
                )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Intersect the pruned vocabularies of runs on disjoint recording sets."
    )
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--min-recall", type=float, default=0.50)
    parser.add_argument("--out", default="runs/bank_of_record.json")
    args = parser.parse_args(None if argv is None else list(argv))

    if len(args.run_dirs) < 2:
        print(
            "FATAL: an intersection of one run is that run's prune. Give at least two runs on "
            "disjoint recording sets, or use anchor_report.",
            file=sys.stderr,
        )
        return 2

    affinities = {entry.name: entry.room_affinity for entry in CANDIDATE_VOCABULARY}
    anchors = {
        entry.name: ROOM_OF_ANCHOR[entry.anchor_object] for entry in CANDIDATE_VOCABULARY
    }

    ranges: List[Tuple[str, Tuple[int, int]]] = []
    columns: List[Tuple[str, Dict[str, float], Sequence[str]]] = []
    for raw in args.run_dirs:
        run_dir = pathlib.Path(raw)
        provenance = load_provenance(run_dir)
        ranges.append((run_dir.name, clip_range(provenance)))
        report = summarise(
            load_rows(run_dir / "rows.jsonl"), affinities=affinities, anchors=anchors
        )
        kept, _cut = prune(
            report,
            min_recall=args.min_recall,
            recall_level="anchor",
            allowed_affinities=("strong", "moderate"),
        )
        recalls = {item.name: float(item.anchor_recall) for item in report.per_class}
        columns.append((run_dir.name, recalls, kept))

    assert_disjoint(ranges)

    print("")
    print("=== the runs, and the recordings each staged ===")
    for name, (low, high) in ranges:
        print("  {:20s} ESC-50 recordings {}..{}".format(name, low, high))
    print("  Disjoint, so a class clearing the bar in every column did so on audio the other")
    print("  columns never saw. That is the held-out validation, one class at a time.")

    every = sorted(affinities)
    intersection = sorted(set.intersection(*[set(kept) for _n, _r, kept in columns]))
    union = sorted(set.union(*[set(kept) for _n, _r, kept in columns]))

    print("")
    print("=== anchor recall per class, per run (bar {:.2f}) ===".format(args.min_recall))
    header = "  {:18s} {:8s}".format("class", "affinity")
    for name, _recalls, _kept in columns:
        header += " {:>14s}".format(name[:14])
    print(header + "   verdict")
    for name in every:
        line = "  {:18s} {:8s}".format(name, affinities[name])
        for _run, recalls, kept in columns:
            mark = "*" if name in kept else " "
            line += " {:>13.3f}{}".format(recalls.get(name, float("nan")), mark)
        if name in intersection:
            verdict = "KEEP"
        elif name in union:
            verdict = "DISPUTED - cut"
        elif affinities[name] == "weak":
            verdict = "cut (affinity)"
        else:
            verdict = "cut (separation)"
        print(line + "   " + verdict)
    print("  * = cleared both cuts in that run.")

    print("")
    print("=== the bank of record ===")
    print("  intersection: {} classes".format(len(intersection)))
    print("  union:        {} classes".format(len(union)))
    disputed = sorted(set(union) - set(intersection))
    print(
        "  DISPUTED (kept by some runs, cut by others): {}".format(
            ", ".join(disputed) or "none"
        )
    )
    print("  A disputed class is not a marginal call to be settled by judgement. It is a class")
    print("  whose recall depends on which recordings it drew, which is the one thing the")
    print("  heard/not-heard column must not be confounded by.")

    by_room: Dict[str, List[str]] = {}
    for name in intersection:
        by_room.setdefault(anchors[name], []).append(name)
    print("")
    print("=== splittability of the bank of record ===")
    unsplittable = []
    for room in sorted(set(anchors.values())):
        names = sorted(by_room.get(room, []))
        if len(names) >= 2:
            verdict = "splittable {}/{}".format(len(names) // 2, len(names) - len(names) // 2)
        else:
            verdict = "CANNOT BE SPLIT"
            unsplittable.append(room)
        print("  {:12s} {:2d}  {:24s} {}".format(room, len(names), verdict, ", ".join(names)))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as sink:
        json.dump(
            {
                "kept": intersection,
                "disputed": disputed,
                "min_recall": args.min_recall,
                "recall_level": "anchor",
                "allowed_affinities": ["strong", "moderate"],
                "taxonomy": "room",
                "derived_from": [
                    {"run": name, "clip_range": list(rng), "kept": sorted(kept)}
                    for (name, rng), (_n, _r, kept) in zip(ranges, columns)
                ],
                "by_anchor": {
                    room: sorted(by_room.get(room, []))
                    for room in sorted(set(anchors.values()))
                },
            },
            sink,
            indent=2,
            sort_keys=True,
        )
    print("")
    print("  written: {}".format(out))
    print("  Pass it to anchor_report as --bank. Scoring either INPUT run against it is not")
    print("  unbiased -- both chose it. A fresh unbiased number needs a third recording set.")
    print("")

    if unsplittable:
        print(
            "EXIT NONZERO: {} cannot be split heard/not-heard, so it would appear in one "
            "column only and the columns would differ by room rather than by memory.".format(
                ", ".join(unsplittable)
            )
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
