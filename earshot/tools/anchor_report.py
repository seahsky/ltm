"""Re-score a finished gate run at the ANCHOR level, and apply both of ADR-0018's cuts.

    python -m earshot.tools.anchor_report runs/<tag>            # both tables
    python -m earshot.tools.anchor_report runs/<tag> --min-recall 0.5

Read-only, no GPU, seconds. It reads `rows.jsonl` and recomputes; it never re-renders.

**Why it exists.** `clapsmoke-3` reported class top-1 0.692 and pruned on it, and both were
the wrong question. The agent navigates to an OBJECT, so a class confused for a sibling of
the same anchor costs it nothing: all 60 of `snoring`'s misses landed on `breathing`, and all
53 of `clock_tick`'s landed on `clock_alarm`, so class recalls of 0.500 and 0.558 understated
what the agent would have done. Anchor accuracy is the number the task rests on.

**And the prune ignored half its own rule.** ADR-0018 disqualifies a weak-affinity class
whatever its recall, because the semantic store cannot learn an association that is not there.
`coughing` scored a perfect 1.000 and is still disqualified: people cough on every one of the
six objects. The first run kept it, and five other weak classes with it.

The two cuts are independent and both are printed with their own counts, on the rule that a
class dropped for want of DATA, for want of SEPARATION, and for want of AFFINITY are three
different findings that must not be merged into one number.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional, Sequence

from earshot.audio.separation import GateRow, anchor_top1_of, prune, summarise
from earshot.audio.vocabulary import CANDIDATE_VOCABULARY, ROOM_OF_ANCHOR

__all__ = ["load_rows", "main"]


def load_rows(path: pathlib.Path) -> List[GateRow]:
    """Every row of a run's `rows.jsonl`, as `GateRow`s that re-validate on construction.

    Re-validating is the point rather than a cost: a row whose absent class turns out to be
    in the prompt bank raises here, so a vacuous forced-failure arm cannot survive a re-score
    any more than it survived the run.
    """
    rows: List[GateRow] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                rows.append(
                    GateRow(
                        true_class=str(raw["true_class"]),
                        in_vocabulary=bool(raw["in_vocabulary"]),
                        distance_m=float(raw["distance_m"]),
                        scene=str(raw["scene"]),
                        recording_index=int(raw["recording_index"]),
                        scores={str(k): float(v) for k, v in raw["scores"].items()},
                        normal_cosine=float(raw["normal_cosine"]),
                    )
                )
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError("{}:{} is not a gate row: {}".format(path, number, exc)) from exc
    if not rows:
        raise ValueError("{} holds no rows; NOT_RUN is red".format(path))
    return rows


def _format(report: Any, rows: Sequence[GateRow], anchors: Dict[str, str], min_recall: float) -> str:
    lines: List[str] = []
    known = [row for row in rows if row.in_vocabulary]

    lines.append("")
    lines.append("=== the two numbers, side by side ===")
    lines.append(
        "  class  top-1: {:.3f}   over {} rows, {} classes (chance {:.3f})".format(
            report.top1_accuracy, report.n_rows, report.n_classes, report.chance_accuracy
        )
    )
    n_anchors = len({anchors[name] for name in {row.true_class for row in known}})
    lines.append(
        "  ANCHOR top-1: {:.3f}   over {} rows, {} anchors (chance {:.3f})".format(
            report.anchor_top1_accuracy, report.n_rows, n_anchors, 1.0 / max(1, n_anchors)
        )
    )
    lines.append(
        "  The anchor number is the one the task rests on: the agent navigates to an object."
    )

    lines.append("")
    lines.append("-- per anchor --")
    for item in sorted(report.per_anchor, key=lambda entry: -entry.accuracy):
        confusion = (
            "{} x{}".format(item.top_confusion[0], item.top_confusion[1])
            if item.top_confusion
            else "-"
        )
        lines.append(
            "  {:12s} n={:5d}  classes={:2d}  accuracy={:.3f}  top-confusion={}".format(
                item.anchor, item.n, item.n_classes, item.accuracy, confusion
            )
        )

    lines.append("")
    lines.append("-- per class: what the anchor view rescues --")
    lines.append(
        "  {:18s} {:8s} {:8s} {:8s} {:8s}".format(
            "class", "affinity", "class", "anchor", "rescued"
        )
    )
    by_class = {item.name: item for item in report.per_class}
    for name in sorted(by_class, key=lambda key: -by_class[key].recall):
        item = by_class[name]
        subset = [row for row in known if row.true_class == name]
        hits = sum(
            1 for row in subset if anchor_top1_of(row, anchors)[0] == anchor_top1_of(row, anchors)[1]
        )
        anchor_recall = hits / len(subset)
        rescued = anchor_recall - item.recall
        lines.append(
            "  {:18s} {:8s} {:8.3f} {:8.3f} {:+8.3f}".format(
                name, item.affinity, item.recall, anchor_recall, rescued
            )
        )

    kept_recall, cut_recall = prune(report, min_recall=min_recall)
    kept_both, cut_both = prune(
        report, min_recall=min_recall, allowed_affinities=("strong", "moderate")
    )
    lines.append("")
    lines.append("=== the vocabulary, under each cut ===")
    lines.append(
        "  recall only (>= {:.2f}):            {} kept   <- what clapsmoke-3 reported".format(
            min_recall, len(kept_recall)
        )
    )
    lines.append(
        "  recall AND affinity (ADR-0018):    {} kept   <- what ADR-0018 actually requires".format(
            len(kept_both)
        )
    )
    lines.append("")
    lines.append("  kept: {}".format(", ".join(sorted(kept_both)) or "NOTHING"))
    dropped_for_affinity = sorted(set(kept_recall) - set(kept_both))
    lines.append(
        "  dropped for AFFINITY despite passing recall: {}".format(
            ", ".join(dropped_for_affinity) or "none"
        )
    )
    lines.append("  dropped for SEPARATION: {}".format(", ".join(sorted(cut_recall)) or "none"))

    by_anchor: Dict[str, List[str]] = {}
    for name in kept_both:
        by_anchor.setdefault(anchors[name], []).append(name)
    lines.append("")
    lines.append("-- the surviving vocabulary by anchor: CAN IT BE SPLIT heard/not-heard? --")
    # From the MAP, not from the object table: in room mode the anchors are rooms, and
    # reading the object table here would print six objects with no classes under them.
    for anchor in sorted(set(anchors.values())):
        names = sorted(by_anchor.get(anchor, []))
        if not names:
            verdict = "NO CLASS SURVIVES — this anchor cannot appear in either column"
        elif len(names) < 2:
            verdict = "ONE class — cannot be split; it belongs to one column only"
        else:
            verdict = "splittable {}/{}".format(len(names) // 2, len(names) - len(names) // 2)
        lines.append("  {:12s} {:2d}  {:40s} {}".format(anchor, len(names), verdict, ", ".join(names)))

    lines.append("")
    lines.append(
        "  A heard/not-heard split needs at least 2 classes at an anchor. Anchors with one or"
    )
    lines.append(
        "  none appear in only one column, so the two columns face different objects and the"
    )
    lines.append("  comparison is confounded by object difficulty rather than by memory.")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-score a gate run at the anchor level and apply both ADR-0018 cuts."
    )
    parser.add_argument("run_dir")
    # `object` is the taxonomy ADR-0018 shipped. `room` is the one clapsmoke-3's confusion
    # structure argues for: `plant` scored 0.383 with 187 of 480 rows going to `toilet`,
    # because water sounds mean BATHROOM. Re-scoring costs nothing -- only the anchor map
    # changes and no audio is re-rendered -- so the taxonomy is decided on data rather than
    # on the author's second guess.
    parser.add_argument("--anchors", choices=("object", "room", "both"), default="both")
    parser.add_argument("--min-recall", type=float, default=0.50)
    parser.add_argument("--n-bands", type=int, default=4)
    args = parser.parse_args(None if argv is None else list(argv))

    run_dir = pathlib.Path(args.run_dir)
    rows_path = run_dir / "rows.jsonl"
    if not rows_path.is_file():
        print("FATAL: {} has no rows.jsonl".format(run_dir), file=sys.stderr)
        return 2

    rows = load_rows(rows_path)
    affinities = {entry.name: entry.affinity for entry in CANDIDATE_VOCABULARY}
    by_object = {entry.name: entry.anchor_object for entry in CANDIDATE_VOCABULARY}
    by_room = {
        entry.name: ROOM_OF_ANCHOR[entry.anchor_object] for entry in CANDIDATE_VOCABULARY
    }

    wanted = (
        [("object", by_object), ("room", by_room)]
        if args.anchors == "both"
        else [(args.anchors, by_object if args.anchors == "object" else by_room)]
    )

    written = []
    for label, anchors in wanted:
        report = summarise(rows, affinities=affinities, anchors=anchors, n_bands=args.n_bands)
        print("")
        print("######################################################################")
        print("### TAXONOMY: {} ".format(label.upper()))
        print("######################################################################")
        print(_format(report, rows, anchors, args.min_recall))
        out = run_dir / "anchor_report_{}.json".format(label)
        with out.open("w", encoding="utf-8") as sink:
            json.dump(report.as_dict(), sink, indent=2, sort_keys=True)
        written.append(str(out))

    print("  written: {}".format(", ".join(written)))
    print("")
    print("  The two taxonomies score the SAME rows. Nothing was re-rendered, so any")
    print("  difference between them is the grouping and only the grouping.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
