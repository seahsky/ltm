"""Score a finished gate run at the ANCHOR level and apply all three of ADR-0018's cuts.

    python -m earshot.tools.anchor_report runs/<tag>                 # the room taxonomy
    python -m earshot.tools.anchor_report runs/<tag> --anchors both  # and the retired one
    python -m earshot.tools.anchor_report runs/<tag> --min-recall 0.5

Read-only, no GPU, seconds. It reads `rows.jsonl` and recomputes; it never re-renders. Stage 4
of `tools/clap_gate.sh` calls it, so a live run and a re-score of that run apply the same cut
by construction rather than by two implementations agreeing.

**Why it exists.** `clapsmoke-3` reported class top-1 0.692 and pruned on it, and both were
the wrong question. The agent navigates to a ROOM, so a class confused for a sibling of the
same room costs it nothing: all 60 of `snoring`'s misses landed on `breathing`, and all 53 of
`clock_tick`'s landed on `clock_alarm`, so class recalls of 0.500 and 0.558 understated what
the agent would have done. Anchor accuracy is the number the task rests on.

**And the first prune applied one of its own three rules.** A weak-affinity class is
disqualified whatever its recall, because the semantic store cannot learn an association that
is not there. `coughing` scored a perfect 1.000 and is still disqualified: people cough in
every room. The run kept it and two other weak classes.

The cuts are independent and each is printed with its own count, on the rule that a class
dropped for want of DATA, for want of SEPARATION, and for want of AFFINITY are three different
findings that must not be merged into one number.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Mapping, Optional, Sequence

from earshot.audio.separation import GateRow, SeparationReport, prune, summarise
from earshot.audio.vocabulary import CANDIDATE_VOCABULARY, ROOM_OF_ANCHOR

__all__ = ["load_rows", "main"]

# The taxonomy of record, as of the 2026-08-20 amendment to ADR-0018. `pruned_vocabulary.json`
# is written from this one only: two files carrying two different surviving vocabularies is how
# a later run reads the wrong one.
TAXONOMY_OF_RECORD = "room"


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


def _cuts(report: SeparationReport, min_recall: float) -> Dict[str, Sequence[str]]:
    """The surviving vocabulary under each bar, so the looser one can be checked.

    ADR-0018's bar is `adr`: anchor recall AND affinity. The other two are here to be
    subtracted from it, not to be quoted.
    """
    adr_kept, _ = prune(
        report,
        min_recall=min_recall,
        recall_level="anchor",
        allowed_affinities=("strong", "moderate"),
    )
    anchor_only, _ = prune(report, min_recall=min_recall, recall_level="anchor")
    class_only, _ = prune(report, min_recall=min_recall, recall_level="class")
    class_and_affinity, _ = prune(
        report,
        min_recall=min_recall,
        recall_level="class",
        allowed_affinities=("strong", "moderate"),
    )
    return {
        "adr": adr_kept,
        "anchor_only": anchor_only,
        "class_only": class_only,
        "class_and_affinity": class_and_affinity,
    }


def _format(
    report: SeparationReport,
    anchors: Mapping[str, str],
    min_recall: float,
) -> str:
    lines: List[str] = []

    lines.append("")
    lines.append("=== the two numbers, side by side ===")
    lines.append(
        "  class  top-1: {:.3f}   over {} rows, {} classes (chance {:.3f})".format(
            report.top1_accuracy, report.n_rows, report.n_classes, report.chance_accuracy
        )
    )
    n_anchors = len(report.per_anchor)
    lines.append(
        "  ANCHOR top-1: {:.3f}   over {} rows, {} anchors (chance {:.3f})".format(
            report.anchor_top1_accuracy, report.n_rows, n_anchors, 1.0 / max(1, n_anchors)
        )
    )
    lines.append(
        "  The anchor number is the one the task rests on: the agent navigates to a room."
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
        "  {:18s} {:8s} {:>8s} {:>8s} {:>9s}".format(
            "class", "affinity", "class", "anchor", "rescued"
        )
    )
    for item in sorted(report.per_class, key=lambda entry: -entry.recall):
        lines.append(
            "  {:18s} {:8s} {:8.3f} {:8.3f} {:+9.3f}".format(
                item.name,
                item.affinity,
                item.recall,
                item.anchor_recall,
                item.anchor_recall - item.recall,
            )
        )

    cuts = _cuts(report, min_recall)
    lines.append("")
    lines.append("=== the vocabulary, under each bar (min_recall {:.2f}) ===".format(min_recall))
    lines.append(
        "  class recall only:                 {:2d} kept   <- what clapsmoke-3 reported".format(
            len(cuts["class_only"])
        )
    )
    lines.append(
        "  class recall AND affinity:         {:2d} kept   <- the stricter bar, kept "
        "scoreable".format(len(cuts["class_and_affinity"]))
    )
    lines.append(
        "  anchor recall only:                {:2d} kept".format(len(cuts["anchor_only"]))
    )
    lines.append(
        "  ANCHOR recall AND affinity:        {:2d} kept   <- ADR-0018, and the bar of "
        "record".format(len(cuts["adr"]))
    )

    kept = list(cuts["adr"])
    lines.append("")
    lines.append("  kept: {}".format(", ".join(sorted(kept)) or "NOTHING"))
    lines.append(
        "  rescued by scoring at the anchor: {}".format(
            ", ".join(sorted(set(kept) - set(cuts["class_and_affinity"]))) or "none"
        )
    )
    lines.append(
        "  dropped for AFFINITY despite passing anchor recall: {}".format(
            ", ".join(sorted(set(cuts["anchor_only"]) - set(kept))) or "none"
        )
    )
    lines.append(
        "  dropped for SEPARATION even at the anchor: {}".format(
            ", ".join(
                sorted(
                    {item.name for item in report.per_class}
                    - set(cuts["anchor_only"])
                    - {item.name for item in report.per_class if item.n < 8}
                )
            )
            or "none"
        )
    )
    lines.append(
        "  dropped for TOO FEW ROWS (not a separation finding): {}".format(
            ", ".join(sorted(item.name for item in report.per_class if item.n < 8)) or "none"
        )
    )

    # The strong-versus-weak read. The grades are declared judgements (ADR-0018) and were NOT
    # derived from recall, so this says whether the gate agreed with the table -- a check on
    # the vocabulary's design rather than on CLAP.
    lines.append("")
    lines.append("-- by declared affinity: does the gate agree with the table? --")
    for grade in ("strong", "moderate", "weak", "unknown"):
        at_grade = [item for item in report.per_class if item.affinity == grade]
        if not at_grade:
            continue
        mean_class = sum(item.recall for item in at_grade) / len(at_grade)
        mean_anchor = sum(item.anchor_recall for item in at_grade) / len(at_grade)
        survived = len([item for item in at_grade if item.name in kept])
        lines.append(
            "  {:8s} n_classes={:2d}  mean class={:.3f}  mean anchor={:.3f}  survived={}/{}".format(
                grade, len(at_grade), mean_class, mean_anchor, survived, len(at_grade)
            )
        )

    by_anchor: Dict[str, List[str]] = {}
    for name in kept:
        by_anchor.setdefault(anchors[name], []).append(name)
    lines.append("")
    lines.append("-- the surviving vocabulary by anchor: CAN IT BE SPLIT heard/not-heard? --")
    # From the MAP, not from the object table: in room mode the anchors are rooms, and reading
    # the object table here would print six objects with no classes under them.
    for anchor in sorted(set(anchors.values())):
        names = sorted(by_anchor.get(anchor, []))
        if not names:
            verdict = "NO CLASS SURVIVES - this anchor cannot appear in either column"
        elif len(names) < 2:
            verdict = "ONE class - cannot be split; it belongs to one column only"
        else:
            verdict = "splittable {}/{}".format(len(names) // 2, len(names) - len(names) // 2)
        lines.append(
            "  {:12s} {:2d}  {:44s} {}".format(anchor, len(names), verdict, ", ".join(names))
        )

    lines.append("")
    lines.append(
        "  A heard/not-heard split needs at least 2 classes at an anchor. Anchors with one or"
    )
    lines.append(
        "  none appear in only one column, so the two columns face different rooms and the"
    )
    lines.append("  comparison is confounded by room difficulty rather than by memory.")

    lines.append("")
    lines.append(
        "FORCED-FAILURE ARM: EER {:.3f}  (0.500 = the two arms are on top of each other)".format(
            report.rejection.eer
        )
    )
    lines.append("")
    return "\n".join(lines)


def _scenes_failed(run_dir: pathlib.Path) -> Sequence[str]:
    """Scenes the live gate could not load. Absent file means this is a bare re-score."""
    path = run_dir / "separation.json"
    if not path.is_file():
        return ()
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return tuple(payload.get("scenes_failed") or ())


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score a gate run at the anchor level and apply all three ADR-0018 cuts."
    )
    parser.add_argument("run_dir")
    # `room` is the taxonomy as of the 2026-08-20 amendment and the default. `object` is kept
    # so the superseded grouping stays scoreable: clapsmoke-3 measured 0.764 for objects
    # against 0.779 for rooms, and the near-tie is itself the finding. The room grouping was
    # adopted for SPLITTABILITY, not for accuracy, and a reader who assumes otherwise should be
    # able to check. Re-scoring costs nothing; no audio is re-rendered.
    parser.add_argument("--anchors", choices=("object", "room", "both"), default="room")
    parser.add_argument("--min-recall", type=float, default=0.50)
    parser.add_argument("--n-bands", type=int, default=4)
    args = parser.parse_args(None if argv is None else list(argv))

    run_dir = pathlib.Path(args.run_dir)
    rows_path = run_dir / "rows.jsonl"
    if not rows_path.is_file():
        print("FATAL: {} has no rows.jsonl".format(run_dir), file=sys.stderr)
        return 2

    rows = load_rows(rows_path)
    affinities = {entry.name: entry.room_affinity for entry in CANDIDATE_VOCABULARY}
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
        print(_format(report, anchors, args.min_recall))
        out = run_dir / "anchor_report_{}.json".format(label)
        with out.open("w", encoding="utf-8") as sink:
            json.dump(report.as_dict(), sink, indent=2, sort_keys=True)
        written.append(str(out))

        if label == TAXONOMY_OF_RECORD:
            cuts = _cuts(report, args.min_recall)
            out = run_dir / "pruned_vocabulary.json"
            with out.open("w", encoding="utf-8") as sink:
                json.dump(
                    {
                        "min_recall": args.min_recall,
                        "recall_level": "anchor",
                        "allowed_affinities": ["strong", "moderate"],
                        "taxonomy": TAXONOMY_OF_RECORD,
                        "kept": sorted(cuts["adr"]),
                        "kept_under_class_recall": sorted(cuts["class_and_affinity"]),
                        "by_anchor": {
                            anchor: sorted(
                                name for name in cuts["adr"] if anchors[name] == anchor
                            )
                            for anchor in sorted(set(anchors.values()))
                        },
                    },
                    sink,
                    indent=2,
                    sort_keys=True,
                )
            written.append(str(out))

    print("  written: {}".format(", ".join(written)))
    if len(wanted) > 1:
        print("")
        print("  The two taxonomies score the SAME rows. Nothing was re-rendered, so any")
        print("  difference between them is the grouping and only the grouping.")

    failed = _scenes_failed(run_dir)
    if failed:
        print("")
        print("SCENES THAT FAILED TO LOAD ({}): {}".format(len(failed), " ".join(failed)))
        print("  Continuing is not passing - this run is INCOMPLETE.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
