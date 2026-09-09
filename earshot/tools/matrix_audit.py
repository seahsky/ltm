"""What a finished matrix sweep can already answer, read back off its own artefacts.

The matrix-1 review (2026-09-05) ended with four questions that need no GPU and no
re-run — every answer is in `runs/<tag>/prior/store.json` and the per-episode audits the
four cells wrote. This tool asks them, in the review's own order:

**A. THE STORE.** Which assigned scenes the prior pass completed, which it left
incomplete, which failed to load — and, for a store written before `pass_provenance`
recorded incompletes, which scenes are in NO list at all, which is the silent case the
review named D3: a scene the sweep ran with its seen cells byte-identical to its unseen
cells and no error anywhere.

**B. THE SEEN AXIS, PER PAIR.** For each (seen, unseen) arm pair sharing a semantic
condition, how often the recalled prior actually differed. The clean signature of a live
episodic store is `memory_prior_instances` differing at a shared category — the seen cell
resolves through the tour's narrowed candidate set, the unseen cell through the scene's
full ObjectNav table — because the voted CATEGORY comes from the semantic store both
cells share, and a category flip is render noise, not the seen axis.

**C. WHAT THE NOT_HEARD CELLS WERE TOLD.** The review's D2: `without_class` strips only
the run's own class, `predict_category` has no abstain, so a `not_heard` episode should
show a confident WRONG category, nearly never `no_prediction`. This section is that
claim, counted on the real audits.

**D. DETERMINISM, FOR FREE.** In a scene the episodic store holds ZERO rows for, the
seen and unseen cells of one semantic condition ran byte-identical inputs. Every
discordant SOURCE_REACHED pair in such a scene is therefore apparatus noise, measured
without spending a run on it — the check `repeat-1` cost a full re-run to make.

**E. WHETHER AN ABSTAIN THRESHOLD EXISTS.** The question D2 leaves open. The vote
carries its own mean cosine (`memory_prior_confidence`), so if a wrong recall scores
lower than a right one, `_vote` can decline to answer and a `not_heard` cell becomes an
arm with NO prior rather than an arm with a confidently wrong one. If the two
distributions overlap, no single floor does that, and only a `NONE` arm controls for it.
The grading key is the sweep's own `assignment.tsv`: a recall is CORRECT when the voted
category is the anchor object of the class that scene actually ran. Confidence reaches
the audit only on a RESOLVED prior, so every miss is outside this section by
construction — including the `unreachable` sub-arm — and the section prints how many
episodes that leaves it blind to.

`--gate-scenes` is the enforcement half, for `matrix_sweep.sh`: exit 2 unless every
assigned scene is in the store's `scenes_complete`. A prior pass that silently dropped a
scene must stop the sweep before the cells run, because "NOT_RUN is never green".

Read-only throughout. Nothing here writes, renders, or needs a simulator.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "DEFAULT_ARMS",
    "scene_of",
    "store_coverage",
    "gate_missing",
    "load_memory_rows",
    "seen_axis_divergence",
    "prior_distribution",
    "discordance_where_identical",
    "load_assignment",
    "anchors_by_scene",
    "graded_confidences",
    "abstain_table",
    "main",
]

# `MemoryCondition`'s own values, in the sweep's own order — the same names
# `matrix_sweep.sh` uses as directory names.
DEFAULT_ARMS: Tuple[str, ...] = (
    "heard_seen",
    "heard_unseen",
    "not_heard_seen",
    "not_heard_unseen",
)

# The (seen, unseen) pairs that share a semantic condition — the only pairs in which a
# prior difference can be attributed to the episodic store rather than to `without_class`.
SEEN_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("heard_seen", "heard_unseen"),
    ("not_heard_seen", "not_heard_unseen"),
)


def scene_of(entry: Any) -> str:
    """The scene name out of a provenance list element.

    `scenes_complete` holds plain strings; `scenes_incomplete` and `scenes_failed` hold
    `SceneTourOutcome.as_dict()` mappings. One accessor, so a caller never branches on
    which list it happens to be reading.
    """
    if isinstance(entry, Mapping):
        return str(entry["scene"])
    return str(entry)


def store_coverage(
    provenance: Mapping[str, Any],
    rows_by_scene: Mapping[str, int],
    assigned: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Every assigned scene, sorted into what the prior pass did with it. Pure.

    `unaccounted` is the list that must be empty: a scene in none of complete /
    incomplete / failed. A store written by `pass_provenance` cannot produce one; a store
    written before it (matrix-1's own) reports its silent incompletes here, which is the
    whole reason this function takes the provenance rather than trusting it.

    `complete_without_rows` is the defensive cross-check the other way: a scene the
    provenance calls complete whose episodic rows never made it into the store would be
    a store/provenance disagreement, and neither side of one is trustworthy.
    """
    complete = [scene_of(e) for e in provenance.get("scenes_complete", [])]
    incomplete = [scene_of(e) for e in provenance.get("scenes_incomplete", [])]
    failed = [scene_of(e) for e in provenance.get("scenes_failed", [])]
    requested = [scene_of(e) for e in provenance.get("scenes_requested", [])]
    scenes = list(assigned) if assigned is not None else requested

    # The recorded REASON, kept beside the name. `scenes_incomplete` exists precisely so
    # a scene that did not tour says why, and a coverage report that prints only the name
    # sends the reader back to the log for the one fact the store already holds.
    detail: Dict[str, Dict[str, Any]] = {}
    for entry in list(provenance.get("scenes_incomplete", ())) + list(
        provenance.get("scenes_failed", ())
    ):
        if isinstance(entry, Mapping):
            detail[scene_of(entry)] = dict(entry)

    accounted = set(complete) | set(incomplete) | set(failed)
    return {
        "assigned": scenes,
        "complete": complete,
        "incomplete": incomplete,
        "failed": failed,
        "detail": detail,
        "unaccounted": sorted(set(scenes) - accounted),
        "records_incomplete": "scenes_incomplete" in provenance,
        "complete_without_rows": sorted(
            scene for scene in complete if int(rows_by_scene.get(scene, 0)) == 0
        ),
        "rows_by_scene": {scene: int(rows_by_scene.get(scene, 0)) for scene in scenes},
    }


def gate_missing(coverage: Mapping[str, Any]) -> List[str]:
    """The assigned scenes a sweep must NOT run its cells over: everything the prior
    pass did not complete, however it failed to. Empty means the gate is green. Pure."""
    complete = set(coverage["complete"])
    return sorted(scene for scene in coverage["assigned"] if scene not in complete)


def load_memory_rows(arm_dir: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """`{scene: {episode_index: row}}` for one arm directory — the only disk reader here.

    Each row carries the memory fields the audit wrote (`memory_prior_category`,
    `memory_prior_miss`, `memory_condition`, and the three `memory_prior_*` metrics) next
    to the same `reached`/`source` pair `episode_diff.load_outcomes` reads, so the
    determinism section can verify its pairs the way `episode_diff` does: same scene,
    same index, same `source_xyz`, or it is not a pair.
    """
    from earshot.report.artifacts import episode_paths, read_audit, run_paths
    from earshot.task.smoke import episode_indices
    from earshot.tools.funnel_diff import HEADLINE_STAGE

    root, _ = run_paths(arm_dir)
    if not root.is_dir():
        raise ValueError("{} is not a directory".format(arm_dir))
    if run_paths(root)[1].is_dir():
        raise ValueError(
            "{} looks like a SCENE directory, not an arm — pass the arm directory that "
            "holds one subdirectory per scene".format(arm_dir)
        )
    out: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        rows: Dict[int, Dict[str, Any]] = {}
        for index in episode_indices(str(scene_dir)):
            _, audit_path = episode_paths(scene_dir, index)
            audit = read_audit(audit_path)
            source = audit.source_xyz
            metrics = dict(audit.metrics or {})
            rows[int(index)] = {
                "reached": audit.funnel_stage >= HEADLINE_STAGE,
                "source": None if source is None else tuple(
                    float(v) for v in source.as_tuple()
                ),
                "condition": audit.memory_condition,
                "category": audit.memory_prior_category,
                "miss": audit.memory_prior_miss,
                "instances": metrics.get("memory_prior_instances"),
                "distance_m": metrics.get("memory_prior_distance_m"),
                # Present only on a RESOLVED prior: `MemoryPrior.as_metrics` is what
                # writes it, and a miss builds no `MemoryPrior`. So a `None` here means
                # "the vote's score was never recorded", NOT "the vote scored zero" --
                # `test_task_memory_arm` holds that distinction on the writing side.
                "confidence": metrics.get("memory_prior_confidence"),
            }
        out[scene_dir.name] = rows
    return out


def _paired(
    seen: Mapping[str, Mapping[int, Mapping[str, Any]]],
    unseen: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> List[Tuple[str, int, Mapping[str, Any], Mapping[str, Any]]]:
    """Every (scene, index) both arms hold with a matching source. `episode_diff`'s own
    discipline: an episode that cannot be verified as the same task is dropped and the
    caller reports how many were."""
    pairs: List[Tuple[str, int, Mapping[str, Any], Mapping[str, Any]]] = []
    for scene in sorted(set(seen) & set(unseen)):
        for index in sorted(set(seen[scene]) & set(unseen[scene])):
            a, b = seen[scene][index], unseen[scene][index]
            if a["source"] is None or b["source"] is None:
                continue
            if any(abs(x - y) > 1e-6 for x, y in zip(a["source"], b["source"])):
                continue
            pairs.append((scene, index, a, b))
    return pairs


def seen_axis_divergence(
    seen: Mapping[str, Mapping[int, Mapping[str, Any]]],
    unseen: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> Dict[str, Any]:
    """How often the seen arm's prior differed from the unseen arm's, component by
    component, over verified pairs. Pure.

    `instances_differ_same_category` is the number that measures the seen axis: same
    voted category (so the semantic half agreed), different candidate count (so the
    episodic narrowing was live). `category_differs` is reported beside it but cannot be
    attributed — the two arms rendered their own audio, so the vote itself carries
    render noise.
    """
    pairs = _paired(seen, unseen)
    consulted = [
        (scene, a, b)
        for scene, _index, a, b in pairs
        if (a["category"] is not None or a["miss"] is not None)
        and (b["category"] is not None or b["miss"] is not None)
    ]
    category_differs = sum(1 for _s, a, b in consulted if a["category"] != b["category"])
    same_category = [
        (scene, a, b)
        for scene, a, b in consulted
        if a["category"] is not None and a["category"] == b["category"]
    ]

    def _ne(x: Optional[float], y: Optional[float]) -> bool:
        if x is None or y is None:
            return (x is None) != (y is None)
        return abs(float(x) - float(y)) > 1e-6

    instances_differ = [
        scene for scene, a, b in same_category if _ne(a["instances"], b["instances"])
    ]
    distance_differs = sum(
        1 for _s, a, b in same_category if _ne(a["distance_m"], b["distance_m"])
    )
    return {
        "pairs": len(pairs),
        "consulted_both": len(consulted),
        "category_differs": category_differs,
        "same_category": len(same_category),
        "instances_differ_same_category": len(instances_differ),
        "instances_differ_scenes": sorted(set(instances_differ)),
        "distance_differs_same_category": distance_differs,
    }


def prior_distribution(
    arm: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> Dict[str, Any]:
    """What this arm's episodes were told by the memory: category tallies, miss tallies,
    and the episodes on which the prior was never consulted. Pure."""
    categories: Dict[str, int] = {}
    misses: Dict[str, int] = {}
    never = 0
    total = 0
    for rows in arm.values():
        for row in rows.values():
            total += 1
            if row["category"] is not None:
                categories[row["category"]] = categories.get(row["category"], 0) + 1
            elif row["miss"] is not None:
                misses[row["miss"]] = misses.get(row["miss"], 0) + 1
            else:
                never += 1
    return {
        "episodes": total,
        "categories": dict(sorted(categories.items())),
        "misses": dict(sorted(misses.items())),
        "never_consulted": never,
    }


def discordance_where_identical(
    seen: Mapping[str, Mapping[int, Mapping[str, Any]]],
    unseen: Mapping[str, Mapping[int, Mapping[str, Any]]],
    rows_by_scene: Mapping[str, int],
) -> Dict[str, Any]:
    """SOURCE_REACHED discordance split by whether the episodic store could have made the
    two arms differ at all. Pure.

    In a scene with ZERO episodic rows, `stores_for_cell`'s seen and unseen branches
    return element-identical stores, so the two arms of one semantic condition ran the
    same inputs — every discordant pair there is the apparatus flipping on its own, the
    fact `repeat-1` bought with a full re-run. The `with_rows` split is context, not a
    control: those arms ran genuinely different inputs.
    """
    zero = {"pairs": 0, "discordant": 0, "scenes": {}}  # type: Dict[str, Any]
    with_rows = {"pairs": 0, "discordant": 0, "scenes": {}}  # type: Dict[str, Any]
    for scene, _index, a, b in _paired(seen, unseen):
        bucket = zero if int(rows_by_scene.get(scene, 0)) == 0 else with_rows
        bucket["pairs"] += 1
        if bool(a["reached"]) != bool(b["reached"]):
            bucket["discordant"] += 1
            bucket["scenes"][scene] = bucket["scenes"].get(scene, 0) + 1
    return {"zero_row_scenes": zero, "scenes_with_rows": with_rows}


def load_assignment(path: str) -> Dict[str, str]:
    """`{scene: anomaly_class}` off the sweep's own `assignment.tsv`.

    Section E's grading key, and it is the sweep's DECISION rather than a reconstruction
    of it: `anchor_yield --emit-assignment` wrote this file and `matrix_sweep.sh`'s cell
    loop read the same two columns back, so a scene graded here ran the class it is
    graded against. A malformed line raises rather than being skipped — a grading key
    that silently loses a scene grades that scene's episodes against nothing.
    """
    out: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            parts = text.split()
            if len(parts) != 2:
                raise ValueError(
                    "{}:{}: expected `scene<TAB>class`, got {!r}".format(
                        path, number, text
                    )
                )
            out[parts[0]] = parts[1]
    return out


def anchors_by_scene(assignment: Mapping[str, str]) -> Dict[str, Optional[str]]:
    """`{scene: the category a CORRECT recall names}` — each class's anchor object.

    `None` for a class that anchors nowhere (`glass_break`). That is not the same fact as
    an absent scene: an episode of such a class has no right answer to be graded against,
    so it is excluded from section E rather than counted as a wrong recall.
    """
    from earshot.task.prior_build import anchor_of_run_class

    return {scene: anchor_of_run_class(name) for scene, name in assignment.items()}


def graded_confidences(
    arm: Mapping[str, Mapping[int, Mapping[str, Any]]],
    anchors: Mapping[str, Optional[str]],
) -> Dict[str, Any]:
    """This arm's recall confidences, split into the ones that named the right category
    and the ones that did not. Both lists come back SORTED, for `abstain_table`. Pure.

    `ungraded` is the section's own blind spot, counted rather than dropped:
    `no_confidence` is every episode whose prior missed or was never consulted, and
    `unknown_anchor` is every episode of a class with no anchor at all.
    """
    correct: List[float] = []
    wrong: List[float] = []
    ungraded = {"no_confidence": 0, "unknown_anchor": 0}
    for scene, rows in arm.items():
        anchor = anchors.get(scene)
        for row in rows.values():
            if row.get("category") is None or row.get("confidence") is None:
                ungraded["no_confidence"] += 1
                continue
            if anchor is None:
                ungraded["unknown_anchor"] += 1
                continue
            bucket = correct if row["category"] == anchor else wrong
            bucket.append(float(row["confidence"]))
    return {"correct": sorted(correct), "wrong": sorted(wrong), "ungraded": ungraded}


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    """The q-quantile of an ALREADY SORTED sequence, by nearest rank. Pure."""
    if not values:
        return None
    index = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return float(values[index])


def abstain_table(
    correct: Sequence[float], wrong: Sequence[float]
) -> Dict[str, Any]:
    """Whether a confidence floor can drop wrong recalls without dropping right ones.

    Both sequences must already be sorted. A threshold abstains on every vote scoring
    strictly BELOW it, so:

    - `free` is the largest threshold that loses no correct recall at all — `min(correct)`
      — and what it drops is the number of wrong priors an abstain rule buys for nothing.
    - `best` maximises (wrong dropped rate) − (correct lost rate), which is Youden's J,
      the standard single-threshold summary. Ties go to the LOWER threshold, so the
      reported operating point is the least aggressive one that achieves it. The ranking
      is done on the INTEGER `dropped * n_correct - lost * n_wrong`, which J is that
      number over a constant: computing J in floating point first made `1 - 2/3` exceed
      `2/3 - 1/3` by one ulp and handed the tie to the most aggressive threshold instead
      of the least.
    - `separable` is the strong claim: every wrong recall scored below every right one.

    Pure. Returns `None` rows when either side is empty, because a rate over zero
    episodes is not a rate and printing 0.0 for one would read as a measured floor.
    """
    n_correct, n_wrong = len(correct), len(wrong)
    summary: Dict[str, Any] = {
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "correct_q": [_quantile(correct, q) for q in (0.0, 0.25, 0.5, 0.75, 1.0)],
        "wrong_q": [_quantile(wrong, q) for q in (0.0, 0.25, 0.5, 0.75, 1.0)],
        "free": None,
        "best": None,
        "separable": None,
    }
    if not n_correct or not n_wrong:
        return summary

    def _row(threshold: float) -> Dict[str, float]:
        dropped = sum(1 for value in wrong if value < threshold)
        lost = sum(1 for value in correct if value < threshold)
        return {
            "threshold": float(threshold),
            "wrong_dropped": dropped,
            "wrong_rate": dropped / n_wrong,
            "correct_lost": lost,
            "correct_rate": lost / n_correct,
            "j": dropped / n_wrong - lost / n_correct,
            # J's numerator over the common denominator `n_wrong * n_correct`. Integer,
            # so the ranking below is exact and the tie-break is the one documented.
            "rank": dropped * n_correct - lost * n_wrong,
        }

    candidates = sorted(set(correct) | set(wrong))
    summary["free"] = _row(correct[0])
    summary["best"] = max(
        (_row(value) for value in candidates),
        key=lambda row: (row["rank"], -row["threshold"]),
    )
    summary["separable"] = wrong[-1] < correct[0]
    return summary


def _fmt_quantiles(quantiles: Sequence[Optional[float]]) -> str:
    return "min/p25/median/p75/max " + "/".join(
        "-" if value is None else "{:.4f}".format(value) for value in quantiles
    )


def _why(row: Optional[Mapping[str, Any]]) -> str:
    """The recorded reason a scene did not tour, as a trailing clause. Pure.

    `SceneTourOutcome.as_dict`'s three informative fields, in the order that answers the
    question: how far the tour got, how much it stored, and the load error if there was
    one. An empty string when the store recorded nothing — a pre-`pass_provenance` store,
    where the absence IS the finding and a fabricated reason would hide it.
    """
    if not row:
        return ""
    parts: List[str] = []
    reached = row.get("rooms_reached")
    if reached is not None:
        # A LIST of room names, not a count -- which rooms the tour got to is the fact
        # that names the missing legs, and a count would throw it away.
        parts.append("rooms reached: {}".format(
            ", ".join(str(room) for room in reached) or "<none>"
        ))
    if row.get("n_observations") is not None:
        parts.append("{} observation(s) stored".format(row["n_observations"]))
    attempts = row.get("start_attempts")
    if attempts is not None and int(attempts) > 1:
        # Only when it retried: a 1 here is the default single draw and says nothing.
        parts.append("{} start(s) drawn".format(int(attempts)))
    dropped = row.get("unreachable")
    if dropped is not None:
        # The line that tells a 0-of-0 tour apart from a partial one: no candidates at all
        # (the assignment and the tour disagree) versus candidates the navmesh refused.
        parts.append(
            "no candidate stop offered" if not dropped
            else "{} candidate(s) unroutable ({})".format(
                len(dropped),
                ", ".join(sorted({str(item.get("room")) for item in dropped})),
            )
        )
    if row.get("error"):
        parts.append("error: {}".format(row["error"]))
    return " -- {}".format(", ".join(parts)) if parts else ""


def _print_coverage(coverage: Mapping[str, Any], say: Any) -> None:
    say("A. THE STORE")
    say("  assigned {} scene(s): complete {}, incomplete {}, failed {}".format(
        len(coverage["assigned"]),
        len(coverage["complete"]),
        len(coverage["incomplete"]),
        len(coverage["failed"]),
    ))
    if not coverage["records_incomplete"]:
        say("  NOTE: this store predates `scenes_incomplete` — a silently dropped scene")
        say("        shows only in the unaccounted list below")
    for name in ("incomplete", "failed"):
        for scene in coverage[name]:
            say("  {}: {}{}".format(
                name.upper(), scene, _why(coverage.get("detail", {}).get(scene)),
            ))
    for scene in coverage["unaccounted"]:
        say("  UNACCOUNTED (in no list — the D3 silent case): {}".format(scene))
    for scene in coverage["complete_without_rows"]:
        say("  DISAGREEMENT: {} is marked complete but holds 0 episodic rows".format(scene))
    say("  episodic rows by scene: {}".format(
        " ".join(
            "{}={}".format(scene, coverage["rows_by_scene"][scene])
            for scene in coverage["assigned"]
        ) or "<none>"
    ))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read a finished matrix sweep back: store coverage, seen-axis "
                    "liveness, not_heard prior distribution, and free determinism. "
                    "--gate-scenes turns the coverage half into a nonzero-exit gate."
    )
    parser.add_argument(
        "run_dir", nargs="?", default=None,
        help="the sweep's tag directory (runs/<tag>) holding <arm>/<scene>/",
    )
    parser.add_argument(
        "--arms", default=" ".join(DEFAULT_ARMS),
        help="space-separated arm directory names (default: the four cells)",
    )
    parser.add_argument(
        "--store", default=None,
        help="path to the prior pass's store.json (default: <run_dir>/prior/store.json)",
    )
    parser.add_argument(
        "--assignment", default=None,
        help="path to the sweep's assignment.tsv (default: <run_dir>/assignment.tsv). "
             "Section E's grading key: which class each scene actually ran",
    )
    parser.add_argument(
        "--gate-scenes", default=None,
        help="space-separated assigned scenes; exit 2 unless every one is in the "
             "store's scenes_complete. Needs --store (or run_dir). Prints nothing else.",
    )
    args = parser.parse_args(None if argv is None else list(argv))

    if args.run_dir is None and args.store is None:
        parser.error("pass a run_dir, or --store with --gate-scenes")
    store_path = args.store or str(pathlib.Path(args.run_dir) / "prior" / "store.json")

    from earshot.task.memory_build import load_stores

    _semantic, episodic, provenance = load_stores(store_path)
    rows_by_scene: Dict[str, int] = {}
    for entry in episodic.entries:
        rows_by_scene[entry.scene] = rows_by_scene.get(entry.scene, 0) + 1

    if args.gate_scenes is not None:
        coverage = store_coverage(
            provenance, rows_by_scene, assigned=args.gate_scenes.split()
        )
        missing = gate_missing(coverage)
        if missing:
            print("COVERAGE GATE RED: the prior pass did not complete {} of {} assigned "
                  "scene(s): {}".format(
                      len(missing), len(coverage["assigned"]), " ".join(missing)))
            print("A cell run over these scenes has seen == unseen by construction. "
                  "NOT_RUN is never green — fix the tour or drop the scenes from the "
                  "assignment before sweeping.")
            return 2
        print("coverage gate green: all {} assigned scene(s) complete in {}".format(
            len(coverage["assigned"]), store_path))
        return 0

    say = print
    coverage = store_coverage(provenance, rows_by_scene)
    _print_coverage(coverage, say)

    if args.run_dir is None:
        # `--store` on its own is a complete question -- "what did the prior pass
        # cover?" -- and is how a store is read BEFORE any cell has run over it. It used
        # to print section A and then die on `Path(None)`, which made the tool look
        # broken at exactly the moment it had just answered.
        say("")
        say("(no run_dir given, so sections B-E are not run: they read the sweep's own "
            "arm directories, which do not exist until the cells have)")
        return 0

    arms = {}
    for arm in args.arms.split():
        arm_dir = pathlib.Path(args.run_dir) / arm
        if not arm_dir.is_dir():
            say("  (arm {} has no directory — skipped)".format(arm))
            continue
        arms[arm] = load_memory_rows(str(arm_dir))

    say("")
    say("B. THE SEEN AXIS (a live episodic store shows as instances differing at a "
        "shared category)")
    for seen_name, unseen_name in SEEN_PAIRS:
        if seen_name not in arms or unseen_name not in arms:
            continue
        d = seen_axis_divergence(arms[seen_name], arms[unseen_name])
        say("  {} vs {}: {} verified pair(s), {} consulted in both".format(
            seen_name, unseen_name, d["pairs"], d["consulted_both"]))
        say("    same voted category {} — instances differ on {} (scenes: {}), "
            "distance differs on {}".format(
                d["same_category"],
                d["instances_differ_same_category"],
                " ".join(d["instances_differ_scenes"]) or "<none>",
                d["distance_differs_same_category"],
            ))
        say("    voted category differs on {} (render noise + semantics, "
            "not attributable to the seen axis)".format(d["category_differs"]))

    say("")
    say("C. WHAT EACH ARM'S PRIOR SAID (D2: not_heard should be a wrong CATEGORY, "
        "almost never no_prediction)")
    for arm in args.arms.split():
        if arm not in arms:
            continue
        dist = prior_distribution(arms[arm])
        say("  {}: {} episode(s) — categories {}; misses {}; never consulted {}".format(
            arm, dist["episodes"],
            dist["categories"] or "<none>",
            dist["misses"] or "<none>",
            dist["never_consulted"],
        ))

    say("")
    say("D. DETERMINISM FOR FREE (zero-row scenes ran seen == unseen byte-identical)")
    for seen_name, unseen_name in SEEN_PAIRS:
        if seen_name not in arms or unseen_name not in arms:
            continue
        split = discordance_where_identical(
            arms[seen_name], arms[unseen_name], rows_by_scene
        )
        zero, with_rows = split["zero_row_scenes"], split["scenes_with_rows"]
        say("  {} vs {}: identical-input scenes {} pair(s), {} discordant{}".format(
            seen_name, unseen_name, zero["pairs"], zero["discordant"],
            "" if not zero["scenes"] else " ({})".format(
                " ".join("{}={}".format(s, n) for s, n in sorted(zero["scenes"].items()))
            ),
        ))
        say("    (context, not a control: scenes with rows {} pair(s), {} "
            "discordant)".format(with_rows["pairs"], with_rows["discordant"]))
        if zero["pairs"] and zero["discordant"]:
            say("    -> the apparatus flips on identical inputs: {:.1f}% of these "
                "pairs".format(100.0 * zero["discordant"] / zero["pairs"]))
        elif zero["pairs"]:
            say("    -> no flips on identical inputs — consistent with a deterministic "
                "run path")

    say("")
    say("E. CAN THE VOTE ABSTAIN (a CORRECT recall's confidence against a WRONG one's)")
    assignment_path = args.assignment or str(
        pathlib.Path(args.run_dir) / "assignment.tsv"
    )
    if not pathlib.Path(assignment_path).is_file():
        say("  SKIPPED: no grading key at {}".format(assignment_path))
        say("  Section E needs the sweep's own assignment.tsv to know which category a")
        say("  correct recall would have named. Pass --assignment to point at it.")
        return 0

    anchors = anchors_by_scene(load_assignment(assignment_path))
    say("  graded on {}".format(assignment_path))
    pooled_correct: List[float] = []
    pooled_wrong: List[float] = []
    for arm in args.arms.split():
        if arm not in arms:
            continue
        graded = graded_confidences(arms[arm], anchors)
        pooled_correct.extend(graded["correct"])
        pooled_wrong.extend(graded["wrong"])
        say("  {}: {} correct, {} wrong, {} with no confidence recorded{}".format(
            arm,
            len(graded["correct"]),
            len(graded["wrong"]),
            graded["ungraded"]["no_confidence"],
            "" if not graded["ungraded"]["unknown_anchor"] else
            ", {} of a class that anchors nowhere".format(
                graded["ungraded"]["unknown_anchor"]
            ),
        ))

    pooled_correct.sort()
    pooled_wrong.sort()
    table = abstain_table(pooled_correct, pooled_wrong)
    say("  pooled CORRECT n={:<4} {}".format(
        table["n_correct"], _fmt_quantiles(table["correct_q"])))
    say("  pooled WRONG   n={:<4} {}".format(
        table["n_wrong"], _fmt_quantiles(table["wrong_q"])))
    if table["free"] is None:
        say("  -> one side is empty, so no floor is measurable from this sweep. A "
            "threshold needs both a right recall and a wrong one to sit between.")
        return 0

    free, best = table["free"], table["best"]
    say("  free floor {:.4f} (the largest that loses NO correct recall): drops {} of {} "
        "wrong ({:.1f}%)".format(
            free["threshold"], free["wrong_dropped"], table["n_wrong"],
            100.0 * free["wrong_rate"]))
    say("  best floor {:.4f} (max J): drops {}/{} wrong ({:.1f}%), loses {}/{} correct "
        "({:.1f}%), J={:.3f}".format(
            best["threshold"], best["wrong_dropped"], table["n_wrong"],
            100.0 * best["wrong_rate"], best["correct_lost"], table["n_correct"],
            100.0 * best["correct_rate"], best["j"]))
    if table["separable"]:
        say("  -> SEPARABLE: every wrong recall scored below every right one. An abstain "
            "floor is a real rule, and `not_heard` can be made an arm with no prior.")
    else:
        say("  -> the distributions OVERLAP: no single floor separates them. An abstain "
            "rule buys the free drop above and then trades right recalls for wrong ones, "
            "so a NONE arm stays the only clean control for a wrong prior.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
