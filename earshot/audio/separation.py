"""The CLAP separation gate's pure core: score rows in, a verdict and a pruned vocabulary out.

`CONTEXT.md` defines the **CLAP separation gate** as the measurement that must clear before
anything reads the **inferred goal class**. This module is the arithmetic half of it. The
rendering half is `task/clap_gate.py`, which needs a simulator; everything here is plain
Python over rows, so it unit-tests on a Mac and the numbers a box run prints are checkable
without one.

**Two arms, because a gate with one arm is what this map keeps finding.** ADR-0014 requires
the healthy path passing AND the induced failure firing, and `audio/clap.py`'s own history
is the case for it: the calibration that shipped `ANOMALY_GATE_DELTA` ran on offline-convolved
audio, and the one arc that exercised the gate live had it reject 0 of 8 -- indistinguishable
from a gate that discriminates nothing. So this reports:

- **closed-set top-1 accuracy**, per class and BANDED BY DISTANCE: can CLAP tell the candidate
  classes apart at all, once a real reverberant IR has been convolved through them.
- **open-set EER**, separating in-vocabulary clips from `vocabulary.ABSENT_CLASSES`: can CLAP
  tell "I know this one" from "I have never been told about this one". The absent classes are
  never in the prompt bank, so this arm cannot pass by accident.

**The distance banding is the point, not a garnish.** A gate that reports one accuracy over a
1-to-8 m band hides the only thing the number is needed for. If CLAP holds at 2 m and collapses
at 6 m, the design is not dead -- `AudioConfig.audible_band_m` moves and the sounding window
gets longer. A single scalar cannot say that.

**Chance is reported beside accuracy, always.** Twenty classes put chance at 0.05, and an
accuracy quoted without it is the shape of number this repo has twice mistaken for a result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "GateRow",
    "ClassResult",
    "AnchorResult",
    "AbsentResult",
    "BandResult",
    "RejectionResult",
    "SeparationReport",
    "top1_of",
    "true_margin_of",
    "decision_score_of",
    "equal_error_rate",
    "summarise",
    "anchor_top1_of",
    "prune",
    "restrict_to",
]


@dataclass(frozen=True)
class GateRow:
    """One rendered clip scored against the whole prompt bank.

    `true_class` is what was actually rendered. `in_vocabulary` says whether the prompt bank
    contained it: rows where it is False are the forced-failure arm, and their `true_class`
    is an `ABSENT_CLASSES` name that appears in no `scores` key by construction.

    `scores` is the cosine against every candidate prompt. `normal_cosine` is the best cosine
    against `clap.NORMAL_PROMPTS`, which is the reference the open-set decision is made
    against -- without it, "is this one of mine" has no scale, because CLAP cosines are not
    calibrated across prompts.
    """

    true_class: str
    in_vocabulary: bool
    distance_m: float
    scene: str
    recording_index: int
    scores: Mapping[str, float]
    normal_cosine: float

    def __post_init__(self) -> None:
        if not self.scores:
            raise ValueError(
                "row for {!r} carries no scores; a row scored against an empty prompt bank "
                "is not a measurement".format(self.true_class)
            )
        if self.in_vocabulary and self.true_class not in self.scores:
            raise ValueError(
                "row claims {!r} is in the vocabulary but the prompt bank did not score it; "
                "present: {}".format(self.true_class, ", ".join(sorted(self.scores)))
            )
        if not self.in_vocabulary and self.true_class in self.scores:
            raise ValueError(
                "row claims {!r} is ABSENT but it was scored -- the forced-failure arm is "
                "vacuous if the absent class is in the prompt bank".format(self.true_class)
            )
        if not math.isfinite(float(self.distance_m)):
            raise ValueError(
                "row for {!r} has a non-finite distance; an unroutable pose must be dropped "
                "before it reaches the report, never defaulted to zero".format(self.true_class)
            )


@dataclass(frozen=True)
class ClassResult:
    """One candidate class's closed-set behaviour over every row that rendered it.

    `recall` is class-level: did CLAP name the class. `anchor_recall` is the same rows scored
    at the level the task pays at: did the winning class point at the right ANCHOR. The two
    diverge wherever a class's misses land on its own siblings, and the divergence is large.
    `clapgate-1` measured `pouring_water` at 0.354 class and 1.000 anchor, every miss inside
    its own bathroom.

    `anchor_recall` is None when `summarise` was given no anchor map. NOT 0.0: an unsupplied
    map and a class that never once found its anchor are different facts.
    """

    name: str
    affinity: str
    n: int
    recall: float
    mean_true_margin: float
    top_confusion: Optional[Tuple[str, int]]
    anchor_recall: Optional[float] = None


@dataclass(frozen=True)
class AnchorResult:
    """One anchor OBJECT's accuracy: did the inferred class point at the right thing.

    This is the number the task cares about and `ClassResult.recall` is not. The agent
    navigates to an object, so a class confused for a SIBLING of the same anchor costs it
    nothing: `snoring` heard as `breathing` still sends it to the bed. The clapsmoke-3 gate
    made that concrete -- all 60 of `snoring`'s misses landed on `breathing`, and all 53 of
    `clock_tick`'s landed on `clock_alarm`, so a class recall of 0.500 and 0.558 understated
    what the agent would actually have done by a wide margin.
    """

    anchor: str
    n: int
    accuracy: float
    n_classes: int
    top_confusion: Optional[Tuple[str, int]]


@dataclass(frozen=True)
class BandResult:
    """Closed-set accuracy inside one distance band. The curve, not the scalar."""

    near_m: float
    far_m: float
    n: int
    top1_accuracy: float
    mean_true_margin: float


@dataclass(frozen=True)
class AbsentResult:
    """One absent class's own rejection rate, so a bad EER can be attributed.

    An aggregate EER cannot tell "the open-set rule discriminates nothing" from "some of the
    negatives are near-duplicates of an in-vocabulary class". `clapgate-2` settled that
    question and REFUTED the guess that came with it: the EER moved 0.232 to 0.318 in the same
    commit that promoted `rain`, `crickets` and `chirping_birds` into the absent set, so those
    three looked like the cause. They are not. `rain` rejects at 0.786 and `crickets` at
    0.815, both near the top. The hardest negative is `chainsaw` at 0.351, which had been in
    the absent set from the start.

    `top_match` is why. It names the candidate class an absent clip most often looks like, and
    a chainsaw looks like a `vacuum_cleaner`: continuous motor noise against continuous motor
    noise. The open-set failures are TWINNED rather than diffuse, which is a different problem
    with a different fix.

    `mean_decision_score` is on the same scale as the threshold, so the two are comparable
    by eye.
    """

    name: str
    n: int
    rejection_rate: float
    mean_decision_score: float
    top_match: Optional[Tuple[str, int]] = None


@dataclass(frozen=True)
class RejectionResult:
    """The forced-failure arm: in-vocabulary accepted, absent rejected, and the EER between.

    `eer` is threshold-free and is the number to quote. `rejection_rate` and
    `false_rejection_rate` are that same separation read at one chosen threshold, and they are
    here because a rate is what a reader checks against "0 of 8".

    `per_absent` breaks the rejection rate down by negative, because an aggregate that a few
    hard negatives dominate is a statement about the negatives and not about the gate.
    """

    n_in_vocabulary: int
    n_absent: int
    eer: float
    threshold_at_eer: float
    rejection_rate: float
    false_rejection_rate: float
    per_absent: Tuple[AbsentResult, ...] = ()


@dataclass(frozen=True)
class SeparationReport:
    """Everything the gate measured. `passes` is deliberately not a field -- see `prune`."""

    n_rows: int
    n_classes: int
    top1_accuracy: float
    chance_accuracy: float
    mean_true_margin: float
    per_class: Tuple[ClassResult, ...]
    per_band: Tuple[BandResult, ...]
    confusion: Mapping[Tuple[str, str], int]
    rejection: RejectionResult
    # Empty when no anchor map was supplied. NOT zero: an unsupplied anchor map and an
    # anchor accuracy of 0.0 are different facts and must not read the same.
    anchor_top1_accuracy: Optional[float] = None
    per_anchor: Tuple[AnchorResult, ...] = ()

    def as_dict(self) -> Dict[str, object]:
        """The artefact form. Confusion keys are joined because JSON has no tuple key."""
        return {
            "n_rows": self.n_rows,
            "n_classes": self.n_classes,
            "top1_accuracy": self.top1_accuracy,
            "chance_accuracy": self.chance_accuracy,
            "mean_true_margin": self.mean_true_margin,
            "per_class": [
                {
                    "name": item.name,
                    "affinity": item.affinity,
                    "n": item.n,
                    "recall": item.recall,
                    "mean_true_margin": item.mean_true_margin,
                    "top_confusion": list(item.top_confusion) if item.top_confusion else None,
                    "anchor_recall": item.anchor_recall,
                }
                for item in self.per_class
            ],
            "per_band": [
                {
                    "near_m": item.near_m,
                    "far_m": item.far_m,
                    "n": item.n,
                    "top1_accuracy": item.top1_accuracy,
                    "mean_true_margin": item.mean_true_margin,
                }
                for item in self.per_band
            ],
            "confusion": {
                "{}->{}".format(true, predicted): count
                for (true, predicted), count in sorted(self.confusion.items())
            },
            "anchor_top1_accuracy": self.anchor_top1_accuracy,
            "per_anchor": [
                {
                    "anchor": item.anchor,
                    "n": item.n,
                    "accuracy": item.accuracy,
                    "n_classes": item.n_classes,
                    "top_confusion": list(item.top_confusion) if item.top_confusion else None,
                }
                for item in self.per_anchor
            ],
            "rejection": {
                "n_in_vocabulary": self.rejection.n_in_vocabulary,
                "n_absent": self.rejection.n_absent,
                "eer": self.rejection.eer,
                "threshold_at_eer": self.rejection.threshold_at_eer,
                "rejection_rate": self.rejection.rejection_rate,
                "false_rejection_rate": self.rejection.false_rejection_rate,
                "per_absent": [
                    {
                        "name": item.name,
                        "n": item.n,
                        "rejection_rate": item.rejection_rate,
                        "mean_decision_score": item.mean_decision_score,
                        "top_match": list(item.top_match) if item.top_match else None,
                    }
                    for item in self.rejection.per_absent
                ],
            },
        }


def top1_of(row: GateRow) -> str:
    """The argmax class over the prompt bank. Ties break on the name, so it is deterministic."""
    return min(row.scores, key=lambda name: (-float(row.scores[name]), name))


def true_margin_of(row: GateRow) -> float:
    """`score[true] - max(score[every other class])`, the closed-set headroom.

    Positive means the true class won. The size says by how much, which is what separates a
    gate that is working from one that is one recording away from not working. Raises on an
    absent row: an absent class has no true score, and returning 0.0 would put a
    forced-failure row into the closed-set mean as a neutral value.
    """
    if not row.in_vocabulary:
        raise ValueError(
            "{!r} is an absent class -- it has no true-class score, so a closed-set margin is "
            "undefined for it".format(row.true_class)
        )
    others = [float(value) for name, value in row.scores.items() if name != row.true_class]
    if not others:
        raise ValueError(
            "prompt bank holds only {!r}; a margin against nothing is not a measurement".format(
                row.true_class
            )
        )
    return float(row.scores[row.true_class]) - max(others)


def decision_score_of(row: GateRow) -> float:
    """The open-set score: best candidate cosine minus the best `NORMAL_PROMPTS` cosine.

    High means "this looks more like one of my classes than like a generic room noise". The
    normal bank is the scale: raw CLAP cosines are not comparable across prompts, so an
    absolute floor alone reproduces the failure `audio/clap.py` documents, where the clean-clip
    threshold rejected the convolved alarm outright.
    """
    best = max(float(value) for value in row.scores.values())
    return best - float(row.normal_cosine)


def equal_error_rate(
    positives: Sequence[float], negatives: Sequence[float]
) -> Tuple[float, float]:
    """`(eer, threshold)` separating `positives` (accept) from `negatives` (reject).

    Threshold-free by construction: it sweeps every observed score as a candidate threshold and
    returns the point where the false-accept and false-reject rates are closest. This is ticket
    13's pattern, and it is used here rather than a fixed tau because the gate's job is to
    MEASURE the separation, not to apply a constant carried from a different audio domain.

    An EER of 0.5 means the two distributions are on top of each other, which is what a gate
    that discriminates nothing looks like. An EER near 0.0 means they are cleanly apart.
    """
    if not positives or not negatives:
        raise ValueError(
            "EER needs both arms: got {} positives and {} negatives. A gate measured on one "
            "arm is exactly the vacuous gate this exists to rule out.".format(
                len(positives), len(negatives)
            )
        )
    candidates = sorted({float(value) for value in list(positives) + list(negatives)})
    best: Optional[Tuple[float, float, float]] = None
    for threshold in candidates:
        # Accept when score >= threshold.
        false_reject = sum(1 for value in positives if float(value) < threshold) / len(positives)
        false_accept = sum(1 for value in negatives if float(value) >= threshold) / len(negatives)
        gap = abs(false_reject - false_accept)
        midpoint = (false_reject + false_accept) / 2.0
        if best is None or gap < best[0]:
            best = (gap, midpoint, threshold)
    assert best is not None
    return best[1], best[2]


def anchor_top1_of(row: GateRow, anchors: Mapping[str, str]) -> Tuple[str, str]:
    """`(true_anchor, predicted_anchor)` for one in-vocabulary row.

    Raises on a class the map does not carry rather than defaulting it to its own name: a
    silently self-anchored class scores a free hit and inflates the one number the design
    decisions rest on.
    """
    if not row.in_vocabulary:
        raise ValueError(
            "{!r} is an absent class and has no true anchor".format(row.true_class)
        )
    predicted = top1_of(row)
    for name in (row.true_class, predicted):
        if name not in anchors:
            raise KeyError(
                "no anchor for class {!r}; the anchor map must cover every class in the "
                "prompt bank or the accuracy is computed over a subset that nothing "
                "declares".format(name)
            )
    return anchors[row.true_class], anchors[predicted]


def _band_edges(rows: Sequence[GateRow], n_bands: int) -> List[Tuple[float, float]]:
    """Log-spaced edges spanning the observed distances, matching `calibration.band_poses`.

    Log rather than linear for the reason `band_poses` gives: level falls roughly with the log
    of distance, so linear bands crowd the near field into one bucket.
    """
    distances = sorted(float(row.distance_m) for row in rows if float(row.distance_m) > 0.0)
    if not distances:
        raise ValueError(
            "no row carries a positive distance; the band curve is the gate's main output and "
            "cannot be computed from distances that were never recorded"
        )
    near, far = distances[0], distances[-1]
    if not (near < far):
        return [(near, far)]
    count = max(1, int(n_bands))
    step = (math.log(far) - math.log(near)) / count
    edges = [
        (math.exp(math.log(near) + step * i), math.exp(math.log(near) + step * (i + 1)))
        for i in range(count)
    ]
    # The last upper edge is pinned to the OBSERVED maximum rather than recomputed. exp(log)
    # round-trips to 7.499999999999999 for an observed 7.5, and the farthest row then fails
    # the `== far` test and vanishes from the curve -- three of twelve rows silently absent,
    # which reads as a band that was never sampled rather than as arithmetic drift.
    edges[-1] = (edges[-1][0], far)
    return edges


def summarise(
    rows: Iterable[GateRow],
    *,
    affinities: Optional[Mapping[str, str]] = None,
    anchors: Optional[Mapping[str, str]] = None,
    n_bands: int = 4,
) -> SeparationReport:
    """Fold gate rows into the report. Raises rather than reporting an unmeasurable arm.

    `affinities` maps class name to its declared grade, so the per-class table can be read
    strong-versus-weak without this module importing the vocabulary and inheriting its table.

    `anchors` maps class name to its anchor OBJECT and is optional only because a caller may
    genuinely not have one. Supply it whenever you can: anchor accuracy is the number the task
    rests on, and class recall systematically understates it wherever sibling classes share an
    anchor.
    """
    materialised = list(rows)
    if not materialised:
        raise ValueError("no rows to summarise; an empty gate run is NOT_RUN, which is red")

    known = [row for row in materialised if row.in_vocabulary]
    absent = [row for row in materialised if not row.in_vocabulary]
    if not known:
        raise ValueError("every row is an absent class -- there is no healthy arm to pass")
    if not absent:
        raise ValueError(
            "no row carries an absent class, so the forced-failure arm did not run. ADR-0014 "
            "requires both arms and CLAUDE.md makes a criterion that could not be evaluated "
            "red, so this is a failure rather than a partial result."
        )

    grades = dict(affinities or {})
    class_names = sorted({row.true_class for row in known})
    # Folded once, up front, so the per-class loop reads it rather than every caller
    # recomputing an anchor recall of its own and the two drifting apart. `anchor_report`
    # did exactly that before this field existed.
    anchor_hits: Dict[str, int] = {}
    if anchors is not None:
        for row in known:
            true_anchor, predicted_anchor = anchor_top1_of(row, anchors)
            if true_anchor == predicted_anchor:
                anchor_hits[row.true_class] = anchor_hits.get(row.true_class, 0) + 1
    confusion: Dict[Tuple[str, str], int] = {}
    for row in known:
        key = (row.true_class, top1_of(row))
        confusion[key] = confusion.get(key, 0) + 1

    per_class: List[ClassResult] = []
    for name in class_names:
        subset = [row for row in known if row.true_class == name]
        hits = sum(1 for row in subset if top1_of(row) == name)
        margins = [true_margin_of(row) for row in subset]
        wrong = {
            predicted: count
            for (true, predicted), count in confusion.items()
            if true == name and predicted != name
        }
        top_confusion = (
            max(sorted(wrong.items()), key=lambda item: item[1]) if wrong else None
        )
        per_class.append(
            ClassResult(
                name=name,
                affinity=grades.get(name, "unknown"),
                n=len(subset),
                recall=hits / len(subset),
                mean_true_margin=sum(margins) / len(margins),
                top_confusion=top_confusion,
                anchor_recall=(
                    None if anchors is None else anchor_hits.get(name, 0) / len(subset)
                ),
            )
        )

    per_band: List[BandResult] = []
    edges = _band_edges(known, n_bands)
    for index, (near, far) in enumerate(edges):
        # Half-open bands, closed at the top edge only for the LAST one, so the farthest row
        # lands in a band instead of falling off the end and no row is counted twice.
        is_last = index == len(edges) - 1
        subset = [
            row
            for row in known
            if near <= float(row.distance_m) < far
            or (is_last and near <= float(row.distance_m) <= far)
        ]
        if not subset:
            continue
        hits = sum(1 for row in subset if top1_of(row) == row.true_class)
        margins = [true_margin_of(row) for row in subset]
        per_band.append(
            BandResult(
                near_m=near,
                far_m=far,
                n=len(subset),
                top1_accuracy=hits / len(subset),
                mean_true_margin=sum(margins) / len(margins),
            )
        )

    positives = [decision_score_of(row) for row in known]
    negatives = [decision_score_of(row) for row in absent]
    eer, threshold = equal_error_rate(positives, negatives)
    per_absent: List[AbsentResult] = []
    for name in sorted({row.true_class for row in absent}):
        subset = [row for row in absent if row.true_class == name]
        scores = [decision_score_of(row) for row in subset]
        matches: Dict[str, int] = {}
        for row in subset:
            winner = top1_of(row)
            matches[winner] = matches.get(winner, 0) + 1
        per_absent.append(
            AbsentResult(
                name=name,
                n=len(scores),
                rejection_rate=sum(1 for value in scores if value < threshold) / len(scores),
                mean_decision_score=sum(scores) / len(scores),
                top_match=max(sorted(matches.items()), key=lambda item: item[1]),
            )
        )
    rejection = RejectionResult(
        n_in_vocabulary=len(known),
        n_absent=len(absent),
        eer=eer,
        threshold_at_eer=threshold,
        rejection_rate=sum(1 for value in negatives if value < threshold) / len(negatives),
        false_rejection_rate=sum(1 for value in positives if value < threshold) / len(positives),
        per_absent=tuple(per_absent),
    )

    per_anchor: List[AnchorResult] = []
    anchor_accuracy: Optional[float] = None
    if anchors is not None:
        pairs = [anchor_top1_of(row, anchors) for row in known]
        anchor_accuracy = sum(1 for true, predicted in pairs if true == predicted) / len(pairs)
        for anchor in sorted({true for true, _predicted in pairs}):
            subset = [pair for pair in pairs if pair[0] == anchor]
            wrong: Dict[str, int] = {}
            for _true, predicted in subset:
                if predicted != anchor:
                    wrong[predicted] = wrong.get(predicted, 0) + 1
            per_anchor.append(
                AnchorResult(
                    anchor=anchor,
                    n=len(subset),
                    accuracy=sum(1 for _t, p in subset if p == anchor) / len(subset),
                    n_classes=len(
                        {row.true_class for row in known if anchors[row.true_class] == anchor}
                    ),
                    top_confusion=(
                        max(sorted(wrong.items()), key=lambda item: item[1]) if wrong else None
                    ),
                )
            )

    total_hits = sum(1 for row in known if top1_of(row) == row.true_class)
    all_margins = [true_margin_of(row) for row in known]
    n_classes = len(next(iter(known)).scores)
    return SeparationReport(
        n_rows=len(known),
        n_classes=n_classes,
        top1_accuracy=total_hits / len(known),
        chance_accuracy=1.0 / n_classes,
        mean_true_margin=sum(all_margins) / len(all_margins),
        per_class=tuple(per_class),
        per_band=tuple(per_band),
        confusion=confusion,
        rejection=rejection,
        anchor_top1_accuracy=anchor_accuracy,
        per_anchor=tuple(per_anchor),
    )


def restrict_to(rows: Iterable[GateRow], names: Sequence[str]) -> List[GateRow]:
    """The same rows re-scored against a SMALLER prompt bank. Pure; nothing is re-rendered.

    The gate scores every clip against the whole candidate bank, but the system ships the
    PRUNED bank. Those are different measurements and the difference is not a detail: on
    `clapgate-2` the cut removes `vacuum_cleaner`, which is what `chainsaw` was being mistaken
    for, so the hardest negative loses its twin. An accuracy quoted from the candidate bank
    describes a configuration nothing will run.

    In-vocabulary rows whose class was cut are DROPPED, because a bank that does not carry a
    class cannot be asked about it. Absent rows are all kept: the forced-failure arm is the
    same question against a smaller bank.

    **This is selection on the outcome.** The bank was chosen using these rows, so a recall
    measured on them afterwards is optimistically biased. The unbiased number needs held-out
    recordings, which ESC-50 has 40 of per class against the 8 a run stages. Report the
    restricted numbers as a direction, never as the gate's result.
    """
    keep = list(dict.fromkeys(names))
    if not keep:
        raise ValueError("cannot restrict to an empty bank; there would be nothing to score")
    kept_set = set(keep)
    out: List[GateRow] = []
    for row in rows:
        missing = kept_set - set(row.scores)
        if missing:
            raise KeyError(
                "row for {!r} was never scored against {}; a bank cannot be restricted to a "
                "class the run did not measure".format(row.true_class, ", ".join(sorted(missing)))
            )
        if row.in_vocabulary and row.true_class not in kept_set:
            continue
        out.append(
            GateRow(
                true_class=row.true_class,
                in_vocabulary=row.in_vocabulary,
                distance_m=row.distance_m,
                scene=row.scene,
                recording_index=row.recording_index,
                scores={name: float(row.scores[name]) for name in keep},
                normal_cosine=row.normal_cosine,
            )
        )
    return out


def prune(
    report: SeparationReport,
    *,
    min_recall: float,
    recall_level: str,
    min_n: int = 8,
    allowed_affinities: Optional[Sequence[str]] = None,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """`(kept, cut)` class names. THREE cuts, and they are independent.

    A class with fewer than `min_n` rows is CUT rather than kept on a small-sample recall, and
    the caller is expected to print the count: a class dropped for want of data and a class
    dropped for want of separation are different findings, and merging them is how a sweep
    reports coverage it did not have.

    `allowed_affinities` applies ADR-0018's OTHER requirement, which the first gate run
    ignored: a weak-affinity class is disqualified whatever its recall, because the semantic
    store cannot learn an association that is not there. `coughing` scored a perfect 1.000 in
    clapsmoke-3 and is still disqualified -- people cough in every room. Leave it `None` to
    skip the affinity cut, and say so when you report the result.

    `recall_level` says WHICH recall the separation cut reads, and it has no default because
    the two bars disagree and choosing between them is a decision:

    - `"anchor"` is what ADR-0018 takes, on `clapgate-1`. The agent navigates to a ROOM, so a
      class confused for a sibling of the same room costs it nothing. Cutting on class recall
      discarded `pouring_water` at 0.354 despite an anchor recall of 1.000, which prices a
      cost the task never pays and took a quarter of the bathroom vocabulary with it.
    - `"class"` is the stricter bar, kept scoreable so the looser one can be checked against
      it. Take it if the claim being defended includes the agent NAMING the sound.

    Asking for `"anchor"` on a report summarised without an anchor map raises. A recall the
    report could not measure must not read as a pass: CLAUDE.md makes NOT_RUN red.

    There is no default `min_recall` either. The bar is a decision, it belongs in the run that
    states it, and a default here would quietly become the bar everywhere.
    """
    if recall_level not in ("class", "anchor"):
        raise ValueError(
            "recall_level must be 'class' or 'anchor', got {!r}. There is no default: the two "
            "bars disagree by design, so a run has to say which one it cut on.".format(
                recall_level
            )
        )
    allowed = None if allowed_affinities is None else set(allowed_affinities)
    kept: List[str] = []
    cut: List[str] = []
    for item in report.per_class:
        if recall_level == "class":
            measured = item.recall
        elif item.anchor_recall is None:
            raise ValueError(
                "class {!r} carries no anchor recall, so this report was summarised without "
                "an anchor map and cannot be pruned at the anchor level".format(item.name)
            )
        else:
            measured = item.anchor_recall
        too_few = item.n < int(min_n)
        too_confused = measured < float(min_recall)
        wrong_affinity = allowed is not None and item.affinity not in allowed
        if too_few or too_confused or wrong_affinity:
            cut.append(item.name)
        else:
            kept.append(item.name)
    return tuple(kept), tuple(cut)
