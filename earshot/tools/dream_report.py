"""What a DREAM arm's memory actually did, read off the sweep's own episode audits.

**`dream-1` RAN AND ITS CENTRAL QUANTITY WAS NEVER PRINTED.** The sweep wrote
``dream_omega_e_spread``, ``dream_me_cosine_*`` and ``dream_rows_added`` onto all 282
episodes, and the two readers that exist — ``window_report`` and ``episode_diff`` — read
Find-SR and the paired outcome. Both are correct and neither can see a single number the
DREAM instrumentation was added for. This module is that reader.

It answers four questions, in the order that decides what is worth running next:

**A. DID `omega_t` MOVE.** Eq. 23 is the paper's mechanism: a query that reweights the
three memory levels step by step. The box measured ``omega^E`` at 0.4775-0.5055 over one
24-step walk, which is uniform to three decimals, and predicted nothing else could happen
while ``M^E``'s keys sat at 0.909-0.989 cosine of each other. A mean of 0.5 is what a live
omega and a dead one both report, so the SPREAD is the statistic and the mean is context.

**B. WAS THE PRECONDITION EVER THERE.** ``dream_me_cosine_*`` is how much of the key space
``M^E`` uses, measured on the memory as it stood at the START of each episode. Reported
against how many rows that memory held, because "the keys are degenerate" and "the memory
had two rows" are different findings and a pooled median cannot separate them.

**C. DID `eta` RETAIN ANYTHING, AND WAS IT eta THAT DID.** ``dream_rows_added`` per
episode. An arm where it is always 0 consolidated nothing all night and every other
number is about an empty memory. An arm where it never falls is a threshold that is not
thresholding. ``dream_rows_added`` CANNOT SAY WHICH RULE RETAINED, though: eta passing
twelve segments and a cap of twelve truncating forty write the same twelve rows. So
``dream_segments_over_eta`` — the count before ``max_retained`` truncates — is read
beside it, and the section says which of the two selected the memory. That is ADR-0024's
open question, and `eta_pass.sh` exists to answer it.

**D. HOW FAR THE MEMORY EVER REACHED.** The sweep runs one ``python -m earshot`` per
scene, so ``run()``'s carry-over is within a scene and every scene starts from empty. That
is measured here rather than asserted: the count of episodes that ran against an empty
``M^E`` and the largest ``M^E`` any episode ever saw are what say whether this run could
have tested a memory built across scenes at all.

ABSENT IS NEVER ZERO, the rule this tree has paid for twice. An episode missing
``dream_informed_steps`` did not record the field; an episode carrying 0 recorded that
omega was never defined. They are counted apart everywhere below, and a mean is printed
only over the episodes that had the number, beside the count of the ones that did not.

The episode enumeration is ``smoke.episode_indices`` and is not re-derived here.
``pilot-1`` spent 42 minutes of V100 time on a second implementation of that glob (it
looked for ``audit.json``; the writer produces ``ep0000.audit.json``) and reported three
dead arms over 120 episodes that were on disk the whole time. A reader that finds nothing
prints exactly what an empty run prints, which is why the naming lives in one function.

Read-only, no GPU, seconds.

    python -m earshot.tools.dream_report runs/dream-1
    python -m earshot.tools.dream_report runs/dream-1 --arm dream
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from earshot.report.artifacts import episode_paths, read_audit, run_paths
from earshot.report.audit import EpisodeAudit, FunnelStage
from earshot.task.smoke import episode_indices
from earshot.tools.window_report import scene_dirs

__all__ = [
    "FLAT_OMEGA_SPREAD",
    "KNOB_KEYS",
    "ROW_BANDS",
    "EpisodeRow",
    "read_rows",
    "scene_dirs_or_flat",
    "band_of",
    "present",
    "format_omega",
    "format_precondition",
    "format_retention",
    "format_reach",
    "format_cost",
    "format_knobs",
    "format_report",
    "main",
]

# The floor this module calls flat, stated once so the verdict is not a judgement made in
# prose. `omega^E` is a softmax share over three levels: it starts at 1/3 when all three
# resolve and at 1/2 when two do, and the spread is how far it travelled across a whole
# episode. Below 0.05 the three levels were fixed weights for 250 steps, which is eq. 24
# with constants in it and not eq. 23.
#
# It is NOT an assertion. A flat omega is a finding about the design, and a tool that
# exited nonzero on it would be reporting a run failure that did not happen.
FLAT_OMEGA_SPREAD = 0.05

# The knob names `DreamKnobs.as_metrics` writes. Held as a literal rather than derived by
# importing `task.dream`, because this module must stay readable on a machine with no
# torch: `tools/` readers are run after a sweep, often on a laptop, and a numpy-and-torch
# import chain to enumerate eighteen strings is a cost with no buyer.
#
# `test_dream_report.py` builds a REAL `DreamKnobs` and asserts this tuple equals its
# `as_metrics` names exactly, so the literal cannot drift from the writer. A knob that
# reaches every audit and no reader is a knob whose value is unrecoverable from the run.
KNOB_KEYS: Tuple[str, ...] = (
    "dream_stm_horizon",
    "dream_stm_decay",
    "dream_present_weight",
    "dream_coherence",
    "dream_min_segment",
    "dream_max_segment",
    "dream_alpha",
    "dream_beta",
    "dream_gamma",
    "dream_eta",
    "dream_max_retained",
    "dream_min_support",
    "dream_k_experience",
    "dream_k_pattern",
    "dream_k_knowledge",
    "dream_temperature",
    "dream_lambda_plan",
    "dream_lambda_memory",
    "dream_lambda_feasibility",
)

# How many rows `M^E` held when the episode started. Bands rather than a scatter because
# the question is whether degeneracy FALLS as the memory fills, and 282 raw pairs do not
# answer that by eye. The first band is its own: an episode against an empty memory has no
# key spread at all and must never be averaged in with one that has.
ROW_BANDS: Tuple[Tuple[int, Optional[int], str], ...] = (
    (0, 0, "0 (empty)"),
    (1, 1, "1"),
    (2, 4, "2-4"),
    (5, 9, "5-9"),
    (10, 19, "10-19"),
    (20, 49, "20-49"),
    (50, None, "50+"),
)


@dataclass(frozen=True)
class EpisodeRow:
    """One episode's DREAM record. Pure data, every field ``Optional``.

    ``None`` means the audit did not carry the key. Every consumer below counts those
    rather than defaulting them, because a DREAM arm read with the wrong arm name and a
    DREAM arm whose memory did nothing produce the same zeros otherwise.
    """

    scene: str
    index: int
    reached: bool
    informed_steps: Optional[float]
    omega_e_spread: Optional[float]
    omega_e_mean: Optional[float]
    omega_e_min: Optional[float]
    omega_e_max: Optional[float]
    omega_p_mean: Optional[float]
    omega_k_mean: Optional[float]
    me_cosine_min: Optional[float]
    me_cosine_mean: Optional[float]
    me_cosine_max: Optional[float]
    experience_rows: Optional[float]
    pattern_rows: Optional[float]
    rows_added: Optional[float]
    # How many segments cleared eta BEFORE `max_retained` truncated. The pair
    # (this, `rows_added`) is what separates "eta is the retention rule" from
    # "the cap is, and eta is decoration" — ADR-0024's open question.
    segments_over_eta: Optional[float]
    tau_steps: Optional[float]
    importance_min: Optional[float]
    importance_max: Optional[float]
    step_s_mean: Optional[float]
    step_s_worst: Optional[float]
    # Only the `KNOB_KEYS` this episode actually carried. A knob absent from the mapping
    # was not recorded, which is different from a knob recorded as 0.
    knobs: Mapping[str, float]

    @property
    def rows_at_start(self) -> Optional[float]:
        """How many ``M^E`` rows this episode's retrievals were made against.

        The runner writes the count AFTER consolidation and the delta beside it, so the
        memory the episode actually used is the difference. ``None`` if either half is
        missing: inferring one from the other would put a number on a record that does
        not hold one.
        """
        if self.experience_rows is None or self.rows_added is None:
            return None
        return self.experience_rows - self.rows_added


def _metric(audit: EpisodeAudit, key: str) -> Optional[float]:
    value = audit.metrics.get(key)
    return None if value is None else float(value)


def scene_dirs_or_flat(arm_dir: str) -> Tuple[pathlib.Path, ...]:
    """The arm's scene directories, or the directory ITSELF when it holds one run.

    A sweep writes ``<tag>/<arm>/<scene>/episodes/``, which is what ``scene_dirs``
    finds. One invocation of the runner writes ``<run-dir>/episodes/`` and NO scene
    directory at all — the shape ``eta_pass.sh`` produces, and the shape CLAUDE.md's
    one-episode command has always produced.

    ``eta-1`` ran 15 episodes, wrote them in that shape, and this reader printed "NO
    EPISODES ON DISK" and exited 2 over data that was on disk the whole time. That is
    ``pilot-1``'s failure from the other side, and the same rule answers it: a reader
    that finds nothing must be wrong about the layout before the run is called empty.
    """
    scenes = scene_dirs(arm_dir)
    if scenes:
        return scenes
    root = pathlib.Path(arm_dir)
    return (root,) if run_paths(root)[1].is_dir() else ()


def read_rows(arm_dir: str) -> Tuple[EpisodeRow, ...]:
    """Every episode under one arm directory, in scene then index order.

    The scene name comes from the directory, matching ``window_report``: a scene that
    built no episodes still has a directory, and the sweep's shape has to survive the
    reading. A FLAT run directory is named after the tag rather than the scene, so there
    — and only there — the name comes off the audit instead.
    """
    rows: List[EpisodeRow] = []
    root = pathlib.Path(arm_dir)
    for scene in scene_dirs_or_flat(arm_dir):
        flat = scene == root
        # `window_report.load_arm_audits` pools an arm's audits and drops which scene
        # each came from, and section D is entirely about the per-scene shape. So the
        # three pieces it is built from are called here instead — `episode_indices` is
        # still the one function in this tree that knows what an episode file is named,
        # which is the part pilot-1's heredoc got wrong and the part not re-derived.
        for index in episode_indices(str(scene)):
            _agent_path, audit_path = episode_paths(str(scene), index)
            audit = read_audit(audit_path)
            rows.append(
                EpisodeRow(
                    scene=(
                        str(audit.scene_id or scene.name) if flat else scene.name
                    ),
                    index=int(audit.episode_index),
                    # `>= SOURCE_REACHED`, the SAME definition `window_report`
                    # uses, quoted from its comment: two definitions of
                    # "reached" in one repo is how a reader comes to quote the
                    # wrong one. `test_dream_report.py` asserts the two agree.
                    reached=audit.funnel_stage >= FunnelStage.SOURCE_REACHED,
                    informed_steps=_metric(audit, "dream_informed_steps"),
                    omega_e_spread=_metric(audit, "dream_omega_e_spread"),
                    omega_e_mean=_metric(audit, "dream_omega_e_mean"),
                    omega_e_min=_metric(audit, "dream_omega_e_min"),
                    omega_e_max=_metric(audit, "dream_omega_e_max"),
                    omega_p_mean=_metric(audit, "dream_omega_p_mean"),
                    omega_k_mean=_metric(audit, "dream_omega_k_mean"),
                    me_cosine_min=_metric(audit, "dream_me_cosine_min"),
                    me_cosine_mean=_metric(audit, "dream_me_cosine_mean"),
                    me_cosine_max=_metric(audit, "dream_me_cosine_max"),
                    experience_rows=_metric(audit, "dream_experience_rows"),
                    pattern_rows=_metric(audit, "dream_pattern_rows"),
                    rows_added=_metric(audit, "dream_rows_added"),
                    segments_over_eta=_metric(
                        audit, "dream_segments_over_eta"),
                    tau_steps=_metric(audit, "dream_tau_steps"),
                    importance_min=_metric(audit, "dream_importance_min"),
                    importance_max=_metric(audit, "dream_importance_max"),
                    step_s_mean=_metric(audit, "dream_step_s_mean"),
                    step_s_worst=_metric(audit, "dream_step_s_worst"),
                    knobs={
                        key: value
                        for key, value in (
                            (name, _metric(audit, name)) for name in KNOB_KEYS
                        )
                        if value is not None
                    },
                )
            )
    return tuple(rows)


def present(values: Sequence[Optional[float]]) -> Tuple[Tuple[float, ...], int]:
    """``(the values that exist, how many were absent)``.

    The one shape every column here is built from, carried over from
    ``window_report._present`` for the same reason: a comprehension that filters ``None``
    and drops the count is the reader that prints a confident median over three of 282
    episodes and says nothing about the other 279.
    """
    kept = tuple(value for value in values if value is not None)
    return kept, len(values) - len(kept)


def band_of(rows_at_start: float) -> str:
    """Which ``ROW_BANDS`` label a memory size falls in."""
    size = int(rows_at_start)
    for low, high, label in ROW_BANDS:
        if size >= low and (high is None or size <= high):
            return label
    return ROW_BANDS[-1][2]


def _stats(values: Sequence[float]) -> str:
    if not values:
        return "no episode carried it"
    return "min {:.4f}  median {:.4f}  max {:.4f}".format(
        min(values), statistics.median(values), max(values)
    )


def format_omega(rows: Sequence[EpisodeRow]) -> str:
    """Section A. Whether eq. 23 did anything, and the three ways it could fail to."""
    out: List[str] = []
    out.append("A. omega_t — DID THE MEMORY WEIGHTING MOVE (eq. 23)")

    informed, unrecorded = present([row.informed_steps for row in rows])
    out.append(
        "   episodes:                    {}".format(len(rows))
    )
    out.append(
        "   recorded dream_informed_steps: {}   NOT RECORDED: {}".format(
            len(informed), unrecorded
        )
    )
    if unrecorded == len(rows):
        out.append("")
        out.append(
            "   THE ARM RECORDED omega NOWHERE. Either this is not a DREAM arm, or it "
            "ran\n   before the instrumentation landed (PR #117). Nothing below can be "
            "read."
        )
        return "\n".join(out)

    never = [row for row in rows if row.informed_steps == 0.0]
    live = [row for row in rows if row.informed_steps not in (None, 0.0)]
    out.append(
        "   of those, omega NEVER DEFINED (0 informed steps): {}".format(len(never))
    )
    out.append(
        "              omega defined on >=1 step:             {}".format(len(live))
    )
    if informed:
        out.append(
            "   informed steps per episode:  {}".format(_stats(informed))
        )

    spreads, spread_absent = present([row.omega_e_spread for row in live])
    out.append("")
    out.append(
        "   omega^E SPREAD within an episode (max - min over its informed steps):"
    )
    out.append("     {}".format(_stats(spreads)))
    if spread_absent:
        out.append(
            "     absent on {} of the {} informed episode(s)".format(
                spread_absent, len(live)
            )
        )
    if spreads:
        exactly_zero = sum(1 for value in spreads if value == 0.0)
        flat = sum(1 for value in spreads if value < FLAT_OMEGA_SPREAD)
        out.append(
            "     spread exactly 0.0: {}   below the {:.2f} flat floor: {} of {} "
            "({:.0f}%)".format(
                exactly_zero, FLAT_OMEGA_SPREAD, flat, len(spreads),
                100.0 * flat / len(spreads),
            )
        )
        out.append("")
        if flat == len(spreads):
            out.append(
                "     VERDICT: FLAT ON EVERY EPISODE. eq. 23 resolved to fixed weights "
                "for the\n     whole sweep, so eq. 24 fused the three levels at "
                "constants. Whatever this\n     arm's Find-SR is, it is not a "
                "measurement of dynamic memory weighting."
            )
        elif flat:
            out.append(
                "     VERDICT: MIXED. {} of {} episodes moved omega by more than "
                "{:.2f}.\n     Those are the episodes any effect has to live in; the "
                "rest ran at fixed\n     weights.".format(
                    len(spreads) - flat, len(spreads), FLAT_OMEGA_SPREAD
                )
            )
        else:
            out.append(
                "     VERDICT: LIVE. Every episode moved omega by more than {:.2f}. "
                "The\n     mechanism is doing something; whether it helps is the "
                "paired diff's\n     question, not this one's.".format(
                    FLAT_OMEGA_SPREAD
                )
            )

    for label, key in (
        ("omega^E", "omega_e_mean"),
        ("omega^P", "omega_p_mean"),
        ("omega^K", "omega_k_mean"),
    ):
        values, absent = present([getattr(row, key) for row in live])
        if values:
            out.append(
                "   {} mean per episode:     {}{}".format(
                    label,
                    _stats(values),
                    "   (absent on {})".format(absent) if absent else "",
                )
            )

    # HOW MANY LEVELS EXISTED TO BE WEIGHTED. `_softmax` EXCLUDES a level that retrieved
    # nothing rather than handing it a 0.0 logit, so a softmax over one live level returns
    # 1.0 for it and eq. 24 fuses that level alone. A reader that stops at "omega^E is
    # flat" reports a weighting that did not vary; the truth in that case is that there
    # was nothing to vary BETWEEN, which is a different defect with a different fix.
    #
    # `dream-1` is why this block exists. It printed omega^E median 1.0000, omega^P median
    # 0.0000 and omega^K max 0.0000 and left the reader to notice that eq. 23 had one
    # element in it.
    lives = [
        sum(
            1
            for value in (row.omega_e_mean, row.omega_p_mean, row.omega_k_mean)
            if value is not None and value > 0.0
        )
        for row in live
        if row.omega_e_mean is not None
    ]
    if lives:
        out.append("")
        out.append("   LEVELS WITH ANY WEIGHT AT ALL, per episode (eq. 24 fuses these):")
        for count in (1, 2, 3):
            out.append(
                "     exactly {} of M^E/M^P/M^K: {} episode(s)".format(
                    count, sum(1 for value in lives if value == count)
                )
            )
        singletons = sum(1 for value in lives if value == 1)
        if singletons:
            out.append("")
            out.append(
                "     {} of {} episode(s) HAD ONLY ONE LIVE LEVEL. Their omega is 1.0 by"
                .format(singletons, len(lives))
            )
            out.append(
                "     arithmetic and their spread is exactly 0.0 for the same reason: a"
            )
            out.append(
                "     softmax over one element. Those episodes did not run a weighting"
            )
            out.append(
                "     that failed to move — they ran no weighting, and reading them as"
            )
            out.append(
                "     flat weights points at the wrong fix. The empty levels are the fix."
            )

    for label, key, what in (
        ("M^P", "omega_p_mean", "abstraction (eq. 16) never produced a pattern"),
        ("M^K", "omega_k_mean", "the semantic store was never populated"),
    ):
        values, _absent = present([getattr(row, key) for row in live])
        if values and max(values) == 0.0:
            out.append("")
            out.append(
                "     {} CARRIED NO WEIGHT ON ANY EPISODE — {}, so".format(label, what)
            )
            out.append(
                "     eq. 22's level had nothing to retrieve on all {} of them.".format(
                    len(values)
                )
            )

    # DESCRIPTIVE AND NOT CAUSAL, and it says so: episodes differ in scene, distance and
    # class as well as in omega, so this split is a place to look and never a finding.
    if spreads and len(spreads) > 3:
        cut = statistics.median(spreads)
        above = [row for row in live if (row.omega_e_spread or 0.0) > cut]
        below = [row for row in live if (row.omega_e_spread or 0.0) <= cut]
        if above and below:
            out.append("")
            out.append(
                "   DESCRIPTIVE ONLY — episodes split at the median spread {:.4f}:"
                .format(cut)
            )
            out.append(
                "     omega moved more: {} of {} reached ({:.1f}%)".format(
                    sum(1 for row in above if row.reached), len(above),
                    100.0 * sum(1 for row in above if row.reached) / len(above),
                )
            )
            out.append(
                "     omega moved less: {} of {} reached ({:.1f}%)".format(
                    sum(1 for row in below if row.reached), len(below),
                    100.0 * sum(1 for row in below if row.reached) / len(below),
                )
            )
            out.append(
                "     These episodes are not matched on scene, distance or class, so "
                "this is\n     a place to look and not a comparison. The paired arm "
                "diff is the test."
            )
            out.append(
                "     AND THE SPLIT MAY BE THE SELECTION ITSELF: omega can only "
                "move when a\n     SECOND level is live, and M^P appears only "
                "after the scene has\n     already produced `min_support` reached "
                "episodes at one anchor. Where\n     that is what separates the "
                "two groups, 'omega moved' is reading out\n     which scenes were "
                "already working."
            )
    return "\n".join(out)


def format_precondition(rows: Sequence[EpisodeRow]) -> str:
    """Section B. How much of the key space ``M^E`` used, against how big it was."""
    out: List[str] = []
    out.append("B. THE PRECONDITION — HOW DEGENERATE ARE M^E's KEYS")
    out.append(
        "   A retrieval cannot discriminate between keys that are all the same key. The"
    )
    out.append(
        "   box measured 0.909-0.989 over ten rows from one walk and omega was flat as a"
    )
    out.append("   consequence. Banded by memory size: pooling the two hides which it is.")
    out.append("")

    banded: Dict[str, List[EpisodeRow]] = {}
    unbanded = 0
    for row in rows:
        start = row.rows_at_start
        if start is None:
            unbanded += 1
            continue
        banded.setdefault(band_of(start), []).append(row)

    if not banded:
        out.append(
            "   NO EPISODE CARRIED A ROW COUNT ({} of {} missing it), so there is no"
            .format(unbanded, len(rows))
        )
        out.append("   memory size to band the key spread against.")
        return "\n".join(out)
    out.append(
        "   M^E rows at      episodes   cos min (median)  cos mean (median)  "
        "cos max (median)"
    )
    out.append(
        "   episode start"
    )
    for _low, _high, label in ROW_BANDS:
        members = banded.get(label)
        if not members:
            continue
        mins, _ = present([row.me_cosine_min for row in members])
        means, _ = present([row.me_cosine_mean for row in members])
        maxes, _ = present([row.me_cosine_max for row in members])

        def cell(values: Sequence[float]) -> str:
            if not values:
                return "      --     "
            return "    {:.4f}   ".format(statistics.median(values))

        out.append(
            "   {:<16} {:>8}  {}  {}  {}".format(
                label, len(members), cell(mins), cell(means), cell(maxes)
            )
        )
    if unbanded:
        out.append(
            "   {} episode(s) carried no row count and are in no band.".format(unbanded)
        )
    empty_band = banded.get("0 (empty)", [])
    if empty_band:
        out.append("")
        out.append(
            "   The '0 (empty)' row is DASHES BY CONSTRUCTION: one key has no pairwise"
        )
        out.append(
            "   cosine, so key_spread returns None there rather than 1.0. Those {} "
            "episode(s)".format(len(empty_band))
        )
        out.append("   ran with no memory at all — see section D.")
    return "\n".join(out)


def format_retention(rows: Sequence[EpisodeRow]) -> str:
    """Section C. Whether ``eta`` retained, whether it refused, and whether it was
    ``eta`` rather than ``max_retained`` that chose what the memory holds."""
    out: List[str] = []
    out.append("C. RETENTION — IS eta (eq. 13) DOING ANYTHING")

    added, absent = present([row.rows_added for row in rows])
    if not added:
        out.append(
            "   NO EPISODE RECORDED dream_rows_added ({} of {} missing it).".format(
                absent, len(rows)
            )
        )
        return "\n".join(out)
    out.append("   rows added per episode:   {}".format(_stats(added)))
    if absent:
        out.append("   absent on {} of {} episode(s)".format(absent, len(rows)))
    zero = sum(1 for value in added if value == 0.0)
    out.append(
        "   episodes that retained NOTHING: {} of {} ({:.0f}%)".format(
            zero, len(added), 100.0 * zero / len(added)
        )
    )
    out.append("   total rows written over the arm: {:.0f}".format(sum(added)))

    # WHICH RULE ACTUALLY RETAINED. `rows_added` alone cannot say: eta passing twelve
    # segments and a cap of twelve truncating forty both write twelve rows. The runner
    # counts the segments over eta BEFORE the cap for exactly this, and until now
    # nothing read it — the `dream-1` shape, where a run's own decisive number reached
    # no reader.
    over, over_absent = present([row.segments_over_eta for row in rows])
    caps = sorted({
        value
        for value in (row.knobs.get("dream_max_retained") for row in rows)
        if value is not None
    })
    cap_bound: Optional[int] = None
    if over:
        out.append(
            "   segments over eta per episode, BEFORE the cap: {}".format(_stats(over))
        )
        if over_absent:
            out.append(
                "     absent on {} of {} episode(s)".format(over_absent, len(rows))
            )
        if len(caps) == 1:
            cap_bound = sum(1 for value in over if value >= caps[0])
            out.append(
                "   the cap (--dream-max-retained) is {:.0f}, and it BOUND on {} of {} "
                "episode(s)".format(caps[0], cap_bound, len(over))
            )
        elif len(caps) > 1:
            out.append(
                "   MIXED CAPS in one directory ({}) — two configurations, so there "
                "is no single cap\n   to judge this against.".format(
                    ", ".join("{:.0f}".format(value) for value in caps))
            )
        else:
            out.append(
                "   no episode recorded --dream-max-retained, so the count above "
                "cannot be\n   compared against the cap it was written to be compared "
                "against."
            )
    elif over_absent == len(rows):
        out.append(
            "   dream_segments_over_eta RECORDED NOWHERE. This run predates the "
            "counter, so\n   whether eta or the cap did the retaining cannot be "
            "decided from this run."
        )

    if cap_bound is not None and cap_bound * 2 > len(over):
        out.append("")
        out.append(
            "   THE CAP IS THE RETENTION RULE, NOT eta. More segments cleared eta than "
            "the cap"
        )
        out.append(
            "   admits on {} of {} episode(s), so eq. 13 selected D* and the cap then "
            "took the".format(cap_bound, len(over))
        )
        out.append(
            "   top {:.0f} by I_j. That is a top-k rule wearing a threshold's name. "
            "RAISE eta".format(caps[0])
        )
        out.append(
            "   until this line reads a minority, or write the ADR that adopts top-k on"
        )
        out.append("   purpose. Do NOT quietly raise the cap.")
    elif zero == len(added):
        out.append("")
        out.append(
            "   eta RETAINED NOTHING, ALL NIGHT. Every other number in this report is"
        )
        out.append("   about an empty memory.")
    elif cap_bound == 0:
        out.append("")
        out.append(
            "   eta IS THE RETENTION RULE. The cap never bound, so every row written "
            "was one"
        )
        out.append(
            "   eq. 13 chose. The cap is the bound it was added to be and nothing more."
        )
    elif zero == 0:
        out.append("")
        out.append(
            "   eta REFUSED NOTHING, ALL NIGHT. Every episode's segments cleared the"
        )
        out.append(
            "   threshold, so eq. 13 selected the whole of D rather than the part of it"
        )
        out.append(
            "   above eta. That is a threshold set below its own data, and the "
            "importance"
        )
        out.append("   range beside it is what it should have been set against.")

    lows, _ = present([row.importance_min for row in rows])
    highs, _ = present([row.importance_max for row in rows])
    if lows and highs:
        out.append(
            "   I_j actually seen:        min {:.4f}   max {:.4f}   "
            "(eta was set at the knob in section F)".format(min(lows), max(highs))
        )

    patterns, _ = present([row.pattern_rows for row in rows])
    if patterns:
        out.append(
            "   M^P rows (eq. 16) per episode: {}".format(_stats(patterns))
        )
        if max(patterns) == 0.0:
            out.append(
                "     M^P WAS EMPTY ON EVERY EPISODE — abstraction never met "
                "min-support, so\n     eq. 22's level had nothing to retrieve and "
                "omega^P weighted an absence."
            )
    return "\n".join(out)


def format_reach(rows: Sequence[EpisodeRow]) -> str:
    """Section D. How large a memory this sweep ever actually built."""
    out: List[str] = []
    out.append("D. HOW FAR THE MEMORY EVER REACHED")

    per_scene: Dict[str, List[EpisodeRow]] = {}
    for row in rows:
        per_scene.setdefault(row.scene, []).append(row)

    sizes, absent = present([row.experience_rows for row in rows])
    if not sizes:
        out.append(
            "   NO EPISODE RECORDED dream_experience_rows ({} missing).".format(absent)
        )
        return "\n".join(out)

    starts, _ = present([row.rows_at_start for row in rows])
    empty = sum(1 for value in starts if value == 0.0)
    out.append(
        "   episodes that ran against an EMPTY M^E: {} of {}".format(empty, len(rows))
    )
    out.append(
        "   largest M^E any episode ever saw:      {:.0f} row(s)".format(max(starts))
        if starts
        else "   no episode carried a start size"
    )
    out.append("")
    out.append("   scene                  episodes   M^E max   started empty")
    scenes_starting_empty = 0
    for scene in sorted(per_scene):
        members = per_scene[scene]
        scene_sizes, _ = present([row.experience_rows for row in members])
        scene_starts, _ = present([row.rows_at_start for row in members])
        n_empty = sum(1 for value in scene_starts if value == 0.0)
        if n_empty:
            scenes_starting_empty += 1
        out.append(
            "   {:<20} {:>9}  {:>8}  {:>13}".format(
                scene,
                len(members),
                "{:.0f}".format(max(scene_sizes)) if scene_sizes else "--",
                n_empty,
            )
        )
    out.append("")
    if scenes_starting_empty == len(per_scene) and len(per_scene) > 1:
        out.append(
            "   EVERY SCENE STARTED FROM EMPTY. The sweep invokes `python -m earshot`"
        )
        out.append(
            "   once per scene, so run()'s carry-over is WITHIN a scene: M^E is built"
        )
        out.append(
            "   and thrown away {} times rather than built once across the sweep. This"
            .format(len(per_scene))
        )
        out.append(
            "   run therefore cannot answer whether a memory spanning many scenes "
            "escapes"
        )
        out.append(
            "   the key degeneracy in section B — it never built one. Carrying the "
            "store"
        )
        out.append("   across scenes is a driver change, not a knob.")
    return "\n".join(out)


def format_cost(rows: Sequence[EpisodeRow]) -> str:
    """Section E. What the DREAM step cost in the field, against the box's estimate."""
    out: List[str] = []
    out.append("E. WHAT IT COST")
    means, absent = present([row.step_s_mean for row in rows])
    worsts, _ = present([row.step_s_worst for row in rows])
    if not means:
        out.append(
            "   NO EPISODE RECORDED dream_step_s_mean ({} missing).".format(absent)
        )
        return "\n".join(out)
    out.append("   DREAM step, per-episode mean:  {}".format(_stats(means)))
    if worsts:
        out.append("   DREAM step, per-episode worst: {}".format(_stats(worsts)))
    out.append(
        "   The box's pre-run estimate was 0.057 s/step, which is what the sweep's wall"
    )
    out.append(
        "   clock was sized from. Criterion 7 audits audio_render_s and never sees this"
    )
    out.append("   cost, so this line is the only place it is checked against the plan.")
    return "\n".join(out)


def format_knobs(rows: Sequence[EpisodeRow]) -> str:
    """Section F. What the arm was configured with, read off the episodes themselves.

    From the audits and never from the driver: a driver edited after a run would
    otherwise rewrite what the run did, and this repo has already lost a box run to a
    fix that was not in the commit the box checked out.
    """
    out: List[str] = []
    out.append("F. THE KNOBS THIS ARM RAN WITH")
    mixed: List[str] = []
    missing: List[str] = []
    for key in KNOB_KEYS:
        values, absent = present([row.knobs.get(key) for row in rows])
        if not values:
            missing.append(key)
            continue
        distinct = sorted(set(values))
        if len(distinct) > 1:
            mixed.append(key)
        out.append(
            "   {:<26} {}{}".format(
                key.replace("dream_", ""),
                ", ".join("{:g}".format(value) for value in distinct),
                "   (absent on {} episode(s))".format(absent) if absent else "",
            )
        )
    if missing:
        out.append(
            "   NOT RECORDED ANYWHERE: {}".format(", ".join(missing))
        )
    if mixed:
        out.append("")
        out.append(
            "   MIXED ARM. {} took more than one value across these episodes, so this"
            .format(", ".join(mixed))
        )
        out.append(
            "   directory holds more than one configuration and its aggregate numbers"
        )
        out.append("   are over two different runs.")
    return "\n".join(out)


def format_report(rows: Sequence[EpisodeRow], *, arm: str, arm_dir: str) -> str:
    """The whole readout."""
    header = [
        "DREAM MEMORY READOUT — arm '{}' at {}".format(arm, arm_dir),
        "=" * 74,
        "",
    ]
    if not rows:
        header.append(
            "NO EPISODES ON DISK under {}. Three layouts are read: <tag>/<arm>/"
            "<scene>/,\none arm directory holding <scene>/, and one run directory "
            "holding episodes/ itself.\nNone of them matched, so either the arm name "
            "is wrong or the run wrote nothing.".format(arm_dir)
        )
        return "\n".join(header)
    body = [
        format_omega(rows),
        "",
        format_precondition(rows),
        "",
        format_retention(rows),
        "",
        format_reach(rows),
        "",
        format_cost(rows),
        "",
        format_knobs(rows),
    ]
    return "\n".join(header + body)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m earshot.tools.dream_report",
        description=(
            "What a DREAM arm's memory did: omega_t's spread, M^E's key degeneracy, "
            "eta's retention and how large a memory the sweep ever built. Read-only."
        ),
    )
    parser.add_argument(
        "run_dir",
        help=(
            "a sweep directory (runs/<tag>, which holds <arm>/<scene>/), one arm "
            "directory directly, or ONE RUN directory holding episodes/ itself — the "
            "shape a single `python -m earshot --run-dir` invocation writes"
        ),
    )
    parser.add_argument(
        "--arm",
        default="dream",
        help="arm subdirectory to read (default: dream). Ignored if run_dir IS the arm.",
    )
    args = parser.parse_args(argv)

    root = pathlib.Path(args.run_dir)
    arm_dir = root / args.arm
    if not arm_dir.is_dir():
        # `run_dir` may already BE the arm. Falling back rather than failing means the
        # tool reads both layouts the sweeps in this tree produce.
        arm_dir = root
    rows = read_rows(str(arm_dir))
    print(format_report(rows, arm=args.arm, arm_dir=str(arm_dir)))
    if not rows:
        return 2
    _informed, unrecorded = present([row.informed_steps for row in rows])
    if unrecorded == len(rows):
        # The arm recorded the central quantity nowhere, which is unreadable rather than
        # flat. Same rule as `placement_report`: a field absent everywhere exits nonzero.
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
