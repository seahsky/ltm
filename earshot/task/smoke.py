"""Task spec §8's nine acceptance criteria, judged off a run directory.

    python -m earshot.task.smoke --run-dir runs/<tag>              # every episode
    python -m earshot.task.smoke --run-dir runs/<tag> --episode 3  # just one

**The default is every episode, and it was not always.** ``judge_run_dir`` answers the
nine over ONE episode, index 0, which is the right shape for the one-episode smoke this
module was built for and the wrong shape for anything else. The yield-1 sweep ran twenty
episodes in each of twenty scenes and evaluated none of them; had it called this module it
would have judged episode 0 of twenty, and criterion 5 passed in 8 of those 20 — a coin
flip printed as a gate. ``tally`` makes n the denominator, which is also the first time
criteria 1/2/3/7 assert anything: "audio rendered at every step" is a claim about a run,
not about the first episode of one.

**The gate ticket 10's irreversible deletion commit hangs off.** Every criterion is
answered from the artefacts a run already writes — ``env_report.json``, and each episode's
``report.json`` / ``audit.json`` — rather than from a run that is still in memory. Two
reasons, both learned on this map: an artefact is what a later reader has, and a judge that
watched the run could pass on state the files do not carry, which is the shape of a green
that cannot be reproduced.

``judge()`` is **pure** — records in, verdict out — so ticket 19's third row applies: the
question *given a failing measurement, does the gate go red* is Mac-testable with injected
records, needs no box, and is the assertion ticket 13's version-blind skip would have
failed. The CLI is a thin reader around it.

**A criterion that could not be evaluated is never green.** ``CriterionStatus.NOT_RUN``
exists for the same reason ``env_check``'s does: ticket 16 found a canary that was never
armed reading as a pass, and ticket 13 found a probe that skipped and reported success.
Absence is red here, and it says which measurement was missing.

Criterion 9 (hermeticity) is **structurally** NOT_RUN in this module: it is a re-run of the
whole thing with both old trees moved out of the repo, so it is a property of two runs and
ticket 27 owns it. Reporting it as a red criterion rather than omitting it is deliberate —
a nine-point gate that silently judges eight is how "smoke green" and "the deletion is
safe" come apart.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple

from earshot.report.artifacts import ENV_REPORT_NAME, read_episode, run_paths
from earshot.report.audit import EpisodeAudit, FunnelStage

__all__ = [
    "CriterionStatus",
    "Criterion",
    "SmokeVerdict",
    "CriterionTally",
    "RunVerdict",
    "RATE_CRITERIA",
    "judge",
    "judge_run_dir",
    "judge_every_episode",
    "tally",
    "main",
]

# §5.1's nine keys. A report missing one is criterion 6 red; a *null* in one §5.1 permits
# to be absent is data. Held here rather than derived from `AgentReport.__dataclass_fields__`
# because the criterion is about the schema the spec names, and a field renamed on both
# sides at once would keep a derived check green.
REPORT_KEYS = (
    "primary_completed",
    "heard_at_step",
    "room",
    "anomaly_class",
    "stopped_at_pose",
    "visual_confirm_object",
    "investigate_aborted",
    "resumed",
    "n_benign_ignored",
)

# Ticket 12's mesh floor, restated where the gate reads it. `> 0` is not enough: a
# degenerate mesh gives the same direct-path-only IR as an empty one.
MIN_SCENE_VERTICES = 10_000

# Criterion 9's evidence file, written into the run directory by
# `earshot/tools/hermeticity_gate.sh`. The name and schema string are duplicated from
# `earshot/tools/reset_manifest.py` on purpose: `tools/` sits outside the layer graph and
# `task/` may not import it, so the record crosses that boundary as JSON exactly the way
# `env_report.json` does. `tests/mac/test_reset_manifest.py` asserts the two agree.
HERMETICITY_NAME = "hermeticity.json"
HERMETICITY_SCHEMA = "earshot.hermeticity/1"

# Criteria that are a RATE over n rather than an assertion that must hold on every
# episode. Only criterion 5 qualifies, and the line is between what the harness did and
# what the agent did: 1/2/3/7 say the audio rendered live at every step within its
# ceiling, 4 says the onset can only have been the anomaly, 6/8/9 say the artefacts and
# the environment are what they claim — every one of those must hold in all n or the run
# is not measuring what it says. Criterion 5 says the agent reached the source and got
# back, which is the capability under study; demanding 20/20 of it turns the gate into a
# performance bar and it goes red for the one reason that is a finding rather than a bug.
#
# It is not exempt, only re-shaped: `tally` still fails it at 0/n. That is ADR-0014's
# vacuous arm — a loop that never once ran is a loop that is not wired, and the funnel
# printing stage 6 for an episode that never reached the source is exactly how this
# criterion was over-credited before ticket 26 enforced the ladder.
RATE_CRITERIA = frozenset({5})


class CriterionStatus(Enum):
    """PASS, FAIL, or NOT_RUN — and only PASS is green."""

    PASS = "pass"
    FAIL = "fail"
    NOT_RUN = "not_run"


@dataclass(frozen=True)
class Criterion:
    """One of §8's nine, with the measurement that decided it.

    ``detail`` carries numbers rather than a verdict restated in words, because the box
    trips on this map have been decidable only when they printed what they measured
    (ADR-0014).
    """

    number: int
    name: str
    status: CriterionStatus
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status is CriterionStatus.PASS

    def line(self) -> str:
        mark = {
            CriterionStatus.PASS: "PASS",
            CriterionStatus.FAIL: "FAIL",
            CriterionStatus.NOT_RUN: "NOT RUN",
        }[self.status]
        return "  {}. {:<7} {:<34} {}".format(self.number, mark, self.name, self.detail)


@dataclass(frozen=True)
class SmokeVerdict:
    criteria: Tuple[Criterion, ...] = ()
    notes: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def failed(self) -> Tuple[int, ...]:
        return tuple(c.number for c in self.criteria if not c.ok)

    @property
    def green(self) -> bool:
        return bool(self.criteria) and not self.failed

    def summary(self) -> str:
        lines = ["task spec §8 — smoke-green acceptance criteria:"]
        lines.extend(c.line() for c in self.criteria)
        lines.extend("  note: {}".format(n) for n in self.notes)
        lines.append(
            "SMOKE GREEN" if self.green else "SMOKE RED — criteria {}".format(
                ", ".join(str(n) for n in self.failed)
            )
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class CriterionTally:
    """One criterion counted across every episode of a run.

    ``details`` keeps the measurement off the first few episodes that did not pass,
    because ADR-0014's rule is that a box result is only decidable when it prints what it
    measured, and "criterion 7: 18/20" without a number is a verdict with the evidence
    thrown away. Bounded rather than complete: twenty identical failures say the same
    thing three times over.
    """

    number: int
    name: str
    n_pass: int = 0
    n_fail: int = 0
    n_not_run: int = 0
    details: Tuple[str, ...] = ()

    @property
    def n(self) -> int:
        return self.n_pass + self.n_fail + self.n_not_run

    @property
    def ok(self) -> bool:
        """Green iff every episode passed — unless this is a rate, which needs one."""
        if self.n == 0:
            return False
        if self.number in RATE_CRITERIA:
            return self.n_pass > 0
        return self.n_pass == self.n

    def line(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        rate = "{}/{}".format(self.n_pass, self.n)
        if self.number in RATE_CRITERIA and self.n:
            rate += " ({:.0%})".format(self.n_pass / self.n)
        head = "  {}. {:<7} {:<34} {}".format(self.number, mark, self.name, rate)
        if self.n_not_run:
            head += "  [{} NOT RUN]".format(self.n_not_run)
        if self.details:
            head += "\n" + "\n".join("       {}".format(d) for d in self.details)
        return head


@dataclass(frozen=True)
class RunVerdict:
    """§8's nine criteria over every episode of one run, as counts rather than one verdict.

    ``judge_run_dir`` answers "did episode 0 pass", which at a 40% loop rate is a coin
    flip dressed as a gate — the yield-1 sweep ran twenty episodes a scene and judged
    none of them. This is the same nine criteria with n as the denominator, so criteria
    1/2/3/7 become a real assertion (they must hold at every step of every episode) and
    criterion 5 becomes the funnel it always was.
    """

    n_episodes: int = 0
    tallies: Tuple[CriterionTally, ...] = ()
    notes: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def failed(self) -> Tuple[int, ...]:
        return tuple(t.number for t in self.tallies if not t.ok)

    @property
    def green(self) -> bool:
        return bool(self.tallies) and not self.failed

    def summary(self) -> str:
        lines = ["task spec §8 — acceptance criteria over {} episode(s):".format(
            self.n_episodes)]
        lines.extend(t.line() for t in self.tallies)
        lines.extend("  note: {}".format(n) for n in self.notes)
        lines.append(
            "GREEN" if self.green else "RED — criteria {}".format(
                ", ".join(str(n) for n in self.failed)))
        return "\n".join(lines)


# How many failing measurements a tally keeps per criterion. Three is enough to tell one
# systematic failure from three unrelated ones and short enough to read in an email.
MAX_TALLY_DETAILS = 3

# Notes are per-episode, and two kinds arrive mixed: §8's required disclosures, which are
# identical across episodes and collapse to one line, and the per-episode collision counts,
# which carry different numbers every time and would otherwise print n of them. `judge`
# emits the disclosures first, so first-seen order keeps them and the cap falls on the
# numeric ones. The count of what was dropped is printed rather than elided silently —
# CLAUDE.md's rule about a bounded report that reads as a complete one.
MAX_TALLY_NOTES = 6


def tally(verdicts: Sequence[SmokeVerdict],
          labels: Optional[Sequence[Any]] = None) -> RunVerdict:
    """Pool per-episode verdicts into per-criterion counts. Pure.

    Pure for the same reason ``judge`` is (ticket 19's third row): *given n episodes of
    which some fail, does the run go red* is answerable on a Mac with injected verdicts,
    and it is the assertion that a gate judging episode 0 could never make.

    Criterion numbering comes from the verdicts rather than from a constant, so a
    criterion added to ``judge`` shows up here without a second edit — but an episode
    that is missing one is a shorter tuple, and ``n`` per criterion is therefore counted
    rather than assumed equal to ``n_episodes``.

    ``labels`` names the episode each verdict came from, and the first version omitted it.
    The detour-1 run printed criterion 5 three times as *"funnel stage 4
    (INVESTIGATE_ENTERED) — CHECK and RESUME must both be reached"*, identical and
    anonymous, which is a verdict with the measurement thrown away — the exact failure
    this tally exists to prevent (ADR-0014). "ep 7: funnel stage 4" is a line you can go
    and read the audit for.
    """
    tags = list(labels) if labels is not None else []
    order: list = []
    counts: dict = {}
    for position, verdict in enumerate(verdicts):
        tag = ("ep {}: ".format(tags[position])
               if position < len(tags) else "")
        for criterion in verdict.criteria:
            key = int(criterion.number)
            if key not in counts:
                order.append(key)
                counts[key] = {"name": criterion.name, "pass": 0, "fail": 0,
                               "not_run": 0, "details": []}
            row = counts[key]
            if criterion.status is CriterionStatus.PASS:
                row["pass"] += 1
            else:
                row["fail" if criterion.status is CriterionStatus.FAIL else "not_run"] += 1
                if len(row["details"]) < MAX_TALLY_DETAILS and criterion.detail:
                    row["details"].append(tag + criterion.detail)

    tallies = tuple(
        CriterionTally(
            number=key,
            name=counts[key]["name"],
            n_pass=counts[key]["pass"],
            n_fail=counts[key]["fail"],
            n_not_run=counts[key]["not_run"],
            details=tuple(counts[key]["details"]),
        )
        for key in sorted(order)
    )
    # Notes are per-episode disclosures (the oracle-STOP one is required by §8), and
    # twenty copies of the same sentence is not a disclosure. De-duplicated in first-seen
    # order rather than sorted, so the required disclosure keeps its position.
    notes: list = []
    for verdict in verdicts:
        for note in verdict.notes:
            if note not in notes:
                notes.append(note)
    if len(notes) > MAX_TALLY_NOTES:
        dropped = len(notes) - MAX_TALLY_NOTES
        notes = notes[:MAX_TALLY_NOTES] + [
            "{} further per-episode note(s) not shown; the full set is in each "
            "episode's audit.json".format(dropped)]
    return RunVerdict(n_episodes=len(verdicts), tallies=tallies, notes=tuple(notes))


def _metric(audit: EpisodeAudit, key: str) -> Optional[float]:
    value = audit.metrics.get(key)
    return None if value is None else float(value)


def _live_every_step(audit: EpisodeAudit) -> Criterion:
    """Criterion 1. **The loop counters, not ``World.n_renders``** (ticket 25).

    Arming the guard, §2.5's start-pose audibility check and the calibration sweep all
    render legitimately *before* the first step, so the lifetime counter fails this for a
    reason that is not a defect.
    """
    rendered = _metric(audit, "n_renders_in_loop")
    stepped = _metric(audit, "n_loop_steps")
    if rendered is None or stepped is None:
        return Criterion(1, "audio live and every-step", CriterionStatus.NOT_RUN,
                         "n_renders_in_loop / n_loop_steps missing from metrics")
    if stepped <= 0:
        return Criterion(1, "audio live and every-step", CriterionStatus.FAIL,
                         "the loop took no steps, so equality is vacuous")
    status = CriterionStatus.PASS if rendered == stepped else CriterionStatus.FAIL
    return Criterion(1, "audio live and every-step", status,
                     "{:.0f} renders / {:.0f} loop steps".format(rendered, stepped))


def _context_sound(audit: EpisodeAudit) -> Criterion:
    """Criterion 2: ticket 12's guard armed, green through ticket 16's invariants."""
    ctx = audit.audio_context
    if ctx is None:
        return Criterion(2, "audio context sound", CriterionStatus.NOT_RUN,
                         "no audio_context on the audit — the guard did not arm")
    problems = []
    if int(ctx.n_vertices) < MIN_SCENE_VERTICES:
        problems.append("mesh {} verts < {}".format(ctx.n_vertices, MIN_SCENE_VERTICES))
    if not ctx.log_canary_seen:
        # Ticket 16: severity is the stream, and a canary that never armed reads exactly
        # like a clean log. Unverified is not satisfied.
        problems.append("log canary never seen")
    if ctx.rlr_engine_error:
        problems.append("RLR engine error in the log")
    if ctx.fatal_log_lines:
        problems.append("{} fatal log line(s)".format(len(ctx.fatal_log_lines)))
    detail = "{} verts (submitted {}), canary {}".format(
        ctx.n_vertices, ctx.submitted_n_vertices, "seen" if ctx.log_canary_seen else "ABSENT"
    )
    if problems:
        return Criterion(2, "audio context sound", CriterionStatus.FAIL,
                         "; ".join(problems))
    return Criterion(2, "audio context sound", CriterionStatus.PASS, detail)


def _ir_real(audit: EpisodeAudit) -> Criterion:
    """Criterion 3: non-silent, scene-dependent, trimmed to the decay."""
    ctx = audit.audio_context
    if ctx is None or ctx.ir_shape is None:
        return Criterion(3, "the IR is real", CriterionStatus.NOT_RUN,
                         "no IR shape recorded")
    shape = tuple(int(v) for v in ctx.ir_shape)
    peak = float(ctx.ir_peak_abs)
    if peak <= 0.0:
        return Criterion(3, "the IR is real", CriterionStatus.FAIL,
                         "IR peak is {:.3g} — silent".format(peak))
    if len(shape) != 2 or shape[0] != 2 or shape[1] <= 0:
        return Criterion(3, "the IR is real", CriterionStatus.FAIL,
                         "IR shape {} is not a binaural (2, N)".format(shape))
    return Criterion(3, "the IR is real", CriterionStatus.PASS,
                     "shape {}, peak {:.4g}".format(shape, peak))


def _provenance(
    audit: EpisodeAudit,
    *,
    t_anom: Optional[int],
    policy: Optional[str] = None,
    pre_onset_rms_tol: Optional[float] = None,
) -> Criterion:
    """Criterion 4 (§3.1), and it is read off the record rather than off a clean exit.

    ``assert_provenance`` raises, so an audit that exists at all *looks* like proof it
    passed — unless it was never called. ``provenance_asserted`` is what makes the
    difference visible, and zero pre-onset readings with ``t_anom > 0`` means the first
    invariant is **unverified, not satisfied** (ticket 16).

    ``t_anom`` arrives from two places now and both are used. The configured value is a
    *pin* and is ``None`` on a run that derived one per episode, so the effective number
    lives on the audit beside the source position the same builder chose. Where a run
    pinned one, the two must agree: an episode whose source started at a different step
    from the one the run asked for is a build that did not do what its configuration
    says, which is the failure this gate exists to catch.

    **ADR-0017's window is checked here rather than as a tenth criterion.** A tenth would
    be NOT_RUN -- and therefore red -- on every audit written before the window existed,
    and this criterion's subject already is *the measured signal is what the task built*.
    The window half runs after every check above and never changes their verdict on a
    record that has no window.

    What it asserts is the PER-STEP TRACE, never the config echo: ``source_playing`` must
    be true on exactly ``[opens_at, offset_step)`` and false from the offset step on.
    That check is entirely missing today -- a source that failed to stop, or one whose
    window closed before it opened, produces a green criterion 4, which is the "the check
    was a log line" shape ADR-0009 and the ``anommxv`` invalidation exist to close.

    **The LEVEL half is measured from ``cue_tail_steps`` since ADR-0019, and a record
    without one PASSES rather than being judged.** The trace half is unaffected and still
    binds on every record. What moved is the fence post: ``tail_steps`` is the 5 s clip
    readout emptying and asserting the level against it waited four steps too long, so the
    criterion now reads the cue tail -- the room, one step wide.

    A record that carries a window and no ``cue_tail_steps`` predates the split, and its
    RMS trace is in the CLIP domain. Judging a clip-domain trace at a cue-domain fence
    post can only go wrongly red, so that case returns PASS with a detail saying the level
    is not asserted -- the same branch shape this file already owns for "ended inside its
    own tail". Two things it deliberately is NOT: it is not added to the ``missing`` guard
    (that would turn every audit written on this branch before today red, and that guard
    must keep failing for its own reason), and it is not re-derived here from
    ``hop_samples`` and ``max_ir_samples`` (a second copy of a definition this tree has
    been bitten by twice, which would also assert a cue bound against a clip trace).
    """
    onset = audit.onset
    if onset is None:
        return Criterion(4, "provenance did not raise", CriterionStatus.NOT_RUN,
                         "no onset record")
    if not onset.provenance_asserted:
        return Criterion(4, "provenance did not raise", CriterionStatus.FAIL,
                         "assert_provenance never ran — its silence is not a pass")
    if t_anom is not None and audit.t_anom is not None and int(t_anom) != int(audit.t_anom):
        return Criterion(4, "provenance did not raise", CriterionStatus.FAIL,
                         "the run pinned t_anom {} but the episode recorded {}".format(
                             int(t_anom), int(audit.t_anom)))
    effective = audit.t_anom if audit.t_anom is not None else t_anom
    if effective is None:
        return Criterion(4, "provenance did not raise", CriterionStatus.NOT_RUN,
                         "neither the run nor the episode records a t_anom, so §3.1's "
                         "first invariant has no bound to be checked against")
    if int(effective) > 0 and int(onset.n_pre_onset_readings) <= 0:
        return Criterion(4, "provenance did not raise", CriterionStatus.FAIL,
                         "t_anom is {} but there were no pre-onset readings, so §3.1's "
                         "first invariant is unverified".format(int(effective)))
    detail = "onset step {}, {} pre-onset readings, t_anom {}".format(
        onset.onset_step, onset.n_pre_onset_readings, int(effective))

    window = audit.sounding_window
    if window is None:
        # Every audit written before ADR-0017. Its verdict must not move: a criterion
        # that changed its answer on an unchanged record is a criterion nobody can
        # compare across runs.
        return Criterion(4, "provenance did not raise", CriterionStatus.PASS, detail)
    if policy is not None and window.policy is not None and str(policy) != str(window.policy):
        return Criterion(4, "provenance did not raise", CriterionStatus.FAIL,
                         "the run asked for the {!r} sounding policy but the episode "
                         "recorded {!r}".format(str(policy), str(window.policy)))
    if window.offset_step is None:
        # The CONTINUOUS control arm. There is nothing to check, and saying so is not
        # the same as passing vacuously.
        return Criterion(4, "provenance did not raise", CriterionStatus.PASS,
                         detail + " — continuous arm, no offset step")

    opens_at = int(window.opens_at)
    offset_step = int(window.offset_step)
    for row in audit.steps:
        index = int(row.step)
        if opens_at <= index < offset_step and not row.source_playing:
            return Criterion(4, "provenance did not raise", CriterionStatus.FAIL,
                             "the sounding window is [{}, {}) but step {} recorded the "
                             "source silent inside it".format(
                                 opens_at, offset_step, index))
        if index >= offset_step and row.source_playing:
            return Criterion(4, "provenance did not raise", CriterionStatus.FAIL,
                             "the source was asked to stop at step {} but step {} "
                             "recorded it still sounding".format(offset_step, index))
    detail += ", window [{}, {})".format(opens_at, offset_step)

    # The signal half, and it deliberately WAITS. The naive symmetric mirror of
    # `onset.py`'s pre-t_anom invariant — "after the offset step the RMS is the bed" —
    # is FALSE for exactly the steps the reverb tail exists for: the room is still ringing
    # on the first silent step and the agent's reading reaches the bed only at the end of
    # `cue_tail_steps`. Waiting that long is what makes the assertion true, and the number
    # comes off the record rather than off a constant because the IR's width is scene- and
    # pose-dependent and this tree caps it nowhere.
    #
    # NOT skipping this is not a NOT_RUN: the criterion IS evaluated — the trace check
    # above always runs — and this is an additional assertion whose premise (the episode
    # outlived its own tail) an episode may simply not meet.
    #
    # `tail_steps` is still what the `missing` guard below demands, because that guard is
    # about the record being whole and every record `run_episode` ever wrote carries it.
    tail_steps = window.tail_steps
    bed_rms = audit.calibration.bed_rms if audit.calibration is not None else None
    last = audit.steps[-1] if audit.steps else None
    # THE RECORD'S OWN CONSISTENCY, and it is RED rather than skipped. `run_episode`
    # writes `tail_steps`, a `CalibrationRecord` and at least one `StepRecord` on every
    # episode that carries a window at all -- one constructor, one call -- so a window
    # WITHOUT them is a record no build of this tree can produce: hand-edited, truncated
    # mid-write, or spliced from two runs. This returned PASS, which is the gate reporting
    # green on an artefact it could not read; and being unreachable from any fixture, it
    # was also the only thing standing between a missing `bed_rms` and the `float(None)`
    # TypeError seven lines down, which turns a gate run into a traceback rather than a
    # verdict.
    missing = [
        name
        for name, value in (
            ("tail_steps", tail_steps),
            ("a calibration record", bed_rms),
            ("any step records", last),
        )
        if value is None
    ]
    if missing:
        return Criterion(4, "provenance did not raise", CriterionStatus.FAIL,
                         detail + ", but the record carries a sounding window and no {} "
                                  "-- run_episode writes those together, so this audit "
                                  "did not come from one run".format(" or ".join(missing)))
    # NOT the same case and NOT red. `pre_onset_rms_tol` is the run's CONFIGURATION and
    # arrives beside the records rather than inside them (`judge`'s own rule: a gate that
    # took its bound from the thing it is bounding passes by construction), so a `judge`
    # called with no `run_config` -- `judge_run_dir` on a directory whose
    # `env_report.json` is missing -- has no bound to measure against. A missing input to
    # the gate is not evidence about the run.
    if pre_onset_rms_tol is None:
        return Criterion(4, "provenance did not raise", CriterionStatus.PASS,
                         detail + ", no pre_onset_rms_tol configured to check the "
                                  "silent phase's level against")
    # THE FENCE POST IS THE CUE'S SINCE ADR-0019, and the record's own `cue_tail_steps`
    # is where it comes from. `tail_steps` is the CLIP readout emptying -- the analysis
    # window, which an anechoic 1-sample IR reproduces to 1.1 points -- so asserting the
    # level against it waited four steps too long on every episode and called a
    # five-second moving average a reverb tail.
    cue_tail_steps = window.cue_tail_steps
    if cue_tail_steps is None:
        # The same shape as "ended inside its own tail" below: the criterion IS evaluated
        # -- the trace check above always ran -- and this is an additional assertion whose
        # premise the RECORD does not meet.
        return Criterion(4, "provenance did not raise", CriterionStatus.PASS,
                         detail + ", the record carries no cue tail, so the silent "
                                  "phase's level is not asserted -- this audit predates "
                                  "the split readout and its RMS trace is the 5 s clip "
                                  "readout, which the cue tail's fence post would judge "
                                  "at the wrong step")
    # The cue tail counts from the LAST SOUNDING step, and the offset step is the first
    # SILENT one -- so the reading is exactly the bed at `offset_step + cue_tail_steps - 1`.
    # Reading it from the offset step over-stated the room's post-offset lifetime by a
    # step and threw away a step of assertable evidence on every episode.
    silent_to_bed = offset_step + int(cue_tail_steps) - 1
    if int(last.step) < silent_to_bed:
        return Criterion(4, "provenance did not raise", CriterionStatus.PASS,
                         detail + ", ended at step {} before the {}-step cue tail ran out "
                                  "at step {}, so the silent phase's level is not "
                                  "asserted".format(
                                      int(last.step), int(cue_tail_steps), silent_to_bed))
    tolerance = abs(float(bed_rms)) * float(pre_onset_rms_tol)
    if tolerance <= 0.0:
        tolerance = abs(float(pre_onset_rms_tol))
    if abs(float(last.measured_rms) - float(bed_rms)) > tolerance:
        return Criterion(4, "provenance did not raise", CriterionStatus.FAIL,
                         "the cue tail ran out at step {} ({} steps after the LAST "
                         "SOUNDING step, {} after the offset step) but the final reading "
                         "is {:.6g} against a bed of {:.6g}".format(
                             silent_to_bed, int(cue_tail_steps), int(cue_tail_steps) - 1,
                             float(last.measured_rms), float(bed_rms)))
    # Printed, not gated. `cue_tail_steps` is arithmetic off the IR's width; this is the
    # number of silent-phase steps the agent could actually still tell from the bed, and
    # a zero means the silence arrived as a hard cut -- which is a fact about the clip and
    # the loop rather than a broken run, so it belongs in the detail an operator reads
    # rather than in a red verdict this gate cannot justify.
    audible = window.post_offset_audible_steps
    return Criterion(4, "provenance did not raise", CriterionStatus.PASS,
                     detail + ", silent phase decayed to the bed at step {} ({} steps "
                              "after the last sounding step), {} audible".format(
                                  silent_to_bed, int(cue_tail_steps),
                                  "audible-step count not recorded"
                                  if audible is None else "{} steps".format(int(audible))))


def _full_loop(audit: EpisodeAudit) -> Criterion:
    """Criterion 5: CHECK and RESUME must **both** be reached.

    Read off ``funnel_stage``, whose ladder ticket 26 made nesting-enforced — the first
    version promoted an aborted detour to stage 6 and printed 6/6 for an episode that
    never reached the source, which is this criterion asserting a loop that did not run.
    """
    stage = audit.funnel_stage
    detail = "funnel stage {} ({})".format(int(stage), stage.name)
    if stage >= FunnelStage.PRIMARY_RESUMED:
        return Criterion(5, "the full loop ran", CriterionStatus.PASS, detail)
    return Criterion(5, "the full loop ran", CriterionStatus.FAIL,
                     detail + " — CHECK and RESUME must both be reached")


def _report_populated(report: Mapping[str, Any]) -> Criterion:
    """Criterion 6: §5.1's schema, fully populated.

    *Populated* is the key set, not the absence of nulls: §5.1 permits
    ``visual_confirm_object`` to be absent, and ``anomaly_class`` is null by design when
    the smoke runs without CLAP — copying it off the dataset would be the task telling the
    agent what it heard.
    """
    missing = [key for key in REPORT_KEYS if key not in report]
    extra = [key for key in report if key not in REPORT_KEYS]
    if missing:
        return Criterion(6, "a report was emitted", CriterionStatus.FAIL,
                         "missing §5.1 key(s): {}".format(", ".join(missing)))
    if extra:
        return Criterion(6, "a report was emitted", CriterionStatus.FAIL,
                         "keys outside §5.1's nine: {}".format(", ".join(extra)))
    return Criterion(6, "a report was emitted", CriterionStatus.PASS,
                     "all {} of §5.1's keys".format(len(REPORT_KEYS)))


def _within_ceiling(audit: EpisodeAudit, *, ceiling_s: Optional[float]) -> Criterion:
    """Criterion 7: per-step audio wall-clock, recorded and inside a **stated** ceiling.

    Set generously and never at ticket 06's 27.2 ms: that sweep measured 2.3x pose
    variance against ticket 04 on the same scene, so a tight bound fails for a reason that
    is not a regression.
    """
    if ceiling_s is None:
        return Criterion(7, "audio wall-clock inside its ceiling", CriterionStatus.NOT_RUN,
                         "no audio_step_ceiling_s in the run config — an unstated "
                         "ceiling is not a bound")
    summary = audit.audio_render_summary()
    if not summary:
        return Criterion(7, "audio wall-clock inside its ceiling", CriterionStatus.NOT_RUN,
                         "no per-step audio timings recorded")
    worst = float(summary["max_s"])
    status = CriterionStatus.PASS if worst <= float(ceiling_s) else CriterionStatus.FAIL
    return Criterion(7, "audio wall-clock inside its ceiling", status,
                     "max {:.4g} s, mean {:.4g} s over {:.0f} steps, ceiling {:.4g} s".format(
                         worst, summary["mean_s"], summary["n"], ceiling_s))


def _env_green(env_report: Optional[Mapping[str, Any]]) -> Criterion:
    """Criterion 8 (ticket 17): ``env_check`` passed, read off the file the run wrote."""
    if env_report is None:
        return Criterion(8, "env_check passed", CriterionStatus.NOT_RUN,
                         "no env_report.json in the run directory")
    probes = env_report.get("probes") or []
    missing = list(env_report.get("missing") or [])
    failed = [p.get("name") for p in probes if p.get("status") != "pass"]
    if missing or failed:
        return Criterion(8, "env_check passed", CriterionStatus.FAIL,
                         "failed {}; missing {}".format(failed or "none", missing or "none"))
    if not probes:
        return Criterion(8, "env_check passed", CriterionStatus.NOT_RUN,
                         "env_report.json records no probes at all")
    return Criterion(8, "env_check passed", CriterionStatus.PASS,
                     "{} probe(s), all pass".format(len(probes)))


def _hermeticity(record: Optional[Mapping[str, Any]], run_dir: str = "") -> Criterion:
    """Criterion 9: this run happened with everything phase 3 deletes already gone.

    It was NOT_RUN by construction until ticket 27, on the reasoning that hermeticity is a
    property of two runs and a judge over one run directory cannot answer it. That was
    half right. The comparison is not what makes the claim — *absence during this run* is,
    and absence is a property of one run's environment that simply was not written down.
    So the gate records it: ``earshot/tools/reset_manifest.py`` verifies the delete set is
    gone immediately before the run and again immediately after, and writes both halves
    into the run directory as ``hermeticity.json``. Without that file this stays NOT_RUN,
    which is what an ordinary run produces — a baseline run cannot read green here by
    being handed to this judge.

    **Two checks, and only one of them lives here.** Is the record complete and about this
    run — that is this function. Does the manifest actually name everything the deletion
    commit removes — that is ``tests/mac/test_reset_manifest.py``, against ``git
    ls-files``. Restating the delete set here would be ticket 24's one-rule-in-two-
    languages, and the copy that drifted would be the one gating the irreversible commit.

    ``complete`` is recomputed from the two halves rather than read off the top-level flag,
    because a record is a small file and a top-level ``true`` is one edit away.
    """
    if not record:
        return Criterion(9, "hermeticity", CriterionStatus.NOT_RUN,
                         "no {} in the run directory — this was an ordinary run, not a "
                         "re-run with the delete set moved out "
                         "(earshot/tools/hermeticity_gate.sh)".format(HERMETICITY_NAME))
    if record.get("schema") != HERMETICITY_SCHEMA:
        return Criterion(9, "hermeticity", CriterionStatus.FAIL,
                         "unknown record schema {!r}".format(record.get("schema")))

    entries = [e.get("path") for e in (record.get("entries") or [])]
    if not entries:
        return Criterion(9, "hermeticity", CriterionStatus.FAIL,
                         "the record names no paths, so it verified nothing")

    recorded_dir = str(record.get("run_dir") or "")
    if recorded_dir and pathlib.Path(recorded_dir).name != pathlib.Path(run_dir).name:
        # Catches a record copied forward from another run — the shape behind this
        # project's own incident of a run directory quoted against another run's numbers.
        # It does not catch a copy between two runs sharing a tag; nothing here could.
        return Criterion(9, "hermeticity", CriterionStatus.FAIL,
                         "record belongs to run {!r}, judged {!r}".format(
                             recorded_dir, run_dir))

    for half in ("before", "after"):
        blob = record.get(half) or {}
        if not blob:
            return Criterion(9, "hermeticity", CriterionStatus.FAIL,
                             "no {!r} verification — absence for the whole run is what "
                             "the two halves are for".format(half))
        still = list(blob.get("still_present") or [])
        if still or not blob.get("complete"):
            return Criterion(9, "hermeticity", CriterionStatus.FAIL,
                             "{} the run, still present: {}".format(
                                 half, ", ".join(still) or "unstated"))
        checked = list(blob.get("checked") or [])
        unchecked = [p for p in entries if p not in checked]
        if unchecked:
            return Criterion(9, "hermeticity", CriterionStatus.FAIL,
                             "{} the run, never checked: {}".format(
                                 half, ", ".join(unchecked)))

    # The box suite on either side of the move, when the gate ran it. A test green with
    # the old trees and red without them is a leak, and it is the only thing here that
    # licenses that word — the gate's first box run called a pre-existing CLAP failure a
    # leak, which it could not have known. A failure on BOTH sides is a sick environment:
    # loud, and deliberately not a hermeticity verdict.
    box = record.get("box_compare")
    if box:
        if not box.get("comparable", True):
            return Criterion(9, "hermeticity", CriterionStatus.FAIL,
                             "the two box-suite runs collected different tests, so "
                             "'no leaks' would be an absence of evidence")
        leaks = list(box.get("leaks") or [])
        if leaks:
            return Criterion(9, "hermeticity", CriterionStatus.FAIL,
                             "{} test(s) pass with the old trees and fail without "
                             "them: {}".format(len(leaks), ", ".join(leaks)))

    detail = "{} path(s) verified absent before and after the run".format(len(entries))
    pre_existing = list((box or {}).get("pre_existing") or [])
    if pre_existing:
        detail += "; no leaks, but {} pre-existing box failure(s) unrelated to the " \
                  "move: {}".format(len(pre_existing), ", ".join(pre_existing))
    return Criterion(9, "hermeticity", CriterionStatus.PASS, detail)


def judge(
    *,
    report: Mapping[str, Any],
    audit: EpisodeAudit,
    env_report: Optional[Mapping[str, Any]] = None,
    run_config: Optional[Mapping[str, Any]] = None,
    hermeticity: Optional[Mapping[str, Any]] = None,
    run_dir: str = "",
) -> SmokeVerdict:
    """§8's nine criteria over one episode's records. Pure.

    ``run_config`` supplies the numbers a criterion is measured *against* rather than
    from — criterion 7's ceiling, and §3.1's ``t_anom`` where a run pinned one — which is
    why they arrive beside the records rather than being read out of them. A gate that
    took its own bound from the thing it is bounding would pass by construction.

    ``t_anom`` is the one place that is now a reconciliation rather than a lookup, and it
    is stated because it weakens the rule above. It is derived per episode, so on an
    unpinned run the configuration does not know it and criterion 4 falls back to the
    record. What it buys back is a second check the old shape could not make: where a run
    *did* pin one, the pin and the record must agree.
    """
    cfg = dict(run_config or {})
    criteria = (
        _live_every_step(audit),
        _context_sound(audit),
        _ir_real(audit),
        _provenance(
            audit,
            t_anom=cfg.get("t_anom"),
            policy=cfg.get("sounding_policy"),
            pre_onset_rms_tol=(cfg.get("audio") or {}).get("pre_onset_rms_tol"),
        ),
        _full_loop(audit),
        _report_populated(report),
        _within_ceiling(audit, ceiling_s=cfg.get("audio_step_ceiling_s")),
        _env_green(env_report),
        _hermeticity(hermeticity, run_dir),
    )
    notes = []
    if audit.localization_arm != "realizable":
        notes.append(
            "§8 says the smoke runs the REALIZABLE arm; this episode ran {!r}, so the "
            "live-audio steering path is unexercised".format(audit.localization_arm)
        )
    if audit.detector_arm == "oracle":
        notes.append(
            "required disclosure: the run used an ORACLE STOP, so goal detection is not "
            "exercised and find numbers are not capability numbers (§8)"
        )
    forward = audit.forward_summary()
    if forward.get("n_forward"):
        notes.append(
            "{:.0f} of {:.0f} forwards collided; {:.2f} m displaced".format(
                forward.get("n_collided", 0.0),
                forward["n_forward"],
                forward.get("total_displacement_m", 0.0),
            )
        )
    return SmokeVerdict(criteria=criteria, notes=tuple(notes))


def judge_run_dir(run_dir: str, *, index: int = 0) -> SmokeVerdict:
    """Read one episode's artefacts out of a run directory and judge them.

    ``env_report.json`` is written flat — ``EnvReport.as_dict()``'s own keys, plus
    ``run_config`` and ``scene`` beside them (``task/runner.run``) — so ``probes`` and
    ``missing`` are read from the top level and criterion 7's ceiling and §3.1's
    ``t_anom`` come out of the nested ``run_config``. That last part is what puts the
    numbers a STOP was gated on in the run record rather than in a shell history, which is
    ticket 25's reason for writing them there at all.
    """
    report, audit = read_episode(run_dir, index)
    root, _ = run_paths(run_dir)
    env_path = root / ENV_REPORT_NAME
    env_report = None
    run_config = None
    if env_path.exists():
        payload = json.loads(env_path.read_text())
        env_report = payload
        run_config = payload.get("run_config")
    # Criterion 9's evidence, written by the hermeticity gate. Absent on an ordinary run,
    # which is the correct reading: NOT_RUN, never green.
    herm_path = root / HERMETICITY_NAME
    hermeticity = json.loads(herm_path.read_text()) if herm_path.exists() else None
    return judge(
        report=report.as_dict(),
        audit=audit,
        env_report=env_report,
        run_config=run_config,
        hermeticity=hermeticity,
        run_dir=str(root),
    )


def episode_indices(run_dir: str) -> Tuple[int, ...]:
    """Every episode index with an audit record under ``run_dir``, in order.

    Read off the filenames rather than off ``summary.json``'s ``n_episodes``, because
    ``write_run_summary`` is written last and a run that crashed part way through has
    episodes on disk and no summary over them. Judging what is there is the point: a gate
    that needs the summary cannot judge the run that failed to produce one.
    """
    _, episodes = run_paths(run_dir)
    if not episodes.is_dir():
        return ()
    found = []
    for path in episodes.glob("ep*.audit.json"):
        try:
            found.append(int(path.name[2:6]))
        except ValueError:  # not one of ours; ignore rather than crash the gate
            continue
    return tuple(sorted(found))


def judge_every_episode(run_dir: str) -> RunVerdict:
    """Judge every episode in ``run_dir`` and tally the nine criteria over all of them."""
    indices = episode_indices(run_dir)
    return tally([judge_run_dir(run_dir, index=i) for i in indices], labels=indices)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, help="a directory `python -m earshot` wrote")
    parser.add_argument("--episode", type=int, default=None,
                        help="judge ONE episode by index; default judges every episode "
                             "in the run and tallies the criteria over all of them")
    args = parser.parse_args(argv)

    if not pathlib.Path(args.run_dir).is_dir():
        print("no such run directory: {}".format(args.run_dir))
        return 2
    if args.episode is not None:
        verdict = judge_run_dir(args.run_dir, index=args.episode)
        print(verdict.summary())
        return 0 if verdict.green else 1
    if not episode_indices(args.run_dir):
        # NOT_RUN, never green: a run directory with no episodes is a gate with nothing
        # to judge, and that has to read differently from nine passes.
        print("no episode records under {} — nothing to judge".format(args.run_dir))
        return 2
    run_verdict = judge_every_episode(args.run_dir)
    print(run_verdict.summary())
    return 0 if run_verdict.green else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
