"""Task spec §8's nine acceptance criteria, judged off a run directory.

    python -m earshot.task.smoke --run-dir runs/<tag>

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
    "judge",
    "judge_run_dir",
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


def _provenance(audit: EpisodeAudit, *, t_anom: Optional[int]) -> Criterion:
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
    return Criterion(4, "provenance did not raise", CriterionStatus.PASS,
                     "onset step {}, {} pre-onset readings, t_anom {}".format(
                         onset.onset_step, onset.n_pre_onset_readings, int(effective)))


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

    return Criterion(9, "hermeticity", CriterionStatus.PASS,
                     "{} path(s) verified absent before and after the run".format(
                         len(entries)))


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
        _provenance(audit, t_anom=cfg.get("t_anom")),
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, help="a directory `python -m earshot` wrote")
    parser.add_argument("--episode", type=int, default=0, help="episode index to judge")
    args = parser.parse_args(argv)

    if not pathlib.Path(args.run_dir).is_dir():
        print("no such run directory: {}".format(args.run_dir))
        return 2
    verdict = judge_run_dir(args.run_dir, index=args.episode)
    print(verdict.summary())
    return 0 if verdict.green else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
