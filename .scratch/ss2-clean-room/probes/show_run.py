"""What a run directory actually says, and whether its own readers can read it back.

The first end-to-end episode ran on 2026-08-05 and printed a 6/6 funnel. The console
trace disagrees with that: six INVESTIGATE entries at steps 30, 72, 114, 156, 198, 240
with resumes exactly 40 steps later — ``ControllerConfig.investigate_max_steps`` — which
is the abort path, not arrival. So this exists to read the artefacts rather than the
progress lines, and to separate what the record states from what the ladder inferred.

Three things it does, in order of what they settle:

  1. **Dumps both artefacts.** Raw JSON, so a missing or misspelled field shows as
     missing rather than as a default. ``EpisodeAudit.steps`` is summarised instead —
     250 per-step records would bury everything else — with the transitions sampled.
  2. **Round-trips them through the tree's own readers.** ``read_episode`` is what
     ticket 26's smoke will use, and a file that writes but does not parse is a failure
     the dump alone would not show. Reported, never fatal: the dump above it is the
     useful part when parsing is what broke.
  3. **Recomputes the funnel from the record.** ``_funnel_stage`` is a monotone ladder
     resting on "an episode that resumed necessarily investigated"
     (``task/runner.py:787``), and ``controller.py:321`` reaches RESUME with
     ``investigate_aborted=True`` and ``investigated`` still false. Where the two
     disagree, the printed stage over-credits. This says so out loud rather than
     leaving it to be noticed.

Derived numbers are labelled ``derived:`` — they are this probe's arithmetic over the
record, not the tree's own accounting, and nothing here should be quoted as if the
runner had reported it.

    python .scratch/ss2-clean-room/probes/show_run.py [run_dir]
"""

import json
import pathlib
import sys
import traceback

DEFAULT_RUN_DIR = "runs/ss2-first-episode-3"

# `EpisodeAudit.steps` — summarised rather than dumped. Named here so a rename shows up
# as "no step array found" instead of a silently empty summary.
STEP_KEY = "steps"
SAMPLE_EDGE = 3


def rule(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def dump(obj, indent=2):
    print(json.dumps(obj, indent=indent, default=str, sort_keys=False))


def summarise_steps(steps):
    """Counts, the transitions, and a sample — never all 250 records."""
    print("  n_records          : {}".format(len(steps)))
    if not steps:
        return

    playing = [s for s in steps if s.get("source_playing")]
    visible = [s for s in steps if s.get("source_is_visible")]
    actions = {}
    for record in steps:
        actions[record.get("action")] = actions.get(record.get("action"), 0) + 1
    rms = [float(s["measured_rms"]) for s in steps if s.get("measured_rms") is not None]
    lateral = [s.get("lateral_sign") for s in steps]

    print("  source_playing     : {} of {}".format(len(playing), len(steps)))
    print("  source_is_visible  : {} true, {} null".format(
        len(visible), sum(1 for s in steps if s.get("source_is_visible") is None)))
    print("  actions            : {}".format(
        ", ".join("{}={}".format(k, v) for k, v in sorted(
            actions.items(), key=lambda kv: (kv[0] is None, kv[0])))))
    print("  lateral_sign       : -1={} 0={} +1={} null={}".format(
        lateral.count(-1), lateral.count(0), lateral.count(1),
        sum(1 for value in lateral if value is None)))
    if rms:
        print("  measured_rms       : min {:.6g} / mean {:.6g} / max {:.6g}".format(
            min(rms), sum(rms) / len(rms), max(rms)))
        # The pre-onset readings are the bed alone (§3.1). A spread here means something
        # reached the sensor before the source did.
        pre = [float(s["measured_rms"]) for s in steps
               if not s.get("source_playing") and s.get("measured_rms") is not None]
        if pre:
            print("  derived: pre-onset rms spread: {:.6g} .. {:.6g} over {} step(s)".format(
                min(pre), max(pre), len(pre)))

    print("  first/last records :")
    for record in list(steps)[:SAMPLE_EDGE] + ["..."] + list(steps)[-SAMPLE_EDGE:]:
        print("    {}".format(record))


def stage_label(value):
    """``6`` -> ``6 (PRIMARY_RESUMED)``.

    ``EpisodeAudit.as_dict`` serialises the stage as its integer, which is the right
    thing on disk — the ladder's ORDER is the meaning, and a name would have to be
    re-mapped to compare two stages — and unreadable in a dump. Resolved through the
    tree's own enum when it is importable, printed raw when it is not.
    """
    try:
        from earshot.report.audit import FunnelStage

        return "{} ({})".format(value, FunnelStage(int(value)).name)
    except Exception:
        return repr(value)


def recompute_funnel(report, audit):
    """The ladder, cross-read against the flags it was built from.

    **This section used to accuse the ladder and no longer does.** It was written while
    ``_funnel_stage`` read each flag independently, so the budget abort — which sets
    ``resumed`` with ``investigated`` False — promoted a stage-4 episode to 6 and the
    first box run printed 6/6 over five aborted detours. The ladder is nesting-enforced
    now and stage 6 requires reaching the source, so an ``aborted`` + ``resumed`` pair is
    the ordinary shape of a detour that gave up, not a symptom. Kept because the pair is
    still worth reading beside the stage, and because a probe that keeps crying wolf
    teaches the reader to stop looking.

    ``investigated`` is in neither artefact, so this cross-reads what is there rather than
    recomputing the stage.
    """
    rule("derived: the funnel, cross-read against the record")
    stated = audit.get("funnel_stage")
    aborted = report.get("investigate_aborted")
    resumed = report.get("resumed")
    onset = (audit.get("onset") or {}).get("onset_step")
    t_anom = audit.get("t_anom")

    print("  stated funnel_stage      : {}".format(stage_label(stated)))
    print("  report.investigate_aborted: {}".format(aborted))
    print("  report.resumed            : {}".format(resumed))
    print("  onset.onset_step          : {}".format(onset))
    print("  audit.t_anom              : {}  (derived per episode)".format(t_anom))

    n_steps = len(audit.get("steps") or [])
    if t_anom is not None and n_steps and n_steps <= int(t_anom) + 1:
        print(
            "\n  THE ANOMALY ARRIVED TOO LATE TO BE ONE. The episode ran {} steps and the\n"
            "  source started sounding at {}, so there was no search left to interrupt.\n"
            "  `derive_t_anom` exists to make this impossible; seeing it means the find\n"
            "  ended earlier than the straight-line bound said it could.".format(
                n_steps, t_anom)
        )
    elif stated is not None and int(stated) >= 6:
        print("\n  Stage 6: the detour reached the source and the primary resumed.")
    elif aborted and resumed:
        print(
            "\n  The detour hit its step budget and resumed without reaching the source.\n"
            "  Stage 4 is the truthful reading and the ladder now reports it as one."
        )
    elif aborted:
        print("\n  Detour aborted and did not resume — stage 4 at most.")
    elif resumed:
        print("\n  Resumed with no abort recorded.")
    else:
        print("\n  Neither flag set; nothing to reconcile.")


def show_episode(path_agent, path_audit):
    report = json.loads(path_agent.read_text(encoding="utf-8"))
    audit = json.loads(path_audit.read_text(encoding="utf-8"))

    rule("§5.1 AgentReport (testimony) — {}".format(path_agent.name))
    dump(report)

    rule("§5.2 EpisodeAudit (analyst) — {}".format(path_audit.name))
    steps = audit.get(STEP_KEY)
    without_steps = {k: v for k, v in audit.items() if k != STEP_KEY}
    dump(without_steps)

    print("\n  {}:".format(STEP_KEY))
    if steps is None:
        print("  NO STEP ARRAY under {!r}. Keys present: {}".format(
            STEP_KEY, sorted(audit)))
    else:
        summarise_steps(steps)

    recompute_funnel(report, audit)

    # §5 keeps the two artefacts disjoint: the agent testifies to what it can know, the
    # audit holds the privileged watches. A key in both is a leak, and it is cheaper to
    # notice here than in a paper table.
    shared = sorted(set(report) & set(audit))
    rule("derived: testimony/audit key overlap")
    print("  {}".format(shared if shared else "none — disjoint, as §5 requires"))


def main():
    run_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUN_DIR)
    rule("run directory: {}".format(run_dir))
    if not run_dir.exists():
        print("  does not exist")
        return 1
    for path in sorted(run_dir.rglob("*")):
        print("  {:>10}  {}".format(
            path.stat().st_size if path.is_file() else "<dir>",
            path.relative_to(run_dir)))

    env_report = run_dir / "env_report.json"
    if env_report.exists():
        rule("env_report.json — the resolved environment AND the resolved config")
        dump(json.loads(env_report.read_text(encoding="utf-8")))

    # `report.artifacts.episode_paths` names these `epNNNN.agent.json` /
    # `epNNNN.audit.json`. Paired on that stem rather than on listing order, and a
    # half-written episode is reported rather than skipped: one artefact without the
    # other is exactly the state a crash between the two writes leaves behind.
    episodes_dir = run_dir / "episodes"
    agents = sorted(episodes_dir.glob("ep*.agent.json")) if episodes_dir.exists() else []
    if not agents:
        print("\n  no episode artefacts under {}".format(episodes_dir))
        return 1
    for agent_path in agents:
        audit_path = agent_path.with_name(
            agent_path.name.replace(".agent.json", ".audit.json")
        )
        if not audit_path.exists():
            rule("{} has no audit beside it".format(agent_path.name))
            print("  the testimony was written and the audit was not — a crash between "
                  "the two writes, or a rename that touched one of them")
            continue
        show_episode(agent_path, audit_path)

    rule("round-trip through the tree's own readers (what ticket 26's smoke uses)")
    # The import is separated from the read so a missing PYTHONPATH does not get
    # reported as an unparseable artefact. One is this probe's environment; the other is
    # a finding.
    try:
        from earshot.report.artifacts import read_episode
    except ImportError as exc:
        print("  SKIPPED — `earshot` is not importable here ({}). Run from the repo "
              "root, or export PYTHONPATH=$PWD. This says nothing about the "
              "artefacts.".format(exc))
        return 0
    try:
        report, audit = read_episode(str(run_dir), 0)
        print("  read_episode(…, 0) OK")
        print("  funnel_stage      : {} ({})".format(
            int(audit.funnel_stage), audit.funnel_stage.name))
        print("  dist_at_stop      : {}".format(audit.dist_at_stop))
        print("  n step records    : {}".format(len(audit.steps)))
        print("  primary_completed : {}".format(report.primary_completed))
        print("  investigate_aborted: {}".format(report.investigate_aborted))
    except Exception:
        # Formatted onto stdout rather than `print_exc`'s stderr: this output gets
        # pasted, and a traceback that lands in the other stream arrives out of order or
        # not at all.
        print("  READ FAILED — the files exist but the tree cannot parse them back:")
        for line in traceback.format_exc().splitlines():
            print("    " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
