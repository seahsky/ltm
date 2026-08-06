"""The **only** module in the tree that writes anything (ADR-0013).

```
runs/<tag>/
├── env_report.json          per-run
└── episodes/
    ├── ep0000.agent.json    §5.1 only
    ├── ep0000.audit.json    §5.2, audio_context nested
    └── …
```

Splitting the testimony into its own file buys a property the paper can use: a reviewer
can be handed ``ep0000.agent.json`` and **physically cannot see the answer key**. The
realizability claim stops being a source-level argument and becomes a demonstrable
artefact. ``tests/mac/test_report_artifacts.py`` asserts that on the bytes, not on the
types — the type boundary is ``test_report_boundary.py``'s job and this is the other
half of it.

Everything writes **atomically** (temp file in the destination directory, then
``os.replace``) and **refuses to overwrite**. Both are answers to the same incident: the
audit found committed run directories holding a different run's data, quoted against
numbers they did not come from. A half-written JSON and a silently re-used ``--tag`` are
the two ways that happens, and neither leaves a trace afterwards.

``read_episode`` exists because ticket 26's smoke asserts against these.
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
from typing import Any, Mapping, Tuple, Union

from .agent import AgentReport
from .audit import EpisodeAudit

__all__ = [
    "ArtifactExistsError",
    "run_paths",
    "episode_paths",
    "write_env_report",
    "write_episode",
    "write_run_summary",
    "read_agent_report",
    "read_audit",
    "read_episode",
]

PathLike = Union[str, "os.PathLike"]

ENV_REPORT_NAME = "env_report.json"
EPISODES_DIR = "episodes"

# The whole-run record. The name is not free: `earshot/tools/notify/notify_email.py`
# digests `runs/*/summary.json` into the emailed report, and because the clean room never
# wrote one, every run report since the rebuild has said "No summary.json updated during
# this run — none found". The carried notifier was looking for a file the new tree had
# stopped producing, which is the same carried-verbatim seam that broke `nrun`'s own
# dispatch paths.
RUN_SUMMARY_NAME = "summary.json"


class ArtifactExistsError(RuntimeError):
    """A run directory already holds this artefact, and writing would replace it.

    Raised rather than overwritten. Re-using a ``--tag`` mixes two runs into one
    directory with no record that it happened, which is how a set of committed results
    came to be quoted against numbers from a different run.
    """


def run_paths(run_dir: PathLike) -> Tuple[pathlib.Path, pathlib.Path]:
    """``(run_dir, run_dir/episodes)`` as paths. Creates neither."""
    root = pathlib.Path(run_dir)
    return root, root / EPISODES_DIR


def episode_paths(run_dir: PathLike, index: int) -> Tuple[pathlib.Path, pathlib.Path]:
    """``(agent_json, audit_json)`` for one episode. Creates neither.

    Zero-padded to four digits so a lexical listing is a chronological one — the shape
    every ``runs/`` consumer in this repo already assumes.
    """
    _, episodes = run_paths(run_dir)
    stem = "ep{:04d}".format(int(index))
    return episodes / (stem + ".agent.json"), episodes / (stem + ".audit.json")


def _write_json(path: pathlib.Path, payload: Any, *, overwrite: bool) -> pathlib.Path:
    """Atomic write into ``path``'s own directory, refusing an existing file.

    The temp file is created in the destination directory rather than in ``/tmp`` so
    ``os.replace`` stays a rename within one filesystem — across a boundary it is a
    copy, and a copy is not atomic, which is the failure this exists to prevent.
    """
    if path.exists() and not overwrite:
        raise ArtifactExistsError(
            "{} already exists. Re-using a run tag mixes two runs into one directory "
            "with nothing on disk saying so — pass a fresh tag, or overwrite=True if "
            "replacing it is the intent.".format(path)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            # fsync before the rename: os.replace is atomic with respect to the
            # directory entry, not with respect to the bytes reaching the disk.
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, str(path))
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return path


def write_env_report(
    run_dir: PathLike, report: Mapping[str, Any], *, overwrite: bool = False
) -> pathlib.Path:
    """``runs/<tag>/env_report.json`` — ticket 17's assertion, recorded.

    Takes a **mapping**, not an ``EnvReport``. ``earshot/env_check.py`` sits at layer
    ``()`` — it imports nothing inside the package, because it has to be able to run
    when the tree is otherwise unusable — so this module cannot name its type and
    ``env_check`` cannot call this module. The wiring layer (``task/``, ticket 25) does
    ``write_env_report(run_dir, assert_env().as_dict())``, which is also why
    ``python -m earshot.env_check --strict`` prints rather than writes: one writer.
    """
    root, _ = run_paths(run_dir)
    return _write_json(root / ENV_REPORT_NAME, dict(report), overwrite=overwrite)


def write_run_summary(
    run_dir: PathLike, summary: Mapping[str, Any], *, overwrite: bool = False
) -> pathlib.Path:
    """``runs/<tag>/summary.json`` — what the whole run reached, and what it could not build.

    A mapping for the same reason ``write_env_report`` takes one: ``RunSummary`` lives in
    ``task/``, which imports this module, so naming its type here would invert the layer
    graph. ``task/runner.run`` calls ``write_run_summary(run_dir, summary.as_dict())``.

    Written **last**, after every episode, because a summary is a claim about a completed
    run: a crash half way through should leave the episodes it did write and no summary
    over them, rather than a summary describing a run that did not finish.
    """
    root, _ = run_paths(run_dir)
    return _write_json(root / RUN_SUMMARY_NAME, dict(summary), overwrite=overwrite)


def write_episode(
    run_dir: PathLike,
    index: int,
    agent_report: AgentReport,
    audit: EpisodeAudit,
    *,
    overwrite: bool = False,
) -> Tuple[pathlib.Path, pathlib.Path]:
    """Both artefacts for one episode, testimony first.

    Written together and never separately: an audit with no testimony beside it is an
    answer key with nothing to check, and a testimony with no audit is a claim with no
    evidence. Ordering is testimony-then-audit so a crash between the two leaves the
    file that cannot mislead.

    ``audit.episode_index`` is checked against ``index`` because the two are the same
    fact stored twice, and a mismatch means the caller's loop counter and the record it
    built have come apart — which is the renumbering class of bug that silently dropped
    pairs from an earlier analyzer.
    """
    if int(audit.episode_index) != int(index):
        raise ValueError(
            "episode_index {} on the audit record does not match the index {} it is "
            "being written under".format(audit.episode_index, index)
        )
    agent_path, audit_path = episode_paths(run_dir, index)
    _write_json(agent_path, agent_report.as_dict(), overwrite=overwrite)
    _write_json(audit_path, audit.as_dict(), overwrite=overwrite)
    return agent_path, audit_path


def read_agent_report(path: PathLike) -> AgentReport:
    """Read one testimony file. The reviewer's entry point, and it needs nothing else."""
    with open(str(path), encoding="utf-8") as stream:
        return AgentReport.from_dict(json.load(stream))


def read_audit(path: PathLike) -> EpisodeAudit:
    with open(str(path), encoding="utf-8") as stream:
        return EpisodeAudit.from_dict(json.load(stream))


def read_episode(run_dir: PathLike, index: int) -> Tuple[AgentReport, EpisodeAudit]:
    """Both halves of one episode, for ticket 26's smoke assertions."""
    agent_path, audit_path = episode_paths(run_dir, index)
    return read_agent_report(agent_path), read_audit(audit_path)
