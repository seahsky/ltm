"""What ticket 10 phase 3 deletes, as data — and the evidence that a run ran without it.

    python -m earshot.tools.reset_manifest --print-paths
    python -m earshot.tools.reset_manifest --verify-absent
    python -m earshot.tools.reset_manifest --write-record --run-dir runs/<tag> ...

Two jobs, and they are the same list, which is the point.

**The delete list.** Ticket 27 names it in prose. Prose is what the deletion commit is
executed from, and a list that lives only in a ticket cannot be checked against the tree
it describes. Here every entry carries the number of tracked files it covered when it was
audited, so `test_reset_manifest.py` goes red when the tree moves under it and the reset
is re-audited by a human rather than widening silently. An irreversible commit is the one
place where "the list changed and nobody looked" is unacceptable.

**The hermeticity evidence.** Ticket 10 phase 2 requires the smoke green *with the old
trees moved out*, because "the smoke is green" and "the smoke is green without the old
trees present" are different claims and only the second licenses the delete. That is a
property of a run's *environment*, not of its records — so unless the run directory
carries proof, criterion 9 is the operator's word, and a judge reading a run directory
cannot tell a hermetic run from an ordinary one. It would pass on the baseline run.

So the gate verifies absence **twice, bracketing the run**, and writes the result into the
run directory as `hermeticity.json`. Both verifications must be complete: the trees were
gone before the first step and still gone after the last, which a single check at the top
cannot say. `task/smoke.py`'s criterion 9 reads that record and nothing else.

**Ticket 27's list is widened here, deliberately, and this is the one place it differs
from the ticket.** Phase 2 as written moves out `embodied_memory/` and `dialogue_memory/`
only, while phase 3 deletes three more groups — `scripts/`, the MSC readmes and drivers,
and `data/msc/`. The gate's own argument (a static grep misses `importlib`, a `sys.path`
append, a hardcoded path in a config, a data path only the old tree knew about) applies
to every one of them, and the narrow version is not hypothetical: ticket 27 found the
carried `earshot/tools/notify/` trio still executing `$REPO_ROOT/scripts/notify_email.py`,
inside the group phase 2 did not move. A gate that covers two of five groups would have
been green and the deletion would have broken the box's own launcher.

`tools/` is outside the layer graph and outside the running program — ADR-0013's tree
walks `audio/`, `agent/`, `sim/`, `task/`, `report/`, and this is a build-and-reset tool
that no live module imports. `task/smoke.py` does **not** import it: the record crosses
that boundary as JSON, the same way `env_report.json` does.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "ManifestEntry",
    "DELETE_SET",
    "KEEP_PINS",
    "RECORD_NAME",
    "delete_paths",
    "verify",
    "build_record",
    "main",
]

RECORD_NAME = "hermeticity.json"

# `pathlib.Path | str` is a 3.9 problem in a runtime-evaluated annotation; the alias keeps
# the signatures readable without reaching for it.
PathLike = Any


@dataclass(frozen=True)
class ManifestEntry:
    """One path phase 3 deletes.

    ``tracked_files`` is the ``git ls-files`` count at audit time (2026-08-05), and it is
    a pin rather than a comment — see the module docstring.
    """

    path: str
    tracked_files: int
    reason: str


# Audited 2026-08-05 against `git ls-files`, on the commit ticket 26 resolved at.
DELETE_SET: Tuple[ManifestEntry, ...] = (
    ManifestEntry(
        "embodied_memory", 163,
        "the tree the clean room replaces; every live behaviour it had is ported or "
        "deliberately dropped (tickets 07, 10, 21-25)",
    ),
    ManifestEntry(
        "dialogue_memory", 30,
        "15 .py and 15 tracked .pyc, one of which (evaluation) has had no .py for a "
        "long time. The retrieval stack is vendored inert at earshot/reference/memory/",
    ),
    ManifestEntry(
        "scripts", 51,
        "deleted WHOLESALE. Ticket 27's 'except the three notify files already carried' "
        "names why nothing is lost, not an exception to the delete: the trio lives at "
        "earshot/tools/notify/. Keeping the originals would leave two divergent copies, "
        "and after ticket 27's path fix the copies here are the stale ones. The runbook "
        "extraction (ticket 14) is what makes the other 48 safe to drop",
    ),
    ManifestEntry(
        "README_LTM_MSC_EVAL.md", 1, "how to run the deleted dialogue path",
    ),
    ManifestEntry(
        "README_MSC_EVAL.md", 1, "how to run the deleted dialogue path",
    ),
    ManifestEntry(
        "run_msc_baseline.sh", 1, "driver for the deleted dialogue path",
    ),
    ManifestEntry(
        "run_msc_full_eval.sh", 1, "driver for the deleted dialogue path",
    ),
    ManifestEntry(
        "run_msc_quick_eval.sh", 1, "driver for the deleted dialogue path",
    ),
    ManifestEntry(
        "data/msc", 4,
        "the MSC dataset. `data/` is gitignored but these four .json are tracked, so "
        "they are a source-tree deletion like the rest and not a 1.2 GB data call",
    ),
)

# Named survivors. Not the whole keep list — the repo is mostly keep — but the entries a
# mis-typed delete path would take with it, and the ones ticket 27 lists by name.
# `test_reset_manifest.py` asserts each exists and none sits inside a delete entry.
KEEP_PINS: Tuple[str, ...] = (
    "earshot",
    "docs",
    "docs/adr",
    "docs/race-box-runbook.md",
    "PHASE2_ABLATION_REPORT.md",
    "ICRA2027_PAPER_DRAFT.md",
    "CONTEXT.md",
    "Research Proposal_Embodied Agent.md",
    "MSC_BENCHMARK_REPORT.md",
    "models/README.md",
    ".scratch/ss2-clean-room",
)


def delete_paths() -> Tuple[str, ...]:
    """The delete set as plain repo-relative paths, for the shell to iterate."""
    return tuple(entry.path for entry in DELETE_SET)


def verify(root: PathLike, *, when: str) -> Dict[str, Any]:
    """Is every delete-set path absent from ``root`` right now?

    Returns the evidence, not a bool: which paths were checked, which were still present,
    and when. A gate that recorded only "yes" would leave a later reader with the same
    problem this module exists to fix.
    """
    base = pathlib.Path(root)
    present = [e.path for e in DELETE_SET if (base / e.path).exists()]
    return {
        "when": when,
        "at": time.time(),
        "checked": list(delete_paths()),
        "still_present": present,
        "complete": not present,
    }


def build_record(
    *,
    run_dir: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    commit: str = "unknown",
    holding_dir: str = "",
) -> Dict[str, Any]:
    """The `hermeticity.json` payload: two bracketing verifications and their provenance.

    ``run_dir`` is carried so criterion 9 can refuse a record that belongs to a different
    run. This project has an incident behind that check — run directories quoted against
    another run's numbers — and a hermeticity record is exactly the kind of small file
    that gets copied forward to make a gate go green.
    """
    return {
        "schema": "earshot.hermeticity/1",
        "run_dir": run_dir,
        "commit": commit,
        "holding_dir": holding_dir,
        "entries": [
            {"path": e.path, "tracked_files": e.tracked_files, "reason": e.reason}
            for e in DELETE_SET
        ],
        "before": dict(before),
        "after": dict(after),
        "complete": bool(before.get("complete")) and bool(after.get("complete")),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".", help="repo root to check against")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print-paths", action="store_true",
                      help="the delete set, one repo-relative path per line")
    mode.add_argument("--verify-absent", action="store_true",
                      help="exit 0 iff every delete-set path is absent; prints the evidence")
    mode.add_argument("--write-record", action="store_true",
                      help="merge two --verify-absent blobs into the run's hermeticity.json")
    parser.add_argument("--when", default="before", help="label for --verify-absent")
    parser.add_argument("--run-dir", help="run directory, for --write-record")
    parser.add_argument("--before", help="path to the pre-run --verify-absent blob")
    parser.add_argument("--after", help="path to the post-run --verify-absent blob")
    parser.add_argument("--commit", default="unknown")
    parser.add_argument("--holding-dir", default="")
    args = parser.parse_args(argv)

    if args.print_paths:
        for path in delete_paths():
            print(path)
        return 0

    if args.verify_absent:
        evidence = verify(args.root, when=args.when)
        print(json.dumps(evidence, indent=2))
        if not evidence["complete"]:
            print(
                "STILL PRESENT: {}".format(", ".join(evidence["still_present"])),
                file=sys.stderr,
            )
            return 1
        return 0

    # --write-record
    missing: List[str] = [
        name for name, value in (("--run-dir", args.run_dir), ("--before", args.before),
                                 ("--after", args.after)) if not value
    ]
    if missing:
        parser.error("--write-record needs {}".format(", ".join(missing)))
    before = json.loads(pathlib.Path(args.before).read_text(encoding="utf-8"))
    after = json.loads(pathlib.Path(args.after).read_text(encoding="utf-8"))
    record = build_record(
        run_dir=args.run_dir, before=before, after=after,
        commit=args.commit, holding_dir=args.holding_dir,
    )
    out = pathlib.Path(args.run_dir) / RECORD_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("wrote {} (complete={})".format(out, record["complete"]))
    return 0 if record["complete"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
