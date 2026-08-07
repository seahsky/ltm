"""What `nrun` left running on the box — discovered from the process table, nothing else.

    python -m earshot.tools.nrun_tasks           # the table an operator reads
    python -m earshot.tools.nrun_tasks --json    # the same, for `./earshot.sh`

`nrun` (in `earshot/tools/notify/notify-run.sh`) is a shell function that `nohup`s
`bash notify-run.sh <cmd>` into the background and **records nothing** — no pid file, no
registry, only a pid echoed to a terminal that is usually gone by the time you care. So
discovery here is a scan of `ps`, deliberately: a registry would only ever see tasks
launched by a `nrun` that had been modified to write one, which is every task except the
ones already running when you go looking, and the ones started from the other SSH session.
The scan sees all of them, and it needs no change to a file whose comment block records
what carrying it wrong already cost.

Two things the scan gets exactly right that a timestamp guess would not:

**The log.** `ps` cannot say which `runs/nrun-<ts>.out` belongs to which pid, and matching
on the timestamp in the name breaks the moment two tasks start in the same minute. But the
`.out` file *is* the process's stdout, so `/proc/<pid>/fd/1` is the answer rather than an
inference. Linux-only, which this whole tool is.

**Who dies.** `nrun` does not `setsid`. Under an interactive shell, job control gives the
background job its own process group and a group kill is complete — it reaps the wrapper,
the `tee`, and the python child holding the GPU. Called from a *non-interactive* shell
there is no job control, the job inherits its caller's group, and that same group kill
takes the caller with it. `group_kill_is_safe()` is the check that tells those two apart
(`pgid == pid` ⇔ the wrapper leads its own group); when it is false the caller must walk
the pid tree instead. A kill that reaps your login shell and a kill that reports success
while the CUDA child survives are the two failure modes this module exists to prevent, and
both are unit-tested against captured `ps` output.

Everything above is a pure function over text so it is testable on a Mac with no box; the
only IO is `run_ps()` and `read_log_tail()`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "PS_ARGS",
    "ProcRow",
    "NrunTask",
    "parse_ps",
    "wrapper_rows",
    "wrapped_command",
    "descendants",
    "build_tasks",
    "group_kill_is_safe",
    "kill_targets",
    "resolve_log_path",
    "scrape_exit_code",
    "summarize_log",
    "format_tasks",
    "format_plan",
    "main",
]

# Fixed-arity fields first, `args` last, because only `args` may contain spaces. `lstart`
# is always exactly five tokens ("Thu Aug  7 18:01:22"), so the split below is arity-safe
# rather than heuristic. `etimes` (whole seconds) is asked for instead of `etime` so the
# elapsed time is a number here and formatted once, in one place.
PS_ARGS = ("ps", "-eo", "pid=,ppid=,pgid=,etimes=,lstart=,args=")

_N_FIXED = 4
_N_LSTART = 5

# The wrapper is `bash <somewhere>/notify-run.sh <the real command>`. The menu script and
# the `pgrep` that finds it also carry the string, so a match is not enough — the row must
# be a *wrapper*, i.e. have a wrapped command after the script path.
_SCRIPT = "notify-run.sh"

# `notify_email.py` prints its Resend subject line into the wrapper's stdout, and the
# subject carries the exit code: "[notify] email sent to a@b (✅ [ltm] tag — exit 0 (2m))".
# That is the ONLY place a finished run's exit code reaches the .out file — notify-run.sh
# itself never prints it. So an unconfigured emailer means an unknown status, and this
# returns None for it rather than guessing zero. A run whose status could not be read is
# not a run that passed.
_EXIT_RE = re.compile(r"exit (-?\d+)")


@dataclasses.dataclass(frozen=True)
class ProcRow:
    """One line of `ps`, parsed. `args` is the full command line."""

    pid: int
    ppid: int
    pgid: int
    etimes: int
    started: str
    args: str


@dataclasses.dataclass(frozen=True)
class NrunTask:
    """A live `nrun` wrapper and everything it is holding open."""

    pid: int
    pgid: int
    etimes: int
    started: str
    command: str
    children: Tuple[int, ...]
    log: Optional[str] = None


def parse_ps(text: str) -> List[ProcRow]:
    """Parse `PS_ARGS` output. Lines that do not have the fixed arity are skipped."""
    rows: List[ProcRow] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split(None, _N_FIXED + _N_LSTART)
        if len(parts) < _N_FIXED + _N_LSTART + 1:
            continue
        try:
            pid, ppid, pgid, etimes = (int(p) for p in parts[:_N_FIXED])
        except ValueError:
            continue
        rows.append(ProcRow(
            pid=pid,
            ppid=ppid,
            pgid=pgid,
            etimes=etimes,
            started=" ".join(parts[_N_FIXED:_N_FIXED + _N_LSTART]),
            args=parts[-1],
        ))
    return rows


def _tokens(args: str) -> List[str]:
    return args.split()


def _script_index(args: str) -> int:
    """Index of the `notify-run.sh` token in `args`, or -1."""
    for i, tok in enumerate(_tokens(args)):
        if tok.endswith(_SCRIPT):
            return i
    return -1


def wrapper_rows(rows: Sequence[ProcRow]) -> List[ProcRow]:
    """The rows that are a `nrun` wrapper — not the menu, not the grep that found it."""
    out = []
    for row in rows:
        i = _script_index(row.args)
        # i < 0: not a wrapper. i == last: `notify-run.sh` with no command, which is the
        # usage-error path, not a run. Either way there is nothing to report or kill.
        if i >= 0 and i < len(_tokens(row.args)) - 1:
            out.append(row)
    return out


def wrapped_command(row: ProcRow) -> str:
    """`bash /x/notify-run.sh python -m earshot ...` → `python -m earshot ...`."""
    i = _script_index(row.args)
    if i < 0:
        return row.args
    return " ".join(_tokens(row.args)[i + 1:])


def descendants(rows: Sequence[ProcRow], pid: int) -> Tuple[int, ...]:
    """Every pid below `pid`, transitively, in `ps` order."""
    by_parent: Dict[int, List[int]] = {}
    for row in rows:
        by_parent.setdefault(row.ppid, []).append(row.pid)
    found: List[int] = []
    seen = {pid}
    stack = list(by_parent.get(pid, ()))
    while stack:
        child = stack.pop(0)
        # A cycle is impossible in a real process tree, but `ps` output is a snapshot
        # taken over time and this must terminate on anything.
        if child in seen:
            continue
        seen.add(child)
        found.append(child)
        stack.extend(by_parent.get(child, ()))
    return tuple(found)


def build_tasks(
    rows: Sequence[ProcRow],
    resolve_log: Optional[Callable[[int], Optional[str]]] = None,
) -> List[NrunTask]:
    """The live tasks, oldest first — the order they will finish in."""
    resolve = resolve_log if resolve_log is not None else resolve_log_path
    tasks = [
        NrunTask(
            pid=row.pid,
            pgid=row.pgid,
            etimes=row.etimes,
            started=row.started,
            command=wrapped_command(row),
            children=descendants(rows, row.pid),
            log=resolve(row.pid),
        )
        for row in wrapper_rows(rows)
    ]
    return sorted(tasks, key=lambda t: (-t.etimes, t.pid))


def group_kill_is_safe(task: NrunTask) -> bool:
    """True ⇔ the wrapper leads its own process group, so `kill -- -PGID` reaps it alone.

    False means `nrun` was called from a non-interactive shell and the job shares its
    caller's group: a group kill there would take the caller down with it.
    """
    return task.pid == task.pgid


def kill_targets(task: NrunTask) -> Tuple[str, Tuple[int, ...]]:
    """`("group", (pgid,))` when that is safe, else `("tree", (pid, *children))`.

    The tree fallback is deliberately not "just the wrapper": killing the wrapper alone
    orphans the python child, which keeps the GPU and disappears from this listing — a
    kill that reports success and freed nothing.
    """
    if group_kill_is_safe(task):
        return ("group", (task.pgid,))
    return ("tree", (task.pid,) + task.children)


def resolve_log_path(
    pid: int,
    readlink: Optional[Callable[[str], str]] = None,
) -> Optional[str]:
    """The file `pid`'s stdout points at, via `/proc/<pid>/fd/1`, or None.

    None for a pipe, a socket, a tty or an unreadable `/proc` entry — all of which are
    "this is not a `nrun` .out file", and none of which are worth guessing about.
    """
    link = readlink if readlink is not None else os.readlink
    try:
        target = link("/proc/{}/fd/1".format(pid))
    except OSError:
        return None
    if not target.startswith("/") or target.startswith("/dev/"):
        return None
    return target


def scrape_exit_code(text: str) -> Optional[int]:
    """The exit code of a finished run, from its `.out` tail, or None if unrecorded."""
    for line in reversed(text.splitlines()):
        if "[notify]" not in line:
            continue
        match = _EXIT_RE.search(line)
        if match:
            return int(match.group(1))
    return None


def summarize_log(name: str, mtime: float, text: str) -> Dict[str, object]:
    """One finished-run row: what it was, when it stopped, and how — if that was written."""
    exit_code = scrape_exit_code(text)
    last = ""
    for line in reversed(text.splitlines()):
        if line.strip():
            last = line.strip()
            break
    return {
        "name": name,
        "mtime": mtime,
        "exit_code": exit_code,
        # An unread status is reported as unknown, never as success: notify-run.sh does
        # not print the exit code, so a run with the emailer unconfigured has none to read.
        "status": "?" if exit_code is None else ("ok" if exit_code == 0 else "FAILED"),
        "last_line": last,
    }


def format_elapsed(seconds: int) -> str:
    hours, rem = divmod(max(0, seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return "{}h{:02d}m".format(hours, minutes)
    if minutes:
        return "{}m{:02d}s".format(minutes, secs)
    return "{}s".format(secs)


def format_tasks(tasks: Sequence[NrunTask]) -> str:
    if not tasks:
        return "no nrun tasks running."
    lines = ["  #    pid   elapsed  command",
             "  " + "-" * 70]
    for i, task in enumerate(tasks, 1):
        lines.append("  {:<3} {:>6}  {:>8}  {}".format(
            i, task.pid, format_elapsed(task.etimes), task.command[:44]))
        lines.append("       {:>6}  {:>8}  log: {}".format(
            "", "", task.log or "(stdout is not a file)"))
        if not group_kill_is_safe(task):
            lines.append("       {:>6}  {:>8}  ! shares pgid {} with its caller — "
                         "kill walks the pid tree".format("", "", task.pgid))
    return "\n".join(lines)


def format_plan(task: NrunTask) -> str:
    """One task as `KEY value` lines — the seam `./earshot.sh` reads before killing.

    Line-oriented and key-first so bash can `read -r key rest` it. The alternative, having
    the menu pick fields out of the JSON, means either a fragile grep or an `eval`, and the
    string most likely to be mangled by either is the command line printed in the "stop
    this?" confirmation — the one string that must not lie.
    """
    mode, targets = kill_targets(task)
    return "\n".join([
        "PID {}".format(task.pid),
        "PGID {}".format(task.pgid),
        "STARTED {}".format(task.started),
        "ELAPSED {}".format(format_elapsed(task.etimes)),
        "COMMAND {}".format(task.command),
        "LOG {}".format(task.log or ""),
        "MODE {}".format(mode),
        "TARGETS {}".format(" ".join(str(t) for t in targets)),
    ])


# --- the thin IO edge -------------------------------------------------------


def run_ps(runner: Optional[Callable[[Sequence[str]], str]] = None) -> List[ProcRow]:
    def _default(argv: Sequence[str]) -> str:
        return subprocess.run(
            list(argv), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            check=False,
        ).stdout.decode("utf-8", "replace")

    return parse_ps((runner or _default)(PS_ARGS))


def recent_logs(log_dir: str, limit: int = 5, tail_bytes: int = 8192) -> List[Dict[str, object]]:
    """The last `limit` finished `nrun-*.out` files, newest first."""
    base = pathlib.Path(log_dir)
    if not base.is_dir():
        return []
    paths = sorted(base.glob("nrun-*.out"), key=lambda p: p.stat().st_mtime, reverse=True)
    rows = []
    for path in paths[:limit]:
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - tail_bytes))
            text = handle.read().decode("utf-8", "replace")
        rows.append(summarize_log(str(path), path.stat().st_mtime, text))
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true",
                        help="emit tasks and recent logs as JSON")
    parser.add_argument("--count", action="store_true",
                        help="print how many tasks are running, and nothing else")
    parser.add_argument("--plan", type=int, metavar="N",
                        help="print task N (1-based) as KEY-value lines, incl. its kill "
                             "mode and targets; exit 1 if there is no task N")
    parser.add_argument("--log-dir", default="runs",
                        help="where nrun-*.out files live (default: runs)")
    parser.add_argument("--recent", type=int, default=5,
                        help="how many finished logs to list (default: 5)")
    args = parser.parse_args(argv)

    if sys.platform != "linux":
        print("nrun_tasks reads /proc and Linux `ps` — it is box-only.", file=sys.stderr)
        return 2

    tasks = build_tasks(run_ps())

    if args.count:
        print(len(tasks))
        return 0

    if args.plan is not None:
        if not 1 <= args.plan <= len(tasks):
            print("no task {} (there are {})".format(args.plan, len(tasks)), file=sys.stderr)
            return 1
        print(format_plan(tasks[args.plan - 1]))
        return 0

    recent = recent_logs(args.log_dir, args.recent) if not tasks else []

    if args.json:
        print(json.dumps({
            "tasks": [dataclasses.asdict(t) for t in tasks],
            "recent": recent,
        }, indent=2))
        return 0

    print(format_tasks(tasks))
    if recent:
        print("\nrecent finished runs:")
        for row in recent:
            print("  {:<7} {}".format(row["status"], row["name"]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
