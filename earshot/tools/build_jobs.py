"""How many compilers this machine can actually hold, rather than how many cores it has.

    python earshot/tools/build_jobs.py          # the number on stdout, the reasoning on stderr

**The bug this exists to kill.** `bootstrap_ss2.sh` set the habitat-sim build's
parallelism straight from `nproc`::

    export CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)"

Two things are wrong with that on this box and either one takes the VM down.

`nproc` honours the CPU **affinity mask** and ignores a cgroup CPU **quota**. In a
container limited to four CPUs on a large node it reports the node's count, so a pod
that is allowed four cores starts sixty-four compilers.

And core count is the wrong budget anyway. habitat-sim pulls in magnum and bullet, whose
heaviest translation units peak in the low gigabytes each, so what bounds a parallel
build is memory. Four `cc1plus` at ~2 GB is ~8 GB, which is already over a modest VM's
head — and a build that OOMs the host does not fail, it kills the machine. `nrun` then
sends no mail at all, so the symptom is silence rather than a red run.

**Memory was never measured on this box.** The inventory behind `docs/race-box-runbook.md`
§1 recorded GLIBC, cores, GPU, disk and mesh coverage; neither `bootstrap_ss2.sh` nor
`env_check` contained a single read of `MemTotal`, a cgroup limit or a `ulimit`. The one
resource that can kill the host was the one nobody wrote down.

It is written down here and in the bootstrap's own log, and deliberately **not** in
`env_check`: ADR-0013 keeps that module a leaf importing nothing intra-package, because
it is what a half-built tree runs to find out what it is missing. The layering gate
rejected the first draft of this fix for exactly that, and it was right — the alternative
was a second copy of the cgroup arithmetic below. This file is `tools/`, which
`NON_AGENT_ROOTS` calls operator tooling, and sizing an operator's build is what it is.

Everything here is pure and takes the file *contents* as arguments, so the whole
cgroup-v1-versus-v2 matrix is exercised on a Mac against injected text (ADR-0014). Only
`main` touches the filesystem.

Run **by path, not by `-m`**: at the point `bootstrap_ss2.sh` needs this number the env is
half-built, and executing the file directly never imports the `earshot` package.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence, Tuple

__all__ = [
    "GB_PER_JOB",
    "affinity_cpus",
    "cpu_quota",
    "memory_limit",
    "build_jobs",
    "main",
]

# provenance: fake — gigabytes of RSS to reserve per concurrent compiler. NOT measured on
# this box; it is a conservative round number for a C++ tree of magnum's and bullet's
# shape, chosen because the cost of being wrong is asymmetric. Too high wastes build
# minutes on a machine that had the headroom. Too low kills the host, loses the run, and
# sends no report. `--gb-per-job` moves it, and a measured value should replace this
# comment the first time someone watches a build's peak RSS.
GB_PER_JOB = 2.0

_BYTES_PER_GB = 1024 ** 3


def affinity_cpus() -> int:
    """CPUs this process may actually run on — what `nproc` reports, and its ceiling.

    ``sched_getaffinity`` where the platform has it (Linux, so the box), falling back to
    ``os.cpu_count``. This is only ever the *upper* bound: it cannot see a cgroup quota,
    which is the half `cpu_quota` supplies.
    """
    getter = getattr(os, "sched_getaffinity", None)
    if getter is not None:
        try:
            return max(1, len(getter(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def cpu_quota(
    cpu_max: Optional[str] = None,
    cfs_quota: Optional[str] = None,
    cfs_period: Optional[str] = None,
) -> Optional[int]:
    """Whole CPUs the cgroup allows, or ``None`` for no quota. Pure.

    Both cgroup generations, because a box can be either and guessing wrong reads as
    "unlimited" — the failure direction that crashes:

    - **v2** ``cpu.max`` is ``"<quota> <period>"``, or ``"max <period>"`` when unset.
    - **v1** ``cpu.cfs_quota_us`` is ``-1`` when unset, against ``cpu.cfs_period_us``.

    Rounded DOWN and floored at 1: a 3.5-CPU quota runs three compilers, never four.
    """
    if cpu_max:
        parts = cpu_max.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                quota, period = int(parts[0]), int(parts[1])
            except ValueError:
                return None
            if quota > 0 and period > 0:
                return max(1, quota // period)
        return None
    if cfs_quota and cfs_period:
        try:
            quota, period = int(cfs_quota.strip()), int(cfs_period.strip())
        except ValueError:
            return None
        if quota > 0 and period > 0:
            return max(1, quota // period)
    return None


def memory_limit(
    meminfo: Optional[str] = None,
    v2_max: Optional[str] = None,
    v1_limit: Optional[str] = None,
) -> Optional[int]:
    """Bytes of RAM this build may use, or ``None`` if nothing could be read. Pure.

    The smaller of what the host has free and any cgroup limit — the cgroup is what kills
    the process, the host is what kills the machine, and a build has to survive both.

    **``MemAvailable``, not ``MemTotal``, wherever the kernel offers it.** Total counts
    memory already spoken for by the OS, the page cache and whatever else is running, so
    budgeting against it hands out compilers for RAM that is not there: on an 8 GiB box
    it authorises four 2 GiB compilers for exactly 8 GiB and leaves nothing for the
    kernel or for `ld`, which on this tree is itself a multi-gigabyte process at the end
    of the build. ``MemAvailable`` is the kernel's own estimate of what can be allocated
    without swapping, which is the question being asked here. ``MemTotal`` is the
    fallback for a kernel too old to publish it (pre-3.14).

    **A cgroup limit at or above ``MemTotal`` is not a limit**, which is how the v1
    sentinel is handled without hardcoding it. v1 writes a huge value
    (``0x7FFFFFFFFFFFF000``-ish) to mean unlimited, and the number varies by kernel and
    page size; comparing against the host's own memory recognises every spelling of it
    and needs no magic constant. That comparison uses ``MemTotal`` deliberately —
    ``MemAvailable`` is always below it, and using the smaller number would let a real
    cgroup limit be mistaken for the sentinel.
    """
    total: Optional[int] = None
    available: Optional[int] = None
    for line in (meminfo or "").splitlines():
        for label, setter in (("MemTotal:", "total"), ("MemAvailable:", "available")):
            if line.startswith(label):
                fields = line.split()
                if len(fields) >= 2:
                    try:
                        value = int(fields[1]) * 1024  # /proc/meminfo is in kB
                    except ValueError:
                        continue
                    if setter == "total":
                        total = value
                    else:
                        available = value
    usable = available if available is not None else total

    limit: Optional[int] = None
    for raw in (v2_max, v1_limit):
        if not raw or raw.strip() == "max":
            continue
        try:
            value = int(raw.strip())
        except ValueError:
            continue
        if value > 0 and (limit is None or value < limit):
            limit = value

    if usable is not None and limit is not None:
        # The sentinel test is against TOTAL; the budget is against USABLE.
        if total is not None and limit >= total:
            return usable
        return min(usable, limit)
    if limit is not None:
        # No meminfo to sanity-check against, so the sentinel cannot be recognised.
        # Reported as-is; `build_jobs` still caps against the CPU budget.
        return limit
    return usable


def build_jobs(
    cpus: int,
    quota: Optional[int],
    mem_bytes: Optional[int],
    *,
    gb_per_job: float = GB_PER_JOB,
) -> Tuple[int, str]:
    """Concurrent compilers this machine can hold, and the sentence explaining it. Pure.

    Returns ``(jobs, reason)`` from one computation rather than leaving a caller to
    re-derive the explanation — a reason that can disagree with its number is worse than
    no reason, and this one is printed into the build log where it will be read months
    later by someone asking why a build took an hour.

    Floored at 1. A one-job build is slow; a build sized for memory the host does not
    have takes the host with it, and only one of those is recoverable.
    """
    ceiling = max(1, int(cpus))
    reason = "{} cpu(s) by affinity".format(ceiling)
    if quota is not None and quota < ceiling:
        reason = "{} cpu(s) by cgroup quota (affinity said {})".format(quota, ceiling)
        ceiling = quota

    if mem_bytes is not None and gb_per_job > 0:
        by_memory = int(mem_bytes // int(gb_per_job * _BYTES_PER_GB))
        gib = mem_bytes / _BYTES_PER_GB
        if by_memory < ceiling:
            return max(1, by_memory), (
                "{} job(s): {:.1f} GiB usable at {:.1f} GiB per compiler, which is "
                "below the {}".format(max(1, by_memory), gib, gb_per_job, reason))
        return ceiling, "{} job(s): {}, and {:.1f} GiB holds it".format(
            ceiling, reason, gib)

    # No memory reading at all. The CPU budget is the only bound left, and this says so
    # rather than presenting a guess as a measurement.
    return ceiling, (
        "{} job(s): {}, memory UNREADABLE so it is not bounding this — if the build "
        "kills the host, this is the line that was wrong".format(ceiling, reason))


def _read(path: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--gb-per-job", type=float, default=GB_PER_JOB,
        help="gigabytes of RSS to reserve per compiler (default {:.1f})".format(
            GB_PER_JOB))
    args = parser.parse_args(argv)

    jobs, reason = build_jobs(
        affinity_cpus(),
        cpu_quota(
            _read("/sys/fs/cgroup/cpu.max"),
            _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
            _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
        ),
        memory_limit(
            _read("/proc/meminfo"),
            _read("/sys/fs/cgroup/memory.max"),
            _read("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
        ),
        gb_per_job=args.gb_per_job,
    )
    # The number alone on stdout so a shell can capture it with `$(...)`; the reasoning
    # on stderr so it lands in the build log beside it without being captured.
    print(jobs)
    sys.stderr.write("  build parallelism: {}\n".format(reason))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
