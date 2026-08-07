"""Sizing a build against memory rather than cores — the arithmetic that crashed the box.

`bootstrap_ss2.sh` set `CMAKE_BUILD_PARALLEL_LEVEL` from `nproc`, and `nproc` cannot see
a cgroup CPU quota. The whole cgroup matrix is text on disk, so every case below is
exercised against injected content on a Mac: v1 and v2, quota set and unset, the v1
unlimited sentinel, and the memory bound biting before the CPU one.

**Both arms, per ADR-0014.** The healthy path is the easy half. What actually protects
the box is the arm where the reading is absent or hostile — an unreadable cgroup must not
silently read as "unlimited", because unlimited is the direction that kills the host.
"""

import ast
import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.tools.build_jobs import (
    GB_PER_JOB,
    affinity_cpus,
    build_jobs,
    cpu_quota,
    memory_limit,
)

GIB = 1024 ** 3


def meminfo(gib, available=None):
    """`/proc/meminfo`, with MemAvailable defaulting to all of MemTotal.

    The default keeps the cgroup tests reading as "a box with N GiB"; the cases that care
    about the total/available split pass `available` explicitly.
    """
    lines = ["MemTotal:       {} kB".format(int(gib * 1048576)), "MemFree: 100 kB"]
    lines.append("MemAvailable:   {} kB".format(
        int((gib if available is None else available) * 1048576)))
    return "\n".join(lines) + "\n"


class TestCpuQuota(unittest.TestCase):
    def test_v2_quota_is_whole_cpus(self):
        self.assertEqual(cpu_quota(cpu_max="400000 100000"), 4)

    def test_v2_unset_reads_as_no_quota(self):
        self.assertIsNone(cpu_quota(cpu_max="max 100000"))

    def test_v1_quota_is_whole_cpus(self):
        self.assertEqual(cpu_quota(cfs_quota="400000", cfs_period="100000"), 4)

    def test_v1_unset_is_minus_one_and_reads_as_no_quota(self):
        self.assertIsNone(cpu_quota(cfs_quota="-1", cfs_period="100000"))

    def test_a_fractional_quota_rounds_down_never_up(self):
        """3.5 CPUs runs three compilers. Rounding up is the direction that OOMs."""
        self.assertEqual(cpu_quota(cpu_max="350000 100000"), 3)

    def test_nothing_readable_is_no_quota_not_a_guess(self):
        self.assertIsNone(cpu_quota())

    def test_garbage_does_not_become_a_number(self):
        self.assertIsNone(cpu_quota(cpu_max="not a quota"))
        self.assertIsNone(cpu_quota(cfs_quota="x", cfs_period="y"))


class TestMemoryLimit(unittest.TestCase):
    def test_memtotal_alone(self):
        self.assertEqual(memory_limit(meminfo(16)), 16 * GIB)

    def test_memavailable_wins_over_memtotal(self):
        """Total counts RAM already spoken for; budgeting against it hands out compilers
        for memory that is not there."""
        self.assertEqual(memory_limit(meminfo(16, available=6)), 6 * GIB)

    def test_a_kernel_too_old_for_memavailable_falls_back_to_total(self):
        self.assertEqual(memory_limit("MemTotal: 16777216 kB\n"), 16 * GIB)

    def test_the_v1_sentinel_is_judged_against_total_not_available(self):
        """A real cgroup limit sits below MemTotal but can sit ABOVE MemAvailable, and
        comparing against the smaller number would read it as the unlimited sentinel."""
        self.assertEqual(
            memory_limit(meminfo(64, available=4), v1_limit=str(8 * GIB)), 4 * GIB)

    def test_a_cgroup_limit_below_memtotal_wins(self):
        """The container is what kills the process; the host is what kills the machine."""
        self.assertEqual(memory_limit(meminfo(64), v2_max=str(8 * GIB)), 8 * GIB)

    def test_the_v1_unlimited_sentinel_is_recognised_without_hardcoding_it(self):
        """v1 writes a huge value for 'no limit' and it varies by kernel and page size.

        Comparing against the host's own memory recognises every spelling of it.
        """
        self.assertEqual(
            memory_limit(meminfo(16), v1_limit="9223372036854771712"), 16 * GIB)

    def test_a_v2_max_string_is_not_a_limit(self):
        self.assertEqual(memory_limit(meminfo(16), v2_max="max"), 16 * GIB)

    def test_nothing_readable_is_none_not_zero(self):
        """Absent and zero are different claims, and only one is a measurement."""
        self.assertIsNone(memory_limit())

    def test_meminfo_without_memtotal_is_none(self):
        self.assertIsNone(memory_limit("MemFree: 100 kB\n"))


class TestBuildJobs(unittest.TestCase):
    def test_memory_bounds_the_build_before_cores_do(self):
        """The box's shape: cores to spare, RAM that cannot hold one compiler each."""
        jobs, reason = build_jobs(16, None, 8 * GIB, gb_per_job=2.0)
        self.assertEqual(jobs, 4)
        self.assertIn("per compiler", reason)

    def test_a_cgroup_quota_overrides_what_affinity_reported(self):
        """The `nproc` bug itself: 64 visible, 4 allowed, and 64 compilers kill the pod."""
        jobs, reason = build_jobs(64, 4, 256 * GIB, gb_per_job=2.0)
        self.assertEqual(jobs, 4)
        self.assertIn("cgroup quota", reason)
        self.assertIn("64", reason)  # the gap is the diagnosis, so it stays in the line

    def test_cores_bound_it_when_memory_is_plentiful(self):
        jobs, _ = build_jobs(4, None, 256 * GIB, gb_per_job=2.0)
        self.assertEqual(jobs, 4)

    def test_a_machine_too_small_for_one_compiler_still_gets_one(self):
        """Floored at 1: a slow build is recoverable, a dead host is not."""
        jobs, _ = build_jobs(8, None, 1 * GIB, gb_per_job=2.0)
        self.assertEqual(jobs, 1)

    def test_unreadable_memory_says_so_rather_than_presenting_a_guess(self):
        """The forced-failure arm. If this line is ever wrong, it names itself."""
        jobs, reason = build_jobs(4, None, None)
        self.assertEqual(jobs, 4)
        self.assertIn("UNREADABLE", reason)

    def test_the_reason_always_carries_the_number_it_explains(self):
        """A reason that can disagree with its number is worse than no reason."""
        for cpus, quota, mem in ((16, None, 8 * GIB), (64, 4, 256 * GIB),
                                 (4, None, 256 * GIB), (8, None, 1 * GIB),
                                 (4, None, None)):
            jobs, reason = build_jobs(cpus, quota, mem)
            self.assertIn(str(jobs), reason)

    def test_the_default_reserve_is_the_documented_one(self):
        """`bootstrap_ss2.sh` calls this with no override, so the default IS the setting."""
        self.assertEqual(build_jobs(64, None, int(GB_PER_JOB * 8 * GIB))[0], 8)


class TestAffinity(unittest.TestCase):
    def test_it_reports_at_least_one_cpu_on_any_platform(self):
        """Runs on the Mac too, where sched_getaffinity does not exist."""
        self.assertGreaterEqual(affinity_cpus(), 1)


class TestItStaysOperatorTooling(unittest.TestCase):
    def test_env_check_does_not_import_this(self):
        """ADR-0013 keeps `env_check` a leaf, and the first draft of this fix broke it.

        Recording the memory reading in `env_check.describe_environment` was the obvious
        home and the layering gate rejected it: the module a half-built tree runs to find
        out what it is missing cannot depend on the tree. Asserted here as well as there
        so the reason travels with the code that tempted it, and so re-adding the import
        fails in the file whose author would be re-adding it.
        """
        import earshot.env_check as env_check

        with open(env_check.__file__, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        # The IMPORTS, not the text: `describe_environment`'s docstring names this module
        # on purpose, to say why it is absent, and a substring check would fail on the
        # very comment that records the decision.
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        self.assertEqual([name for name in imported if "build_jobs" in name], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
