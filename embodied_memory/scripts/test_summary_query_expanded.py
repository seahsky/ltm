"""
TDD for the ``n_query_expanded`` summary counter.

The query-side instance fix (``LTM_QUERY_EXPANSION`` → ``expand_query`` in
``propose_memory_candidates``) is A/B'd against the baseline S3 on the
anomaly-response matrix. The A/B is only meaningful if the expansion actually
FIRED in the variant arm — two runs that both left it OFF would compare as a
vacuous tie. ``MemoryBridge.stats()`` already carries ``n_query_expanded`` per
run, but ``RunSummary``/``summary.json`` did NOT surface it, so the driver could
not assert the arm fired. These tests pin that ``RunSummary`` carries the counter
and serializes it (mirroring ``n_audio_writes``), so the driver's post-run guard
can read it from ``summary.json``.

Run: KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
       /opt/anaconda3/envs/ltm-embodied/bin/python \
       embodied_memory/scripts/test_summary_query_expanded.py
"""
from __future__ import annotations

import sys

from embodied_memory.episode_runner import RunSummary


def case_default_zero_and_serialized():
    s = RunSummary()
    assert s.n_query_expanded == 0, s.n_query_expanded
    d = s.to_dict()
    assert "n_query_expanded" in d, sorted(d.keys())
    assert d["n_query_expanded"] == 0, d["n_query_expanded"]
    print("  case_default_zero_and_serialized: OK")


def case_set_value_serializes():
    s = RunSummary()
    s.n_query_expanded = 7
    assert s.to_dict()["n_query_expanded"] == 7, s.to_dict()["n_query_expanded"]
    print("  case_set_value_serializes: OK")


def main() -> int:
    print("running n_query_expanded summary tests…")
    case_default_zero_and_serialized()
    case_set_value_serializes()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
