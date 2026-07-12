"""
TDD for surfacing the anomaly-response controller report into the per-episode
``summary.episodes`` row (P1.1 — Anomaly-response SR as a first-class metric).

The controller's ``anomaly_report`` (investigated / resumed / primary_completed /
…) is written onto ``ep_log`` (``episode_runner`` ~L2297) and persisted into each
``episode_NNN.json``, but only the standalone ``diagnose_anomaly_controller.py``
census reads it — it is NOT projected into ``summary.json``'s ``episodes`` rows.
So Anomaly-response SR could not be computed from ``summary.json`` alone.

``anomaly_report_summary_fields(ep_log)`` is the pure projection seam (agreed in
the grilling session): it returns the metric-bearing controller fields when a
report is present, and ``{}`` otherwise so the objectnav / audiogoal / revisit
summary rows stay byte-identical.

Run: KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
       /opt/anaconda3/envs/ltm-embodied/bin/python \
       embodied_memory/scripts/test_summary_anomaly_report.py
"""
from __future__ import annotations

import sys

from embodied_memory.episode_runner import anomaly_report_summary_fields


def case_present_report_surfaces_controller_fields():
    # An anomaly_response episode: the controller diverted, checked the source,
    # resumed, and completed the primary at 1.0 m. The report is the independent
    # source of truth; the projection must carry its metric fields verbatim.
    ep_log = {
        "n_steps": 42,
        "anomaly_report": {
            "primary_completed": False,     # 0.1 m ring — strict, usually False
            "primary_completed_1m": True,   # 1.0 m ring — the benchmark ring
            "investigated": True,
            "investigate_aborted": False,
            "resumed": True,
            "n_benign_ignored": 2,
            "reported": True,               # an extra field the row must NOT need
        },
    }
    row = anomaly_report_summary_fields(ep_log)
    assert row["investigated"] is True, row
    assert row["resumed"] is True, row
    assert row["investigate_aborted"] is False, row
    assert row["primary_completed"] is False, row
    assert row["primary_completed_1m"] is True, row
    assert row["n_benign_ignored"] == 2, row
    print("  case_present_report_surfaces_controller_fields: OK")


def case_no_report_is_empty_byte_identical():
    # objectnav / audiogoal / revisit episodes carry no anomaly_report. The
    # projection must add NOTHING (empty dict) so those summary rows are
    # unchanged — the default-path byte-identical invariant.
    for ep_log in ({}, {"n_steps": 10, "success_1m": True}):
        assert anomaly_report_summary_fields(ep_log) == {}, ep_log
    # An explicit falsy report (never happens in practice) is also treated as
    # "no report" rather than surfacing null fields.
    assert anomaly_report_summary_fields({"anomaly_report": None}) == {}
    print("  case_no_report_is_empty_byte_identical: OK")


def main() -> int:
    print("running anomaly_report summary-projection tests…")
    case_present_report_surfaces_controller_fields()
    case_no_report_is_empty_byte_identical()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
