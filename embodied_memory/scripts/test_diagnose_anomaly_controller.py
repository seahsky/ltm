"""
TDD for diagnose_anomaly_controller.py — confirms the interrupt-resume state
machine actually RAN in a --task anomaly_response run, by reading the
``anomaly_report`` the E5-S7 hook writes into each episode_NNN.json
(investigated / resumed / primary_completed / primary_completed_1m /
investigate_aborted / n_benign_ignored). Pure JSON logic, no sim/GPU.

Run: KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
       /opt/anaconda3/envs/ltm-embodied/bin/python \
       embodied_memory/scripts/test_diagnose_anomaly_controller.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_anomaly_controller as dc  # noqa: E402


def _ep(report=None, **top):
    d = dict(top)
    if report is not None:
        d["anomaly_report"] = report
    return d


_FULL = {
    "investigated": True, "resumed": True, "investigate_aborted": False,
    "primary_completed": False, "primary_completed_1m": True,
    "n_benign_ignored": 0, "anomaly_class": "alarm",
}


# ----------------------------------------------------------------------
# episode_row
# ----------------------------------------------------------------------
def case_row_extracts_report_fields():
    r = dc.episode_row(_ep(report=_FULL, n_steps=24, success_1m=True, soft_spl=0.3),
                       "episode_001.json")
    assert r["has_report"] is True
    assert r["investigated"] is True and r["resumed"] is True
    assert r["primary_completed_1m"] is True
    assert r["n_steps"] == 24
    assert r["anomaly_class"] == "alarm"


def case_row_no_report():
    r = dc.episode_row(_ep(n_steps=10, success_1m=False), "episode_000.json")
    assert r["has_report"] is False
    assert r["investigated"] is None and r["resumed"] is None


# ----------------------------------------------------------------------
# aggregate
# ----------------------------------------------------------------------
def case_aggregate_rates_and_counts():
    rows = [
        dc.episode_row(_ep(report=_FULL, n_steps=24), "e1"),
        dc.episode_row(_ep(report={**_FULL, "resumed": False, "investigated": False,
                                   "investigate_aborted": True}, n_steps=40), "e2"),
        dc.episode_row(_ep(n_steps=8), "e3"),   # no report
    ]
    a = dc.aggregate(rows)
    assert a["n_episodes"] == 3
    assert a["n_with_report"] == 2
    assert a["n_investigated"] == 1
    assert a["n_resumed"] == 1
    assert a["n_aborted"] == 1
    assert abs(a["investigate_rate"] - 0.5) < 1e-9  # 1 of 2 reports investigated


# ----------------------------------------------------------------------
# verdict
# ----------------------------------------------------------------------
def case_verdict_no_report():
    a = dc.aggregate([dc.episode_row(_ep(n_steps=5), "e0")])
    v, _ = dc.verdict(a)
    assert v == "NO_ANOMALY_REPORT", v


def case_verdict_no_interrupt():
    rows = [dc.episode_row(_ep(report={**_FULL, "investigated": False, "resumed": False,
                                       "investigate_aborted": False}), "e0")]
    v, _ = dc.verdict(dc.aggregate(rows))
    assert v == "NO_INTERRUPT", v


def case_verdict_controller_ran():
    rows = [dc.episode_row(_ep(report=_FULL), "e0"),
            dc.episode_row(_ep(report=_FULL), "e1")]
    v, rec = dc.verdict(dc.aggregate(rows))
    assert v == "CONTROLLER_RAN", v
    assert "INVESTIGATE" in rec


def case_verdict_partial_when_investigate_but_no_resume():
    rows = [dc.episode_row(_ep(report={**_FULL, "investigated": True, "resumed": False}), "e0")]
    v, _ = dc.verdict(dc.aggregate(rows))
    assert v == "PARTIAL", v


def main() -> int:
    cases = [
        case_row_extracts_report_fields,
        case_row_no_report,
        case_aggregate_rates_and_counts,
        case_verdict_no_report,
        case_verdict_no_interrupt,
        case_verdict_controller_ran,
        case_verdict_partial_when_investigate_but_no_resume,
    ]
    print(f"running {len(cases)} diagnose_anomaly_controller cases…")
    for c in cases:
        c()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
