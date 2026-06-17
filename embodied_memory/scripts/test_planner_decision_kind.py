"""
Sanity test for ``remembr_backbone.planner_decision_kind`` /
``planner_retrieve_calls`` — the per-decision PlannerTrace classifier that
answers "is the LLM too dumb to know it needs memory?".

The classifier buckets each planner decision into:
  stub | stop | grounding_rejected | goto | explore | budget_defer | none

which distinguishes the two failure modes behind ``n_remembr_chosen=0``:
the planner never tried to recall (explore / none / retrieve_calls=0) vs it
tried and its grounded pick lost the rerank (goto) vs it answered but
grounding rejected the timestep (grounding_rejected).

Loads ``remembr_backbone.py`` faiss-free via ``spec_from_file_location``
(stub the package, real-load the numpy-only ``frontier_planner`` first), the
same pattern as ``test_propose_candidates.py``.

    python embodied_memory/scripts/test_planner_decision_kind.py
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

_EMB_DIR = Path(__file__).resolve().parent.parent  # …/embodied_memory


def _load_remembr_backbone():
    if "embodied_memory" not in sys.modules:
        pkg = types.ModuleType("embodied_memory")
        pkg.__path__ = [str(_EMB_DIR)]
        sys.modules["embodied_memory"] = pkg

    def _real_load(modname: str):
        spec = importlib.util.spec_from_file_location(
            f"embodied_memory.{modname}", _EMB_DIR / f"{modname}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"embodied_memory.{modname}"] = mod
        spec.loader.exec_module(mod)
        return mod

    _real_load("frontier_planner")          # numpy-only; satisfies the relative import
    return _real_load("remembr_backbone")


_rb = _load_remembr_backbone()
planner_decision_kind = _rb.planner_decision_kind
planner_retrieve_calls = _rb.planner_retrieve_calls


def _trace(tool_calls, stub_mode=False):
    return SimpleNamespace(tool_calls=list(tool_calls), stub_mode=stub_mode)


def _t(name, **kw):
    d = {"tool": name}
    d.update(kw)
    return d


def case_stub_mode_is_stub():
    assert planner_decision_kind(_trace([], stub_mode=True)) == "stub"


def case_empty_trace_is_none():
    assert planner_decision_kind(_trace([])) == "none"


def case_answer_explore_is_explore():
    assert planner_decision_kind(_trace([
        _t("retrieve_from_text", query="bed", n_hits=2),
        _t("answer_explore", reply="ANSWER: explore"),
    ])) == "explore"


def case_answer_goto_grounded_is_goto():
    assert planner_decision_kind(_trace([
        _t("retrieve_from_text", query="bed", n_hits=3),
        _t("answer_goto", t=12, reply="ANSWER: goto_t=12"),
    ])) == "goto"


def case_goto_rejected_unknown_t_is_grounding_rejected():
    assert planner_decision_kind(_trace([
        _t("answer_goto", t=999),
        _t("goto_rejected_unknown_t", t=999),
    ])) == "grounding_rejected"


def case_goto_rejected_zero_displacement_is_grounding_rejected():
    assert planner_decision_kind(_trace([
        _t("answer_goto", t=4),
        _t("goto_rejected_zero_displacement", t=4, dist=0.1),
    ])) == "grounding_rejected"


def case_answer_xy_rejected_far_is_grounding_rejected():
    assert planner_decision_kind(_trace([
        _t("answer_xy", reply="ANSWER: x=1,z=2"),
        _t("answer_xy_rejected_far_from_memory", snap_d=3.2),
    ])) == "grounding_rejected"


def case_budget_exhausted_is_budget_defer():
    assert planner_decision_kind(_trace([
        _t("retrieve_from_text", query="bed", n_hits=0),
        _t("retrieve_from_time", arg="5", n_hits=0),
        _t("budget_exhausted_defer"),
    ])) == "budget_defer"


def case_stop_check_is_stop():
    assert planner_decision_kind(_trace([
        _t("stop_check", match="bed", cos=1.0, dist_m=0.4),
    ])) == "stop"


def case_rejected_takes_priority_over_goto():
    # A rejected goto trace carries BOTH answer_goto and a *_rejected_* marker;
    # it must classify as grounding_rejected, not goto.
    assert planner_decision_kind(_trace([
        _t("answer_goto", t=7),
        _t("goto_rejected_unknown_t", t=7),
    ])) == "grounding_rejected"


def case_retrieve_calls_counts_all_retrieve_tools():
    assert planner_retrieve_calls(_trace([
        _t("retrieve_from_text", query="bed", n_hits=2),
        _t("retrieve_from_position", arg="1,0,2", n_hits=1),
        _t("retrieve_from_time", arg="9", n_hits=0),
        _t("answer_goto", t=2),
    ])) == 3


def case_retrieve_calls_zero_when_never_queried():
    # The strongest "too dumb to recall" signal: answered explore with no
    # retrieval attempt at all.
    assert planner_retrieve_calls(_trace([_t("answer_explore")])) == 0


def main() -> int:
    cases = [v for k, v in sorted(globals().items()) if k.startswith("case_")]
    print(f"running {len(cases)} planner_decision_kind cases…")
    for c in cases:
        c()
        print(f"  {c.__name__}: OK")
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
