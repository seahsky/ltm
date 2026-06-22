"""TDD for the M4 temporal-context head.

The temporal-context head (the ICRA-2027 paper's named novelty) gives the MORE
RECENTLY consolidated same-category sightings a small additive rerank bonus —
recency ~ reliability in a lifelong map. It is env-gated (``LTM_TEMPORAL_CONTEXT``)
and DEFAULT-OFF, so the standard objectnav/revisit/audiogoal path stays
byte-identical; it is A/B-ablated (temporal-on vs off) on the warm matrix.

This exercises the pure ``_temporal_recency_bonus`` helper in
``embodied_memory.memory_bridge``. It loads ``memory_bridge.py`` with the heavy
``dialogue_memory``/faiss + perception/room_resolver deps stubbed (same spec-load
pattern as ``test_propose_candidates``), so it runs without the model stack.

    python embodied_memory/scripts/test_temporal_context.py
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

_EMB_DIR = Path(__file__).resolve().parent.parent       # …/embodied_memory
_REPO = _EMB_DIR.parent


def _stub_mod(name, attrs, is_pkg=False, path=None):
    m = types.ModuleType(name)
    if is_pkg:
        m.__path__ = path or []
    for a in attrs:
        setattr(m, a, type(a, (), {}))
    sys.modules[name] = m
    return m


def _bootstrap():
    # embodied_memory package shell so the relative imports in memory_bridge
    # (.frontier_planner etc.) resolve to our stubs.
    if "embodied_memory" not in sys.modules:
        pkg = types.ModuleType("embodied_memory")
        pkg.__path__ = [str(_EMB_DIR)]
        sys.modules["embodied_memory"] = pkg
    # dialogue_memory package + submodules — stubbed (no faiss).
    _stub_mod("dialogue_memory", [], is_pkg=True, path=[str(_REPO / "dialogue_memory")])
    _stub_mod("dialogue_memory.consolidation", ["DialogueConsolidation", "DialogueSegment"])
    _stub_mod("dialogue_memory.ltm", ["HierarchicalLTM", "MemoryEntry"])
    _stub_mod("dialogue_memory.pattern_cluster", ["MidLayerMemory", "PatternClusterer"])
    _stub_mod("dialogue_memory.reranking",
              ["CoherenceScorer", "HistorySuccessScorer", "MemorySimilarityScorer",
               "RerankingResult", "ResponseReranker", "Scorer", "ScoredResponse"])
    # embodied_memory submodules memory_bridge imports — lightweight stubs.
    _stub_mod("embodied_memory.frontier_planner", ["FrontierCandidate"])
    _stub_mod("embodied_memory.perception", ["Keyframe"])
    rr = types.ModuleType("embodied_memory.room_resolver")
    rr.preferred_room = lambda *a, **k: None
    rr.resolve_room = lambda *a, **k: None
    sys.modules["embodied_memory.room_resolver"] = rr
    te = types.ModuleType("embodied_memory.text_encode_util")
    te.cosine_sim = lambda a, b: 0.0
    # Stage-1 query-expansion is default-OFF in these tests; identity stub.
    te.expand_query = lambda q_cat, hit_embeddings=None, **kw: q_cat
    sys.modules["embodied_memory.text_encode_util"] = te

    spec = importlib.util.spec_from_file_location(
        "embodied_memory.memory_bridge", str(_EMB_DIR / "memory_bridge.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["embodied_memory.memory_bridge"] = mod
    spec.loader.exec_module(mod)
    return mod


mb = _bootstrap()
bonus = mb._temporal_recency_bonus


# ----------------------------------------------------------------------
# cases
# ----------------------------------------------------------------------


def case_weight_zero_is_noop():
    # Default weight 0 → no temporal effect → byte-identical scoring.
    assert bonus([10, 20, 30], 0.0) == [0.0, 0.0, 0.0]
    print("  case weight_zero_is_noop: OK")


def case_negative_weight_is_noop():
    assert bonus([10, 20], -0.5) == [0.0, 0.0]
    print("  case negative_weight_is_noop: OK")


def case_single_distinct_step_is_noop():
    # <2 DISTINCT valid steps → no recency signal to spread → all zeros.
    assert bonus([5, 5, 5], 0.06) == [0.0, 0.0, 0.0]
    assert bonus([7], 0.06) == [0.0]
    assert bonus([], 0.06) == []
    print("  case single_distinct_step_is_noop: OK")


def case_linear_recency():
    # Most recent (max step) → +weight; oldest → 0; linear in between.
    b = bonus([10, 20, 30], 0.06)
    assert abs(b[0] - 0.00) < 1e-9, b
    assert abs(b[1] - 0.03) < 1e-9, b
    assert abs(b[2] - 0.06) < 1e-9, b
    print("  case linear_recency: OK")


def case_recency_keyed_on_value_not_position():
    # Order-independent: the bonus follows the step VALUE, not list index.
    b = bonus([30, 10, 20], 0.06)
    assert abs(b[0] - 0.06) < 1e-9, b
    assert abs(b[1] - 0.00) < 1e-9, b
    assert abs(b[2] - 0.03) < 1e-9, b
    print("  case recency_keyed_on_value_not_position: OK")


def case_unknown_step_gets_zero():
    # step_idx -1 (unknown) → 0 bonus; others normalized over the VALID set.
    b = bonus([-1, 10, 20], 0.04)
    assert b[0] == 0.0, b
    assert abs(b[1] - 0.00) < 1e-9, b
    assert abs(b[2] - 0.04) < 1e-9, b
    print("  case unknown_step_gets_zero: OK")


def case_none_step_gets_zero():
    # A None step_idx is treated as unknown, not a crash.
    b = bonus([None, 100, 200], 0.02)
    assert b[0] == 0.0, b
    assert abs(b[2] - 0.02) < 1e-9, b
    print("  case none_step_gets_zero: OK")


def main() -> int:
    print("M4 temporal-context head tests")
    case_weight_zero_is_noop()
    case_negative_weight_is_noop()
    case_single_distinct_step_is_noop()
    case_linear_recency()
    case_recency_keyed_on_value_not_position()
    case_unknown_step_gets_zero()
    case_none_step_gets_zero()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
