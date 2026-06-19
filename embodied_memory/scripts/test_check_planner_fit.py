"""TDD for check_planner_fit.evaluate_fit — the planner/host fit-smoke verdict.

The fit smoke runs ONE cold+warm pair (setting 3, real backbone) and must prove,
cheaply, that the chosen planner/GPU config is viable BEFORE a multi-hour matrix:

  (a) FIT      — no CUDA OOM / crash; every attempted episode completed.
  (b) NAVIGATE — the warm episode cleared the stall floor (n_steps > min_steps;
                 the Run-2 3B regurgitation stalled at ~9 steps).
  (c) LTM FIRES— the warm visit retrieved + chose a memory candidate
                 (n_memory_candidates>0 AND n_memory_chosen>0). This is SBERT,
                 planner-independent, so it is the control that should pass for
                 ANY viable host.
  (d) PARSEABLE— the planner emitted a parseable ANSWER (n_planner_goto +
                 n_planner_explore > 0); a too-small planner that breaks the
                 ANSWER protocol shows all-zero here and silently nulls the eval.

GREEN only if all four hold for the warm episode. Pure-python over a run's
summary.json so it is unit-testable without a GPU.

    python embodied_memory/scripts/test_check_planner_fit.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_planner_fit", str(_HERE / "check_planner_fit.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cpf = _load()


def _warm_ep(**over):
    ep = {
        "episode_id": "chair-warm-1",
        "n_steps": 84,
        "n_memory_candidates": 8,
        "n_memory_chosen": 5,
        "n_planner_goto": 2,
        "n_planner_explore": 3,
    }
    ep.update(over)
    return ep


def _cold_ep(**over):
    ep = {
        "episode_id": "chair-cold-0",
        "n_steps": 30,
        "n_memory_candidates": 0,
        "n_memory_chosen": 0,
        "n_planner_goto": 1,
        "n_planner_explore": 1,
    }
    ep.update(over)
    return ep


def _summary(episodes, **over):
    s = {
        "n_episodes_attempted": 2,
        "n_episodes_completed": 2,
        "notes": [],
        "episodes": episodes,
    }
    s.update(over)
    return s


def case_green_when_all_four_hold():
    ok, lines = cpf.evaluate_fit(_summary([_cold_ep(), _warm_ep()]), min_steps=20)
    assert ok is True, lines
    text = "\n".join(lines)
    assert "GREEN" in text, text
    print("  case green_when_all_four_hold: OK")


def case_oom_in_notes_is_red():
    s = _summary(
        [],  # a crash leaves episodes empty (observed on the T4 OOM)
        n_episodes_completed=0,
        notes=["episode 0 crashed: RuntimeError: Failed to load planner "
               "Qwen/Qwen2.5-7B-Instruct: CUDA out of memory. Tried to allocate ..."],
    )
    ok, lines = cpf.evaluate_fit(s, min_steps=20)
    assert ok is False, lines
    text = "\n".join(lines).lower()
    assert "oom" in text or "out of memory" in text, text
    print("  case oom_in_notes_is_red: OK")


def case_partial_completion_is_red():
    # cold ran, warm crashed (completed 1/2) -> not a clean fit.
    s = _summary([_cold_ep()], n_episodes_completed=1)
    ok, lines = cpf.evaluate_fit(s, min_steps=20)
    assert ok is False, lines
    print("  case partial_completion_is_red: OK")


def case_stall_below_floor_is_red():
    # warm cleared everything but stalled at 9 steps (the 3B regurgitation floor).
    ok, lines = cpf.evaluate_fit(_summary([_cold_ep(), _warm_ep(n_steps=9)]), min_steps=20)
    assert ok is False, lines
    assert any("navigat" in l.lower() and "fail" in l.lower() for l in lines), lines
    print("  case stall_below_floor_is_red: OK")


def case_no_memory_fire_is_red():
    ok, lines = cpf.evaluate_fit(
        _summary([_cold_ep(), _warm_ep(n_memory_chosen=0)]), min_steps=20)
    assert ok is False, lines
    assert any("ltm" in l.lower() and "fail" in l.lower() for l in lines), lines
    print("  case no_memory_fire_is_red: OK")


def case_no_memory_candidates_is_red():
    ok, lines = cpf.evaluate_fit(
        _summary([_cold_ep(), _warm_ep(n_memory_candidates=0, n_memory_chosen=0)]),
        min_steps=20)
    assert ok is False, lines
    print("  case no_memory_candidates_is_red: OK")


def case_planner_unparseable_is_red():
    # small planner breaks the ANSWER protocol -> no goto/explore parsed.
    ok, lines = cpf.evaluate_fit(
        _summary([_cold_ep(), _warm_ep(n_planner_goto=0, n_planner_explore=0)]),
        min_steps=20)
    assert ok is False, lines
    assert any("planner" in l.lower() and "fail" in l.lower() for l in lines), lines
    print("  case planner_unparseable_is_red: OK")


def case_no_warm_episode_is_red():
    # only a cold episode present (warm never ran) -> cannot certify.
    ok, lines = cpf.evaluate_fit(_summary([_cold_ep()]), min_steps=20)
    assert ok is False, lines
    print("  case no_warm_episode_is_red: OK")


def case_warm_found_by_position_when_id_unlabeled():
    # episodes without 'warm' in the id -> fall back to "all after the first".
    cold = _cold_ep(episode_id="0")
    warm = _warm_ep(episode_id="1")
    ok, lines = cpf.evaluate_fit(_summary([cold, warm]), min_steps=20)
    assert ok is True, lines
    print("  case warm_found_by_position_when_id_unlabeled: OK")


def main() -> int:
    print("check_planner_fit.evaluate_fit sanity tests")
    case_green_when_all_four_hold()
    case_oom_in_notes_is_red()
    case_partial_completion_is_red()
    case_stall_below_floor_is_red()
    case_no_memory_fire_is_red()
    case_no_memory_candidates_is_red()
    case_planner_unparseable_is_red()
    case_no_warm_episode_is_red()
    case_warm_found_by_position_when_id_unlabeled()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
