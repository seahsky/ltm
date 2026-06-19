"""
TDD for diagnose_audio_doa_calib — the $0 pre-flight gate for the audio-DOA
rerank head (LTM_AUDIO_DOA / stage S2). Pure python, no Habitat/model/logs: each
case builds a synthetic episode_*.json-shaped dict and asserts the RECOMMEND
verdict (GO / RECALL-GAP / FRAME-BROKEN / CO-LINEAR / INSUFFICIENT-DATA).

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
        python embodied_memory/scripts/test_diagnose_audio_doa_calib.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_audio_doa_calib as dg  # noqa: E402


# ---- synthetic-log builders -------------------------------------------------
def _step(idx, sign=None, pos=(0.0, 0.0, 0.0), yaw=0.0):
    return {"step_idx": idx, "agent_pos": list(pos), "agent_yaw": yaw,
            "audio_lateral_sign": sign, "audio_energy": 0.5}


def _mem(world_xy):
    return {"id": 1, "world_xy": list(world_xy), "distance_m": 1.0,
            "bearing_rad": 0.0, "cluster_size": 1, "raw_score": 0.4, "source": "memory"}


def _decision(idx, mem_world_xys, *, pos=None, yaw=None, sign=None):
    # Decision-level pose + heard sign (the primary path: logged at the
    # audio-processed decision step). Omit to test the step-join fallback.
    d = {"step_idx": idx,
         "candidates": [_mem(xy) for xy in mem_world_xys]
         + [{"id": 99, "world_xy": [0.0, 1.0], "source": "frontier"}]}
    if pos is not None:
        d["agent_pos"] = list(pos)
    if yaw is not None:
        d["agent_yaw"] = yaw
    if sign is not None:
        d["audio_lateral_sign"] = sign
    return d


def _ep(decisions, steps, source=None, target=None):
    return {"episode_id": "e", "scene_id": "s", "steps": steps,
            "decisions": decisions, "source_position": source, "target_position": target}


# ---- helper-level geometry --------------------------------------------------
def case_right_sign_convention():
    # agent at origin, yaw 0: a point at +x has sin(rel)>0 -> LEFT(-1);
    # -x -> RIGHT(+1); straight ahead -> 0. (Convention measured, not asserted,
    # in the live verdict via both hypotheses — here we pin the helper.)
    assert dg.right_sign_from_bearing(dg.bearing_agent_frame([0, 0, 0], 0.0, (3.0, 3.0))) == -1
    assert dg.right_sign_from_bearing(dg.bearing_agent_frame([0, 0, 0], 0.0, (-3.0, 3.0))) == 1
    assert dg.right_sign_from_bearing(dg.bearing_agent_frame([0, 0, 0], 0.0, (0.0, 3.0))) == 0
    print("  case right_sign_convention: OK")


def case_correct_present_and_separation_helpers():
    tgt = (2.0, 2.0)
    d = _decision(0, [(2.0, 2.0), (-3.0, 2.0)])  # near goal + far on opposite side
    assert dg.correct_present(d, tgt)
    assert dg.opposite_side_present(d, [0, 0, 0], 0.0, tgt)
    d2 = _decision(0, [(2.0, 2.0), (3.0, 5.0)])  # near + far on SAME side
    assert dg.correct_present(d2, tgt)
    assert not dg.opposite_side_present(d2, [0, 0, 0], 0.0, tgt)
    print("  case correct_present_and_separation_helpers: OK")


# ---- verdict cases ----------------------------------------------------------
def case_go():
    tgt = [2.0, 0.0, 2.0]
    src = [2.0, 0.0, 2.0]
    rs = dg.right_sign_from_bearing(dg.bearing_agent_frame([0, 0, 0], 0.0, (src[0], src[2])))
    steps = [_step(0)]                                           # only for start_yaw (drift)
    decs = [_decision(0, [(2.0, 2.0), (-3.0, 2.0)], pos=(0, 0, 0), yaw=0.0, sign=rs),
            _decision(1, [(2.0, 2.0), (-3.0, 2.0)], pos=(0, 0, 0), yaw=0.0, sign=rs)]
    v, _ = dg.recommend(dg.aggregate([_ep(decs, steps, src, tgt)]))
    assert v == "GO", v
    print("  case go: OK")


def case_recall_gap():
    tgt = [2.0, 0.0, 2.0]
    src = [2.0, 0.0, 2.0]
    steps = [_step(0)]
    decs = [_decision(0, [(-5.0, -5.0)], pos=(0, 0, 0), yaw=0.0, sign=-1),
            _decision(1, [(-6.0, -6.0)], pos=(0, 0, 0), yaw=0.0, sign=-1)]  # none near goal
    v, _ = dg.recommend(dg.aggregate([_ep(decs, steps, src, tgt)]))
    assert v == "RECALL-GAP", v
    print("  case recall_gap: OK")


def case_frame_broken():
    tgt = [2.0, 0.0, 2.0]
    src = [2.0, 0.0, 2.0]
    rs = dg.right_sign_from_bearing(dg.bearing_agent_frame([0, 0, 0], 0.0, (src[0], src[2])))
    # even split: 2 decisions heard==rs, 2 heard==-rs -> agreeA=agreeB=0.5 -> max 0.5 < 0.60
    signs = [rs, rs, -rs, -rs]
    steps = [_step(0)]
    decs = [_decision(i, [(2.0, 2.0), (-3.0, 2.0)], pos=(0, 0, 0), yaw=0.0, sign=signs[i])
            for i in range(4)]                                  # presence+sep fine
    v, _ = dg.recommend(dg.aggregate([_ep(decs, steps, src, tgt)]))
    assert v == "FRAME-BROKEN", v
    print("  case frame_broken: OK")


def case_co_linear():
    tgt = [2.0, 0.0, 2.0]
    src = [2.0, 0.0, 2.0]
    rs = dg.right_sign_from_bearing(dg.bearing_agent_frame([0, 0, 0], 0.0, (src[0], src[2])))
    steps = [_step(0)]
    decs = [_decision(0, [(2.0, 2.0), (3.0, 5.0)], pos=(0, 0, 0), yaw=0.0, sign=rs),
            _decision(1, [(2.0, 2.0), (3.0, 5.0)], pos=(0, 0, 0), yaw=0.0, sign=rs)]  # SAME side
    v, _ = dg.recommend(dg.aggregate([_ep(decs, steps, src, tgt)]))
    assert v == "CO-LINEAR", v
    print("  case co_linear: OK")


def case_back_compat_step_join():
    # OLDER logs: decision lacks the pose/sign fields, but a keyframe step at the
    # same step_idx carries them -> the diagnostic falls back to the joined step.
    tgt = [2.0, 0.0, 2.0]
    src = [2.0, 0.0, 2.0]
    rs = dg.right_sign_from_bearing(dg.bearing_agent_frame([0, 0, 0], 0.0, (src[0], src[2])))
    steps = [_step(0, sign=rs), _step(1, sign=rs)]              # sign on the step, not decision
    decs = [_decision(0, [(2.0, 2.0), (-3.0, 2.0)]),           # no decision-level fields
            _decision(1, [(2.0, 2.0), (-3.0, 2.0)])]
    v, _ = dg.recommend(dg.aggregate([_ep(decs, steps, src, tgt)]))
    assert v == "GO", v
    print("  case back_compat_step_join: OK")


def case_insufficient_when_no_gt():
    # fires present but no GT source/target -> cannot label -> INSUFFICIENT-DATA
    steps = [_step(0)]
    decs = [_decision(0, [(2.0, 2.0)], pos=(0, 0, 0), yaw=0.0, sign=-1)]
    v, _ = dg.recommend(dg.aggregate([_ep(decs, steps, source=None, target=None)]))
    assert v == "INSUFFICIENT-DATA", v
    print("  case insufficient_when_no_gt: OK")


def case_insufficient_when_no_audio_sign():
    # GT present + correct recalled, but no audio_lateral_sign anywhere (the bug
    # the first RACE run exposed: source/target logged but the heard sign absent)
    tgt = [2.0, 0.0, 2.0]
    src = [2.0, 0.0, 2.0]
    steps = [_step(0)]
    decs = [_decision(0, [(2.0, 2.0)], pos=(0, 0, 0), yaw=0.0),
            _decision(1, [(2.0, 2.0)], pos=(0, 0, 0), yaw=0.0)]
    v, _ = dg.recommend(dg.aggregate([_ep(decs, steps, src, tgt)]))
    assert v == "INSUFFICIENT-DATA", v
    print("  case insufficient_when_no_audio_sign: OK")


def main() -> int:
    cases = [
        case_right_sign_convention,
        case_correct_present_and_separation_helpers,
        case_go,
        case_recall_gap,
        case_frame_broken,
        case_co_linear,
        case_back_compat_step_join,
        case_insufficient_when_no_gt,
        case_insufficient_when_no_audio_sign,
    ]
    print(f"running {len(cases)} diagnose_audio_doa_calib cases…")
    for c in cases:
        c()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
