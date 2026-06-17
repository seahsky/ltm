"""
audiogoal_retrieval_smoke — M0c GO/NO-GO: prove the anomaly→LTM→waypoint
retrieval path discriminates the right instance, BEFORE any episode runs.

This is the cheap kill for the highest plan risk ("CLAP flat / retrieval picks
the wrong instance"). It exercises the *real* SBERT retrieval seam
(``EmbodiedMemoryBridge.propose_memory_candidates``) AND the *real* source-aware
reranker (``FrontierPhysicsScorer``) — both reused VERBATIM, the same code the
live runner calls — on a hand-built 3-entry fine LTM:

    heard anomaly class  →  audio.CLASS_TO_OBJECT  →  target object name
    →  the EXISTING SBERT query "there is a {object}"  (memory_bridge:846)
    →  the matching sighting recalled as a memory waypoint;
       a clearly-unrelated sighting filtered at the proposal cosine floor;
       a same-domain distractor proposed but DOMINATED in the rerank.

The CLAP classify step itself (audio bytes → class) is unit-tested in
test_audio.py / test_perception_clap.py and measured for accuracy on RACE in
M3; here we take the class as given and verify the map + retrieval + rerank.

Faithful to the live config: the runner calls propose with the default
min_cosine=0.23 (so a clear non-match is filtered there) and lets the
FrontierPhysicsScorer (_MEM_COS_NULL=0.30 / _FULL=0.42) pick the right instance
among the near-category candidates that survive.

GREEN = crib recalled + top-ranked + wins the rerank; clear non-match filtered.
RED (non-zero exit) prints the cosines/scores — which IS the finding.

    KMP_DUPLICATE_LIB_OK=TRUE TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
        PYTHONPATH=. python embodied_memory/scripts/audiogoal_retrieval_smoke.py
"""
from __future__ import annotations

import sys

import numpy as np

from embodied_memory import audio
from embodied_memory.memory_bridge import (
    EmbodiedMemoryBridge,
    FrontierPhysicsScorer,
)
from embodied_memory.text_encode_util import cosine_sim, l2_normalize_encoder


_SCENE = "wcojb4TFT35"


def _build_bridge():
    from dialogue_memory.encoder import SentenceTransformerEncoder

    enc = SentenceTransformerEncoder(model_name="all-MiniLM-L6-v2")
    encode_fn = l2_normalize_encoder(enc.encode)  # exactly the runner's wiring
    dim = int(enc.embed_dim)
    bridge = EmbodiedMemoryBridge(
        text_embed_dim=dim,
        text_encode_fn=encode_fn,
        disable_stm=False,
        disable_ltm=False,    # S3: full LTM
        disable_rerank=False,
    )
    bridge._current_scene_id = _SCENE
    return bridge, encode_fn


def _seed_fine(bridge, encode_fn, caption, world_xyz, step_idx):
    emb = np.asarray(encode_fn(caption), dtype=np.float32).reshape(-1)
    bridge.ltm.fine.insert(
        emb, content=caption,
        metadata={
            "scene_id": _SCENE,
            "agent_position": [float(world_xyz[0]), float(world_xyz[1]),
                               float(world_xyz[2])],
            "step_idx": int(step_idx),
            "episode_id": "map-pass",
        },
    )
    return emb


def _phys_score(cand) -> float:
    """Score one candidate with the live source-aware reranker."""
    return FrontierPhysicsScorer().score("", None, {"frontier_candidate": cand})


def main() -> int:
    bridge, encode_fn = _build_bridge()

    # Three sightings from a silent mapping pass, captioned the way Qwen-VL
    # would (rich, instance-level): the goal object, a same-domain distractor,
    # and a clearly-unrelated room.
    crib_xyz = np.array([3.0, 1.5, 4.0], dtype=np.float32)
    sofa_xyz = np.array([-4.0, 1.5, 2.0], dtype=np.float32)
    sink_xyz = np.array([1.0, 1.5, -5.0], dtype=np.float32)
    crib_cap = "a white wooden baby crib in the corner of the nursery"
    sofa_cap = "a large grey fabric sofa in the living room"
    sink_cap = "a stainless steel kitchen sink with a faucet"
    crib_emb = _seed_fine(bridge, encode_fn, crib_cap, crib_xyz, 11)
    sofa_emb = _seed_fine(bridge, encode_fn, sofa_cap, sofa_xyz, 23)
    sink_emb = _seed_fine(bridge, encode_fn, sink_cap, sink_xyz, 31)
    assert len(bridge.ltm.fine) == 3, len(bridge.ltm.fine)

    # The anomaly fires: CLAP would classify it baby_cry → object "crib".
    anomaly_class = "baby_cry"
    target = audio.CLASS_TO_OBJECT[anomaly_class]
    assert target == "crib", target

    query = encode_fn(f"there is a {target}")
    cos = {"crib": cosine_sim(query, crib_emb),
           "sofa": cosine_sim(query, sofa_emb),
           "sink": cosine_sim(query, sink_emb)}
    print(f"  class={anomaly_class} → target={target!r}  query='there is a {target}'")
    print(f"  cosines: crib={cos['crib']:.3f}  sofa={cos['sofa']:.3f}  "
          f"sink={cos['sink']:.3f}")

    agent_pos = np.array([0.0, 1.5, 0.0], dtype=np.float32)
    cands = bridge.propose_memory_candidates(
        agent_pos=agent_pos, agent_yaw=0.0, target_category=target,
        planner_world_xys=[], top_k=3,   # all other args at live defaults
    )
    mem = [c for c in cands if c.source == "memory"]

    def _which(c):
        for name, xyz in (("crib", crib_xyz), ("sofa", sofa_xyz), ("sink", sink_xyz)):
            if float(np.linalg.norm(c.world_xy - xyz[[0, 2]])) < 0.05:
                return name
        return "?"

    labeled = [(_which(c), float(c.raw_score), _phys_score(c)) for c in mem]
    print(f"  proposed memory waypoints (name, cos, rerank_score): "
          + ", ".join(f"{n}({r:.3f},{s:.3f})" for n, r, s in labeled))

    names = {n for n, _, _ in labeled}
    fails = []
    # 1. the crib sighting must be recalled as a memory waypoint
    if "crib" not in names:
        fails.append("crib was NOT recalled as a memory waypoint")
    # 2. the clearly-unrelated sink sighting must be filtered at the proposal floor
    if "sink" in names:
        fails.append(f"clear non-match 'sink' (cos {cos['sink']:.3f}) leaked past "
                     f"the proposal floor")
    # 3. strong instance discrimination: crib cosine dominates the distractor
    if not (cos["crib"] - cos["sofa"] > 0.3):
        fails.append(f"weak discrimination: crib {cos['crib']:.3f} vs sofa "
                     f"{cos['sofa']:.3f} (margin <= 0.3)")
    # 4. crib must WIN the live source-aware rerank (right instance chosen)
    if labeled:
        winner = max(labeled, key=lambda t: t[2])
        if winner[0] != "crib":
            fails.append(f"rerank winner is {winner[0]!r}, not the crib")

    if fails:
        for f in fails:
            print(f"RED: {f}")
        return 1

    crib_score = next(s for n, _, s in labeled if n == "crib")
    print(f"GREEN: anomaly '{anomaly_class}' recalled the crib (cos "
          f"{cos['crib']:.3f}, rerank {crib_score:.3f}) as the chosen waypoint; "
          f"clear non-match filtered; same-domain sofa dominated. Retrieval "
          f"discriminates the right instance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
