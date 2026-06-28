"""
build_instance_caption_corpus — Phase-0 corpus builder for the CapRL captioner-swap
GATE. Renders HM3D keyframes labeled by physical INSTANCE (which specific
chair/sofa) and captions each with BOTH the current captioner (Qwen2-VL-2B) and a
CANDIDATE (CapRL-3B), then writes a captions-by-instance JSON that
``diagnose_sbert_cosines.py --compare-captions`` scores for within- vs
between-instance SBERT separation.

The gate decides — for $0 of matrix compute — whether the swap is worth a GPU
ablation: if the candidate widens the instance separation, the bottleneck was the
caption (swap justified); if not, the ceiling is the embedding/query and we pivot
to a retriever fix instead. See PHASE-0 plan in the report.

Runs in the ``ltm-embodied`` conda env (has habitat_sim + transformers). Captioning
two VLMs over a few hundred frames is a CHEAP one-off GPU pass, not a full ablation.

    python embodied_memory/scripts/build_instance_caption_corpus.py \
        --scenes wcojb4TFT35 TEEsavR23oF --categories chair bed sofa \
        --n-viewpoints 6 --captioners qwen2-vl-2b=Qwen/Qwen2-VL-2B-Instruct \
        caprl-3b=internlm/CapRL-3B --out runs/phase0-caprl/captions.json

The pure enumeration/sampling logic (no habitat / no transformers) is unit-tested
in test_build_instance_caption_corpus.py; the render + caption steps are
habitat/GPU and RACE-verified.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_CAPTIONERS = {
    "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
    "caprl-3b": "internlm/CapRL-3B",
}
# A terse, instance-discriminative prompt (object + material/color + one spatial
# relation) — the property that raises SBERT within-vs-between instance separation.
# Same prompt for BOTH captioners so the comparison is fair.
DEFAULT_PROMPT = ("Describe the most prominent piece of furniture in one short sentence: "
                  "name it, its color/material, and one spatial relation to a nearby object.")


# ----------------------------------------------------------------------
# Pure logic (no habitat / no transformers) — unit-tested.
# ----------------------------------------------------------------------
def load_content(content_path: str) -> Dict[str, Any]:
    with gzip.open(content_path, "rt") as f:
        return json.load(f)


def find_goal_instances(content: Dict[str, Any], category: str) -> List[Dict[str, Any]]:
    """The list of goal-instance dicts for ``category`` from goals_by_category.

    Keys are ``"<scene>.basis.glb_<category>"``; the suffix match handles
    multi-word categories (tv_monitor) without colliding with other categories.
    """
    suffix = "_" + category
    for key, insts in content.get("goals_by_category", {}).items():
        if key.endswith(suffix) and key[: -len(suffix)].endswith(".basis.glb"):
            return insts
    return []


def sample_viewpoints(view_points: Sequence[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """Pick ``n`` good-visibility but ANGLE-VARIED view_points of one instance.

    Sort by iou desc, keep the top ~60% (good visibility), then take ``n`` evenly
    spaced across them (varied angles, so within-instance captions test genuine
    agreement across views, not n near-duplicates). Deterministic.
    """
    good = [v for v in view_points if float(v.get("iou", 0.0)) > 0.0]
    good.sort(key=lambda v: -float(v.get("iou", 0.0)))
    if len(good) <= n:
        return list(good)
    top = good[: max(n, (len(good) * 3) // 5)]
    if n <= 1:
        return top[:1]
    step = (len(top) - 1) / (n - 1)
    return [top[round(i * step)] for i in range(n)]


def viewpoint_pose(view_point: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    """(position[x,y,z], rotation[x,y,z,w]) from a view_point's agent_state.

    HM3D stores the quaternion scalar-LAST ([x,y,z,w], w≈1 for near-identity).
    """
    st = view_point["agent_state"]
    return list(st["position"]), list(st["rotation"])


def plan_corpus(content: Dict[str, Any], scene: str, categories: Sequence[str],
                n_viewpoints: int) -> List[Dict[str, Any]]:
    """The flat list of (scene, category, object_id, viewpoint_idx, position,
    rotation) render jobs — every job rendered ONCE and captioned by every model.

    Only categories with >= 2 instances are kept (a single instance has no
    between-instance pairs, so it can't contribute to the separation gate).
    """
    jobs: List[Dict[str, Any]] = []
    for cat in categories:
        insts = find_goal_instances(content, cat)
        if len(insts) < 2:
            continue
        for inst in insts:
            obj_id = inst.get("object_id")
            for vp_i, vp in enumerate(sample_viewpoints(inst.get("view_points", []), n_viewpoints)):
                pos, rot = viewpoint_pose(vp)
                jobs.append({"scene": scene, "category": cat, "object_id": obj_id,
                             "viewpoint_idx": vp_i, "position": pos, "rotation": rot})
    return jobs


def parse_captioners(pairs: Sequence[str]) -> Dict[str, str]:
    """``["label=hf/model", ...]`` -> ``{label: model_id}`` (default if empty)."""
    if not pairs:
        return dict(DEFAULT_CAPTIONERS)
    out: Dict[str, str] = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"--captioners entry must be label=model_id, got {p!r}")
        label, model = p.split("=", 1)
        out[label.strip()] = model.strip()
    return out


# ----------------------------------------------------------------------
# Habitat render (RACE / ltm-embodied: needs habitat_sim).
# ----------------------------------------------------------------------
def make_sim(glb_path: str, height: int = 256, width: int = 256, eye_height: float = 0.88):
    import habitat_sim  # noqa: WPS433 (heavy, RACE-only)
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = glb_path
    backend.enable_physics = False
    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [height, width]
    rgb.position = [0.0, eye_height, 0.0]
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb]
    return habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))


def render_rgb_at(sim, position: Sequence[float], rotation_xyzw: Sequence[float]):
    import habitat_sim  # noqa: WPS433
    import numpy as np
    import quaternion  # noqa: F401  (registers np.quaternion)
    st = habitat_sim.AgentState()
    st.position = np.asarray(position, dtype=np.float32)
    x, y, z, w = rotation_xyzw
    st.rotation = np.quaternion(w, x, y, z)
    sim.get_agent(0).set_state(st)
    rgb = np.asarray(sim.get_sensor_observations()["rgb"], dtype=np.uint8)
    return rgb[..., :3]  # drop alpha if present


# ----------------------------------------------------------------------
# Captioner (RACE / ltm-embodied: needs transformers + torch).
# ----------------------------------------------------------------------
def load_captioner(model_id: str):
    """Return (processor, model, device). Loads with an EXPLICIT .to(device) (not
    device_map) so model.device is well-defined and only ONE VLM is resident at a
    time — the caller frees it before loading the next."""
    import torch  # noqa: WPS433
    from transformers import AutoProcessor  # noqa: WPS433
    try:
        from transformers import AutoModelForImageTextToText as _VLM
    except ImportError:  # older transformers
        from transformers import AutoModelForVision2Seq as _VLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = _VLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device).eval()
    return processor, model, device


def caption_rgb(processor, model, device, rgb, prompt: str, max_new_tokens: int = 64) -> str:
    import torch  # noqa: WPS433
    from PIL import Image  # noqa: WPS433
    img = Image.fromarray(rgb)
    messages = [{"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=img, return_tensors="pt").to(device)
    n_in = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    # match the production decode path (remembr_backbone): slice this sequence's new
    # tokens, re-batch, decode. Correct for batch=1 and consistent with the repo.
    new_tokens = out[0, n_in:].unsqueeze(0)
    cap = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    for term in (". ", "\n"):
        i = cap.find(term)
        if i > 0:
            cap = cap[: i + 1]
            break
    return cap.strip()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--categories", nargs="+", default=["chair", "bed", "sofa", "toilet"])
    ap.add_argument("--n-viewpoints", type=int, default=6, help="frames per instance")
    ap.add_argument("--captioners", nargs="+", default=[],
                    help="label=hf/model_id pairs (default: qwen2-vl-2b + caprl-3b)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--split", default="val_mini")
    ap.add_argument("--content-dir", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    content_dir = args.content_dir or f"data/hm3d/datasets/objectnav/hm3d/v1/{args.split}/content"
    captioners = parse_captioners(args.captioners)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    # ---- enumerate render jobs per scene (pure) ----
    jobs_by_scene: Dict[str, List[Dict[str, Any]]] = {}
    for scene in args.scenes:
        cpath = os.path.join(content_dir, f"{scene}.json.gz")
        if not os.path.isfile(cpath):
            print(f"WARN: no content for {scene} at {cpath} — skipping")
            continue
        jobs = plan_corpus(load_content(cpath), scene, args.categories, args.n_viewpoints)
        jobs_by_scene[scene] = jobs
        print(f"  {scene}: {len(jobs)} frames across "
              f"{len({(j['category'], j['object_id']) for j in jobs})} instances")
    n_jobs = sum(len(v) for v in jobs_by_scene.values())
    if n_jobs == 0:
        print("FATAL: 0 render jobs (no scene has >=2-instance categories)")
        return 1
    print(f"  total {n_jobs} frames × {len(captioners)} captioners = {n_jobs * len(captioners)} captions")

    # ---- render ALL frames ONCE (per scene), accumulate in memory ----
    # (a few hundred 256x256x3 frames ~ tens of MB — cheap to hold so we can free
    #  each VLM before loading the next instead of co-residing both on the GPU.)
    frames: List[Tuple[Dict[str, Any], Any]] = []
    for scene, jobs in jobs_by_scene.items():
        glb = _find_glb(scene)
        if glb is None:
            print(f"WARN: no .glb for {scene} — skipping its {len(jobs)} frames")
            continue
        sim = make_sim(glb)
        try:
            for j in jobs:
                frames.append((j, render_rgb_at(sim, j["position"], j["rotation"])))
        finally:
            sim.close()
        print(f"  {scene}: rendered {len(jobs)} frames")
    if not frames:
        print("FATAL: rendered 0 frames")
        return 1

    # ---- caption with each model SEQUENTIALLY (one VLM resident at a time) ----
    import gc  # noqa: WPS433
    import torch  # noqa: WPS433
    records: List[Dict[str, Any]] = []
    for label, model_id in captioners.items():
        print(f"  loading captioner {label} ({model_id}) …")
        proc, model, device = load_captioner(model_id)
        try:
            for j, rgb in frames:
                cap = caption_rgb(proc, model, device, rgb, args.prompt, args.max_new_tokens)
                records.append({"captioner": label, "scene": j["scene"], "category": j["category"],
                                "object_id": j["object_id"], "viewpoint_idx": j["viewpoint_idx"],
                                "caption": cap})
        finally:
            del proc, model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"  {label}: {sum(1 for r in records if r['captioner'] == label)} captions")

    out = {"meta": {"scenes": args.scenes, "categories": args.categories,
                    "captioners": captioners, "n_viewpoints": args.n_viewpoints,
                    "prompt": args.prompt, "n_frames": n_jobs}, "records": records}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"DONE. {len(records)} caption records -> {args.out}")
    print(f"Next: python embodied_memory/scripts/diagnose_sbert_cosines.py "
          f"--compare-captions {args.out} --baseline {next(iter(captioners))} "
          f"--candidate {list(captioners)[-1]}")
    return 0


def _find_glb(scene: str) -> Optional[str]:
    import glob
    hits = glob.glob(f"data/hm3d/**/{scene}.basis.glb", recursive=True)
    if hits:
        return hits[0]
    alt = [p for p in glob.glob(f"data/hm3d/**/*{scene}*.glb", recursive=True)
           if "semantic" not in os.path.basename(p)]
    return alt[0] if alt else None


if __name__ == "__main__":
    sys.exit(main())
