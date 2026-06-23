"""
Sanity test for ``plan_scaleup_cells`` — the (scene, category) cell planner that
expands the AudioGoal revisit ablation to the ~100-cell full-val matrix.

Pure dict/filesystem logic — no Habitat / sim. Verifies category discovery,
deterministic class round-robin (with reuse when classes < categories), mesh
gating, scene subsetting, and --max-cells staging.

    python embodied_memory/scripts/test_plan_scaleup_cells.py
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plan_scaleup_cells as P  # noqa: E402


# ----------------------------------------------------------------------
# Fixtures.
# ----------------------------------------------------------------------
def _write_content(content_dir: str, scene: str, categories) -> str:
    """Write a minimal ObjectNav content .json.gz with one episode per category."""
    os.makedirs(content_dir, exist_ok=True)
    path = os.path.join(content_dir, f"{scene}.json.gz")
    episodes = [{"object_category": c, "episode_id": str(i)} for i, c in enumerate(categories)]
    with gzip.open(path, "wt") as f:
        json.dump({"episodes": episodes}, f)
    return path


def _touch_mesh(mesh_root: str, prefix: str, scene: str) -> None:
    """Create a fake mesh at <mesh_root>/val/<prefix>-<scene>/<scene>.basis.glb."""
    d = os.path.join(mesh_root, "val", f"{prefix}-{scene}")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, f"{scene}.basis.glb"), "w").close()
    # a semantic glb that must NOT be treated as the mesh:
    open(os.path.join(d, f"{scene}.semantic.glb"), "w").close()


# ----------------------------------------------------------------------
# present_categories.
# ----------------------------------------------------------------------
def case_present_categories_preserves_target_order():
    # episodes list categories out of order; output follows TARGET order.
    got = P.present_categories(["plant", "chair", "bed"], P.DEFAULT_CATEGORIES)
    assert got == ["chair", "bed", "plant"], got


def case_present_categories_intersects_only():
    got = P.present_categories(["chair", "fridge", "lamp"], P.DEFAULT_CATEGORIES)
    assert got == ["chair"], got


def case_present_categories_empty_when_none_match():
    assert P.present_categories(["fridge"], P.DEFAULT_CATEGORIES) == []


# ----------------------------------------------------------------------
# assign_classes.
# ----------------------------------------------------------------------
def case_assign_classes_roundrobin_reuses():
    cats = ["chair", "bed", "sofa", "toilet", "tv_monitor"]
    got = P.assign_classes(cats, ["baby_cry", "alarm", "glass_break"])
    assert got == [
        ("chair", "baby_cry"),
        ("bed", "alarm"),
        ("sofa", "glass_break"),
        ("toilet", "baby_cry"),
        ("tv_monitor", "alarm"),
    ], got


def case_assign_classes_each_category_once():
    cats = ["chair", "bed", "sofa", "toilet", "tv_monitor", "plant"]
    got = P.assign_classes(cats, P.DEFAULT_CLASSES)
    assert [c for c, _ in got] == cats  # every category appears exactly once, in order


def case_assign_classes_deterministic():
    a = P.assign_classes(["chair", "bed"], P.DEFAULT_CLASSES)
    b = P.assign_classes(["chair", "bed"], P.DEFAULT_CLASSES)
    assert a == b


def case_assign_classes_empty_classes_raises():
    try:
        P.assign_classes(["chair"], [])
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty classes")


# ----------------------------------------------------------------------
# scene_categories_from_content + discover_scenes.
# ----------------------------------------------------------------------
def case_scene_categories_from_content():
    with tempfile.TemporaryDirectory() as d:
        p = _write_content(d, "sceneA", ["bed", "plant", "fridge", "chair"])
        got = P.scene_categories_from_content(p, P.DEFAULT_CATEGORIES)
        assert got == ["chair", "bed", "plant"], got  # fridge dropped, target order


def case_discover_scenes_sorted():
    with tempfile.TemporaryDirectory() as d:
        _write_content(d, "zzz", ["chair"])
        _write_content(d, "aaa", ["chair"])
        assert P.discover_scenes(d) == ["aaa", "zzz"]


# ----------------------------------------------------------------------
# mesh_present.
# ----------------------------------------------------------------------
def case_mesh_present_finds_basis_glb():
    with tempfile.TemporaryDirectory() as root:
        _touch_mesh(root, "00813", "svBbv1Pavdk")
        assert P.mesh_present("svBbv1Pavdk", root) is True


def case_mesh_present_false_when_absent():
    with tempfile.TemporaryDirectory() as root:
        _touch_mesh(root, "00813", "svBbv1Pavdk")
        assert P.mesh_present("missingScene", root) is False


def case_mesh_present_ignores_semantic_only():
    with tempfile.TemporaryDirectory() as root:
        # only a semantic glb present (no basis.glb) -> not a usable mesh
        d = os.path.join(root, "val", "00999-ghostScene")
        os.makedirs(d)
        open(os.path.join(d, "ghostScene.semantic.glb"), "w").close()
        assert P.mesh_present("ghostScene", root) is False


# ----------------------------------------------------------------------
# plan_cells (integration of the above).
# ----------------------------------------------------------------------
def _two_scene_fixture(tmp):
    content = os.path.join(tmp, "content")
    mesh = os.path.join(tmp, "mesh")
    _write_content(content, "sceneA", ["chair", "bed", "sofa", "toilet", "tv_monitor"])
    _write_content(content, "sceneB", ["chair", "bed", "toilet"])
    _touch_mesh(mesh, "00001", "sceneA")
    _touch_mesh(mesh, "00002", "sceneB")
    return content, mesh


def case_plan_cells_distinct_categories_per_scene():
    with tempfile.TemporaryDirectory() as tmp:
        content, mesh = _two_scene_fixture(tmp)
        plan = P.plan_cells(content, mesh_root=mesh)
        # categories within a scene must be unique (analyzer pairs by (scene,cat)).
        for scene in ("sceneA", "sceneB"):
            cats = [c["category"] for c in plan["cells"] if c["scene"] == scene]
            assert len(cats) == len(set(cats)), (scene, cats)
        assert plan["n_cells"] == 5 + 3
        assert plan["n_scenes"] == 2


def case_plan_cells_reuses_classes_within_scene():
    with tempfile.TemporaryDirectory() as tmp:
        content, mesh = _two_scene_fixture(tmp)
        plan = P.plan_cells(content, mesh_root=mesh, classes=["baby_cry", "alarm", "glass_break"])
        a = [c for c in plan["cells"] if c["scene"] == "sceneA"]
        # 5 categories, 3 classes -> class of cat[0] == class of cat[3] (reuse).
        assert a[0]["anomaly_class"] == a[3]["anomaly_class"] == "baby_cry"


def case_plan_cells_skips_missing_mesh():
    with tempfile.TemporaryDirectory() as tmp:
        content, mesh = _two_scene_fixture(tmp)
        # add a third scene with content but NO mesh -> must be skipped + recorded.
        _write_content(content, "sceneC", ["chair", "bed"])
        plan = P.plan_cells(content, mesh_root=mesh)
        assert "sceneC" not in {c["scene"] for c in plan["cells"]}
        assert plan["skipped_no_mesh"] == ["sceneC"]


def case_plan_cells_no_mesh_root_keeps_all():
    with tempfile.TemporaryDirectory() as tmp:
        content, _ = _two_scene_fixture(tmp)
        _write_content(content, "sceneC", ["chair", "bed"])
        plan = P.plan_cells(content, mesh_root=None)  # mesh gating off
        assert "sceneC" in {c["scene"] for c in plan["cells"]}
        assert plan["skipped_no_mesh"] == []


def case_plan_cells_scene_subset():
    with tempfile.TemporaryDirectory() as tmp:
        content, mesh = _two_scene_fixture(tmp)
        plan = P.plan_cells(content, mesh_root=mesh, scenes=["sceneB"])
        assert {c["scene"] for c in plan["cells"]} == {"sceneB"}
        assert plan["n_cells"] == 3


def case_plan_cells_max_cells_truncates():
    with tempfile.TemporaryDirectory() as tmp:
        content, mesh = _two_scene_fixture(tmp)
        plan = P.plan_cells(content, mesh_root=mesh, max_cells=4)
        assert plan["n_cells"] == 4
        # truncation keeps the first scene's cells first (deterministic).
        assert all(c["scene"] == "sceneA" for c in plan["cells"])


def case_plan_cells_skips_scene_with_no_target_category():
    with tempfile.TemporaryDirectory() as tmp:
        content, mesh = _two_scene_fixture(tmp)
        _write_content(content, "sceneD", ["fridge", "lamp"])  # no target categories
        _touch_mesh(mesh, "00004", "sceneD")
        plan = P.plan_cells(content, mesh_root=mesh)
        assert "sceneD" not in {c["scene"] for c in plan["cells"]}
        assert plan["skipped_no_category"] == ["sceneD"]


def case_plan_cells_skips_unreadable_content():
    with tempfile.TemporaryDirectory() as tmp:
        content, mesh = _two_scene_fixture(tmp)
        # a corrupted (non-gzip) content file for a scene WITH a mesh must be
        # skipped, not abort the whole plan.
        with open(os.path.join(content, "sceneBad.json.gz"), "wb") as f:
            f.write(b"not a gzip file at all")
        _touch_mesh(mesh, "00009", "sceneBad")
        plan = P.plan_cells(content, mesh_root=mesh)
        assert "sceneBad" not in {c["scene"] for c in plan["cells"]}
        assert plan["skipped_unreadable"] == ["sceneBad"]
        # the two good scenes still planned fully (no abort).
        assert plan["n_cells"] == 5 + 3


def case_plan_cells_json_serializable():
    with tempfile.TemporaryDirectory() as tmp:
        content, mesh = _two_scene_fixture(tmp)
        plan = P.plan_cells(content, mesh_root=mesh)
        json.dumps(plan)  # must not raise


def case_format_lines_shape():
    with tempfile.TemporaryDirectory() as tmp:
        content, mesh = _two_scene_fixture(tmp)
        plan = P.plan_cells(content, mesh_root=mesh, scenes=["sceneB"])
        lines = P.format_lines(plan).splitlines()
        assert len(lines) == 3
        for ln in lines:
            parts = ln.split("\t")
            assert len(parts) == 3 and parts[0] == "sceneB"


def case_main_lines_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        content, mesh = _two_scene_fixture(tmp)
        rc = P.main(["--content-dir", content, "--mesh-root", mesh, "--scenes", "sceneB", "--format", "lines"])
        assert rc == 0


def main() -> int:
    cases = [
        case_present_categories_preserves_target_order,
        case_present_categories_intersects_only,
        case_present_categories_empty_when_none_match,
        case_assign_classes_roundrobin_reuses,
        case_assign_classes_each_category_once,
        case_assign_classes_deterministic,
        case_assign_classes_empty_classes_raises,
        case_scene_categories_from_content,
        case_discover_scenes_sorted,
        case_mesh_present_finds_basis_glb,
        case_mesh_present_false_when_absent,
        case_mesh_present_ignores_semantic_only,
        case_plan_cells_distinct_categories_per_scene,
        case_plan_cells_reuses_classes_within_scene,
        case_plan_cells_skips_missing_mesh,
        case_plan_cells_no_mesh_root_keeps_all,
        case_plan_cells_scene_subset,
        case_plan_cells_max_cells_truncates,
        case_plan_cells_skips_scene_with_no_target_category,
        case_plan_cells_skips_unreadable_content,
        case_plan_cells_json_serializable,
        case_format_lines_shape,
        case_main_lines_smoke,
    ]
    print(f"running {len(cases)} plan_scaleup_cells cases…")
    for c in cases:
        c()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
