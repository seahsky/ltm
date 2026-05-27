# Phase C — Multi-Scene 3-Setting Revisit Ablation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the lifelong/revisit eval a single RACE command that runs 2 scenes × {chair, bed} × {S1, S2, S3} and emits a Gate-A verdict with an STM-vs-LTM decomposition.

**Architecture:** Four small, isolated changes — (1) a new import-clean helper `episode_order.py` that pins habitat's episode iterator to `shuffle=False` (guarantees each scene's cold seed precedes its warm visits across a multi-scene run), wired into `habitat_env._build_env`; (2) an S2 (STM-only) decomposition added to `analyze_revisit.py`'s report (Gate-A stays on S3−S1, back-compatible); (3)(4) characterization tests locking the additive multi-scene build and the `(scene_id, episode_id)` warm-pair keying; (5) `race-revisit.sh` generalized to a multi-scene build loop + `for S in 1 2 3` + multi-file episode-count sum + `--scene all`. The tested builder `make_revisit_smoke.py` and the runner `run_hm3d_pol.py` (whose `--scene all` already discovers from `content/`) are **not** modified.

**Tech Stack:** Python (numpy/stdlib for the analysis + tests; no Habitat needed for any unit test — they follow the repo's standalone `case_*`/`main()`/`sys.exit` runner pattern, NOT pytest), bash for the RACE driver, Habitat/CUDA only on RACE for the live run.

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `embodied_memory/episode_order.py` | **create** | Pure helper `pin_no_shuffle(config)` — sets the habitat episode-iterator option, guarded. Import-clean (no faiss/habitat), like `spl_guard.py` / `text_encode_util.py`, so it unit-tests standalone. |
| `embodied_memory/scripts/test_episode_order.py` | **create** | Sanity test for `pin_no_shuffle` against a fake config namespace. |
| `embodied_memory/habitat_env.py` | modify (`_build_env`, ~line 184) | Call `pin_no_shuffle(config)` inside the `read_write(config)` block. |
| `embodied_memory/scripts/analyze_revisit.py` | modify (`print_report`, lines 386–396; add `_print_delta` helper) | Report warm S2−S1 and S3−S2 deltas when an S2 run is present; Gate-A unchanged. |
| `embodied_memory/scripts/test_analyze_revisit.py` | modify (add cases) | S2 decomposition reported; gate unchanged by S2; back-compat (no S2 block); multi-scene warm-pair keying. |
| `embodied_memory/scripts/test_make_revisit_smoke.py` | modify (add case) | Lock the additive two-build-into-one-dir contract Phase C relies on. |
| `scripts/race-revisit.sh` | rewrite | Multi-scene build loop, `for S in 1 2 3`, multi-file n-episodes sum, `--scene all`, analyze all three dirs. |

---

## Task 1: `episode_order.py` — pin the habitat episode iterator to no-shuffle

**Files:**
- Create: `embodied_memory/episode_order.py`
- Test: `embodied_memory/scripts/test_episode_order.py`
- Modify: `embodied_memory/habitat_env.py:184` (wire the call in)

- [ ] **Step 1: Write the failing test**

Create `embodied_memory/scripts/test_episode_order.py`:

```python
"""
Sanity test for ``episode_order.pin_no_shuffle`` — pins habitat's episode
iterator to NOT shuffle so a multi-scene revisit run yields each
(scene, category) group's COLD seed episode before its WARM revisits (the
analyzer assigns visit order by processing order; a shuffled iterator would
mislabel warm/cold and could run a warm visit before its cold sighting was
ever indexed in the LTM).

Stdlib-only (uses a fake config namespace) — runs locally without habitat.

Invoke with::

    python embodied_memory/scripts/test_episode_order.py
"""

from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import episode_order  # noqa: E402


def _cfg_with_iterator_options(shuffle=True):
    """A minimal stand-in for habitat's nested config:
    config.habitat.dataset.episode_iterator_options.shuffle"""
    opts = types.SimpleNamespace(shuffle=shuffle)
    dataset = types.SimpleNamespace(episode_iterator_options=opts)
    habitat = types.SimpleNamespace(dataset=dataset)
    return types.SimpleNamespace(habitat=habitat)


def case_sets_shuffle_false():
    cfg = _cfg_with_iterator_options(shuffle=True)
    ok = episode_order.pin_no_shuffle(cfg)
    assert ok is True, ok
    assert cfg.habitat.dataset.episode_iterator_options.shuffle is False
    print("  case sets_shuffle_false: OK")


def case_missing_key_is_noop():
    # A config lacking episode_iterator_options must not crash — return False.
    cfg = types.SimpleNamespace(habitat=types.SimpleNamespace(dataset=types.SimpleNamespace()))
    ok = episode_order.pin_no_shuffle(cfg)
    assert ok is False, ok
    print("  case missing_key_is_noop: OK")


def case_no_habitat_attr_is_noop():
    ok = episode_order.pin_no_shuffle(types.SimpleNamespace())
    assert ok is False, ok
    print("  case no_habitat_attr_is_noop: OK")


def main() -> int:
    print("episode_order.pin_no_shuffle sanity tests")
    case_sets_shuffle_false()
    case_missing_key_is_noop()
    case_no_habitat_attr_is_noop()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python embodied_memory/scripts/test_episode_order.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'episode_order'`.

- [ ] **Step 3: Write the minimal implementation**

Create `embodied_memory/episode_order.py`:

```python
"""
Pin habitat's episode iterator to NOT shuffle.

The lifelong/revisit eval runs multiple scenes in one process (``--scene all``).
The analyzer assigns each episode a *visit order* by the order the runner
processed it — 0 = "cold" (first sighting of a category in a scene), >=1 =
"warm". For that labelling to be correct, and for a warm visit to run only
*after* its cold sighting was indexed in the persisting LTM, habitat must yield
each (scene, category) group's cold episode first. Habitat's ``group_by_scene``
keeps scenes contiguous; setting ``shuffle = False`` keeps each scene's episodes
in dataset order (the builder writes cold first). The single-scene smoke happened
to order correctly; multi-scene must guarantee it regardless of habitat defaults.

Habitat-free (operates on the passed config object) so it unit-tests without the
sim; the caller (``habitat_env._build_env``) invokes it inside ``read_write``.
"""

from __future__ import annotations


def pin_no_shuffle(config) -> bool:
    """Set ``config.habitat.dataset.episode_iterator_options.shuffle = False``.

    Returns True if the option was set, False if the config lacks that key
    (an older/newer habitat layout) — the caller treats False as a harmless
    no-op, never an error. Must be called inside a ``read_write(config)`` block
    when the config is a frozen omegaconf object.
    """
    try:
        opts = config.habitat.dataset.episode_iterator_options
    except Exception:
        return False
    try:
        opts.shuffle = False
    except Exception:
        return False
    return True
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python embodied_memory/scripts/test_episode_order.py`
Expected: PASS — "All cases passed."

- [ ] **Step 5: Wire it into `habitat_env._build_env`**

In `embodied_memory/habitat_env.py`, inside the `with read_write(config):` block, the line at ~184 currently reads:

```python
            config.habitat.environment.max_episode_steps = int(self.max_steps)
```

Insert immediately after it:

```python

            # Pin the episode iterator to no-shuffle so a multi-scene revisit
            # run yields each scene's COLD seed episode before its WARM visits
            # (the LTM must hold the cold sighting before a warm visit recalls
            # it; the analyzer also labels visit order by processing order).
            from .episode_order import pin_no_shuffle
            pin_no_shuffle(config)
```

- [ ] **Step 6: Verify nothing else broke (syntax + the new test)**

Run: `python -c "import ast; ast.parse(open('embodied_memory/habitat_env.py').read()); print('habitat_env.py parses OK')"`
Run: `python embodied_memory/scripts/test_episode_order.py`
Expected: both succeed. (We can't import `habitat_env` directly off-RACE — it pulls faiss/habitat — so an AST parse is the local check; the live import is exercised on RACE.)

- [ ] **Step 7: Commit**

```bash
git add embodied_memory/episode_order.py embodied_memory/scripts/test_episode_order.py embodied_memory/habitat_env.py
git commit -m "revisit(phase-c): pin episode iterator to no-shuffle for multi-scene cold-first ordering"
```

---

## Task 2: `analyze_revisit.py` — S2 (STM-only) decomposition in the report

**Files:**
- Modify: `embodied_memory/scripts/analyze_revisit.py` (add `_print_delta`; edit `print_report` lines 386–396)
- Test: `embodied_memory/scripts/test_analyze_revisit.py` (add cases)

- [ ] **Step 1: Write the failing tests**

In `embodied_memory/scripts/test_analyze_revisit.py`, add these imports at the top (after the existing `import analyze_revisit as ar`):

```python
import contextlib  # noqa: E402
import io  # noqa: E402
```

Add this run-builder helper after the existing `_ep(...)` helper:

```python
def _run(setting, eps):
    return ar.RevisitRun(name=f"s{setting}", path=f"runs/s{setting}",
                         setting=setting, episodes=eps)
```

Add these four cases (anywhere above `main`):

```python
def case_s2_decomposition_reported():
    s1 = _run(1, [_ep("S", "a", "chair", 0, soft=0.1),
                  _ep("S", "b", "chair", 6, soft=0.2),
                  _ep("S", "d", "chair", 11, soft=0.3)])
    s2 = _run(2, [_ep("S", "a", "chair", 0, soft=0.1),
                  _ep("S", "b", "chair", 6, soft=0.25),
                  _ep("S", "d", "chair", 11, soft=0.35)])
    s3 = _run(3, [_ep("S", "a", "chair", 0, soft=0.9),
                  _ep("S", "b", "chair", 6, soft=0.6, n_mem_chosen=1),
                  _ep("S", "d", "chair", 11, soft=0.5, n_mem_chosen=1)])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ar.print_report([s1, s2, s3], n_bootstrap=500)
    out = buf.getvalue()
    assert "S2 - S1" in out, out
    assert "S3 - S2" in out, out
    assert "S3 - S1" in out, out
    print("  case s2_decomposition_reported: OK")


def _gate_helps_runs():
    # warm S3 > warm S1 and memory fires -> gate (a)
    s1 = _run(1, [_ep("S", "a", "chair", 0, soft=0.1),
                  _ep("S", "b", "chair", 6, soft=0.2),
                  _ep("S", "d", "chair", 11, soft=0.3)])
    s3 = _run(3, [_ep("S", "a", "chair", 0, soft=0.9),
                  _ep("S", "b", "chair", 6, soft=0.6, n_mem_chosen=1),
                  _ep("S", "d", "chair", 11, soft=0.5, n_mem_chosen=1)])
    s2 = _run(2, [_ep("S", "a", "chair", 0, soft=0.1),
                  _ep("S", "b", "chair", 6, soft=0.9),
                  _ep("S", "d", "chair", 11, soft=0.9)])
    return s1, s2, s3


def case_gate_unchanged_by_s2():
    s1, s2, s3 = _gate_helps_runs()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        gate_no_s2 = ar.print_report([s1, s3], n_bootstrap=500)
    s1b, s2b, s3b = _gate_helps_runs()
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        gate_s2 = ar.print_report([s1b, s2b, s3b], n_bootstrap=500)
    assert gate_no_s2 == "a", gate_no_s2
    assert gate_s2 == "a", gate_s2
    assert gate_no_s2 == gate_s2, (gate_no_s2, gate_s2)
    print("  case gate_unchanged_by_s2: OK")


def case_back_compat_no_s2_block():
    s1 = _run(1, [_ep("S", "a", "chair", 0, soft=0.1),
                  _ep("S", "b", "chair", 6, soft=0.2)])
    s3 = _run(3, [_ep("S", "a", "chair", 0, soft=0.9),
                  _ep("S", "b", "chair", 6, soft=0.6, n_mem_chosen=1)])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ar.print_report([s1, s3], n_bootstrap=500)
    out = buf.getvalue()
    assert "S2 - S1" not in out and "S3 - S2" not in out, out
    print("  case back_compat_no_s2_block: OK")


def case_warm_delta_multiscene_no_id_collision():
    # two scenes with the SAME episode_ids; pairing must key on
    # (scene_id, episode_id) so the scenes don't collide into one pair.
    s1 = [
        _ep("S", "chair-cold-0", "chair", 0, soft=0.1),
        _ep("S", "chair-warm-1", "chair", 1, soft=0.2),
        _ep("T", "chair-cold-0", "chair", 0, soft=0.1),
        _ep("T", "chair-warm-1", "chair", 1, soft=0.3),
    ]
    s3 = [
        _ep("S", "chair-cold-0", "chair", 0, soft=0.9),
        _ep("S", "chair-warm-1", "chair", 1, soft=0.6),
        _ep("T", "chair-cold-0", "chair", 0, soft=0.9),
        _ep("T", "chair-warm-1", "chair", 1, soft=0.8),
    ]
    ar.assign_visit_order(s1)
    ar.assign_visit_order(s3)
    res = ar.paired_warm_delta(s1, s3, n_bootstrap=1000)
    assert res["n"] == 2, res["n"]   # NOT collapsed to 1 despite shared ids
    # deltas [0.6-0.2, 0.8-0.3] = [0.4, 0.5] -> mean 0.45
    assert abs(res["mean"] - 0.45) < 1e-9, res["mean"]
    print("  case warm_delta_multiscene_no_id_collision: OK")
```

Register them in `main()` (after `case_classify_gate_b_fires_but_hurts()` and before the loader cases is fine):

```python
    case_s2_decomposition_reported()
    case_gate_unchanged_by_s2()
    case_back_compat_no_s2_block()
    case_warm_delta_multiscene_no_id_collision()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python embodied_memory/scripts/test_analyze_revisit.py`
Expected: `case_warm_delta_multiscene_no_id_collision` PASSES already (it locks existing `(scene_id, episode_id)` keying), but `case_s2_decomposition_reported` FAILS — the current report prints a single `S3 - S1` block with no `S2 - S1` / `S3 - S2` substrings.

- [ ] **Step 3: Add the `_print_delta` helper**

In `embodied_memory/scripts/analyze_revisit.py`, add this function just above `print_report` (after `_fmt_block` / `print_visit_distribution`):

```python
def _print_delta(label: str, res: Dict[str, Any]) -> None:
    """Print one paired-delta block uniformly (used for S3-S1, S2-S1, S3-S2,
    and the cold control)."""
    print(f"  {label}: n={res['n']:d}  mean={res['mean']:+.4f}  "
          f"90% CI=[{res['lo']:+.4f}, {res['hi']:+.4f}]  "
          f"one-sided p(<=0)={res['p_le_zero']:.3f}")
```

- [ ] **Step 4: Rewrite the delta-printing block in `print_report`**

Replace lines 386–396 (from `    s1, s3 = by_setting[1], by_setting[3]` through the `print()` after the COLD line) with:

```python
    s1, s3 = by_setting[1], by_setting[3]
    s2 = by_setting.get(2)

    warm = paired_warm_delta(s1.episodes, s3.episodes, n_bootstrap=n_bootstrap)
    cold = paired_cold_delta(s1.episodes, s3.episodes, n_bootstrap=n_bootstrap)

    print("=== paired soft-SPL delta, bootstrap, 90% CI ===")
    _print_delta("WARM S3 - S1 (full vs memory-off; PRIMARY gate)", warm)
    if s2 is not None:
        warm_s2_s1 = paired_warm_delta(s1.episodes, s2.episodes, n_bootstrap=n_bootstrap)
        warm_s3_s2 = paired_warm_delta(s2.episodes, s3.episodes, n_bootstrap=n_bootstrap)
        _print_delta("WARM S2 - S1 (STM-only effect; module 1)", warm_s2_s1)
        _print_delta("WARM S3 - S2 (LTM-specific: consolidation+LTM+rerank)", warm_s3_s2)
    _print_delta("COLD S3 - S1 (control, expect ~0)", cold)
    print()
```

(`Dict` and `Any` are already imported at the top of the file; `warm` remains the variable the Gate-A classification below uses, so `classify_gate_a` is unchanged.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python embodied_memory/scripts/test_analyze_revisit.py`
Expected: PASS — "All cases passed." (all original cases + the 4 new ones).

- [ ] **Step 6: Commit**

```bash
git add embodied_memory/scripts/analyze_revisit.py embodied_memory/scripts/test_analyze_revisit.py
git commit -m "revisit(phase-c): add S2 (STM-only) decomposition to analyzer; Gate-A stays on S3-S1"
```

---

## Task 3: lock the additive multi-scene build contract

**Files:**
- Test: `embodied_memory/scripts/test_make_revisit_smoke.py` (add one case — no production change; `make_revisit_smoke.py` already writes additively)

- [ ] **Step 1: Write the characterization test**

In `embodied_memory/scripts/test_make_revisit_smoke.py`, add this case (above `main`):

```python
def case_two_builds_into_one_dir_are_additive():
    # Phase C builds each scene into ONE shared out-dir; the per-scene
    # content/<scene>.json.gz writes must be additive (the 2nd build must not
    # clobber the 1st), and the rewritten top-level must re-load with empty
    # episodes + a category map.
    src_a = _src_content()
    src_b = _src_content()
    content_a = mk.build_dataset(src_a, categories=["chair"], n_warm=1)
    content_b = mk.build_dataset(src_b, categories=["bed"], n_warm=1)
    with tempfile.TemporaryDirectory() as d:
        mk.write_dataset(out_dir=d, scene="sceneA", content=content_a, category_maps=src_a)
        top = mk.write_dataset(out_dir=d, scene="sceneB", content=content_b, category_maps=src_b)
        assert os.path.isfile(os.path.join(d, "content", "sceneA.json.gz")), "1st build clobbered"
        assert os.path.isfile(os.path.join(d, "content", "sceneB.json.gz")), "2nd build missing"
        tj = json.load(gzip.open(top))
        assert tj["episodes"] == []
        assert "category_to_task_category_id" in tj
    print("  case two_builds_into_one_dir_are_additive: OK")
```

Register it in `main()` (after `case_write_dataset_roundtrip()`):

```python
    case_two_builds_into_one_dir_are_additive()
```

- [ ] **Step 2: Run the test**

Run: `python embodied_memory/scripts/test_make_revisit_smoke.py`
Expected: PASS — this is a characterization test locking existing behavior (the builder already writes `content/<scene>.json.gz` per-scene and rewrites only the top-level). It must pass against the **unchanged** builder; if it fails, the additive assumption in `race-revisit.sh` (Task 4) is wrong and the build loop must be revisited.

- [ ] **Step 3: Commit**

```bash
git add embodied_memory/scripts/test_make_revisit_smoke.py
git commit -m "revisit(phase-c): lock additive multi-scene build contract (test only)"
```

---

## Task 4: `race-revisit.sh` — generalize to multi-scene, 3-setting

**Files:**
- Rewrite: `scripts/race-revisit.sh`

- [ ] **Step 1: Rewrite the driver**

Replace the entire contents of `scripts/race-revisit.sh` with:

```bash
#!/bin/bash
# scripts/race-revisit.sh — one-shot RACE driver for the multi-scene,
# 3-setting lifelong/revisit ablation (Phase C).
#
# Phase 3 (Run 8) turned the revisit eval GREEN on a SINGLE scene (wcojb4TFT35,
# chair+bed, S1 vs S3). Phase C scales it to MULTIPLE scenes and adds S2
# (STM-only) so the gain can be attributed: S2-S1 = STM module, S3-S2 =
# consolidation+hierarchical-LTM+rerank (the proposal's novel part), S3-S1 =
# headline full system (the gate). Multiple scenes test the proposal's
# cross-environment (跨环境) claim.
#
# Mirrors race-smoke.sh (pull -> setup -> pre-verify -> build -> run -> analyze).
# EXECUTE it (do NOT source) — it activates conda in its own process:
#
#   bash scripts/race-revisit.sh --tag revisit-c1
#
# A bare invocation reproduces the documented Phase-C matrix: both val_mini
# scenes (wcojb4TFT35, TEEsavR23oF) x {chair, bed} x {S1, S2, S3}, n-warm 3.
#
# Critical invariants baked in (each cost a re-run before):
#   * --backbone remembr      — omitting it silently uses the 'frontier' stub.
#   * REMEMBR_STRICT=1         — a missing-weights/stub fallback CRASHES instead
#                                of silently logging a fake (stub_mode) run.
#   * S1/S2/S3 in SEPARATE processes / out-dirs — the LTM persists within a
#                                process, so mixing settings would corrupt it.
#   * --scene all + shuffle=False (pinned in habitat_env via episode_order) —
#                                each scene's COLD seed precedes its WARM visits.
#   * --target any            — runs all dataset episodes.
#
# Aborts early (before the paid run) if git pull, conda setup, the pre-test
# suite, or the dataset build fail. The final Gate-A verdict always prints.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

# --- defaults (a bare run reproduces the documented Phase-C matrix) ---
SCENES="wcojb4TFT35 TEEsavR23oF"
CATS="chair bed"
NWARM="3"
TAG="revisit-c1"
# Empty => auto: run each dataset episode exactly ONCE (one clean cold->warm
# pass across ALL scenes). Habitat wraps around when n_episodes > dataset size,
# which re-runs cold (start-on-goal) episodes and deflates the warm fire-rate.
N_EPISODES=""
TARGET="any"

# --- arg parse ---
while [ $# -gt 0 ]; do
  case "$1" in
    --scenes|--scene)    SCENES="$2"; shift 2 ;;
    --categories|--cats) CATS="$2"; shift 2 ;;
    --n-warm)            NWARM="$2"; shift 2 ;;
    --tag)               TAG="$2"; shift 2 ;;
    --n-episodes)        N_EPISODES="$2"; shift 2 ;;
    --target)            TARGET="$2"; shift 2 ;;
    *) echo "FATAL: unknown arg '$1'"; exit 1 ;;
  esac
done
CATS="${CATS//,/ }"        # accept comma- or space-separated lists
SCENES="${SCENES//,/ }"

VALMINI="data/hm3d/datasets/objectnav/hm3d/v1/val_mini/content"
DS_DIR="data/hm3d/datasets/objectnav/hm3d/v1/revisit_${TAG}"
NAME="revisit_${TAG}"
DS="${DS_DIR}/${NAME}.json.gz"

banner() { printf '\n========== %s ==========\n' "$1"; }

# --- 1. git pull ---
banner "[1/6] git pull --ff-only"
git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }

# --- 2. conda setup (sourced so the env persists in THIS process) ---
banner "[2/6] conda setup (source scripts/race-setup.sh)"
# shellcheck disable=SC1091
source scripts/race-setup.sh || { echo "FATAL: race-setup.sh failed"; exit 1; }

# --- 3. pre-test code verify (free; aborts before any paid run if broken) ---
# Standalone case_*/main() runners (assert-based, sys.exit), NOT pytest test_*.
banner "[3/6] pre-test code verify (analyzer + builder + SPL-guard + encoder + episode-order)"
python embodied_memory/scripts/test_analyze_revisit.py \
  || { echo "FATAL: analyze_revisit sanity suite failed — not spending on the live run."; exit 1; }
python embodied_memory/scripts/test_make_revisit_smoke.py \
  || { echo "FATAL: make_revisit_smoke sanity suite failed — not spending on the live run."; exit 1; }
python embodied_memory/scripts/test_spl_guard.py \
  || { echo "FATAL: spl_guard sanity suite failed — not spending on the live run."; exit 1; }
python embodied_memory/scripts/test_text_encode_util.py \
  || { echo "FATAL: text_encode_util sanity suite failed — not spending on the live run."; exit 1; }
python embodied_memory/scripts/test_episode_order.py \
  || { echo "FATAL: episode_order sanity suite failed — not spending on the live run."; exit 1; }

# --- 4. rebuild controlled-start dataset, ALL scenes into one shared dir ---
# make_revisit_smoke writes content/<scene>.json.gz per-scene (additive across
# calls) and rewrites the top-level <name>.json.gz (harmless: both val_mini
# scenes share the ObjectNav category map; the scene-annotation map is unused).
banner "[4/6] build revisit dataset: scenes=[$SCENES] cats=[$CATS] n-warm=$NWARM -> $DS_DIR"
rm -rf "$DS_DIR"   # fresh build so a stale content/ from an earlier tag can't inflate n-episodes
for SCENE in $SCENES; do
  SRC="${VALMINI}/${SCENE}.json.gz"
  [ -f "$SRC" ] || { echo "FATAL: source episodes missing: $SRC"; exit 1; }
  # shellcheck disable=SC2086
  python embodied_memory/scripts/make_revisit_smoke.py \
      --src "$SRC" --scene "$SCENE" --categories $CATS --n-warm "$NWARM" \
      --out-dir "$DS_DIR" \
    || { echo "FATAL: dataset build failed for scene $SCENE."; exit 1; }
done
[ -f "$DS" ] || { echo "FATAL: expected top-level dataset not written: $DS"; exit 1; }

# Default n-episodes = SUM of episodes across ALL content/*.json.gz (--scene all
# loads every scene; counting one file would truncate the others).
if [ -z "$N_EPISODES" ]; then
  N_EPISODES="$(python -c "import gzip,json,glob,sys; print(sum(len(json.load(gzip.open(f))['episodes']) for f in sorted(glob.glob(sys.argv[1]))))" "${DS_DIR}/content/*.json.gz")" \
    || { echo "FATAL: could not count dataset episodes."; exit 1; }
  echo "  auto n-episodes = $N_EPISODES (one pass over all built scenes)"
fi

# --- 5. run S1/S2/S3 in SEPARATE processes (--scene all over the built scenes) ---
OUT_DIRS=""
for S in 1 2 3; do
  out_dir="runs/${TAG}-s$S"
  banner "[5/6] run: setting=$S backbone=remembr scenes=all -> $out_dir"
  REMEMBR_STRICT=1 python -m embodied_memory.run_hm3d_pol --mode live \
      --backbone remembr --setting "$S" --episodes-path "$DS" \
      --scene all --target "$TARGET" --n-episodes "$N_EPISODES" \
      --out-dir "$out_dir" 2>&1 | tee "${out_dir}.log"
  OUT_DIRS="$OUT_DIRS $out_dir"
done

# --- 6. Gate-A verdict (warm-only paired soft-SPL + S2 decomposition) ---
banner "[6/6] Gate-A analysis: analyze_revisit.py$OUT_DIRS"
# shellcheck disable=SC2086
python embodied_memory/scripts/analyze_revisit.py $OUT_DIRS

banner "DONE — paste everything above (esp. the Gate-A block + S2 decomposition)"
```

- [ ] **Step 2: Verify shell syntax**

Run: `bash -n scripts/race-revisit.sh`
Expected: no output, exit 0.

- [ ] **Step 3: Local build dry-run (no Habitat — exercises steps 4 + the count)**

The build + count path is pure Python and runs on the Mac. Verify both scenes build into one dir and the count sums:

```bash
DS_DIR="data/hm3d/datasets/objectnav/hm3d/v1/revisit_dryrun"
rm -rf "$DS_DIR"
for SCENE in wcojb4TFT35 TEEsavR23oF; do
  python embodied_memory/scripts/make_revisit_smoke.py \
      --src "data/hm3d/datasets/objectnav/hm3d/v1/val_mini/content/${SCENE}.json.gz" \
      --scene "$SCENE" --categories chair bed --n-warm 3 --out-dir "$DS_DIR"
done
ls "$DS_DIR/content/"
python -c "import gzip,json,glob; fs=sorted(glob.glob('$DS_DIR/content/*.json.gz')); print('files', len(fs)); print('total episodes', sum(len(json.load(gzip.open(f))['episodes']) for f in fs))"
rm -rf "$DS_DIR"
```

Expected: two content files (`wcojb4TFT35.json.gz`, `TEEsavR23oF.json.gz`); total episodes ≤ 16 (16 if every category yields 3 reachable warm starts; fewer if some source starts are dropped as < `min_dist` from a goal). Any value > 0 with both files present confirms the build loop + count wiring. (The S1/S2/S3 run itself needs RACE.)

- [ ] **Step 4: Commit**

```bash
git add scripts/race-revisit.sh
git commit -m "revisit(phase-c): multi-scene 3-setting driver (--scene all, S1/S2/S3, episode-count sum)"
```

---

## Task 5: full local verification + push

**Files:** none (verification + push only)

- [ ] **Step 1: Run every sanity suite + syntax checks**

```bash
python embodied_memory/scripts/test_episode_order.py
python embodied_memory/scripts/test_analyze_revisit.py
python embodied_memory/scripts/test_make_revisit_smoke.py
python embodied_memory/scripts/test_spl_guard.py
python embodied_memory/scripts/test_text_encode_util.py
python -c "import ast; ast.parse(open('embodied_memory/habitat_env.py').read()); print('habitat_env OK')"
bash -n scripts/race-revisit.sh && echo "race-revisit.sh OK"
```

Expected: every suite prints "All cases passed."; both syntax checks succeed.

- [ ] **Step 2: Push the branch**

```bash
git push origin lifelong-revisit-eval
```

(Confirm with the user before pushing if they haven't already authorized it.)

---

## RACE verification (operator step — not runnable from the dev sandbox)

```bash
cd <ltm on RACE> && git checkout lifelong-revisit-eval && git pull --ff-only
# optional cheap pre-check: 1 scene, 1 category, all 3 settings
bash scripts/race-revisit.sh --scenes wcojb4TFT35 --categories chair --tag revisit-c0
# full Phase-C matrix (defaults = both scenes, chair+bed, n-warm 3):
bash scripts/race-revisit.sh --tag revisit-c1
```

Paste the Gate-A block + S2 decomposition back. Read:
- **(a) GREEN** — warm fire-rate ≥ 0.25 AND warm S3−S1 > 0 → the effect generalizes across scenes. Report Phase C; consider folding revisit into the standard harness.
- **(b)** fires but warm S3−S1 ≤ 0 → diagnose wrong-instance recall / detour cost before claiming generalization.
- **(c)** rarely fires → a cold seed didn't seat a usable sighting on one of the new scenes; inspect that scene's cold pose / caption.
- **S2 decomposition (ideal):** S3−S2 > 0 (LTM adds value beyond STM) and S2−S1 ≈ 0 (STM alone near-neutral on single-goal ObjectNav) → attributes the gain to the hierarchical LTM specifically.

Each `runs/revisit-c1-s*/episode_*.json` must have `stub_mode: false` (or the run aborts under `REMEMBR_STRICT=1`).

---

## Self-review notes (author)

- **Spec coverage:** change (1) → Task 1; change (2) driver → Task 4; change (3) analyzer S2 → Task 2; change (4) analyzer tests → Task 2; change (5) builder test → Task 3. All five spec changes have a task.
- **No habitat in unit tests:** every new test is numpy/stdlib only and follows the standalone `case_*`/`main()`/`sys.exit` pattern (the repo's convention; pytest would collect zero).
- **Type/name consistency:** `pin_no_shuffle(config) -> bool` defined in Task 1 and called identically in `habitat_env`; `_print_delta(label, res)` defined and used in Task 2; `_run(setting, eps)` and `_gate_helps_runs()` test helpers are self-contained.
- **Characterization tests** (Task 2 multi-scene case, Task 3) pass against unchanged production code by design — called out explicitly so the executor doesn't expect a red-first.
```
