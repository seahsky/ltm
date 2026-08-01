#!/usr/bin/env python3
"""
.scratch/ss2-clean-room/probes/box_inventory.py — ticket 05's box inventory.

Question: what is actually on the RACE V100 right now, and what of it is worth
keeping through the reset?

    nrun python3 .scratch/ss2-clean-room/probes/box_inventory.py

STRICTLY READ-ONLY. It installs nothing, deletes nothing, and writes only under
--out-dir. That is deliberate: this runs on a box holding the only copy of the
HM3D download, and the whole point of the ticket is to stop the reset from
deleting or rebuilding blind.

Stdlib only, so it runs under whatever `python3` is on PATH and needs no env of
its own. Every section is independently guarded — one section failing records an
`error` for that section and cannot hide the seven behind it (same rule as
ticket 04's gate).

Two things here are shortcuts for OTHER tickets, and are marked `opportunistic`
in the report so they are not mistaken for those tickets' own answers:

  * If an env already holds an audio-capable habitat_sim, the probe dumps the
    `AudioSensorSpec` defaults. Ticket 06 blocks on those defaults and ticket 04
    plans an hour-long clean build to get them. Ticket 04 stays authoritative —
    an existing env is exactly the "unknown drift" it refuses to trust — but if
    the numbers agree, that is a strong prior, and if the build here is already
    sound then 04's build step is minutes rather than an hour.
  * GLIBC version. `libRLRAudioPropagation.so` is a prebuilt Linux-x64 binary
    needing GLIBC >= 2.29, which the map's Notes call load-bearing. It is one
    line to check and it fails the whole map if it is wrong.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]

# Build dirs worth looking for by name. `soundspaces-build` is the 2026-06-10
# spike; `ss2-build` is where ticket 04's gate puts its clean build.
KNOWN_BUILD_ROOTS = ["soundspaces-build", "ss2-build", "habitat-build"]

# Roots this project has used for HM3D across its history.
HM3D_ROOT_CANDIDATES = [
    "data/hm3d",
    "data/hm3d-0.2/hm3d",
    "data/scene_datasets",
]

# habitat-sim methods that exist only on RLRAudioPropagationUpdate, not on
# `main`. Presence proves the build is the expected branch generation rather
# than a stale checkout (ticket 04's note).
BRANCH_GEN_METHODS = [
    "sourceIsVisible",
    "getRayEfficiency",
    "setListenerHRTF",
    "writeIRWave",
    "writeSceneMeshOBJ",
]

# Importing habitat_sim is slow (EGL context, large .so). Envs that do not have
# it fail fast; the ones that do are the ones worth waiting for.
IMPORT_PROBE_TIMEOUT = 180
SHORT_CMD_TIMEOUT = 30
DU_TIMEOUT = 240


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def run(cmd, timeout=SHORT_CMD_TIMEOUT, cwd=None):
    """Run a command, never raise. Returns (rc, stdout, stderr)."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s"
    except FileNotFoundError:
        return 127, "", f"not found: {cmd[0]}"
    except Exception as exc:  # noqa: BLE001 — inventory must never die
        return 1, "", repr(exc)


def section(name):
    """Decorator-free guard: wraps a producer so a failure is recorded, not fatal."""

    def wrap(fn):
        started = time.time()
        try:
            value = fn()
        except Exception as exc:  # noqa: BLE001
            value = {"error": f"{type(exc).__name__}: {exc}"}
        if isinstance(value, dict):
            value.setdefault("_elapsed_s", round(time.time() - started, 1))
        print(f"  done: {name} ({round(time.time() - started, 1)}s)", flush=True)
        return value

    return wrap


def human_bytes(n):
    if n is None:
        return None
    for unit in ["B", "K", "M", "G", "T"]:
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}P"


def dir_size_du(path, timeout=DU_TIMEOUT):
    """Directory size in bytes. Returns None on timeout — a slow answer is not
    worth a hang.

    `-sk` rather than `--block-size=1`: the latter is GNU-only and silently
    returns nothing on BSD/macOS, which made every size read `None` when this
    was dry-run off the box.
    """
    rc, out, _ = run(["du", "-skx", str(path)], timeout=timeout)
    if rc != 0 or not out:
        return None
    try:
        return int(out.split()[0]) * 1024
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# The in-env probe
#
# Runs inside each candidate env's own interpreter. Prints exactly one JSON
# object on stdout so the parent can parse it regardless of the import noise
# habitat_sim writes to stderr.
# ---------------------------------------------------------------------------

ENV_PROBE = r'''
import json, sys

out = {"python": sys.version.split()[0], "executable": sys.executable}

def probe(name, fn):
    try:
        out[name] = fn()
    except Exception as exc:
        out[name] = {"error": "%s: %s" % (type(exc).__name__, exc)}

def _numpy():
    import numpy
    return numpy.__version__

def _torch():
    import torch
    return {
        "version": torch.__version__,
        "cuda_build": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

def _transformers():
    import transformers
    info = {"version": transformers.__version__}
    # CLAP is the only model class the rebuilt agent needs from here.
    try:
        from transformers import ClapModel, ClapProcessor  # noqa: F401
        info["clap"] = True
    except Exception as exc:
        info["clap"] = False
        info["clap_error"] = "%s: %s" % (type(exc).__name__, exc)
    return info

def _habitat_lab():
    import habitat
    return {"version": getattr(habitat, "__version__", "?")}

def _habitat_sim():
    # Issue #1813: quaternion must be imported before habitat_sim.
    import quaternion  # noqa: F401
    import habitat_sim, habitat_sim.sensor

    info = {"version": getattr(habitat_sim, "__version__", "?"),
            "file": getattr(habitat_sim, "__file__", "?")}

    # Issue #2340: AudioSensorSpec is bound even in NON-audio builds, so the
    # class is not evidence. Probe the enum MEMBER.
    layout = getattr(habitat_sim.sensor, "RLRAudioPropagationChannelLayoutType", None)
    info["audio_capable"] = bool(layout is not None and hasattr(layout, "Binaural"))
    if layout is not None:
        info["channel_layouts"] = sorted(
            k for k in dir(layout) if not k.startswith("_")
        )

    # Branch-generation check: these exist on RLRAudioPropagationUpdate, not main.
    sensor_cls = getattr(habitat_sim.sensor, "AudioSensor", None)
    info["branch_gen_methods"] = {
        m: bool(sensor_cls is not None and hasattr(sensor_cls, m))
        for m in __BRANCH_GEN__
    }
    return info

def _audio_spec_defaults():
    """Opportunistic: ticket 04 owns this on a CLEAN build. Free to grab here."""
    import quaternion  # noqa: F401
    import habitat_sim, habitat_sim.sensor

    spec = habitat_sim.sensor.AudioSensorSpec()
    res = {}

    def dump(obj, label):
        fields = {}
        for k in dir(obj):
            if k.startswith("_"):
                continue
            try:
                v = getattr(obj, k)
            except Exception:
                continue
            if callable(v):
                continue
            fields[k] = v if isinstance(v, (int, float, bool, str, type(None))) else repr(v)
        res[label] = fields

    dump(spec, "spec")
    ac = getattr(spec, "acousticsConfig", None)
    if ac is not None:
        dump(ac, "acousticsConfig")

    # Ticket 11: irTime was renamed maxIRLength. Which name exists pins the
    # branch generation harder than a version string does.
    ac_fields = res.get("acousticsConfig", {})
    res["name_check"] = {
        "maxIRLength": "maxIRLength" in ac_fields,
        "irTime": "irTime" in ac_fields,
        "directRayCount": "directRayCount" in ac_fields,
    }
    # Ticket 03/04 called these two out by name.
    res["called_out"] = {
        "transmission": ac_fields.get("transmission", "ABSENT"),
        "enableMaterials": res.get("spec", {}).get("enableMaterials", "ABSENT"),
    }
    return res

probe("numpy", _numpy)
probe("habitat_sim", _habitat_sim)
probe("torch", _torch)
probe("transformers", _transformers)
probe("habitat_lab", _habitat_lab)

hs = out.get("habitat_sim")
if isinstance(hs, dict) and hs.get("audio_capable"):
    probe("audio_spec_defaults_opportunistic", _audio_spec_defaults)

print("@@JSON@@" + json.dumps(out))
'''.replace("__BRANCH_GEN__", json.dumps(BRANCH_GEN_METHODS))


def probe_env(python_bin):
    """Run ENV_PROBE under `python_bin`. Never raises."""
    rc, out, err = run([str(python_bin), "-c", ENV_PROBE], timeout=IMPORT_PROBE_TIMEOUT)
    marker = "@@JSON@@"
    if marker in out:
        payload = out.split(marker, 1)[1].strip()
        try:
            parsed = json.loads(payload)
            # habitat_sim writes a lot to stderr even on success; keep only a tail.
            if err:
                parsed["_stderr_tail"] = err[-600:]
            return parsed
        except json.JSONDecodeError as exc:
            return {"error": f"probe emitted unparseable JSON: {exc}", "raw": payload[:600]}
    return {
        "error": f"probe did not complete (rc={rc})",
        "stdout_tail": out[-600:],
        "stderr_tail": err[-600:],
    }


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def sec_host():
    glibc = None
    # `ldd --version` is the portable read; platform.libc_ver() lies on some images.
    rc, out, _ = run(["ldd", "--version"])
    if rc == 0 and out:
        m = re.search(r"(\d+\.\d+)\s*$", out.splitlines()[0])
        if m:
            glibc = m.group(1)
    if glibc is None:
        glibc = platform.libc_ver()[1] or None

    glibc_ok = None
    if glibc:
        try:
            glibc_ok = tuple(int(x) for x in glibc.split(".")) >= (2, 29)
        except ValueError:
            glibc_ok = None

    nproc = os.cpu_count()
    # threadCount is a free speed knob currently pinned to 1 (ticket 06).
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_running_this": sys.version.split()[0],
        "glibc": glibc,
        # libRLRAudioPropagation.so needs >= 2.29. If this is False the whole
        # map is dead where it stands, so it is checked first and loudly.
        "glibc_ok_for_rlr_audio": glibc_ok,
        "cpu_count": nproc,
        "threadcount_headroom": (
            f"threadCount=1 today; {nproc} cores available" if nproc else None
        ),
    }


def sec_gpu():
    fields = [
        "name",
        "driver_version",
        "memory.total",
        "memory.used",
        "memory.free",
        "compute_cap",
    ]
    rc, out, err = run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader",
        ]
    )
    if rc != 0:
        return {"error": f"nvidia-smi failed (rc={rc}): {err or out}"}

    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        gpus.append(dict(zip(fields, parts)))

    # The driver's max supported CUDA decides which torch wheel ticket 04 should
    # install (SS2_TORCH_SPEC is overridable precisely pending this).
    rc2, out2, _ = run(["nvidia-smi"])
    cuda_driver = None
    if rc2 == 0:
        m = re.search(r"CUDA Version:\s*([\d.]+)", out2)
        if m:
            cuda_driver = m.group(1)

    rc3, out3, _ = run(["nvcc", "--version"])
    nvcc = None
    if rc3 == 0:
        m = re.search(r"release ([\d.]+)", out3)
        if m:
            nvcc = m.group(1)

    return {"gpus": gpus, "cuda_driver_max": cuda_driver, "nvcc": nvcc}


def sec_disk():
    out = {}
    total, used, free = shutil.disk_usage(str(REPO_ROOT))
    out["repo_filesystem"] = {
        "total": human_bytes(total),
        "used": human_bytes(used),
        "free": human_bytes(free),
        "free_bytes": free,
    }
    home = Path.home()
    if os.stat(home).st_dev != os.stat(REPO_ROOT).st_dev:
        t2, u2, f2 = shutil.disk_usage(str(home))
        out["home_filesystem"] = {
            "total": human_bytes(t2),
            "used": human_bytes(u2),
            "free": human_bytes(f2),
        }

    # Named consumers only. A blind `du` over $HOME on a box with a conda tree
    # and 1.2 GB of meshes is minutes of nothing useful.
    candidates = [
        REPO_ROOT / "data",
        REPO_ROOT / "runs",
        REPO_ROOT / "models",
        home / ".cache" / "huggingface",
        home / ".cache" / "torch",
        home / ".cache" / "pip",
    ]
    candidates += [home / name for name in KNOWN_BUILD_ROOTS]
    for prefix in ("miniconda3", "anaconda3", "miniforge3"):
        candidates.append(home / prefix / "envs")

    sizes = {}
    for path in candidates:
        if not path.exists():
            continue
        n = dir_size_du(path)
        sizes[str(path)] = {"bytes": n, "human": human_bytes(n)}
    out["large_consumers"] = sizes
    return out


def sec_conda_envs():
    envs = []
    rc, out, _ = run(["conda", "info", "--json"], timeout=60)
    prefixes = []
    if rc == 0 and out:
        try:
            prefixes = json.loads(out).get("envs", [])
        except json.JSONDecodeError:
            prefixes = []
    if not prefixes:
        # conda not on PATH is normal on RACE after a pod restart
        # (scripts/race-setup.sh exists for exactly this). Scan instead.
        home = Path.home()
        for prefix in ("miniconda3", "anaconda3", "miniforge3"):
            envs_dir = home / prefix / "envs"
            if envs_dir.is_dir():
                prefixes.extend(str(p) for p in sorted(envs_dir.iterdir()) if p.is_dir())
            base = home / prefix
            if (base / "bin" / "python").exists():
                prefixes.append(str(base))

    for prefix in prefixes:
        p = Path(prefix)
        py = p / "bin" / "python"
        entry = {"name": p.name, "prefix": str(p), "has_python": py.exists()}
        if py.exists():
            print(f"    probing env: {p.name} ...", flush=True)
            entry["probe"] = probe_env(py)
        envs.append(entry)
    return {"count": len(envs), "envs": envs}


def _git_facts(repo_dir):
    facts = {}
    for key, cmd in [
        ("head", ["git", "rev-parse", "HEAD"]),
        ("branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        ("describe", ["git", "describe", "--all", "--always"]),
    ]:
        rc, out, _ = run(cmd, cwd=str(repo_dir))
        facts[key] = out if rc == 0 else None
    rc, out, _ = run(["git", "status", "--porcelain"], cwd=str(repo_dir))
    facts["dirty"] = bool(out) if rc == 0 else None

    # The submodule SHA is what makes tickets 01/11's parameter sheet
    # falsifiable — the docs were read against that repo's archived main, and
    # habitat-sim pins a commit nobody checked against it.
    rc, out, _ = run(["git", "submodule", "status", "--recursive"], cwd=str(repo_dir))
    if rc == 0 and out:
        subs = {}
        for line in out.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                subs[parts[1]] = parts[0].lstrip("+-U")
        facts["submodules"] = subs
        facts["rlr_audio_propagation_sha"] = next(
            (v for k, v in subs.items() if "rlr-audio-propagation" in k), None
        )
    return facts


def sec_habitat_builds():
    found = []
    home = Path.home()
    roots = [home / name for name in KNOWN_BUILD_ROOTS]
    roots = [r for r in roots if r.exists()]

    for root in roots:
        entry = {"root": str(root), "exists": True}
        for sub in ("habitat-sim", "habitat-lab"):
            d = root / sub
            if not d.is_dir():
                entry[sub] = None
                continue
            info = {"path": str(d), "git": _git_facts(d) if (d / ".git").exists() else None}
            if sub == "habitat-sim":
                info["build_dir_exists"] = (d / "build").is_dir()
                # The prebuilt closed-source binary. Its presence is the single
                # best signal that `--audio` was actually built.
                so_hits = glob.glob(str(d / "**" / "libRLRAudioPropagation*.so"), recursive=True)
                info["rlr_audio_so"] = so_hits[:5]
                info["rlr_audio_so_present"] = bool(so_hits)
                size = dir_size_du(d, timeout=120)
                info["size"] = human_bytes(size)
            entry[sub] = info
        found.append(entry)

    return {"searched": [str(home / n) for n in KNOWN_BUILD_ROOTS], "found": found}


def _scene_ids_from_content(content_dir):
    return sorted(
        Path(p).name[: -len(".json.gz")]
        for p in glob.glob(str(Path(content_dir) / "*.json.gz"))
    )


def sec_hm3d():
    out = {}
    roots = []
    for rel in HM3D_ROOT_CANDIDATES:
        p = REPO_ROOT / rel
        if p.exists():
            roots.append(p)
    out["roots_present"] = [str(p) for p in roots]
    if not roots:
        out["note"] = "no HM3D root found under the repo — check for an out-of-tree copy"
        return out

    hm3d = REPO_ROOT / "data" / "hm3d"

    # --- scene meshes, per split -------------------------------------------
    scene_root = hm3d / "scene_datasets" / "hm3d"
    splits = {}
    if scene_root.is_dir():
        for split_dir in sorted(p for p in scene_root.iterdir() if p.is_dir()):
            basis = glob.glob(str(split_dir / "**" / "*.basis.glb"), recursive=True)
            semantic_glb = glob.glob(str(split_dir / "**" / "*.semantic.glb"), recursive=True)
            semantic_txt = glob.glob(str(split_dir / "**" / "*.semantic.txt"), recursive=True)
            splits[split_dir.name] = {
                "scene_dirs": len([p for p in split_dir.iterdir() if p.is_dir()]),
                "basis_glb": len(basis),
                "semantic_glb": len(semantic_glb),
                "semantic_txt": len(semantic_txt),
                "size": human_bytes(dir_size_du(split_dir, timeout=120)),
            }
    out["scene_dataset_splits"] = splits
    out["scene_dataset_configs"] = [
        Path(p).name
        for p in glob.glob(str(scene_root / "*.scene_dataset_config.json"))
    ]

    # --- episode datasets ---------------------------------------------------
    ep_root = hm3d / "datasets" / "objectnav" / "hm3d" / "v1"
    episodes = {}
    if ep_root.is_dir():
        for d in sorted(p for p in ep_root.iterdir() if p.is_dir()):
            content = d / "content"
            episodes[d.name] = {
                "content_files": len(glob.glob(str(content / "*.json.gz"))) if content.is_dir() else 0,
                "top_level_json_gz": [Path(p).name for p in glob.glob(str(d / "*.json.gz"))],
            }
    out["episode_datasets"] = episodes

    # --- the mesh-coverage cross-check -------------------------------------
    # A prior run found only 2 of 20 val scenes had meshes. That is the single
    # most expensive thing to rediscover after a reset, so check it explicitly.
    val_content = ep_root / "val" / "content"
    if val_content.is_dir():
        want = _scene_ids_from_content(val_content)
        have, missing = [], []
        for sid in want:
            hits = glob.glob(str(scene_root / "*" / f"*{sid}*"))
            mesh = [h for h in hits if glob.glob(os.path.join(h, "*.basis.glb"))]
            (have if mesh else missing).append(sid)
        out["val_mesh_coverage"] = {
            "episode_scenes": len(want),
            "with_mesh": len(have),
            "missing_mesh": missing,
            "verdict": f"{len(have)}/{len(want)} val scenes have meshes",
        }
    else:
        out["val_mesh_coverage"] = {"error": f"no val content dir at {val_content}"}

    return out


def sec_weights():
    home = Path.home()
    out = {}
    hf = home / ".cache" / "huggingface" / "hub"
    if hf.is_dir():
        models = []
        for d in sorted(hf.iterdir()):
            if not d.is_dir() or not d.name.startswith("models--"):
                continue
            models.append(
                {
                    "repo": d.name.replace("models--", "").replace("--", "/"),
                    "size": human_bytes(dir_size_du(d, timeout=90)),
                }
            )
        out["huggingface_hub"] = models
    else:
        out["huggingface_hub"] = []

    torch_cache = home / ".cache" / "torch"
    if torch_cache.is_dir():
        out["torch_cache"] = {"size": human_bytes(dir_size_du(torch_cache, timeout=90))}

    models_dir = REPO_ROOT / "models"
    if models_dir.is_dir():
        out["repo_models_dir"] = {
            "size": human_bytes(dir_size_du(models_dir, timeout=90)),
            "entries": sorted(p.name for p in models_dir.iterdir())[:40],
        }
    return out


def sec_repo():
    facts = _git_facts(REPO_ROOT)
    facts["root"] = str(REPO_ROOT)
    runs = REPO_ROOT / "runs"
    if runs.is_dir():
        facts["runs_dirs"] = len([p for p in runs.iterdir() if p.is_dir()])
    return facts


# ---------------------------------------------------------------------------
# Verdict
#
# The ticket asks for a keep/rebuild/delete call on each item. The script makes
# the calls it can make mechanically and leaves the judgement ones to the human,
# clearly marked, rather than guessing.
# ---------------------------------------------------------------------------


def build_verdict(report):
    calls = []

    def call(item, verdict, why):
        calls.append({"item": item, "call": verdict, "why": why})

    host = report.get("host", {})
    if host.get("glibc_ok_for_rlr_audio") is False:
        call(
            "GLIBC",
            "BLOCKER",
            f"glibc {host.get('glibc')} < 2.29 — libRLRAudioPropagation.so cannot load. "
            "The map's execution environment assumption is wrong.",
        )
    elif host.get("glibc_ok_for_rlr_audio"):
        call("GLIBC", "keep", f"glibc {host.get('glibc')} >= 2.29, audio binary can load")
    else:
        call(
            "GLIBC",
            "UNKNOWN",
            f"could not read a glibc version on {platform.system()}. Expected off-box; "
            "if this appears in a RACE run, treat it as unresolved rather than fine.",
        )

    # An existing sound audio build turns ticket 04's hour into minutes.
    audio_envs = []
    for env in report.get("conda_envs", {}).get("envs", []):
        hs = (env.get("probe") or {}).get("habitat_sim")
        if isinstance(hs, dict) and hs.get("audio_capable"):
            audio_envs.append(env["name"])
    if audio_envs:
        call(
            "existing audio-capable env(s): " + ", ".join(audio_envs),
            "keep (but 04 still builds clean)",
            "Ticket 04 refuses to trust a spike env for the verdict, but this proves "
            "the build works on this box and lets 04 reuse the clone rather than reclone.",
        )
    else:
        call(
            "audio-capable habitat_sim",
            "rebuild",
            "no env on this box imports an audio-capable habitat_sim — 04's build is the full hour",
        )

    cov = report.get("hm3d", {}).get("val_mesh_coverage", {})
    if isinstance(cov, dict) and "with_mesh" in cov:
        if cov.get("missing_mesh"):
            call(
                "HM3D val meshes",
                "keep + top up",
                f"{cov['verdict']}; missing: {', '.join(cov['missing_mesh'][:8])}"
                + (" ..." if len(cov["missing_mesh"]) > 8 else ""),
            )
        else:
            call("HM3D val meshes", "keep", cov["verdict"])

    disk = report.get("disk", {}).get("repo_filesystem", {})
    free_bytes = disk.get("free_bytes")
    if free_bytes is not None:
        # habitat-sim's build tree is the big transient; ~15 GB is a safe floor
        # for a clean build plus a torch wheel.
        enough = free_bytes > 15 * 1024**3
        call(
            "disk",
            "keep" if enough else "BLOCKER",
            f"{disk.get('free')} free on the repo filesystem"
            + ("" if enough else " — under the ~15G a clean build plus torch needs"),
        )

    gpus = report.get("gpu", {}).get("gpus")
    if gpus:
        g = gpus[0]
        call(
            f"GPU: {g.get('name')}",
            "keep",
            f"driver {g.get('driver_version')}, CUDA(driver max) "
            f"{report.get('gpu', {}).get('cuda_driver_max')}, free VRAM {g.get('memory.free')} "
            "— sets SS2_TORCH_SPEC for ticket 04",
        )

    calls.append(
        {
            "item": "spike build dir / old runs / old envs",
            "call": "HUMAN JUDGEMENT",
            "why": "Ticket 10 owns the reset spec. This inventory only reports what is there; "
            "it deletes nothing.",
        }
    )
    return calls


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Ticket 05 — RACE box inventory (read-only)")
    ap.add_argument("--out-dir", default="runs/ss2-box-inventory")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Ticket 05 — RACE box inventory (READ-ONLY: installs nothing, deletes nothing)")
    print(f"repo root : {REPO_ROOT}")
    print(f"out dir   : {out_dir}")
    print("=" * 72, flush=True)

    report = {
        "ticket": "05-race-box-inventory",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "read_only": True,
    }

    report["host"] = section("host")(sec_host)
    report["gpu"] = section("gpu")(sec_gpu)
    report["disk"] = section("disk")(sec_disk)
    report["repo"] = section("repo")(sec_repo)
    report["conda_envs"] = section("conda envs (slow: imports habitat_sim)")(sec_conda_envs)
    report["habitat_builds"] = section("habitat-sim builds")(sec_habitat_builds)
    report["hm3d"] = section("HM3D")(sec_hm3d)
    report["weights"] = section("model weights")(sec_weights)

    report["verdict"] = build_verdict(report)

    json_path = out_dir / "inventory.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=False))

    # A short human-readable summary, because the JSON is long and the thing that
    # gets pasted back into the ticket should be readable.
    lines = []
    add = lines.append
    add("# Ticket 05 — RACE box inventory")
    add("")
    add(f"Generated: {report['generated_utc']}")
    h = report["host"]
    add(f"Host: {h.get('hostname')} — {h.get('platform')}")
    add(f"glibc: {h.get('glibc')} (>=2.29 required: {h.get('glibc_ok_for_rlr_audio')})")
    add(f"CPU cores: {h.get('cpu_count')}")
    g = report.get("gpu", {})
    if g.get("gpus"):
        for gpu in g["gpus"]:
            add(
                f"GPU: {gpu.get('name')} | driver {gpu.get('driver_version')} | "
                f"VRAM free {gpu.get('memory.free')} of {gpu.get('memory.total')}"
            )
        add(f"CUDA (driver max): {g.get('cuda_driver_max')} | nvcc: {g.get('nvcc')}")
    else:
        add(f"GPU: {g.get('error', 'none detected')}")
    d = report.get("disk", {}).get("repo_filesystem", {})
    add(f"Disk (repo fs): {d.get('free')} free of {d.get('total')}")
    add("")
    add("## Envs")
    for env in report.get("conda_envs", {}).get("envs", []):
        pr = env.get("probe") or {}
        hs = pr.get("habitat_sim")
        if isinstance(hs, dict) and "error" not in hs:
            hs_str = f"habitat_sim {hs.get('version')} audio={hs.get('audio_capable')}"
        else:
            hs_str = "no habitat_sim"
        torch_info = pr.get("torch")
        t_str = (
            f"torch {torch_info.get('version')} cuda={torch_info.get('cuda_available')}"
            if isinstance(torch_info, dict) and "error" not in torch_info
            else "no torch"
        )
        add(f"- **{env['name']}** — py {pr.get('python')}, numpy {pr.get('numpy')}, {hs_str}, {t_str}")
    add("")
    add("## HM3D")
    cov = report.get("hm3d", {}).get("val_mesh_coverage", {})
    add(f"- val mesh coverage: {cov.get('verdict', cov.get('error'))}")
    for name, s in (report.get("hm3d", {}).get("scene_dataset_splits") or {}).items():
        add(
            f"- split `{name}`: {s.get('basis_glb')} basis.glb, "
            f"{s.get('semantic_glb')} semantic.glb, {s.get('size')}"
        )
    add("")
    add("## Large consumers")
    for path, s in (report.get("disk", {}).get("large_consumers") or {}).items():
        add(f"- {s.get('human')}  {path}")
    add("")
    add("## Keep / rebuild / delete")
    for c in report["verdict"]:
        add(f"- **{c['call']}** — {c['item']}: {c['why']}")

    summary = "\n".join(lines)
    (out_dir / "inventory.md").write_text(summary + "\n")

    print()
    print(summary)
    print()
    print("=" * 72)
    print(f"Wrote {json_path}")
    print(f"Wrote {out_dir / 'inventory.md'}")
    print("Paste inventory.md back into ticket 05 to resolve it.")
    print("=" * 72)


if __name__ == "__main__":
    main()
