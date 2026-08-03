#!/usr/bin/env python3
"""Ticket 15 — VRAM attribution and the clean room's VRAM budget.

Three stages, deliberately separated because they have different blast radii and
different environment requirements.

``--attribute``  stdlib-only, read-only, runs in ANY env (even base, even while
                 ticket 06's sweep is timing renders underneath it). Answers
                 "what holds the 24 GB".
``--release``    the only destructive path. Refuses to act without ``--yes``,
                 refuses to touch anything it cannot attribute to this user, and
                 protects ticket 06's sweep by cmdline pattern.
``--budget``     needs the ``ss2`` env (torch + habitat_sim). Loads the clean
                 room's stack in the order the runner will and records the
                 driver-visible cost of each component.

Why the stages are split rather than one script:

Ticket 05 measured 8,249 MiB free of 32,768 and declined to act. Ticket 06 is
timing audio renders on the same box right now, and a cost measurement taken
under memory pressure is a validity problem rather than a rounding error. So
attribution has to be safe to run mid-sweep, and release has to be impossible to
run by accident.

Usage on the box::

    eval "$(~/miniconda3/bin/conda shell.bash hook)"
    conda activate ss2
    mkdir -p runs/ss2-vram
    python3 .scratch/ss2-clean-room/probes/vram_probe.py --attribute \
        --out runs/ss2-vram/attribution.json
    # read it, decide, then if there is something to reap:
    python3 .scratch/ss2-clean-room/probes/vram_probe.py --release --yes \
        --out runs/ss2-vram/release.json
    python3 .scratch/ss2-clean-room/probes/vram_probe.py --budget \
        --out runs/ss2-vram/budget.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pwd
import re
import shlex
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

MIB = 1024 ** 2
GIB = 1024 ** 3

# Ticket 06 is timing audio renders on this box. Killing it would destroy a
# measurement the map's destination hangs off, so it is protected by default.
DEFAULT_PROTECT = ("rendercost", "oneenv_gate", "audioguard", "vram_probe")


def banner(msg: str) -> None:
    print("\n=== {} ===".format(msg), flush=True)


def run(cmd: List[str], timeout: int = 60) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except FileNotFoundError:
        return 127, "", "{} not found on PATH".format(cmd[0])
    except subprocess.TimeoutExpired:
        return 124, "", "{} timed out after {}s".format(cmd[0], timeout)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


# ----------------------------------------------------------------------
# stage 1 — attribution
# ----------------------------------------------------------------------


def query_gpu() -> Dict[str, Any]:
    fields = [
        "index",
        "name",
        "memory.total",
        "memory.used",
        "memory.free",
        "utilization.gpu",
        "persistence_mode",
        "compute_mode",
        "ecc.mode.current",
    ]
    rc, out, err = run(
        ["nvidia-smi", "--query-gpu={}".format(",".join(fields)), "--format=csv,noheader,nounits"]
    )
    if rc != 0:
        raise RuntimeError("nvidia-smi --query-gpu failed (rc={}): {}".format(rc, err or out))
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        gpus.append(dict(zip(fields, parts)))
    return {"gpus": gpus}


def query_compute_apps() -> Dict[str, Any]:
    """Compute contexts only.

    A leaked habitat-sim holds an EGL/OpenGL *graphics* context, which does not
    appear here on most driver builds. That is why the full ``nvidia-smi`` text
    is captured too — ticket 05 captured it, scraped the CUDA version out of it,
    and threw the process table away.
    """
    fields = ["pid", "process_name", "used_gpu_memory"]
    rc, out, err = run(
        ["nvidia-smi", "--query-compute-apps={}".format(",".join(fields)),
         "--format=csv,noheader,nounits"]
    )
    if rc != 0:
        return {"supported": False, "error": "rc={}: {}".format(rc, err or out), "apps": []}
    apps = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        row = dict(zip(fields, parts))
        try:
            row["pid"] = int(row["pid"])
            row["used_mib"] = int(row["used_gpu_memory"])
        except (ValueError, KeyError):
            row["used_mib"] = None
        apps.append(row)
    return {"supported": True, "apps": apps}


_PROC_TEXT_RE = re.compile(
    r"^\|\s+(?P<gpu>\d+)\s+N/A\s+N/A\s+(?P<pid>\d+)\s+(?P<type>\S+)\s+(?P<name>.+?)\s+(?P<mem>\d+)MiB"
)
_PROC_TEXT_RE_OLD = re.compile(
    r"^\|\s+(?P<gpu>\d+)\s+(?P<pid>\d+)\s+(?P<type>\S+)\s+(?P<name>.+?)\s+(?P<mem>\d+)MiB"
)


def parse_smi_text(text: str) -> List[Dict[str, Any]]:
    """Parse the process table out of full ``nvidia-smi`` output.

    Both the modern (GPU/GI/CI/PID) and older (GPU/PID) column layouts are tried,
    because the type column ('C', 'G', 'C+G') is the whole point: it is the only
    place a graphics-only context shows up.
    """
    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        m = _PROC_TEXT_RE.match(line) or _PROC_TEXT_RE_OLD.match(line)
        if not m:
            continue
        rows.append(
            {
                "gpu": int(m.group("gpu")),
                "pid": int(m.group("pid")),
                "type": m.group("type"),
                "process_name": m.group("name").strip(),
                "used_mib": int(m.group("mem")),
            }
        )
    return rows


def inspect_pid(pid: int) -> Dict[str, Any]:
    """Turn a bare PID into something a human can make a decision about.

    An unreadable /proc entry is itself evidence: it means the PID belongs to
    another user or another container, which is a different problem from a stale
    process of our own.
    """
    info: Dict[str, Any] = {"pid": pid}
    base = "/proc/{}".format(pid)
    if not os.path.isdir(base):
        info["exists"] = False
        info["note"] = "no /proc entry — PID is in another namespace or already gone"
        return info
    info["exists"] = True

    try:
        with open(base + "/cmdline", "rb") as fh:
            raw = fh.read()
        info["cmdline"] = " ".join(p for p in raw.decode("utf-8", "replace").split("\0") if p)
    except OSError as exc:
        info["cmdline_error"] = str(exc)

    try:
        st = os.stat(base)
        info["uid"] = st.st_uid
        try:
            info["user"] = pwd.getpwuid(st.st_uid).pw_name
        except KeyError:
            info["user"] = str(st.st_uid)
        info["start_epoch"] = st.st_ctime
        info["age_hours"] = round((time.time() - st.st_ctime) / 3600.0, 2)
    except OSError as exc:
        info["stat_error"] = str(exc)

    try:
        with open(base + "/status") as fh:
            for line in fh:
                if line.startswith("State:"):
                    info["state"] = line.split(":", 1)[1].strip()
                elif line.startswith("PPid:"):
                    info["ppid"] = int(line.split(":", 1)[1].strip())
                elif line.startswith("VmRSS:"):
                    info["rss"] = line.split(":", 1)[1].strip()
    except OSError as exc:
        info["status_error"] = str(exc)

    try:
        info["cwd"] = os.readlink(base + "/cwd")
    except OSError as exc:
        info["cwd_error"] = str(exc)

    try:
        info["exe"] = os.readlink(base + "/exe")
    except OSError as exc:
        info["exe_error"] = str(exc)

    return info


def probe_fuser() -> Dict[str, Any]:
    """Fallback attribution when the driver's process table is blind.

    ``fuser`` reports PIDs in *our* namespace holding the device nodes open, so a
    disagreement with nvidia-smi localises the blindness.
    """
    nodes = sorted(glob.glob("/dev/nvidia*"))
    if not nodes:
        return {"available": False, "reason": "no /dev/nvidia* nodes"}
    rc, out, err = run(["fuser", "-v"] + nodes)
    if rc == 127:
        return {"available": False, "reason": err}
    # fuser reports on stderr by design and exits 1 when nothing holds the file.
    pids = sorted({int(tok) for tok in re.findall(r"\b(\d+)\b", out) if tok.isdigit()})
    return {"available": True, "rc": rc, "stdout": out, "stderr": err, "pids": pids}


def attribute() -> Dict[str, Any]:
    banner("stage 1 — attribution (read-only)")
    report: Dict[str, Any] = {"stage": "attribute", "timestamp": time.time()}

    gpu = query_gpu()
    report["gpu"] = gpu
    g0 = gpu["gpus"][0]
    total_mib = int(g0["memory.total"])
    used_mib = int(g0["memory.used"])
    free_mib = int(g0["memory.free"])
    print("  total {} MiB | used {} MiB | free {} MiB".format(total_mib, used_mib, free_mib))
    print("  compute_mode={} persistence={} ecc={}".format(
        g0.get("compute_mode"), g0.get("persistence_mode"), g0.get("ecc.mode.current")))

    rc, smi_text, smi_err = run(["nvidia-smi"])
    if rc != 0:
        raise RuntimeError("bare nvidia-smi failed (rc={}): {}".format(rc, smi_err))
    # Captured verbatim: this is the artifact ticket 05 had in hand and discarded.
    report["nvidia_smi_text"] = smi_text
    text_rows = parse_smi_text(smi_text)
    report["processes_from_text"] = text_rows

    compute = query_compute_apps()
    report["compute_apps"] = compute

    merged: Dict[int, Dict[str, Any]] = {}
    for row in text_rows:
        merged[row["pid"]] = dict(row)
    for row in compute.get("apps", []):
        pid = row.get("pid")
        if isinstance(pid, int):
            merged.setdefault(pid, {"pid": pid, "type": "C"}).update(
                {"process_name": row.get("process_name"), "used_mib": row.get("used_mib")}
            )

    for pid, row in merged.items():
        row["proc"] = inspect_pid(pid)
    report["processes"] = [merged[k] for k in sorted(merged)]

    attributed_mib = sum(int(r.get("used_mib") or 0) for r in report["processes"])
    unattributed_mib = used_mib - attributed_mib
    report["attributed_mib"] = attributed_mib
    report["unattributed_mib"] = unattributed_mib
    report["total_mib"] = total_mib
    report["used_mib"] = used_mib
    report["free_mib"] = free_mib

    print("  attributed to visible processes: {} MiB".format(attributed_mib))
    print("  UNATTRIBUTED: {} MiB".format(unattributed_mib))

    report["fuser"] = probe_fuser()

    # The discriminating verdict. These three cases need different actions, and
    # conflating them is how "just kill it" becomes a support ticket.
    if attributed_mib == 0 and used_mib > 1024:
        verdict = "BLIND"
        detail = (
            "The driver reports {} MiB in use and names no process. This box is a "
            "pod, so the most likely cause is PID-namespace isolation: the holder "
            "is outside this container (another tenant, or the host). Nothing here "
            "can reap it — escalate to RACE, or restart the pod.".format(used_mib)
        )
    elif unattributed_mib > 1024:
        verdict = "PARTIAL"
        detail = (
            "{} MiB is named, {} MiB is not. The named processes are reapable; the "
            "remainder is not visible from inside this namespace.".format(
                attributed_mib, unattributed_mib)
        )
    elif used_mib > 1024:
        verdict = "ATTRIBUTED"
        detail = "All {} MiB in use is accounted for by visible processes.".format(used_mib)
    else:
        verdict = "CLEAR"
        detail = "Nothing meaningful is held. The 24 GB is gone — it was transient."
    report["verdict"] = verdict
    report["verdict_detail"] = detail
    print("  VERDICT: {} — {}".format(verdict, detail))

    return report


# ----------------------------------------------------------------------
# stage 2 — release
# ----------------------------------------------------------------------


def build_kill_list(report: Dict[str, Any], protect: Tuple[str, ...]) -> Dict[str, Any]:
    """Decide what is reapable, and record a reason for every exclusion.

    Conservative on purpose: this runs on a box holding a three-month reservation
    and an in-flight measurement.
    """
    me = os.getuid()
    self_pid = os.getpid()
    candidates: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []

    for row in report.get("processes", []):
        pid = row["pid"]
        proc = row.get("proc", {})
        cmdline = proc.get("cmdline", "") or ""
        reason: Optional[str] = None

        if pid == self_pid:
            reason = "this probe"
        elif not proc.get("exists"):
            reason = "no /proc entry — not ours to kill"
        elif proc.get("uid") != me:
            reason = "owned by uid {} not {}".format(proc.get("uid"), me)
        else:
            hit = next((p for p in protect if p in cmdline), None)
            if hit:
                reason = "protected pattern {!r} — in-flight measurement".format(hit)

        if reason:
            excluded.append({"pid": pid, "used_mib": row.get("used_mib"),
                             "cmdline": cmdline, "reason": reason})
        else:
            candidates.append({"pid": pid, "used_mib": row.get("used_mib"),
                               "cmdline": cmdline, "state": proc.get("state"),
                               "age_hours": proc.get("age_hours")})

    return {"candidates": candidates, "excluded": excluded}


def release(confirm: bool, protect: Tuple[str, ...], grace: float) -> Dict[str, Any]:
    banner("stage 2 — release (DESTRUCTIVE)")
    before = attribute()
    plan = build_kill_list(before, protect)
    report: Dict[str, Any] = {
        "stage": "release",
        "before": {"free_mib": before["free_mib"], "used_mib": before["used_mib"]},
        "plan": plan,
        "protect_patterns": list(protect),
    }

    print("\n  would signal {} process(es):".format(len(plan["candidates"])))
    for c in plan["candidates"]:
        print("    pid {:<8} {:>7} MiB  age {}h  {}".format(
            c["pid"], c["used_mib"], c["age_hours"], (c["cmdline"] or "?")[:100]))
    print("  excluded {}:".format(len(plan["excluded"])))
    for e in plan["excluded"]:
        print("    pid {:<8} {}".format(e["pid"], e["reason"]))

    if not plan["candidates"]:
        report["action"] = "none"
        report["note"] = (
            "Nothing reapable. If the verdict was BLIND the memory is held outside "
            "this namespace and only a pod restart or RACE support can clear it."
        )
        print("\n  nothing to do: {}".format(report["note"]))
        return report

    if not confirm:
        report["action"] = "dry-run"
        print("\n  DRY RUN — re-run with --yes to actually signal these.")
        return report

    signalled = []
    for c in plan["candidates"]:
        try:
            os.kill(c["pid"], signal.SIGTERM)
            signalled.append({"pid": c["pid"], "signal": "SIGTERM"})
        except OSError as exc:
            signalled.append({"pid": c["pid"], "error": str(exc)})
    report["signalled"] = signalled

    deadline = time.time() + grace
    while time.time() < deadline:
        time.sleep(1.0)
        if all(not os.path.isdir("/proc/{}".format(s["pid"])) for s in signalled if "pid" in s):
            break

    survivors = [s["pid"] for s in signalled
                 if "pid" in s and os.path.isdir("/proc/{}".format(s["pid"]))]
    report["survivors_after_sigterm"] = survivors
    if survivors:
        # Left to the human deliberately: SIGKILL on a process holding a GPU
        # context can leave the device in a worse state than the leak did.
        print("\n  {} survived SIGTERM: {}".format(len(survivors), survivors))
        print("  NOT escalating to SIGKILL automatically — decide by hand.")

    after = query_gpu()["gpus"][0]
    report["after"] = {
        "free_mib": int(after["memory.free"]),
        "used_mib": int(after["memory.used"]),
    }
    report["freed_mib"] = report["after"]["free_mib"] - report["before"]["free_mib"]
    report["action"] = "signalled"
    print("\n  free {} MiB -> {} MiB (freed {} MiB)".format(
        report["before"]["free_mib"], report["after"]["free_mib"], report["freed_mib"]))
    return report


# ----------------------------------------------------------------------
# stage 3 — budget
# ----------------------------------------------------------------------


def _free_bytes_driver() -> int:
    """Driver-visible free memory, in bytes.

    ``nvidia-smi`` rather than torch, so this works before CUDA is initialised
    and so the baseline includes anything torch cannot see.
    """
    rc, out, err = run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
    )
    if rc != 0:
        raise RuntimeError("nvidia-smi failed (rc={}): {}".format(rc, err or out))
    return int(out.splitlines()[0].strip()) * MIB


def _find_scene(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    patterns = [
        "data/scene_datasets/hm3d/minival/**/*.basis.glb",
        "data/scene_datasets/hm3d/minival/**/*.glb",
        "data/scene_datasets/hm3d/val/**/*.basis.glb",
    ]
    for pattern in patterns:
        hits = [p for p in glob.glob(pattern, recursive=True)
                if "semantic" not in os.path.basename(p)]
        if hits:
            return sorted(hits)[0]
    raise RuntimeError("no HM3D .glb found — pass --scene explicitly")


class Ledger:
    """Records driver-visible cost per component, in load order.

    Deltas come from ``cudaMemGetInfo`` rather than ``torch.cuda.max_memory_allocated``.
    The allocator view is what ticket 13 recorded for CLAP (0.713 GB) and it
    systematically undercounts a *budget*: it excludes the CUDA context (a few
    hundred MB, paid once), allocator reserve beyond live tensors, and every
    non-torch allocation — which on this stack includes habitat-sim's GL context
    and scene textures, the largest single item nobody has measured.
    """

    def __init__(self) -> None:
        self.baseline = _free_bytes_driver()
        self.last = self.baseline
        self.rows: List[Dict[str, Any]] = []
        print("  baseline free: {:.2f} GiB".format(self.baseline / GIB))

    def mark(self, name: str, note: str = "") -> Dict[str, Any]:
        now = _free_bytes_driver()
        row = {
            "component": name,
            "delta_gib": round((self.last - now) / GIB, 3),
            "cumulative_gib": round((self.baseline - now) / GIB, 3),
            "free_after_gib": round(now / GIB, 3),
            "note": note,
        }
        self.last = now
        self.rows.append(row)
        print("  {:<28} +{:>6.3f} GiB  (cum {:>6.3f}, free {:>6.3f})".format(
            name, row["delta_gib"], row["cumulative_gib"], row["free_after_gib"]))
        return row


def budget(scene: Optional[str], sample_rate: float, with_captioner: bool,
           with_clip: bool, captioner_model: str) -> Dict[str, Any]:
    banner("stage 3 — VRAM budget (needs the ss2 env)")
    report: Dict[str, Any] = {"stage": "budget", "timestamp": time.time()}

    total_mib = int(query_gpu()["gpus"][0]["memory.total"])
    report["total_gib"] = round(total_mib * MIB / GIB, 3)

    ledger = Ledger()
    report["baseline_free_gib"] = round(ledger.baseline / GIB, 3)
    if ledger.baseline < 12 * GIB:
        report["contended"] = True
        print("  WARNING: baseline free is under 12 GiB. This budget is measured "
              "under contention — run --release first or the margin is not the "
              "clean room's margin.")
    else:
        report["contended"] = False

    import torch  # noqa: E402  (deliberately late: the baseline must precede CUDA init)

    report["torch_version"] = torch.__version__
    if not torch.cuda.is_available():
        raise RuntimeError("torch reports no CUDA — wrong env, or the driver is unhappy")

    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    ledger.mark("cuda_context", "paid once per process, invisible to torch's allocator")

    # Order mirrors the runner: simulator first, audio sensor on top of it.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from audio_guard import apply_audio_config  # noqa: E402

    import habitat_sim  # noqa: E402

    scene_path = _find_scene(scene)
    report["scene"] = scene_path

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    for field, value in (("load_semantic_mesh", False), ("enable_physics", False)):
        if hasattr(backend_cfg, field):
            setattr(backend_cfg, field, value)
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
    ledger.mark("habitat_sim + HM3D scene", "GL context, scene mesh and textures")

    spec = habitat_sim.AudioSensorSpec()
    apply_audio_config(
        spec,
        {
            "uuid": "audio_sensor",
            "enableMaterials": False,  # permanently off, ADR-0007
            "acousticsConfig": {"sampleRate": sample_rate},
        },
    )
    spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
    spec.channelLayout.channelCount = 2
    sim.add_sensor(spec)
    ledger.mark("audio sensor (spec added)", "before any render — geometry not yet uploaded")

    import numpy as np  # noqa: E402

    audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
    if sim.pathfinder.is_loaded:
        listener = sim.pathfinder.get_random_navigable_point()
        source = sim.pathfinder.get_random_navigable_point()
    else:
        listener = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        source = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = listener
    agent.set_state(state)
    audio_sensor.setAudioSourceTransform(np.asarray(source, dtype=np.float32))
    obs = sim.get_sensor_observations()
    report["ir_shape"] = list(np.asarray(obs["audio_sensor"]).shape)
    # The mesh upload is lazy (ticket 12): it happens on the first render, not at
    # context creation, so this is the mark that prices the audio engine at all.
    ledger.mark("audio first render", "lazy mesh upload happens here, not at add_sensor")

    report["clap"] = {}
    from transformers import ClapModel, ClapProcessor  # noqa: E402

    clap_id = "laion/clap-htsat-unfused"
    clap = ClapModel.from_pretrained(clap_id).to("cuda").eval()
    ledger.mark("CLAP weights", clap_id)
    wave = np.zeros(int(48000 * 1.0), dtype=np.float32)
    processor = ClapProcessor.from_pretrained(clap_id)
    labels = ["a baby crying", "a smoke alarm", "glass breaking"]
    kw = {"text": labels, "return_tensors": "pt", "padding": True, "sampling_rate": 48000}
    try:
        inputs = processor(audio=[wave], **kw)
    except TypeError:
        inputs = processor(audios=[wave], **kw)
    inputs = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        out = clap(**inputs)
    report["clap"]["logits_shape"] = list(out.logits_per_audio.shape)
    report["clap"]["allocator_peak_gib"] = round(torch.cuda.max_memory_allocated() / GIB, 3)
    ledger.mark("CLAP forward", "activations on top of weights")

    # The two optional loads are non-fatal on purpose. They price components the
    # smoke does not run (ADR-0008: oracle STOP, CLIP gone unless ticket 09
    # revives ADR-0002), so a transformers/dtype failure here must not destroy a
    # ledger that already holds the measured sim + audio + CLAP rows.
    if with_clip:
        clip_id = "openai/clip-vit-base-patch32"
        try:
            from transformers import CLIPModel  # noqa: E402
            CLIPModel.from_pretrained(clip_id).to("cuda").eval()
            ledger.mark("CLIP", "{} — only in the stack if ticket 09 keeps ADR-0002".format(clip_id))
        except Exception as exc:
            report["clip_error"] = "{}: {}".format(type(exc).__name__, exc)
            print("  CLIP FAILED (recorded, not fatal): {}".format(report["clip_error"]))

    if with_captioner:
        try:
            from transformers import AutoModelForVision2Seq  # noqa: E402
            cap = AutoModelForVision2Seq.from_pretrained(
                captioner_model, torch_dtype=torch.float16).to("cuda").eval()
            del cap
            ledger.mark(captioner_model, "R2 only — the smoke runs an oracle STOP (ADR-0008)")
        except Exception as exc:
            report["captioner_error"] = "{}: {}".format(type(exc).__name__, exc)
            print("  captioner FAILED (recorded, not fatal): {}".format(report["captioner_error"]))

    report["ledger"] = ledger.rows
    resident_gib = round((ledger.baseline - _free_bytes_driver()) / GIB, 3)
    report["resident_gib"] = resident_gib
    report["margin_vs_baseline_gib"] = round(ledger.baseline / GIB - resident_gib, 3)
    report["margin_vs_total_gib"] = round(report["total_gib"] - resident_gib, 3)

    print("\n  co-resident stack: {:.3f} GiB".format(resident_gib))
    print("  margin against the {:.2f} GiB baseline free: {:.3f} GiB".format(
        ledger.baseline / GIB, report["margin_vs_baseline_gib"]))
    print("  margin against the {:.2f} GiB card:          {:.3f} GiB".format(
        report["total_gib"], report["margin_vs_total_gib"]))

    sim.close()
    return report


# ----------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attribute", action="store_true", help="stage 1, read-only, any env")
    ap.add_argument("--release", action="store_true", help="stage 2, DESTRUCTIVE")
    ap.add_argument("--budget", action="store_true", help="stage 3, needs the ss2 env")
    ap.add_argument("--yes", action="store_true", help="arm stage 2; without it, dry run")
    ap.add_argument("--protect", action="append", default=None,
                    help="cmdline substring to never signal (repeatable)")
    ap.add_argument("--grace", type=float, default=15.0, help="seconds to wait after SIGTERM")
    ap.add_argument("--scene", default=None, help="explicit HM3D .glb for stage 3")
    ap.add_argument("--sample-rate", type=float, default=48000.0)
    ap.add_argument("--with-captioner", action="store_true",
                    help="price Qwen2-VL-2B (R2 only; the smoke uses an oracle STOP)")
    ap.add_argument("--captioner-model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--with-clip", action="store_true",
                    help="price CLIP (only if ticket 09 keeps the ADR-0002 room classifier)")
    ap.add_argument("--out", default=None, help="write the report JSON here")
    args = ap.parse_args()

    if not (args.attribute or args.release or args.budget):
        ap.error("pick at least one of --attribute / --release / --budget")

    protect = tuple(args.protect) if args.protect else DEFAULT_PROTECT
    report: Dict[str, Any] = {
        "argv": " ".join(shlex.quote(a) for a in sys.argv),
        "cwd": os.getcwd(),
        "python": sys.version.split()[0],
    }
    rc = 0
    try:
        if args.attribute:
            report["attribute"] = attribute()
        if args.release:
            report["release"] = release(args.yes, protect, args.grace)
        if args.budget:
            report["budget"] = budget(args.scene, args.sample_rate,
                                      args.with_captioner, args.with_clip,
                                      args.captioner_model)
    except Exception as exc:  # recorded, then re-raised through the exit code
        report["error"] = "{}: {}".format(type(exc).__name__, exc)
        print("\nFAILED: {}".format(report["error"]), file=sys.stderr, flush=True)
        rc = 1

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        print("\nwrote {}".format(args.out), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
