#!/usr/bin/env python
"""
diagnose_convolved_anomaly_calib — the $0 Gate-0b: can the open-set CLAP gate
separate anomaly-vs-background on RIR-CONVOLVED (+ mixed) audio?

0c proved the gate (audio.is_anomaly), calibrated on CLEAN clips (EER 0.00,
delta 0.137), REJECTS the RIR-convolved alarm — onset never fired unless the
gate was disabled. This diagnostic recalibrates on the EXACT live signal:

  anomaly render  = render_at_pose(grid, cell, anomaly_clip)          (2, L)
  background bed   = bg_gain * diotic(render_at_pose(grid, cell, bg_clip))
  ACCEPT population = is_anomaly(anomaly + bed)      (the live-gate accept)
  REJECT population = is_anomaly(bed alone)          (the pre-t_anom reject)

and finds a (delta, tau, bg_gain) that separates them, or STOPs.

Review-driven guarantees (else it returns a false GO):
  * M1  score at the AUDIBLE-RADIUS band (cells whose anomaly render rms is near
    onset_rms — where the live gate actually fires), NOT the loudest cells.
  * M2  the EXACT live composition: spatial anomaly render + DIOTIC bed, scored
    through is_anomaly on the (2, L) array (the domain the live gate sees).
  * M3  a 2-D (delta, tau) sweep — convolution can depress s_anom absolutely
    (the 0c alarm->glass_break flip is evidence), so delta-only is insufficient.
  * R1/C2  STOP unless some bg_gain BOTH clears onset_rms on the post-convolution
    bed AND separates (else the reject-half is vacuous -> mixture is dressing).
  * M4  the bed class should not be a verbatim NORMAL_PROMPT (prompt-leakage
    inflates GO); we prefer benign clips whose class is not in the prompt bank.
  * m1  an audio-vs-audio PROTOTYPE column (text-vs-convolved-audio is CLAP's
    weakest regime — the pivot target is pre-measured, not discovered later).

Reuses best_threshold / verdict / load_wav from diagnose_normal_anomaly_calib.

NOTE: $0 of *matrix*, NOT $0 of compute/network — it loads CLAP (GPU/CPU) and
needs the ESC-50 clips staged (fetch_anomaly_clips.py --include-benign).

Run:
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python \
    embodied_memory/scripts/diagnose_convolved_anomaly_calib.py \
    --rir-grid 'runs/audiogoal/*_rir_grid.npz' --device cuda
"""
from __future__ import annotations

import argparse
import glob
import os
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagnose_normal_anomaly_calib import best_threshold, verdict  # noqa: E402


# ----------------------------------------------------------------------
# pure decision logic (unit-tested)
# ----------------------------------------------------------------------
def _cuts(vals: Sequence[float], max_cuts: int = 40) -> List[float]:
    import numpy as np
    vs = sorted(set(float(v) for v in vals))
    if len(vs) > max_cuts:
        qs = np.quantile(vs, np.linspace(0.0, 1.0, max_cuts))
        vs = sorted(set(float(q) for q in qs))
    return [min(vs) - 1e-6] + vs   # a below-min sentinel = "accept all on this axis"


def sweep_delta_tau(accept: Sequence[Tuple[float, float]],
                    reject: Sequence[Tuple[float, float]],
                    max_cuts: int = 40) -> Dict[str, Any]:
    """2-D (delta, tau) threshold sweep. accept/reject are (margin, s_anom)
    pairs. Accept a point iff margin >= delta AND s_anom >= tau. Minimize the
    balanced error (FPR + FNR)/2; tie-break to the HIGHER (delta, tau) (the more
    conservative gate). Returns the best {delta, tau, eer, tpr, fpr}."""
    if not accept or not reject:
        return {"ok": False, "reason": "need both populations"}
    na, nr = len(accept), len(reject)
    dcuts = _cuts([m for m, _ in accept] + [m for m, _ in reject], max_cuts)
    tcuts = _cuts([s for _, s in accept] + [s for _, s in reject], max_cuts)
    best_key: Optional[Tuple[float, float, float]] = None
    best: Optional[Dict[str, Any]] = None
    for d in dcuts:
        for t in tcuts:
            tp = sum(1 for m, s in accept if m >= d and s >= t) / na
            fp = sum(1 for m, s in reject if m >= d and s >= t) / nr
            eer = (fp + (1.0 - tp)) / 2.0
            key = (eer, -d, -t)   # min eer, then higher delta, then higher tau
            if best_key is None or key < best_key:
                best_key = key
                best = {"ok": True, "delta": float(d), "tau": float(t),
                        "eer": float(eer), "tpr": float(tp), "fpr": float(fp)}
    return best  # type: ignore[return-value]


def decide_gate(per_gain: List[Dict[str, Any]], onset_rms: float) -> Tuple[str, Dict[str, Any]]:
    """R1-aware verdict over the bg_gain sweep. A gain only QUALIFIES if its
    post-convolution bed RMS >= onset_rms (else the reject half never reaches the
    gate — the mixture is inert dressing). Among qualifying gains pick min EER."""
    qualifying = [g for g in per_gain if float(g.get("bed_rms_med", 0.0)) >= float(onset_rms)]
    if not qualifying:
        return "STOP", {"reason": (f"no bg_gain clears onset_rms={onset_rms:.4f} on the "
                                   "post-convolution bed — the REJECT half is vacuous "
                                   "(the gate never sees the background); the mixture "
                                   "would be inert dressing"), "recommend": None}
    best = min(qualifying, key=lambda g: float(g["eer"]))
    eer = float(best["eer"])
    res = "GO" if eer <= 0.15 else ("BORDERLINE" if eer <= 0.30 else "STOP")
    return res, {"bg_gain": best["bg_gain"], "delta": best["delta"], "tau": best["tau"],
                 "eer": eer, "bed_rms_med": best["bed_rms_med"]}


# ----------------------------------------------------------------------
# render + CLAP scoring (RACE/GPU; imports heavy deps lazily)
# ----------------------------------------------------------------------
def _diotic(sig):
    """Collapse a (2, L) binaural signal to a non-directional (2, L) bed — mono
    mean broadcast to both ears, so the bed carries no lateral/DOA cue (protects
    the anomaly's lateral_sign)."""
    import numpy as np
    m = np.asarray(sig, dtype=np.float32)
    if m.ndim == 1:
        return np.stack([m, m])
    mono = m.mean(axis=0)
    return np.stack([mono, mono])


def _align(bed, length):
    import numpy as np
    b = np.asarray(bed, dtype=np.float32)
    L = b.shape[-1]
    if L == length:
        return b
    if L > length:
        return b[..., :length]
    reps = int(np.ceil(length / max(1, L)))
    return np.tile(b, reps)[..., :length]


def _class_from_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _cos(a, b) -> float:
    import numpy as np
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def _select_audible_cells(grid, an_norm, onset_rms, band_hi, max_cells, n_probe=48):
    """M1: cells whose CONVOLVED-ANOMALY rms is in the audible band
    [onset_rms, band_hi*onset_rms] — where the live gate realistically fires —
    not the loudest cells (which give an optimistic GO the live gate can't hit)."""
    import numpy as np
    from embodied_memory.audio import render_at_pose, rms
    _ce = getattr(grid, "cell_energies", None)   # @property -> ndarray (not a method)
    energies = _ce() if callable(_ce) else _ce
    n = len(grid.cell_positions)
    order = list(np.argsort(energies)[::-1]) if energies is not None else list(range(n))
    probe = order[:n_probe]
    scored = []
    for ci in probe:
        r = float(rms(render_at_pose(grid, grid.cell_positions[ci], an_norm)))
        scored.append((ci, r))
    band = [(ci, r) for ci, r in scored if onset_rms <= r <= band_hi * onset_rms]
    if not band:   # relax: the cells closest to onset_rms from above
        above = sorted([(ci, r) for ci, r in scored if r >= onset_rms], key=lambda x: x[1])
        band = above[:max_cells] if above else sorted(scored, key=lambda x: -x[1])[:max_cells]
    band.sort(key=lambda x: x[1])   # quietest-audible first (worst-case SNR)
    return [ci for ci, _ in band[:max_cells]]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="$0 Gate-0b: CLAP anomaly-vs-bg on convolved audio.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rir-grid", required=True, help="glob of RIR grid .npz (e.g. runs/audiogoal/*_rir_grid.npz)")
    ap.add_argument("--anomaly-dir", default="data/anomaly_audio")
    ap.add_argument("--benign-dir", default="data/benign_audio")
    ap.add_argument("--onset-rms", type=float, default=0.05)
    ap.add_argument("--bg-gains", default="0.0 0.3 0.5 0.7 1.0")
    ap.add_argument("--band-hi", type=float, default=4.0, help="audible band upper = band_hi*onset_rms")
    ap.add_argument("--max-cells", type=int, default=12)
    ap.add_argument("--max-grids", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))

    grids = sorted(glob.glob(args.rir_grid))[: args.max_grids]
    an_paths = sorted(glob.glob(os.path.join(args.anomaly_dir, "*.wav")))
    bn_paths = sorted(glob.glob(os.path.join(args.benign_dir, "*.wav")))
    if not grids or not an_paths or not bn_paths:
        print(f"FATAL: need grids + clips. grids={len(grids)} ({args.rir_grid}), "
              f"anomaly={len(an_paths)} ({args.anomaly_dir}), benign={len(bn_paths)} ({args.benign_dir})")
        print("  stage clips: python embodied_memory/scripts/fetch_anomaly_clips.py --include-benign")
        return 2

    # M4: prefer bed clips whose class is NOT a verbatim NORMAL_PROMPT (avoid
    # prompt-leakage inflating GO). NORMAL_PROMPTS map roughly to these classes.
    _NORMAL_LEAK = {"footsteps", "vacuum", "people_talking", "quiet_room", "background_noise"}
    beds = [p for p in bn_paths if _class_from_name(p) not in _NORMAL_LEAK] or bn_paths
    gains = [float(x) for x in args.bg_gains.split()]

    from embodied_memory import audio
    from embodied_memory.audio import RIRGrid, render_at_pose, rms
    from embodied_memory.audio_task import build_anomaly_clip
    from embodied_memory.perception import CLAPAudioEncoder

    print(f"[gate0b] grids={len(grids)} anomaly={len(an_paths)} bed={len(beds)} "
          f"gains={gains} onset_rms={args.onset_rms} band<= {args.band_hi}x  device={args.device}")
    print(f"[gate0b] beds (non-normal-prompt preferred): {[_class_from_name(p) for p in beds]}")
    enc = CLAPAudioEncoder(device=args.device)

    # per-gain accumulators
    acc: Dict[float, Dict[str, list]] = {g: {"accept": [], "reject": [], "bed_rms": [],
                                             "cls_ok": [], "pa": [], "pr": []} for g in gains}

    for gpath in grids:
        grid = RIRGrid.load(gpath)
        for an_path in an_paths:
            true_cls = _class_from_name(an_path)
            an_norm = build_anomaly_clip(an_path, int(grid.sample_rate))
            try:
                proto = enc.encode_audio(an_norm)   # m1: clean-anomaly audio prototype
            except Exception:
                proto = None
            cells = _select_audible_cells(grid, an_norm, args.onset_rms, args.band_hi, args.max_cells)
            if not cells:
                continue
            an_by_cell = {ci: render_at_pose(grid, grid.cell_positions[ci], an_norm) for ci in cells}
            for bed_path in beds:
                bg_norm = build_anomaly_clip(bed_path, int(grid.sample_rate))
                for ci in cells:
                    an = an_by_cell[ci]
                    bed_d = _diotic(render_at_pose(grid, grid.cell_positions[ci], bg_norm))
                    for g in gains:
                        import numpy as np
                        bed_g = (g * _align(bed_d, an.shape[-1])).astype(np.float32)
                        mix = (an + bed_g).astype(np.float32)
                        _, cls_m, s_m = audio.is_anomaly(mix, int(grid.sample_rate), enc)
                        acc[g]["accept"].append((float(s_m["margin"]), float(s_m["s_anom"])))
                        acc[g]["cls_ok"].append(1.0 if cls_m == true_cls else 0.0)
                        if proto is not None:
                            try:
                                acc[g]["pa"].append(_cos(enc.encode_audio(mix), proto))
                            except Exception:
                                pass
                        if g > 0.0:
                            _, _clsb, s_b = audio.is_anomaly(bed_g, int(grid.sample_rate), enc)
                            acc[g]["reject"].append((float(s_b["margin"]), float(s_b["s_anom"])))
                            acc[g]["bed_rms"].append(float(rms(bed_g)))
                            if proto is not None:
                                try:
                                    acc[g]["pr"].append(_cos(enc.encode_audio(bed_g), proto))
                                except Exception:
                                    pass

    # per-gain stats
    print("\n  bg_gain   n_acc  n_rej   bed_rms_med   EER(text)   delta    tau    cls_ok   EER(proto)")
    print("  " + "-" * 88)
    per_gain = []
    for g in gains:
        a = acc[g]
        bed_med = statistics.median(a["bed_rms"]) if a["bed_rms"] else 0.0
        cls_ok = (sum(a["cls_ok"]) / len(a["cls_ok"])) if a["cls_ok"] else float("nan")
        if a["accept"] and a["reject"]:
            sw = sweep_delta_tau(a["accept"], a["reject"])
            eer, delta, tau = sw["eer"], sw["delta"], sw["tau"]
        else:
            eer, delta, tau = float("nan"), float("nan"), float("nan")
        proto_eer = float("nan")
        if a["pa"] and a["pr"]:
            pstat = best_threshold(a["pa"], a["pr"])
            proto_eer = float(pstat.get("eer", float("nan")))
        per_gain.append({"bg_gain": g, "eer": eer, "delta": delta, "tau": tau,
                         "bed_rms_med": bed_med, "cls_ok": cls_ok, "proto_eer": proto_eer,
                         "n_acc": len(a["accept"]), "n_rej": len(a["reject"])})
        print(f"  {g:>6.2f}  {len(a['accept']):>6} {len(a['reject']):>6}   {bed_med:>10.4f}   "
              f"{eer:>9.3f}   {delta:>6.3f} {tau:>6.3f}   {cls_ok:>6.2f}   {proto_eer:>9.3f}")

    # decide over gains WITH a valid reject population (g > 0)
    decidable = [pg for pg in per_gain if pg["bg_gain"] > 0.0 and pg["n_rej"] > 0
                 and not (pg["eer"] != pg["eer"])]  # drop NaN
    result, rec = decide_gate(decidable, args.onset_rms)

    print()
    print(f"GATE_RESULT={result}")
    if rec.get("recommend") is None and "reason" in rec:
        print(f"  -> {rec['reason']}")
        print("RECOMMEND_DELTA=  RECOMMEND_TAU=  RECOMMEND_BG_GAIN=")
    else:
        print(f"RECOMMEND_DELTA={rec['delta']:.4f}")
        print(f"RECOMMEND_TAU={rec['tau']:.4f}")
        print(f"RECOMMEND_BG_GAIN={rec['bg_gain']:.2f}")
        print(f"  EER={rec['eer']:.3f}  bed_rms_med={rec['bed_rms_med']:.4f}  "
              f"(bed clears onset_rms={args.onset_rms})")
    # class-flip caveat (R2): the gate is binary, but a flipped class corrupts
    # CLASS_TO_OBJECT retrieval — survivable only because the dataset supplies
    # anomaly_object_override. Report it, don't block on it.
    best_cls = max((pg["cls_ok"] for pg in per_gain if pg["cls_ok"] == pg["cls_ok"]), default=float("nan"))
    print(f"  CLASS_CORRECT_RATE(best gain)={best_cls if best_cls==best_cls else float('nan'):.2f} "
          f"(<1.0 = CLAP class-flip on convolved audio; binary gate survives, retrieval uses "
          f"dataset anomaly_object)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
