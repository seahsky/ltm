"""
video_overlay — draw an informative HUD onto an episode-video frame.

The saved first-person clip (``--save-video``) is far more useful for QA/demo when
each frame carries the agent's *state*: task + goal, last action, memory activity
(how many LTM candidates were injected / chosen, the committed waypoint), the VLM
caption, and — for the AudioGoal task — the audio onset / CLAP class / DOA / energy.

``draw_overlay`` is intentionally **pure and bullet-proof**:
  * it never mutates the caller's array (it copies),
  * it never raises (any failure — Pillow missing, malformed hud, odd shape —
    returns the input frame unchanged, so a missing dep can never take down a
    multi-hour ablation),
  * an empty/None hud is a no-op (returns the input frame).

Pillow is imported lazily inside the function (mirrors ``video_recorder``'s
never-crash convention). cv2 is deliberately NOT used (not a repo dependency).

The runner builds the ``hud`` dict (see ``EpisodeRunner._build_hud``) with already
string-mapped fields so this helper stays dependency-free and trivially testable.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

# Energy reference for the audio mini-bar (full bar ≈ this RMS); onset fires ~0.065.
_ENERGY_REF = 0.10
# DOA arrow by lateral sign (world-frame left/right; ASCII for bitmap-font safety).
_DOA = {-1: "<", 0: ".", 1: ">"}


def _load_font(height: int, *, big: bool = False):
    """Best-effort scalable font; falls back to Pillow's bitmap default.

    Module-level so tests can monkeypatch it to exercise the never-raise path.
    """
    from PIL import ImageFont  # lazy

    size = max(10, int(round(height / (8 if big else 26))))
    for path in ("DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def _fmt(v: Any) -> str:
    return "—" if v is None else str(v)


def _fmt_f(v: Any, nd: int = 2) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def _hud_lines(hud: Dict[str, Any]) -> List[str]:
    """The top-left HUD text block (one entry per line)."""
    wp_src = hud.get("wp_source")
    wp = f"{wp_src} {_fmt_f(hud.get('wp_dist'), 1)}m" if wp_src else "—"
    lines = [
        f"[{_fmt(hud.get('task'))} | {_fmt(hud.get('backbone'))}]  t={_fmt(hud.get('step'))}",
        f"goal: {_fmt(hud.get('goal'))}    d2g: {_fmt_f(hud.get('d2g'))}",
        f"{_fmt(hud.get('action'))}   wp: {wp}",
        f"mem {_fmt(hud.get('n_mem_chosen'))}/{_fmt(hud.get('n_mem_inj'))} chosen/inj",
    ]
    cap = hud.get("caption")
    if cap:
        cap = str(cap).replace("\n", " ")
        lines.append("cap: " + (cap[:48] + "…" if len(cap) > 48 else cap))
    return lines


def _audio_line(audio: Dict[str, Any]) -> str:
    """The bottom AudioGoal strip (one line)."""
    detected = bool(audio.get("detected"))
    klass = _fmt(audio.get("klass"))
    doa = _DOA.get(audio.get("lateral"), "?") if audio.get("lateral") is not None else "?"
    try:
        n = int(round(min(1.0, max(0.0, float(audio.get("energy") or 0.0)) / _ENERGY_REF) * 6))
    except (TypeError, ValueError):
        n = 0
    bar = "#" * n + "-" * (6 - n)
    seg = f"AUDIO[{'ON ' if detected else 'off'}] {klass}  DOA {doa}  E[{bar}]"
    if audio.get("onset_fired"):
        seg += "  *ONSET*"
    elif audio.get("onset_step") is not None:
        seg += f"  onset@{audio.get('onset_step')}"
    tgt = audio.get("target")
    if tgt:
        seg += f"  ->{tgt}"
    return seg


def draw_overlay(frame_rgb: np.ndarray, hud: Optional[Dict[str, Any]]) -> np.ndarray:
    """Return a NEW (H,W,3) uint8 frame = ``frame_rgb`` with the ``hud`` drawn on top.

    Pure (no mutation, no I/O). Never raises: on any failure or a None/empty hud,
    returns the input frame unchanged.
    """
    if not hud or not isinstance(hud, dict):
        return frame_rgb
    try:
        from PIL import Image, ImageDraw

        arr = np.ascontiguousarray(frame_rgb)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return frame_rgb
        arr = arr.astype(np.uint8, copy=True)
        h, w = arr.shape[:2]

        base = Image.fromarray(arr, mode="RGB").convert("RGBA")
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        font = _load_font(h)
        pad = max(2, h // 100)
        spacing = max(1, h // 160)

        # --- top-left HUD block (dark box + white text) ---
        text = "\n".join(_hud_lines(hud))
        try:
            bb = draw.multiline_textbbox((pad, pad), text, font=font, spacing=spacing)
        except Exception:  # noqa: BLE001 — very old Pillow: skip the box, still draw text
            bb = (pad, pad, w - pad, pad + h // 3)
        draw.rectangle([0, 0, min(w, bb[2] + 2 * pad), min(h, bb[3] + pad)], fill=(0, 0, 0, 165))
        draw.multiline_text((pad, pad), text, font=font, fill=(255, 255, 255, 255), spacing=spacing)

        # --- bottom AudioGoal strip ---
        audio = hud.get("audio")
        if isinstance(audio, dict):
            seg = _audio_line(audio)
            try:
                ab = draw.textbbox((pad, 0), seg, font=font)
                bar_h = (ab[3] - ab[1]) + 2 * pad
            except Exception:  # noqa: BLE001
                bar_h = max(12, h // 12)
            detected = bool(audio.get("detected"))
            box_c = (10, 70, 10, 175) if detected else (0, 0, 0, 165)
            draw.rectangle([0, h - bar_h, w, h], fill=box_c)
            draw.text((pad, h - bar_h + pad), seg, font=font,
                      fill=((90, 255, 90, 255) if detected else (210, 210, 210, 255)))

        # --- centre STOP banner ---
        if hud.get("is_stop"):
            big = _load_font(h, big=True)
            msg = "STOP"
            try:
                sb = draw.textbbox((0, 0), msg, font=big)
                tw, th = sb[2] - sb[0], sb[3] - sb[1]
            except Exception:  # noqa: BLE001
                tw, th = w // 3, h // 8
            cx, cy = (w - tw) // 2, (h - th) // 2
            draw.rectangle([cx - pad, cy - pad, cx + tw + pad, cy + th + 2 * pad], fill=(0, 0, 0, 140))
            draw.text((cx, cy), msg, font=big, fill=(235, 50, 50, 255))

        out = Image.alpha_composite(base, layer).convert("RGB")
        return np.ascontiguousarray(np.asarray(out, dtype=np.uint8))
    except Exception:  # noqa: BLE001 — overlay must never break a run
        return frame_rgb
