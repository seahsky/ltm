"""
Wiring tests for the coarse head's CLIP room classifier (Stage 5):
``EpisodeRunner._get_room_classifier`` + its call site.

The classifier ALGORITHM is covered by test_room_classifier.py (build + classify
on synthetic embeddings). This suite covers the RUNNER WRAPPER — the thin env-gated
lazy-cache that turns the already-loaded ``clip_encoder`` into the per-anchor room
signal passed to ``propose_coarse_candidates``:

  W1  default ON (coarse on) -> returns a working classifier closure that maps a
      room's image embedding to that room.
  W2  ``LTM_COARSE_ROOM_CLIP=0`` -> None (caption-only A/B baseline).
  W3  caching: built once, the SAME closure is returned and ``encode_text`` is
      called exactly 6 times total (one per room prompt), not per step.
  W4  no clip_encoder / encode_text raises -> None (graceful caption-only fallback,
      never crashes the loop).
  W5  thresholds env-tunable: ``LTM_ROOM_CLIP_MIN_COS`` / ``LTM_ROOM_CLIP_MARGIN``
      flow into the closure (a high min_cos / margin forces abstain).
  W6  source-scan: the coarse call site passes ``room_classifier=self._get_room_classifier()``.

Invoke with::

    python embodied_memory/scripts/test_room_clip_wiring.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

try:
    from embodied_memory.episode_runner import EpisodeRunner
    from embodied_memory.room_resolver import ROOM_TEXT_PROMPTS
except Exception as e:  # heavy deps unavailable locally
    print(f"SKIP test_room_clip_wiring: import failed ({e})")
    sys.exit(0)


_ROOMS = list(ROOM_TEXT_PROMPTS.keys())
_DIM = 16
_PROMPT_TO_ROOM = {p: r for r, p in ROOM_TEXT_PROMPTS.items()}


class _FakeClip:
    """A CLIP stand-in whose encode_text maps each room prompt to a distinct
    orthonormal basis vector, so a room's 'image embedding' (same basis vector)
    classifies to that room with cosine 1."""

    def __init__(self):
        self.n_text_calls = 0

    def encode_text(self, prompt: str) -> np.ndarray:
        self.n_text_calls += 1
        room = _PROMPT_TO_ROOM.get(prompt)
        v = np.zeros(_DIM, dtype=np.float32)
        if room is not None:
            v[_ROOMS.index(room)] = 5.0  # deliberately not unit-norm
        return v


def _img(room: str) -> np.ndarray:
    v = np.zeros(_DIM, dtype=np.float32)
    v[_ROOMS.index(room)] = 1.0
    return v


def _runner_with(clip) -> EpisodeRunner:
    r = EpisodeRunner.__new__(EpisodeRunner)   # bypass heavy __init__
    r.clip_encoder = clip
    r._room_classifier_fn = None
    r._room_text_embeddings = None
    return r


class _EnvGuard:
    """Temporarily set/clear env vars, restoring prior state on exit."""

    def __init__(self, **kv):
        self._kv = kv
        self._saved = {}

    def __enter__(self):
        for k, v in self._kv.items():
            self._saved[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *a):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def case_default_on_returns_working_classifier():
    with _EnvGuard(LTM_COARSE_ROOM_CLIP=None, LTM_ROOM_CLIP_MIN_COS=None,
                   LTM_ROOM_CLIP_MARGIN=None):
        clf = _runner_with(_FakeClip())._get_room_classifier()
        assert clf is not None
        for room in _ROOMS:
            assert clf(_img(room)) == room, room
    print("  case default_on_returns_working_classifier: OK")


def case_env_zero_disables():
    with _EnvGuard(LTM_COARSE_ROOM_CLIP="0"):
        assert _runner_with(_FakeClip())._get_room_classifier() is None
    print("  case env_zero_disables: OK")


def case_builds_once_and_caches():
    with _EnvGuard(LTM_COARSE_ROOM_CLIP=None):
        clip = _FakeClip()
        r = _runner_with(clip)
        c1 = r._get_room_classifier()
        c2 = r._get_room_classifier()
        assert c1 is c2, "classifier should be cached (same closure)"
        assert clip.n_text_calls == len(ROOM_TEXT_PROMPTS), clip.n_text_calls
    print("  case builds_once_and_caches: OK")


def case_no_clip_encoder_is_none():
    with _EnvGuard(LTM_COARSE_ROOM_CLIP=None):
        assert _runner_with(None)._get_room_classifier() is None
    print("  case no_clip_encoder_is_none: OK")


def case_encode_text_failure_is_graceful():
    class _Boom:
        def encode_text(self, prompt):
            raise RuntimeError("no text tower")
    with _EnvGuard(LTM_COARSE_ROOM_CLIP=None):
        assert _runner_with(_Boom())._get_room_classifier() is None
    print("  case encode_text_failure_is_graceful: OK")


def case_thresholds_from_env_force_abstain():
    # a huge required margin -> even a perfect (cos 1, runner-up 0) match abstains,
    # proving LTM_ROOM_CLIP_MARGIN reaches the closure.
    with _EnvGuard(LTM_COARSE_ROOM_CLIP=None, LTM_ROOM_CLIP_MARGIN="2.0",
                   LTM_ROOM_CLIP_MIN_COS=None):
        clf = _runner_with(_FakeClip())._get_room_classifier()
        assert clf(_img(_ROOMS[0])) is None, "margin=2.0 should force abstain"
    # a min_cos above the max possible cosine -> abstain (min_cos plumbed through)
    with _EnvGuard(LTM_COARSE_ROOM_CLIP=None, LTM_ROOM_CLIP_MIN_COS="1.5",
                   LTM_ROOM_CLIP_MARGIN=None):
        clf = _runner_with(_FakeClip())._get_room_classifier()
        assert clf(_img(_ROOMS[0])) is None, "min_cos=1.5 should force abstain"
    print("  case thresholds_from_env_force_abstain: OK")


def case_default_thresholds_are_conservative():
    # MF-1 (post-verification): the runner's in-code env DEFAULTS must be the
    # conservative real-scale values (min_cos 0.25, margin 0.02), NOT the synthetic
    # 0.20/0.005 — a 0.22-cosine frame must abstain by default. Drive it through the
    # real closure: a fake CLIP image whose top room cosine is 0.22.
    import numpy as np
    with _EnvGuard(LTM_COARSE_ROOM_CLIP=None, LTM_ROOM_CLIP_MIN_COS=None,
                   LTM_ROOM_CLIP_MARGIN=None):
        clf = _runner_with(_FakeClip())._get_room_classifier()
        weak = np.zeros(_DIM, dtype=np.float32)
        weak[_ROOMS.index(_ROOMS[0])] = 0.22
        weak[_DIM - 1] = float(np.sqrt(1.0 - 0.22 ** 2))  # rest along an unused dim
        assert clf(weak) is None, "default thresholds must abstain at cos 0.22"
    print("  case default_thresholds_are_conservative: OK")


def case_cos_fn_companion():
    # _get_room_cos_fn returns the RAW top room cosine (no abstain gate), reusing the
    # cached prompt embeddings; None when room-CLIP is off.
    with _EnvGuard(LTM_COARSE_ROOM_CLIP=None):
        cos_fn = _runner_with(_FakeClip())._get_room_cos_fn()
        assert cos_fn is not None
        c = cos_fn(_img(_ROOMS[2]))
        assert abs(c - 1.0) < 1e-5, c
    with _EnvGuard(LTM_COARSE_ROOM_CLIP="0"):
        assert _runner_with(_FakeClip())._get_room_cos_fn() is None
    print("  case cos_fn_companion: OK")


def case_call_site_passes_classifier():
    src = (REPO / "embodied_memory" / "episode_runner.py").read_text()
    assert "room_classifier=self._get_room_classifier()" in src, \
        "coarse call site must pass the CLIP room classifier"
    assert "room_cos_fn=self._get_room_cos_fn()" in src, \
        "coarse call site must pass the cosine probe"
    assert 'os.environ.get("LTM_COARSE_ROOM_CLIP"' in src
    # the diag read must be present so zero-fire runs are interpretable
    assert "_last_coarse_diag" in src
    print("  case call_site_passes_classifier: OK")


def main() -> int:
    print("coarse CLIP room-classifier wiring tests")
    case_default_on_returns_working_classifier()
    case_env_zero_disables()
    case_builds_once_and_caches()
    case_no_clip_encoder_is_none()
    case_encode_text_failure_is_graceful()
    case_thresholds_from_env_force_abstain()
    case_default_thresholds_are_conservative()
    case_cos_fn_companion()
    case_call_site_passes_classifier()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
