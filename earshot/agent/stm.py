"""DREAM §III.C: the short-term episodic memory, and the fusion that feeds it.

    M^S_t = {e_{t-K+1}, ..., e_t}                    (eq. 4)
    e_t   = (z^v_t, z^u_t, p_t, a_{t-1})             (eq. 5)
    z^v_t = f_v(v_t),  z^u_t = f_u(u_t)              (eq. 6)
    z^av_t = f_fuse(z^v_t, z^u_t)                    (eq. 7)

**Reset between episodes, and that is the whole difference from long-term memory.** The
paper says so in one line and it is the line that makes STM safe: nothing here crosses an
episode boundary, so no amount of it can leak the answer from one episode into the next.
`reset()` returns a NEW empty memory rather than clearing in place, which is the same
discipline `occupancy.integrate_depth` follows and for the same reason -- a caller holding
the old object still holds the old episode's history rather than silently watching it
empty.

**`f_fuse` IS CONCATENATION, AND THE BALANCE IS THE POINT.** The paper writes `f_fuse`
without saying what it is. Concatenation of the two unit halves, scaled so the result is
unit norm, is the choice that adds no parameters and loses nothing: both halves survive
intact, so a retrieval that wants only the audio (as `M^K`'s sound-to-object table does)
reads `entry.audio` rather than trying to unmix a sum. `CLIP_MODEL_ID` was chosen for
512-d precisely so the two halves weigh the same here; a lopsided fuse would make `q_t`
(eq. 19) a visual query wearing an audio hat, and the `omega_t` shift the paper's central
hypothesis is about would be measuring the fuse rather than the memory.

**THE ENTRY KEEPS BOTH HALVES AS WELL AS THE FUSION.** Eq. 5 stores `z^v` and `z^u`
separately and eq. 7 derives `z^av` from them, so the entry does the same rather than
keeping only the fused vector. That is not redundancy: eq. 20-22 retrieve from three
memories with different key spaces, and an entry that had thrown its halves away could
only ever query the one the fusion happens to match.

**A LEAF-ADJACENT MODULE (ADR-0013: `"agent": ("agent", "vlm", "types")`).** It holds
vectors, never encoders: the runner owns `ClipEncoder`/`ClapEncoder` and hands their
output in, exactly as `resolve_prior` is handed `heard_embedding` rather than a model. So
this whole file runs on a Mac with no torch, which is what makes eq. 4-7 testable at all.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional, Tuple

import numpy as np

from earshot.types import Pose

__all__ = [
    "StmEntry",
    "ShortTermMemory",
    "fuse",
]


def _unit(vector: Any) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    return values / (float(np.linalg.norm(values)) + 1e-8)


def fuse(visual: Any, audio: Any) -> np.ndarray:
    """``z^av_t = f_fuse(z^v_t, z^u_t)`` (eq. 7): the unit concatenation of both halves.

    Each half is normalised BEFORE concatenating, so neither can dominate by arriving at
    a larger scale, and the pair is then scaled by ``1/sqrt(2)`` so the whole is unit
    norm. Cosine similarity is scale-invariant, so this changes no ranking; it makes the
    fused vector interchangeable with either half wherever a unit vector is expected.

    Raises on an empty half rather than returning a half-width vector: a fused vector of
    the wrong width is a shape error several equations downstream, and the module that
    produced it is the one worth naming.
    """
    left, right = np.asarray(visual), np.asarray(audio)
    if left.size == 0 or right.size == 0:
        raise ValueError(
            "f_fuse needs both halves: got visual of size {} and audio of size {}. An "
            "empty half means its encoder failed, and a fused vector built from one half "
            "would be a valid-looking query for a memory it does not describe".format(
                left.size, right.size
            )
        )
    pair = np.concatenate([_unit(left), _unit(right)])
    return (pair / np.float32(np.sqrt(2.0))).astype(np.float32)


@dataclass(frozen=True)
class StmEntry:
    """``e_t = (z^v_t, z^u_t, p_t, a_{t-1})`` (eq. 5). One step of the current episode.

    `prev_action` is `None` at t=0 and only there -- the paper's ``a_{t-1}`` does not
    exist before the first action, and a sentinel string would make "the episode had not
    acted yet" indistinguishable from an action actually named that.
    """

    visual: np.ndarray
    audio: np.ndarray
    pose: Pose
    prev_action: Optional[str]

    def __post_init__(self) -> None:
        for name in ("visual", "audio"):
            vector = np.array(getattr(self, name), dtype=np.float32, copy=True).reshape(-1)
            if vector.size == 0:
                raise ValueError(
                    "StmEntry was given an empty {} embedding; a 0-length vector cannot "
                    "be an observation and must not reach M^S".format(name)
                )
            vector.flags.writeable = False
            object.__setattr__(self, name, vector)

    @property
    def fused(self) -> np.ndarray:
        """``z^av_t`` for this entry. Derived rather than stored, so it can never
        disagree with the halves it is made of."""
        return fuse(self.visual, self.audio)


@dataclass(frozen=True)
class ShortTermMemory:
    """``M^S_t`` (eq. 4): the last ``horizon`` entries of the CURRENT episode, oldest first.

    Immutable. `push` returns a new memory, `reset` returns an empty one, and neither
    mutates the receiver -- so a caller that kept a reference to step 5's memory still has
    step 5's memory at step 40, which is what makes a retrieval reproducible after the
    fact.
    """

    horizon: int
    entries: Tuple[StmEntry, ...] = ()

    def __post_init__(self) -> None:
        if int(self.horizon) < 1:
            raise ValueError(
                "ShortTermMemory needs a horizon of at least 1; got {}. A zero-horizon "
                "memory would drop every entry the instant it was pushed and make q_t "
                "(eq. 19) a function of the current observation alone, with nothing on "
                "disk saying the K had been set to nothing".format(self.horizon)
            )
        object.__setattr__(self, "horizon", int(self.horizon))
        if len(self.entries) > self.horizon:
            object.__setattr__(self, "entries", tuple(self.entries[-self.horizon:]))

    def push(self, entry: StmEntry) -> "ShortTermMemory":
        """The memory with `entry` appended, evicting the oldest past `horizon`."""
        return replace(self, entries=(self.entries + (entry,))[-self.horizon:])

    def reset(self) -> "ShortTermMemory":
        """A new EMPTY memory with the same horizon. What `run_episode` calls at a boundary.

        The paper: "STM is reset between navigation episodes." It is the one structural
        guarantee that stops this being long-term memory by accident.
        """
        return replace(self, entries=())

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def latest(self) -> Optional[StmEntry]:
        """``e_t``, or `None` on an empty memory. `None` rather than raising: an empty STM
        is the ordinary state at the first step of every episode, not a fault."""
        return self.entries[-1] if self.entries else None

    def context(self, *, decay: float) -> Optional[np.ndarray]:
        """The recency-weighted mean of the fused entries, unit-normalised.

        The STM half of ``q_t = f_q(z^av_t, M^S_t)`` (eq. 19). `decay` in (0, 1] weights
        entry `i` steps back by ``decay ** i``, so `decay=1.0` is a flat mean over the
        window and a smaller value leans on the present. It has NO DEFAULT for the reason
        `resolve_prior`'s `k` has none: a knob with a default is a knob that reaches no
        artefact, so the caller passes it and the audit records it.

        `None` on an empty memory, so the caller falls back to ``z^av_t`` alone rather
        than querying with a zero vector that would match the store's own arithmetic
        rather than anything about the episode.
        """
        if not self.entries:
            return None
        if not 0.0 < float(decay) <= 1.0:
            raise ValueError(
                "decay must be in (0, 1]; got {}. Above 1 the OLDEST entry dominates, "
                "which inverts the recency this is for; at or below 0 the weights are "
                "not a mean at all".format(decay)
            )
        weights = np.array(
            [float(decay) ** index for index in range(len(self.entries) - 1, -1, -1)],
            dtype=np.float32,
        )
        stacked = np.stack([entry.fused for entry in self.entries])
        return _unit(np.tensordot(weights, stacked, axes=(0, 0)))
