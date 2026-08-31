"""The reverb tail: an accumulation buffer, folded per step, and read at TWO widths.

ADR-0017 asked for the buffer and named the reason: ``clips.render_through_ir``
convolves the WHOLE clip through the current pose's IR every step, which models a source
sustaining forever and is fine while ``playing`` is monotone. Cut to a bounded window and
the silence arrives as a hard step to the bed with no tail, which is unphysical. The fix
is a preallocated accumulation buffer that adds each step's convolution into a running
signal, so the source decays out instead of being switched off.

**The model, stated exactly, because it is an invention.** The source emits ``hop``
samples of the (looped) clip per step while the window is open. Each step's ``hop``
samples are convolved through THAT step's IR and added into a running buffer laid out
forward in time. ``hop`` is ``round(AudioConfig.step_seconds * sample_rate)``, and
``step_seconds`` is the invented unit that makes any of this expressible -- see its
provenance box in ``audio/config.py``.

**The clip loops for the duration of the window.** A 5 s recording cannot sound for a
60-step window otherwise, and a source that faded out mid-window would put the window's
duration under the recording's control rather than the task's. The loop has a period:
``phase_folds = N // gcd(N, hop)`` folds, 5 at the box's numbers.

===========================================================================
THE DEFECT ADR-0019 REPAIRS, AND THE SPLIT THAT REPAIRS IT
===========================================================================

**The defect, as measured.** The buffer had ONE readout, the last ``N = len(clip)``
samples. ``N`` is 5 s and a step is 1 s, so the number the agent called its
"instantaneous" RMS was a **five-second moving average**, and its decay after the offset
step was the analysis window emptying rather than the room.

The control that settles it is an anechoic 1-sample IR -- a room with literally no
reverberation -- on the same clip and window. Measured here (5 s white-noise clip,
``N = 220500``, ``hop = 44100``, synthetic IR ``L = 72300`` at RT60 0.8 s), post-offset
readout over the settled level:

    clip readout, room       0.902  0.783  0.642  0.460  0.110  0.000
    clip readout, anechoic   0.895  0.775  0.633  0.448  0.000

The room buys **0.6 to 1.3 points** over the first four steps and one extra step at
0.110. This is structural rather than a badly chosen IR: ticket 06 measured
``L = 72300`` (1.64 s) against a 4.0 s cap while the read window is 5 s, so ``L < N`` in
every configuration this tree can produce and the clip tail can never be
reverberation-dominated.

**The fix is a second readout off the same buffer, one step wide.**

    clip readout   ``buffer[:, :N]``            (2, N)     unchanged, CLAP only
    cue  readout   ``buffer[:, N - hop : N]``   (2, hop)   what arrived DURING this step

The two decay curves side by side are the whole argument, same buffer, same folds:

    cue  readout, room       0.245  0.000                  <- the ROOM
    cue  readout, anechoic   0.000                         <- and its control
    clip readout, room       0.902  0.783  0.642  0.460  0.110  0.000

The cue readout falls to EXACTLY zero at ``cue_tail_steps`` = 3 folds after the last
sounding step against the clip readout's ``clip_tail_steps`` = 7, and the anechoic
control collapses the cue to ``cue_tail_steps`` = 1 -- exactly zero on the offset step
itself, 24.5 points below the room's reading there. **The cue tail IS reverberation and
the anechoic control no longer reproduces it.** That pair is the measurement the split
exists to make, and ``test_audio_tail.py`` runs both arms so it stays a measurement.

**The clip readout is UNCHANGED and feeds CLAP only.** ADR-0018's bank of record (anchor
recall 0.911 / 0.895 over two 27-minute box runs) was measured on clip-length waveforms;
re-deriving it is not in scope, so ``heard_clip_window`` hands CLAP exactly what
``heard_step`` used to return. It is called once per episode rather than once per step.

**The cue is INTERMITTENT, and that is a property rather than an apology.** At a settled
pose the cue RMS cycles with the loop, ``phase_folds`` readings long. Measured:

    5 s white noise     crest 1.0018, min_ratio 0.9982
    0.6 s transient     0.000  0.000  2.236  0.002  0.000  (crest 2.2361, min_ratio 0.0)

A looped 0.6 s alarm on a 5 s period really is silent for 4.4 s in every 5. The old
readout hid that behind the moving average; ``cue_crest`` and ``cue_min_ratio`` measure
it and the calibration record carries them.

===========================================================================
WHAT THE TWO READOUTS DO AND DO NOT SHARE
===========================================================================

**They agree in quadratic mean and disagree step by step.** The ``phase_folds`` cue
windows are disjoint, consecutive and tile the settled period an integer number of times,
so ``cue_level(steady_state_cue_rms(...)) == rms(steady_state_render(...))`` -- measured
at ratio 1.000000000000 in all four configurations this tree ships (tail fixture
800/100/512, runner 2205/441/64 and 2205/441/900, box 220500/44100/72300). That identity
is why ``onset_rms`` does not move across the split.

**The settled level is still above a bare whole-clip render**, by the loop's wrap-around
energy in the first ``L - 1`` samples: measured 1.0014x at RT60 0.2 s, 1.0057x at 0.8 s
and 1.0146x at 2.0 s, on a 5 s clip at the box's numbers. It is a property of the IR's
decay and the clip, not a constant, and it holds for the cue level and the clip readout
alike -- the identity above makes them the same number.

**HOW the level falls is a property of the CLIP as much as of the room.** Post-offset
readout over the settled level, ``sounding_steps = 60``, both readouts, measured:

    white noise       clip  0.902  0.783  0.642  0.460  0.110  0.000
                      cue   0.245  0.000
    1007 Hz tone      clip  0.897  0.778  0.636  0.452  0.068  0.000
                      cue   0.152  0.000
    0.6 s transient   clip  0.001  0.000
                      cue   0.000
    2.5 s transient   clip  0.222  0.041  0.000
                      cue   0.000

A recording whose energy sits inside one hop -- which most of ESC-50 is, ``glass_break``
above all -- rings once every ``phase_folds`` steps, so its last ring can be up to
``phase_folds - 1`` steps before the window closes and the cue reads zero at the offset
step. Physically the last ring really was four steps ago and the room really is quiet by
then; what is invented is the LOOP, which ADR-0017 did not specify and this module chose.

``runner.tail_is_active`` cannot see any of that -- it reads the record, and the record
carries no energy -- so the runner also records ``post_offset_audible_steps``, how many
silent-phase steps the agent's own reading stayed distinguishable from the bed. Since the
split that count is measured on the CUE trace, so it counts steps at which the ROOM was
still audible; its values fall, and that is the correction.

The consequence every downstream check has to respect is unchanged: **``onset_step >
offset_step`` is REACHABLE** -- an agent can first cross threshold on the source's tail --
and no post-offset invariant may assert "the RMS is the bed at the offset step".

**Per-step cost FELL AGAIN.** Measured here, median of nine at the box's numbers: 5.23 ms
for a sounding step composed through the cue readout, against 5.54 ms through the clip
readout (one ``N``-length copy and one ``N``-length mix left the per-step path) and
19.56 ms for a whole-clip ``render_through_ir``. A silent step skips the convolution
entirely. The buffer is 2.34 MB; the cue bed is 352.8 kB against the clip bed's 1.76 MB.
Criterion 7's ceiling is 0.5 s and has been breached once at 0.5335 s; this spends none
of that margin. **These are Mac numbers and a Mac has no renderer beside them**, so
``tests/box/test_sounding_window_box.py`` measures the same bill against a real IR and
prints it -- ADR-0014's rule that a capability is exercised rather than proxied.

This module knows nothing about ``WindowPolicy``: the fold takes a ``bool``. The window
is ``audio/window.py``'s value and the runner is what joins them.

Python 3.9 (the SoundSpaces pin).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from earshot.audio.bed import mix_bed
from earshot.audio.clips import as_binaural, rms

__all__ = [
    "TailError",
    "TailState",
    "CUE_RAMP_STEPS",
    "hop_samples",
    "phase_folds",
    "open_tail",
    "advance_tail",
    "cue_readout",
    "clip_readout",
    "heard_step",
    "heard_clip_window",
    "steady_state_render",
    "steady_state_cue_rms",
    "cue_level",
    "cue_crest",
    "cue_min_ratio",
]


class TailError(RuntimeError):
    """A non-finite IR reached the accumulator.

    Fatal on purpose, and the reason is new with the buffer. ``render_through_ir`` is
    stateless, so a NaN render poisoned exactly one step and the next step recovered;
    an accumulator carries it for ``clip_tail_steps`` steps and every reading after it,
    and ``rms`` of a NaN buffer compares False against ``onset_rms`` forever -- so the
    onset would never fire again and the episode would read as ordinary attrition.
    ``test_audio_clips.py:40`` already names that failure on the stateless path. Same
    discipline as ``ProvenanceError``: raise, because a caller who can carry on past it
    is a caller who will.
    """


# How many sounding folds it takes to WRITE the cue window whole. It is 1 by
# construction: one fold emits exactly ``hop`` samples and the cue window is exactly
# ``hop`` wide, so there is no fill ramp and ``onset_delay_steps`` carries none of the
# 0-to-4 step upward bias the clip readout's ``clip_ramp_steps`` fill imposed.
#
# It is a constant on purpose and is deliberately NOT a record field: a record field
# replaceable by a literal is the hole `TestTheWindowRecordIsTheAccumulatorsOwn
# Measurement` exists to close, and a property that ignored ``self`` and returned 1 would
# invite exactly that.
#
# **FILL and LEVEL-SETTLE are two different numbers and only the clip readout confuses
# them.** The cue window is WRITTEN whole by fold 1 and its LEVEL still approaches steady
# state over ``cue_tail_steps`` folds, because the room's reverberation builds up over
# ``L`` samples. Measured at the box's numbers: cue fold 1 is 0.9696 of the settled level
# and folds 2 onward are inside 0.2% of it, against a clip readout that needs
# ``clip_ramp_steps`` = 5 folds to reach 0.9944. For the clip readout fill (5) and
# level-settle (7) nearly coincide because ``N >> L``; for the cue they are 1 and 3, and
# the dominant term flips from the analysis window to the room.
CUE_RAMP_STEPS: int = 1


def phase_folds(*, window: int, hop: int) -> int:
    """``window // gcd(window, hop)`` -- folds after which the cue readout repeats.

    One definition, read by ``TailState.phase_folds`` and by ``steady_state_cue_rms``.
    ``phase_folds * hop`` is ``lcm(window, hop)``, a whole number of clip periods, which
    is what makes the cue windows tile the settled period exactly.
    """
    n = int(window)
    stride = int(hop)
    if n <= 0 or stride <= 0:
        raise ValueError(
            "phase_folds needs a positive window and hop, got {} and {}".format(
                window, hop
            )
        )
    return n // math.gcd(n, stride)


@dataclass(frozen=True)
class TailState:
    """The episode's accumulated reverb, frozen in and frozen out.

    ``buffer`` is ``(2, window + headroom)`` float32 laid out FORWARD IN TIME: index 0 is
    the OLDEST sample of the read window, index ``window - 1`` is *now*, and index
    ``window`` onward is reverb already committed but not yet arrived. Every index in
    this module depends on that layout, and it is what makes the cue readout a plain
    slice of the clip readout's last ``hop`` samples.

    This follows ``OnsetState``'s shape deliberately -- frozen in, frozen out, the runner
    owns the episode's one mutable slot. A mutable accumulator class was rejected because
    a leaked one would carry one episode's reverb into the next, which is the bug
    ``AudioEpisodeState.reset()`` existed to paper over.
    """

    # `compare=False, repr=False` is mandatory, not tidiness. A bare `==` on a frozen
    # dataclass holding an ndarray raises "truth value of an array is ambiguous", and a
    # 2.3 MB repr in a traceback is unreadable. Equality then falls to the scalar fields,
    # which is what a test comparing two states actually wants to ask.
    buffer: np.ndarray = dataclasses.field(compare=False, repr=False)
    window: int
    hop: int
    phase: int
    max_ir_samples: int
    n_grows: int
    # Which of the folds still inside the CLIP read window were SOUNDING, oldest first,
    # at most ``ceil(window / hop)`` of them. Defaulted so a ``dataclasses.replace`` on
    # an older construction still builds. This exists for one caller: the step CLAP is
    # handed the buffer. See ``clip_source_fill``.
    recent_sounding: Tuple[bool, ...] = ()

    @property
    def clip_source_fill(self) -> float:
        """Fraction of the CLIP read window a SOUNDING fold wrote into. 0.0 to 1.0.

        **What CLAP is actually given, which nothing else records.** The buffer is laid
        out oldest-first and takes ``ceil(N/hop)`` folds to fill, so an agent that crosses
        ``onset_rms`` mid-ramp hands ``is_anomaly`` a clip-length waveform whose older
        part is bed only; symmetrically, an onset on the tail hands it a decay. Both are a
        different domain from the full-length ``render_through_ir`` clips ADR-0018's bank
        of record and the CLAP separation gate were measured on, and the confound is
        invisible unless the fill is written down beside the class.

        **The ``clip_`` prefix is load-bearing.** Everything around this gained a prefix
        at the split; an unprefixed ``source_fill`` would be the one place a reader
        reaches for the wrong window. There is no cue equivalent: ``CUE_RAMP_STEPS`` is 1,
        so a sounding fold fills the cue window whole and the fraction is never partial.

        Reverb SPILL is deliberately not counted. A silent fold's window still holds the
        previous fold's reverb running on, and calling that "source" would report a full
        window for a buffer that holds only decay -- which is the half of the confound
        that matters most.
        """
        if not self.recent_sounding:
            return 0.0
        filled = int(self.hop) * sum(1 for flag in self.recent_sounding if flag)
        return min(1.0, float(filled) / float(self.window))

    @property
    def clip_tail_steps(self) -> int:
        """Steps after the last sounding step before the CLIP readout is exactly zero.

        ``ceil((N + L - 1) / hop)``. Derived from the WIDEST IR actually seen, never from
        a constant. ``clips.py:146-148`` forbids a fixed-width assumption about the IR --
        ticket 06's ``[2, 72300]`` is one scene's measurement, not a cap -- and
        ``tests/box/test_audio_box.py:307-311`` enforces it. Reading this off the record
        rather than pinning it is what lets a more reverberant scene have a longer tail
        and still be described correctly.

        **This number is NOT evidence that the geometric acoustics did any work.** It is
        the analysis window emptying: an anechoic 1-sample IR reproduces the same decay
        curve to within 1.3 points over the first four steps (measured above), and
        ``L < N`` in every configuration this tree can produce. ``cue_tail_steps`` is the
        number that IS evidence. Since ADR-0019 this one bounds what CLAP reads and keeps
        its clause in ``tail_is_active``; it no longer bounds what the agent reads.
        """
        width = max(int(self.max_ir_samples), 1)
        return int(math.ceil((int(self.window) + width - 1) / float(self.hop)))

    @property
    def clip_ramp_steps(self) -> int:
        """Sounding folds before the CLIP read window is full of source. ``ceil(N/hop)``.

        Moved here from ``runner.py``'s inline ``ceil(tail.window / tail.hop)``: that
        comment demanded one definition of the ramp and this is where it goes. Since the
        split its consumer is the CLAP deferral -- ``clip_ramp_steps - 1`` steps at most
        -- rather than a correction to ``onset_delay_steps``, which the cue readout no
        longer needs.
        """
        return int(math.ceil(int(self.window) / float(self.hop)))

    @property
    def cue_tail_steps(self) -> int:
        """Steps after the last sounding step before the CUE readout is exactly zero.

        ``ceil((hop + L - 1) / hop)``, and it is the first number on this state that IS
        evidence the geometric acoustics did work. ``1`` means the IR fits inside one step
        and the silent phase is an honest hard cut; ``> 1`` means the room outlives a
        step. Measured on this Mac: an anechoic 1-sample IR gives 1 and a 72300-sample IR
        at RT60 0.8 s gives 3, and the cue readout really is exactly zero at the third
        silent fold and nonzero at the second (2.18e-04 against a settled 5.08).

        Contrast ``clip_tail_steps``, which the same anechoic control reproduces to 1.3
        points -- the analysis window emptying, and the defect ADR-0019 corrects.
        """
        width = max(int(self.max_ir_samples), 1)
        return int(math.ceil((int(self.hop) + width - 1) / float(self.hop)))

    @property
    def phase_folds(self) -> int:
        """Folds after which the cue readout repeats at a pose held fixed and sounding.

        The loop's period, ``N // gcd(N, hop)`` -- 5 at the box's numbers, 8 at the tail
        fixture's. It bounds how long a bursty clip can keep the cue near silence, so it
        bounds the onset's worst-case delay at ``phase_folds - 1`` steps.
        """
        return phase_folds(window=int(self.window), hop=int(self.hop))


def hop_samples(*, step_seconds: float, sample_rate: int) -> int:
    """How many samples of source one simulator step emits.

    The unit ADR-0017 needed and the tree did not have. ``step_seconds`` is
    ``provenance: fake`` (``audio/config.py``) and this is the only place it turns into
    an index. Since ADR-0019 it is also the CUE readout's width, which is why the record
    needs no separate field for that.
    """
    hop = int(round(float(step_seconds) * int(sample_rate)))
    if hop <= 0:
        raise ValueError(
            "a step must advance at least one sample: step_seconds {} at sample_rate "
            "{} gives a hop of {}".format(step_seconds, sample_rate, hop)
        )
    return hop


def open_tail(*, window: int, hop: int, headroom: int = 0) -> TailState:
    """A silent accumulator for one episode. ``window`` is ``len(clip)``.

    ``headroom`` is a HINT, not a bound: pass ``max(0, ir_len - 1)`` from a render
    already in hand and the buffer never reallocates; pass 0 and it grows once, on the
    first sounding step. ADR-0017's word is "preallocated" and this is
    preallocated-with-an-explicit-grow, because the tree has no ``maxIRLength`` anywhere
    (``spec.py:81-86`` sets it deliberately not at all) and a buffer sized on ticket 06's
    measured 72300 would truncate the tail on a more reverberant scene and produce a
    quiet, plausible, wrong signal.

    ``hop >= window`` is refused rather than allowed. A step that advances at least the
    whole read window makes consecutive CLIP readouts DISJOINT: it is a different sensor,
    one that silently drops the audio between two steps, and it has to be asked for
    explicitly rather than fallen into. Since ADR-0019 there is a third reason on top of
    the two above: at ``hop == window`` the cue readout IS the clip readout and
    ``phase_folds`` is 1, so the split this module is built around collapses and every
    number that distinguishes the two readouts silently becomes the same number.
    """
    n = int(window)
    stride = int(hop)
    if n <= 0:
        raise ValueError(
            "the read window is len(clip) and must be positive, got {}".format(window)
        )
    if stride <= 0:
        raise ValueError(
            "the hop is round(AudioConfig.step_seconds * sample_rate) and must be "
            "positive, got {}".format(hop)
        )
    if stride >= n:
        raise ValueError(
            "a hop of {} samples at or beyond the {}-sample read window makes "
            "consecutive readouts disjoint, so the audio between two steps is dropped "
            "and this is a different sensor. hop is round(AudioConfig.step_seconds * "
            "sample_rate) and the window is len(clip): lower step_seconds or use a "
            "longer clip.".format(stride, n)
        )
    pad = max(0, int(headroom))
    return TailState(
        buffer=np.zeros((2, n + pad), dtype=np.float32),
        window=n,
        hop=stride,
        phase=0,
        max_ir_samples=0,
        n_grows=0,
        recent_sounding=(),
    )


def _looped_chunk(clip: Any, *, phase: int, hop: int) -> Tuple[np.ndarray, int]:
    """``hop`` samples of the clip read cyclically from ``phase``, and the clip's length."""
    signal = np.asarray(clip, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        raise ValueError("anomaly clip is empty -- nothing to emit into the tail")
    index = (int(phase) + np.arange(int(hop))) % signal.size
    return signal[index], int(signal.size)


def _convolve(chunk: np.ndarray, impulse: np.ndarray) -> np.ndarray:
    """``(2, hop + L - 1)`` -- the FULL linear convolution, tail included.

    The same numpy-FFT arithmetic ``render_through_ir`` uses (next power of two,
    ``rfft``/``irfft``, no scipy, because scipy is not on the Mac side). The one
    difference is the trim: that function keeps ``[:, :N]`` and throws the ``L - 1``
    reverb samples away every step, which is precisely the energy this buffer exists to
    carry.
    """
    length = int(chunk.size) + int(impulse.shape[1]) - 1
    n_fft = 1 << int(max(1, length - 1)).bit_length()
    spectrum = np.fft.rfft(chunk, n_fft)
    rendered = np.fft.irfft(
        np.fft.rfft(impulse, n_fft, axis=1) * spectrum, n_fft, axis=1
    )
    return np.ascontiguousarray(rendered[:, :length], dtype=np.float32)


def advance_tail(
    state: TailState, *, ir: Optional[Any], clip: Any, sounding: bool
) -> TailState:
    """Fold one step into the accumulator. Returns a NEW state; never writes into ``state``.

    On a sounding step the emitted ``hop`` samples land at the END of the read window
    (``window - hop`` to ``window - 1``) and their reverb runs on past it, which is what
    the headroom is for. That end region IS the cue readout: what this fold emitted, plus
    whatever earlier folds' reverb has reached it. On a silent step nothing is emitted,
    ``ir`` is not read at all, and the buffer simply slides -- so both readouts keep
    falling as the committed reverb runs out, the cue over ``cue_tail_steps`` and the
    clip over ``clip_tail_steps``.

    **The growth never truncates, and it is counted** (``n_grows``). A truncating
    implementation is the dangerous one: it loses reverb silently and the signal stays
    plausible, which is this map's recurring failure class. There is no fixed IR width to
    size against (``spec.py:81-86`` sets no ``maxIRLength``), so the buffer grows to
    ``window + L - 1`` for the widest IR it has actually been handed.
    """
    buf = state.buffer
    hop = int(state.hop)
    window = int(state.window)

    if sounding:
        if ir is None:
            raise ValueError(
                "a sounding step needs this step's IR: the accumulator convolves the "
                "hop it emits through the pose the agent is standing at, and there is "
                "no last-known-good IR to fall back on"
            )
        impulse = as_binaural(ir)
        if not bool(np.all(np.isfinite(impulse))):
            raise TailError(
                "the IR handed to the accumulator is not finite (shape {}, {} "
                "non-finite samples). render_through_ir is stateless so a NaN poisoned "
                "one step; this buffer carries it for every later step, and rms(NaN) "
                "compares False against onset_rms forever.".format(
                    tuple(impulse.shape), int(np.count_nonzero(~np.isfinite(impulse)))
                )
            )
        n_ir = int(impulse.shape[1])
        need = window + n_ir - 1
    else:
        impulse = None
        n_ir = 0
        need = window

    total = max(int(buf.shape[1]), need)
    n_grows = state.n_grows + (1 if total > int(buf.shape[1]) else 0)

    out = np.zeros((2, total), dtype=np.float32)
    # Safe because `open_tail` refuses hop >= window and window <= buf.shape[1].
    out[:, : int(buf.shape[1]) - hop] = buf[:, hop:]

    phase = int(state.phase)
    if sounding and impulse is not None:
        chunk, n_clip = _looped_chunk(clip, phase=phase, hop=hop)
        conv = _convolve(chunk, impulse)
        start = window - hop
        if start + int(conv.shape[1]) > total:
            raise ValueError(
                "the step's convolution is {} samples written from {} and would run "
                "past the {}-sample buffer -- the grow path above sizes to window + L - "
                "1 = {} and did not fire".format(
                    int(conv.shape[1]), start, total, need
                )
            )
        out[:, start : start + int(conv.shape[1])] += conv
        phase = (phase + hop) % n_clip

    # The folds still inside the CLIP read window, oldest first. `ceil(window / hop)` of
    # them, which is exactly the set whose emitted hops have not scrolled out yet.
    depth = int(math.ceil(window / float(hop)))
    recent = (tuple(state.recent_sounding) + (bool(sounding),))[-depth:]

    return TailState(
        buffer=out,
        window=window,
        hop=hop,
        phase=phase,
        max_ir_samples=max(int(state.max_ir_samples), n_ir),
        n_grows=n_grows,
        recent_sounding=recent,
    )


def cue_readout(state: TailState) -> np.ndarray:
    """``(2, hop)`` float32 -- the samples that arrived at the ears DURING this step.

    **This is what the agent reads.** Its tail is driven by the IR's width ``L`` rather
    than by the analysis window ``N``, so it IS reverberation: it falls to exactly zero
    ``cue_tail_steps`` folds after the last sounding step, and an anechoic 1-sample IR
    collapses that to one fold instead of reproducing the curve. ``rms``, ``lateral_sign``
    and the onset detector all read this, and therefore so do the controller and the
    calibration.

    A COPY and never a view, for the same reason and by the same means as
    ``clip_readout``: ``np.array(..., order="C")`` rather than ``np.ascontiguousarray``.

    **``clip_readout``'s aliasing scar does NOT reproduce here, and the reason is an
    accident of the layout rather than a decision.** ``buffer[:, window - hop : window]``
    is two rows of a wider array, so its strides are ``(buffer_width * 4, 4)`` and it is
    never C-contiguous while ``hop < window`` -- which ``open_tail`` refuses to allow
    otherwise. Measured: ``np.ascontiguousarray`` copies at headroom 0 and at
    ``headroom = L - 1`` alike, where on ``clip_readout``'s full-width slice it aliased.
    ``np.array`` is used anyway, because a guarantee that holds only while nobody widens
    the hop or reshapes the buffer is not a guarantee, and the failure it prevents -- a
    caller writing into what it was told was its own -- cost a whole class of bug once.
    """
    window = int(state.window)
    hop = int(state.hop)
    return np.array(
        state.buffer[:, window - hop : window], dtype=np.float32, order="C"
    )


def clip_readout(state: TailState) -> np.ndarray:
    """``(2, window)`` float32 -- the last ``N`` samples, the CLAP domain.

    Renamed from ``tail_readout`` at ADR-0019 and otherwise unchanged. The old name said
    nothing about which of the two windows it returned, and after the split "the tail" is
    the one word that means two different lengths.

    A COPY and never a view. A caller that wrote into a view would corrupt the next
    step's readout, which is the same class of bug as ``heard_signal``'s not-playing
    branch aliasing the one shared bed buffer (``np.asarray(bed, float32) is bed`` is
    True) -- the branch this replaces.

    **``np.ascontiguousarray`` did NOT hold that** and was measured aliasing: it copies
    only when it has to, so it returned the buffer's own memory for every state whose
    buffer is exactly ``window`` wide and C-contiguous -- which is the state
    ``open_tail(headroom=0)`` starts in and stays in for every step before the first
    sounding fold. That is the whole pre-``t_anom`` phase on any path that cannot hand
    over an IR width up front (``run_episode``'s ``ir_shape is None``, the Mac fake).
    Nothing corrupted anything, because ``mix_bed`` allocates, but the guarantee this
    docstring states was false on the exact path the ramp's zeros make load-bearing.
    ``np.array`` copies unconditionally.
    """
    return np.array(
        state.buffer[:, : int(state.window)], dtype=np.float32, order="C"
    )


def heard_step(
    state: TailState, *, ir: Optional[Any], clip: Any, bed_cue: Any, sounding: bool
) -> Tuple[TailState, np.ndarray]:
    """``(next state, (2, hop) CUE signal)`` -- the per-step composition point.

    This is what replaces ``bed.heard_signal`` once the window can close.
    ``heard_signal`` stays pure, unchanged and **un-called by the runner**, as the
    pre-ADR-0017 control the tail's tests measure against.

    **The kwarg is ``bed_cue`` rather than ``bed``, and the rename is load-bearing.** A
    caller that hands over the clip-length bed dies on ``mix_bed``'s shape refusal
    instead of composing a wrong signal, and the refusal's message names both lengths.
    That guard is the reason there is no ``HeardStep(cue=..., clip=...)`` pair here: a
    pair would force an ``N``-length copy and an ``N``-length mix on EVERY step for a
    value used once per episode, and it would leave the clip signal in a local where a
    later edit can read it by accident.

    Onset, lateral and the controller all read this and only this.

    **§3.1 keeps exactly the content it had, with no special case here.** Before the
    window opens the buffer is all zeros, and ``mix_bed(zeros, bed_cue)`` is the bed to
    the bit, so ``observe_step``'s pre-``t_anom`` invariant -- "the measured RMS is the
    bed level" -- still has an exact expected value. Nothing in this function branches on
    ``t_anom``; the zeros do the work. It holds only because ``bed_cue`` is normalised at
    ``hop`` in its own right: a ``hop``-length SLICE of the clip-length bed is off by a
    measured 3.90% at the runner fixture's numbers against a 5% tolerance, and by 17.73%
    at the tail fixture's worst slice.
    """
    nxt = advance_tail(state, ir=ir, clip=clip, sounding=sounding)
    return nxt, mix_bed(cue_readout(nxt), bed_cue)


def heard_clip_window(state: TailState, *, bed_clip: Any) -> np.ndarray:
    """``(2, window)`` -- the CLAP domain, unchanged from what ``heard_step`` returned.

    Called at the classification step ONLY -- once per episode, not per step -- so the
    per-step bill loses an ``N``-length copy and an ``N``-length add (measured 5.54 ms to
    5.23 ms at the box's numbers), and no caller can read the clip signal by accident.

    Kept as a separate function rather than folded into ``heard_step`` because ADR-0018's
    bank of record was measured on clip-length waveforms and this change deliberately does
    not touch that domain. ``bed_clip`` is the bed built at ``len(clip)``, which is a
    different buffer from ``heard_step``'s ``bed_cue`` -- same seed, each normalised at
    its own length, and therefore NOT sample-aligned. Nothing compares them.
    """
    return mix_bed(clip_readout(state), bed_clip)


def steady_state_render(ir: Any, clip: Any, *, hop: int) -> np.ndarray:
    """``(2, N)`` -- the CLIP readout at a pose held fixed and sounding.

    **It has no production caller since ADR-0019 and that is deliberate.** It is the CLIP
    domain's named control, kept for the same reason ``bed.heard_signal`` and
    ``test_rising_window``'s ``OLD_EPS`` are kept: a control that is deleted is a
    comparison that cannot be made twice. Its live job is the identity below, which is the
    only available proof that the threshold's LEVEL did not move when the sweep changed
    domain --

        cue_level(steady_state_cue_rms(ir, clip, hop=hop))
            == rms(steady_state_render(ir, clip, hop=hop))

    measured at ratio 1.000000000000 in all four configurations this tree ships.

    The returned waveform is one rotation of a period-``N`` signal, so the WAVEFORM is
    phase-dependent while its RMS is not (measured identical to 12 decimal places across
    phases). **Callers may take the RMS; they may not compare the samples.**
    """
    impulse = as_binaural(ir)
    stride = int(hop)
    n_ir = int(impulse.shape[1])
    window = int(np.asarray(clip, dtype=np.float32).reshape(-1).size)
    state = open_tail(window=window, hop=stride, headroom=max(0, n_ir - 1))
    # `TailState.clip_tail_steps` cannot answer this yet -- `max_ir_samples` is 0 until a
    # sounding fold has happened -- so the same expression is written out here. One
    # spare fold past the settle so the return is inside the flat region, not on it.
    folds = int(math.ceil((window + n_ir - 1) / float(stride))) + 1
    for _ in range(folds):
        state = advance_tail(state, ir=impulse, clip=clip, sounding=True)
    return clip_readout(state)


def steady_state_cue_rms(ir: Any, clip: Any, *, hop: int) -> Tuple[float, ...]:
    """The per-fold CUE RMS over one full loop period, oldest phase first.

    ``len`` is ``phase_folds(window=len(clip), hop=hop)`` -- 5 at the box's numbers. This
    is the domain the onset threshold has to be calibrated in since ADR-0019, because it
    is the domain ``observe_step`` reads.

    **The settle is the CLIP settle, and it is CONSERVATIVE rather than necessary.** This
    folds ``clip_tail_steps + 1`` times before collecting -- exactly what
    ``steady_state_render`` folds -- so the two sides of the identity are settled by the
    same expression and a change to one cannot silently under-settle the other. The cue
    readout on its own needs less: it depends only on the last ``cue_tail_steps`` folds,
    so it is settled from there on, and there is no configuration where the clip settle is
    required. Measured at the tail fixture (800/100/512, ``cue_tail_steps`` 7,
    ``clip_tail_steps`` 14) -- a settle of 6 folds already reproduces the phase multiset
    exactly and a settle of 4 is 0.0143% high on the level. The longer settle is kept
    because "the same settle as the function this must equal" is a property a reader can
    check, and "6 is enough at this fixture" is one they would have to re-derive.

    Cost, honestly: settle + ``phase_folds`` folds per pose (13 at the box's numbers)
    against ``steady_state_render``'s 8. ONE live render per pose either way -- the caller
    calls ``render_at`` once and this reuses the IR -- so the sweep's habitat bill is
    unchanged and only its numpy time moves, once per episode.
    """
    impulse = as_binaural(ir)
    stride = int(hop)
    n_ir = int(impulse.shape[1])
    window = int(np.asarray(clip, dtype=np.float32).reshape(-1).size)
    state = open_tail(window=window, hop=stride, headroom=max(0, n_ir - 1))
    folds = int(math.ceil((window + n_ir - 1) / float(stride))) + 1
    for _ in range(folds):
        state = advance_tail(state, ir=impulse, clip=clip, sounding=True)
    phases = []
    for _ in range(phase_folds(window=window, hop=stride)):
        state = advance_tail(state, ir=impulse, clip=clip, sounding=True)
        phases.append(float(rms(cue_readout(state))))
    return tuple(phases)


def cue_level(phases: Sequence[float]) -> float:
    """The quadratic mean of the loop's cue RMSs -- ``sqrt(mean(v**2))``.

    THE choice, written down once so no call site can aggregate differently. It is the
    aggregation that makes ``onset_rms`` invariant across the split: the ``phase_folds``
    cue windows are disjoint, consecutive and tile the settled period an integer number
    of times, so their quadratic mean EQUALS the clip readout's RMS exactly (measured
    ratio 1.000000000000 in four configurations).

    The maximum was rejected -- it raises the threshold by the crest factor, 2.24x for a
    0.6 s transient on a 5 s loop, in a clip-dependent way that makes every historic
    threshold unpriceable. The minimum was rejected -- it puts a bursty clip's low
    percentile at the bed and fails the 6 dB gate for a source that is plainly audible.
    One arbitrary phase was rejected because that is the defect ADR-0019 names.
    """
    values = [float(v) for v in phases]
    if not values:
        raise ValueError(
            "cue_level needs at least one loop phase -- an empty sweep is not a level "
            "of zero, it is a measurement that did not happen"
        )
    return math.sqrt(sum(v * v for v in values) / len(values))


def cue_crest(phases: Sequence[float]) -> float:
    """``max(phases) / cue_level(phases)``. Dimensionless. How intermittent the cue is.

    1.0 for a clip whose energy is flat across the loop, and 2.2361 for a 0.6 s transient
    on a 5 s loop at a 1 s hop (measured on this Mac). This is the number that says the
    per-step reading is intermittent, and the calibration record carries its median over
    the swept poses.
    """
    values = [float(v) for v in phases]
    return max(values) / cue_level(values)


def cue_min_ratio(phases: Sequence[float]) -> float:
    """``min(phases) / cue_level(phases)``. The other half of ``cue_crest``.

    It says how many folds are effectively silent. Measured 0.9982 for 5 s of white noise
    and 0.0000 for a 0.6 s transient, both on a 5 s loop at a 1 s hop -- four folds in
    five carry no source at all for the transient, and the gate can still pass on the
    quadratic mean. That is why this is recorded rather than gated: a refusal here would
    be a second change, and it would make four of ESC-50's five classes unusable.
    """
    values = [float(v) for v in phases]
    return min(values) / cue_level(values)
