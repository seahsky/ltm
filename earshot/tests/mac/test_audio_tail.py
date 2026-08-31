"""``audio/tail.py`` -- the accumulation buffer ADR-0017 says must exist before any SWS,
and the split readout ADR-0019 says it must be read through.

What a failure here costs: a bounded sounding window with no tail cuts the received
signal to the bed in one step, which is unphysical, and every downstream number -- the
climb's rising test, the onset threshold's domain, the detour report's band slopes -- is
then measured on a step function nobody modelled. What a failure in the SPLIT costs is
subtler and was live for a whole branch: reading the agent's "instantaneous" level off a
5 s window while a step is 1 s makes it a five-second moving average, and its decay is
the analysis window emptying rather than the room.

**The IR matters, and one of the tree's fakes is a trap.**
``_task_fakes.FakeAudioSensorHandle.audio_of`` returns a SINGLE sample at index 0 in a
``(2, 64)`` array, so the tail it accumulates is one sample long and every tail assertion
against it would be vacuously green. These tests use ``_audio_fakes.synthetic_ir``
(decaying noise over 512 samples), which is what ``test_audio_clips.py``'s delayed-impulse
test exists for as well.

The constants are chosen so the arithmetic is checkable by hand: ``window = 800``,
``hop = 100``, ``L = 512``. The CLIP read window fills after ``ceil(800/100) = 8`` folds
and empties ``ceil((800 + 511)/100) = 14`` steps after the last sounding one; the CUE
readout is written whole by ONE fold and empties after ``ceil((100 + 511)/100) = 7``. The
loop repeats every ``800 // gcd(800, 100) = 8`` folds.

**The anechoic control runs at the BOX's numbers, not at these**, because the claim it
settles is about ``L < N`` at the shipped configuration and ``L/N`` here is 0.64 against
the box's 0.33. See ``TestTheAnechoicControl``.
"""

import dataclasses
import inspect
import math
import unittest

import numpy as np

import _audio_fakes as fakes
from _interpreter import assert_interpreter  # noqa: F401

import earshot.audio.tail as tail_module
from earshot.audio.bed import bed_signal, heard_signal
from earshot.audio.clips import as_binaural, render_through_ir, rms
from earshot.audio.tail import (
    CUE_RAMP_STEPS,
    TailError,
    TailState,
    advance_tail,
    clip_readout,
    cue_crest,
    cue_level,
    cue_min_ratio,
    cue_readout,
    heard_clip_window,
    heard_step,
    hop_samples,
    open_tail,
    phase_folds,
    steady_state_cue_rms,
    steady_state_render,
)

WINDOW = 800
HOP = 100
IR_SAMPLES = 512
BED_RMS = 1e-3
CLIP_RAMP_STEPS = 8  # ceil(WINDOW / HOP): the CLIP read window is full of source
CLIP_TAIL_STEPS = 14  # ceil((WINDOW + IR_SAMPLES - 1) / HOP): the clip readout is zero
CUE_TAIL_STEPS = 7  # ceil((HOP + IR_SAMPLES - 1) / HOP): the cue readout is zero
PHASE_FOLDS = 8  # WINDOW // gcd(WINDOW, HOP): the cue readout repeats

# The box's shipped numbers, for the one claim that is about them. A 5 s clip at 44100,
# a 1 s step, and ticket 06's measured 72300-sample IR width.
BOX_WINDOW = 220500
BOX_HOP = 44100
BOX_IR_SAMPLES = 72300


def a_clip(seed=3, n=WINDOW):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) * 0.1).astype(np.float32)


def a_room_ir(n_samples=BOX_IR_SAMPLES, rt60=0.8, sample_rate=44100, seed=11):
    """Decaying noise at a stated RT60 -- the box-scale IR the anechoic arm controls for.

    ``synthetic_ir`` is fixed at a 6-nat decay over whatever length it is given, so its
    RT60 moves with ``n_samples`` and it cannot express "1.64 s of support with 0.8 s of
    decay", which is the shape ticket 06 measured and the shape the ``L < N`` argument is
    about.
    """
    rng = np.random.default_rng(seed)
    seconds = np.arange(n_samples) / float(sample_rate)
    envelope = np.power(10.0, -3.0 * seconds / float(rt60))
    base = rng.standard_normal(n_samples) * envelope
    return np.stack([base, base]).astype(np.float32)


def anechoic_ir():
    """A room with literally no reverberation: one sample, so ``L = 1``."""
    delta = np.zeros((2, 1), dtype=np.float32)
    delta[:, 0] = 1.0
    return delta


def sounded(state, ir, clip, folds):
    for _ in range(folds):
        state = advance_tail(state, ir=ir, clip=clip, sounding=True)
    return state


def decay_curves(ir, clip, hop, silent_steps):
    """``(cue curve, clip curve, first exact-zero fold for each)``, settled first.

    Both curves come off ONE buffer, so a difference between them is the readout and
    never the fold.
    """
    n_ir = int(np.asarray(ir).shape[1])
    window = int(len(clip))
    # Past the clip settle -- the longer of the two, so both readouts are flat before the
    # window closes -- and rounded UP TO A WHOLE NUMBER OF LOOP PERIODS. The rounding is
    # not decoration: the cue readout's first silent value depends on which loop phase
    # the window closes at, so a settle that is not period-aligned makes the printed
    # curve depend on an arithmetic accident and stop matching the docstrings. Written
    # out rather than read off the state because `clip_tail_steps` is `ceil(N/hop)` until
    # a sounding fold has happened.
    period = phase_folds(window=window, hop=hop)
    settle = period * int(
        math.ceil((math.ceil((window + n_ir - 1) / float(hop)) + 2) / float(period))
    )
    state = open_tail(window=window, hop=hop, headroom=max(0, n_ir - 1))
    state = sounded(state, ir, clip, settle)
    settled_clip = rms(clip_readout(state))
    settled_cue = cue_level(steady_state_cue_rms(ir, clip, hop=hop))
    cue_curve, clip_curve = [], []
    cue_zero, clip_zero = None, None
    for fold in range(1, silent_steps + 1):
        state = advance_tail(state, ir=None, clip=clip, sounding=False)
        here_cue, here_clip = rms(cue_readout(state)), rms(clip_readout(state))
        cue_curve.append(here_cue / settled_cue)
        clip_curve.append(here_clip / settled_clip)
        if cue_zero is None and here_cue == 0.0:
            cue_zero = fold
        if clip_zero is None and here_clip == 0.0:
            clip_zero = fold
    return cue_curve, clip_curve, cue_zero, clip_zero


def _fold_without_the_finite_check(state, *, ir, clip):
    """``advance_tail`` with the ``TailError`` guard taken out, and nothing else.

    The control arm for the NaN test: an accumulator that lets a bad IR through is not
    a hypothetical, it is this function minus one ``if``.

    **Built out of the real fold's own pieces, which the first version was not.** It
    re-implemented the arithmetic and read ``clip[:hop]`` where ``advance_tail`` reads
    the LOOPED chunk from ``phase``, so the two agreed on fold 0 and diverged on every
    fold after it -- and a control that does not track the function it controls for
    cannot say whether a difference came from the missing ``if`` or from the copy.
    ``test_the_control_arm_is_the_real_fold_minus_one_if`` measures the agreement on a
    finite IR, which is what makes the NaN result below evidence about the guard.
    """
    impulse = as_binaural(ir)
    # THE ONE LINE REMOVED: `if not np.all(np.isfinite(impulse)): raise TailError(...)`.
    n_ir = int(impulse.shape[1])
    need = int(state.window) + n_ir - 1
    total = max(int(state.buffer.shape[1]), need)
    out = np.zeros((2, total), dtype=np.float32)
    out[:, : int(state.buffer.shape[1]) - state.hop] = state.buffer[:, state.hop :]
    chunk, n_clip = tail_module._looped_chunk(clip, phase=state.phase, hop=state.hop)
    conv = tail_module._convolve(chunk, impulse)
    start = int(state.window) - int(state.hop)
    out[:, start : start + int(conv.shape[1])] += conv
    depth = int(math.ceil(int(state.window) / float(state.hop)))
    return dataclasses.replace(
        state,
        buffer=out,
        phase=(int(state.phase) + int(state.hop)) % n_clip,
        max_ir_samples=max(int(state.max_ir_samples), n_ir),
        n_grows=state.n_grows + (1 if total > int(state.buffer.shape[1]) else 0),
        recent_sounding=(tuple(state.recent_sounding) + (True,))[-depth:],
    )


class TestTheTailFold(unittest.TestCase):
    def setUp(self):
        self.ir = fakes.synthetic_ir(n_samples=IR_SAMPLES)
        self.clip = a_clip()
        # TWO beds, each normalised at its own length. `bed_cue` reaches `heard_step` and
        # `bed_clip` reaches `heard_clip_window`; a slice of the second is NOT the first
        # and `TestTheTwoBedsAreNotOneSliced` measures by how much.
        self.bed_cue = bed_signal(HOP, BED_RMS)
        self.bed_clip = bed_signal(WINDOW, BED_RMS)
        self.state = open_tail(window=WINDOW, hop=HOP, headroom=IR_SAMPLES - 1)

    def test_before_the_window_opens_the_heard_signal_is_the_bed_to_the_bit(self):
        """What keeps §3.1's provenance assertion content rather than slack.

        And it holds with NO special case in ``heard_step``: the buffer is zeros before
        the first sounding fold, so ``mix_bed(zeros, bed_cue)`` is the bed exactly. A
        branch on ``t_anom`` inside the composition point would be a second place the
        window is decided.
        """
        state = self.state
        for _ in range(5):
            state, heard = heard_step(
                state, ir=self.ir, clip=self.clip, bed_cue=self.bed_cue, sounding=False
            )
            self.assertEqual(heard.shape, (2, HOP))
            np.testing.assert_array_equal(heard, self.bed_cue)
        self.assertAlmostEqual(rms(heard), BED_RMS, places=9)

    def test_the_heard_signal_is_a_fresh_buffer_rather_than_the_shared_bed(self):
        """``heard_signal``'s not-playing branch returns the bed OBJECT --
        ``np.asarray(bed, float32) is bed`` is True -- so a tail written in place would
        drift the one bed every later provenance check reads, and the ProvenanceError
        would fire at a step that is not the bug."""
        self.assertIs(np.asarray(self.bed_cue, dtype=np.float32), self.bed_cue)

        before = np.array(self.bed_cue)
        _, heard = heard_step(
            self.state, ir=self.ir, clip=self.clip, bed_cue=self.bed_cue, sounding=False
        )
        self.assertIsNot(heard, self.bed_cue)
        heard[:] = 7.0
        np.testing.assert_array_equal(self.bed_cue, before)

    def test_the_clip_level_ramps_over_the_read_window_and_settles(self):
        """The switch-on artefact of the CLIP readout, several steps wide.

        It is the CLAP deferral's business since ADR-0019 and no longer corrects
        ``onset_delay_steps``: the cue readout has no fill ramp at all (see
        ``TestTheCueRamp``). Measured here, the clip readout reaches 0.9526 of the settled
        level at ``CLIP_RAMP_STEPS`` folds and is inside 1e-6 of it from
        ``CLIP_TAIL_STEPS`` on.
        """
        state = self.state
        levels = []
        for _ in range(CLIP_TAIL_STEPS + 4):
            state = advance_tail(state, ir=self.ir, clip=self.clip, sounding=True)
            levels.append(rms(clip_readout(state)))

        for earlier, later in zip(levels[:CLIP_RAMP_STEPS], levels[1:CLIP_RAMP_STEPS]):
            self.assertGreaterEqual(later, earlier)
        # measured 0.9526 of the settled level once the window is full of source
        self.assertGreater(levels[CLIP_RAMP_STEPS - 1], 0.9 * levels[-1])
        for level in levels[CLIP_TAIL_STEPS - 1 :]:
            self.assertAlmostEqual(level, levels[-1], places=6)

    def test_the_source_decays_after_the_offset_step_instead_of_cutting_to_the_bed(self):
        """THE HEALTHY ARM. The silence arrives as a decay, over a length the IR sets.

        Measured on the CUE readout, which is what the agent hears: the first silent step
        is 481x the bed level and it reaches exactly the bed at ``CUE_TAIL_STEPS``, not
        one step earlier. What that makes reachable is unchanged by the split -- an agent
        can cross ``onset_rms`` for the first time AFTER the offset step, so
        ``onset_step > offset_step`` is a real outcome and must never be asserted against.
        """
        state = sounded(self.state, self.ir, self.clip, CLIP_TAIL_STEPS + 2)
        self.assertEqual(state.cue_tail_steps, CUE_TAIL_STEPS)
        self.assertEqual(state.clip_tail_steps, CLIP_TAIL_STEPS)

        state, heard = heard_step(
            state, ir=None, clip=self.clip, bed_cue=self.bed_cue, sounding=False
        )
        self.assertGreater(rms(heard), 100.0 * BED_RMS)

        for _ in range(CUE_TAIL_STEPS - 2):
            state, heard = heard_step(
                state, ir=None, clip=self.clip, bed_cue=self.bed_cue, sounding=False
            )
            self.assertGreater(rms(heard), BED_RMS)
        # ...and not one step early: at CUE_TAIL_STEPS - 1 folds there is still source
        self.assertFalse(np.array_equal(heard, self.bed_cue))

        state, heard = heard_step(
            state, ir=None, clip=self.clip, bed_cue=self.bed_cue, sounding=False
        )
        np.testing.assert_array_equal(heard, self.bed_cue)
        self.assertAlmostEqual(rms(heard), BED_RMS, places=9)

    def test_the_hard_cut_is_the_arm_this_replaces(self):
        """THE FORCED-FAILURE ARM: the same question against the previous mechanism.

        ADR-0017: "the silence arrives as a hard step to the bed with no tail, which is
        unphysical". Same state, same step -- ``bed.heard_signal(playing=False)`` returns
        the bed exactly and the accumulator returns a signal strictly above it.
        """
        state = sounded(self.state, self.ir, self.clip, CLIP_TAIL_STEPS + 2)

        cut = heard_signal(self.ir, self.clip, self.bed_clip, playing=False)
        np.testing.assert_array_equal(cut, self.bed_clip)
        self.assertAlmostEqual(rms(cut), BED_RMS, places=9)

        _, tailed = heard_step(
            state, ir=None, clip=self.clip, bed_cue=self.bed_cue, sounding=False
        )
        self.assertGreater(rms(tailed), rms(cut))
        self.assertGreater(rms(tailed), 100.0 * BED_RMS)

    def test_the_remaining_energy_never_increases_once_the_window_closes(self):
        """The honest post-offset invariant.

        NOT per-step monotone RMS and NOT "the measured RMS is the bed at the offset
        step": the tail is exactly what makes both false, and a symmetric mirror of
        ``onset.py``'s pre-``t_anom`` clause would be wrong by design. What is true is
        that no energy enters a silent step, so what is left can only leave.
        """
        state = sounded(self.state, self.ir, self.clip, CLIP_TAIL_STEPS + 2)
        energies = []
        for _ in range(CLIP_TAIL_STEPS + 2):
            state = advance_tail(state, ir=None, clip=self.clip, sounding=False)
            energies.append(float(np.sum(np.square(state.buffer))))

        self.assertEqual(energies[-1], 0.0)
        for earlier, later in zip(energies, energies[1:]):
            if earlier == 0.0:
                self.assertEqual(later, 0.0)
            else:
                self.assertLess(later, earlier)

    def test_the_fold_never_writes_into_the_state_it_was_given(self):
        """Pure in, pure out. The runner threads this the way it threads ``OnsetState``,
        and a fold that mutated its input would make two consumers of one state disagree.
        """
        state = sounded(self.state, self.ir, self.clip, 3)
        before = np.array(state.buffer)
        after = advance_tail(state, ir=self.ir, clip=self.clip, sounding=True)
        np.testing.assert_array_equal(state.buffer, before)
        self.assertIsNot(after.buffer, state.buffer)
        self.assertEqual(state.phase, 3 * HOP % WINDOW)

    def test_the_clip_readout_is_a_copy_even_when_the_buffer_never_grew(self):
        """``clip_readout`` promises a COPY, and the first implementation did not give one.

        ``np.ascontiguousarray`` copies only when it has to, so it handed back the
        buffer's own memory for exactly the states whose buffer is ``window`` wide and
        C-contiguous -- ``open_tail(headroom=0)`` before the first sounding fold, which
        is the whole pre-``t_anom`` phase whenever no IR width is available up front
        (``run_episode``'s ``ir_shape is None``). The grown case copied and the never-grown
        case aliased, so a test that only ever looked at a settled accumulator was green
        against both.

        Nothing corrupts today, because ``mix_bed`` allocates. The guarantee is asserted
        anyway: it is stated in the docstring, ``clip_readout`` is exported, and the
        failure it prevents -- a caller writing into what it was told was its own -- is
        the same aliasing class as ``heard_signal``'s shared-bed branch.
        """
        never_grown = advance_tail(
            open_tail(window=WINDOW, hop=HOP), ir=None, clip=self.clip, sounding=False
        )
        self.assertEqual(never_grown.buffer.shape, (2, WINDOW))
        readout = clip_readout(never_grown)
        self.assertFalse(
            np.shares_memory(readout, never_grown.buffer),
            "the readout aliases the accumulator's own buffer",
        )
        readout[:] = 7.0
        self.assertEqual(float(np.max(np.abs(never_grown.buffer))), 0.0)

        grown = sounded(self.state, self.ir, self.clip, 3)
        self.assertFalse(np.shares_memory(clip_readout(grown), grown.buffer))

    def test_the_clip_loops_when_the_hop_does_not_divide_it(self):
        """The wrap INSIDE one step's chunk, which every other fixture hides.

        ``phase`` wraps at ``len(clip)``, so when the hop divides the clip length the
        cyclic read never crosses the end and the modulo in ``_looped_chunk`` is a no-op
        -- which is true of every other constant in this file (800/100), of the runner's
        Mac fixture (2205/441) and of the box's own 5 s clip at a 1 s step (220500/44100).
        Removing the wrap therefore left the whole suite green.

        It matters because the module docstring rests on it: *"the clip loops for the
        duration of the window ... a source that faded out mid-window would put the
        window's duration under the recording's control rather than the task's"*. A hop
        that does not divide the clip is the ordinary case for any other ``step_seconds``.

        The IR is a unit impulse so the buffer's newly written region IS the emitted
        chunk, and the expectation is built with ``np.take(mode="wrap")`` rather than
        with the module's own arithmetic.
        """
        hop = 300  # 800 = 2*300 + 200, so the third chunk straddles the clip's end
        state = open_tail(window=WINDOW, hop=hop)

        for fold in range(4):
            phase = state.phase
            state = advance_tail(state, ir=anechoic_ir(), clip=self.clip, sounding=True)
            expected = np.take(self.clip, np.arange(phase, phase + hop), mode="wrap")
            np.testing.assert_allclose(
                state.buffer[0, WINDOW - hop : WINDOW], expected, atol=1e-6,
                err_msg="fold {} from phase {} did not read the clip cyclically".format(
                    fold, phase
                ),
            )
        # ...and the third fold is the one that actually crossed the end.
        self.assertLess(2 * hop, WINDOW)
        self.assertGreater(3 * hop, WINDOW)


class TestTheTwoReadouts(unittest.TestCase):
    """ADR-0019: one buffer, two windows. Everything here is about the pair.

    The defect being repaired: the agent read the last ``N`` samples every step while a
    step is ``hop`` samples long, so its "instantaneous" RMS was an ``N/hop``-step moving
    average and its decay was the analysis window emptying.
    """

    def setUp(self):
        self.ir = fakes.synthetic_ir(n_samples=IR_SAMPLES)
        self.clip = a_clip()

    def test_the_cue_readout_is_the_clip_readouts_last_hop_samples_exactly(self):
        """Two readouts, ONE buffer. They must never be able to diverge.

        A second buffer, a second slice expression, or a rounding difference in the index
        arithmetic would put the agent and CLAP on signals that disagree about the same
        samples -- and nothing downstream compares them, so it would be silent.
        """
        for headroom in (0, IR_SAMPLES - 1):
            state = advance_tail(
                open_tail(window=WINDOW, hop=HOP, headroom=headroom),
                ir=self.ir, clip=self.clip, sounding=True,
            )
            cue, whole = cue_readout(state), clip_readout(state)
            self.assertEqual(cue.shape, (2, HOP))
            self.assertEqual(whole.shape, (2, WINDOW))
            np.testing.assert_array_equal(cue, whole[:, -HOP:], str(headroom))

    def test_the_cue_readout_is_a_copy_at_every_headroom(self):
        """The same guarantee ``clip_readout`` makes, asserted rather than assumed.

        **``clip_readout``'s aliasing scar does NOT reproduce on this slice, and that was
        MEASURED rather than argued.** Replacing ``np.array`` with ``np.ascontiguousarray``
        here leaves the whole suite green: ``buffer[:, window - hop : window]`` is two
        rows of a wider array, strides ``(buffer_width * 4, 4)``, so it is never
        C-contiguous while ``hop < window`` -- which ``open_tail`` refuses to allow
        otherwise -- and ``ascontiguousarray`` therefore always copies.

        So this test is about the guarantee and not about the bug: a copy that happens
        only because of a stride accident stops being a copy the day someone widens the
        hop or reshapes the buffer. Both headrooms and the never-sounded state, because
        the never-grown buffer is the case that aliased for ``clip_readout``.
        """
        never_sounded = advance_tail(
            open_tail(window=WINDOW, hop=HOP), ir=None, clip=self.clip, sounding=False
        )
        self.assertEqual(never_sounded.buffer.shape, (2, WINDOW))
        states = [never_sounded] + [
            advance_tail(
                open_tail(window=WINDOW, hop=HOP, headroom=headroom),
                ir=self.ir, clip=self.clip, sounding=True,
            )
            for headroom in (0, IR_SAMPLES - 1)
        ]
        for state in states:
            slice_of_buffer = state.buffer[:, WINDOW - HOP : WINDOW]
            self.assertFalse(slice_of_buffer.flags["C_CONTIGUOUS"])
            readout = cue_readout(state)
            self.assertFalse(
                np.shares_memory(readout, state.buffer),
                "the cue readout aliases the accumulator's buffer at buffer width "
                "{}".format(state.buffer.shape[1]),
            )
            before = float(np.sum(np.square(state.buffer)))
            readout[:] = 7.0
            self.assertEqual(float(np.sum(np.square(state.buffer))), before)

    def test_the_two_decay_curves_and_the_cue_empties_strictly_first(self):
        """**The measurement the split exists to make.** Both arms, one buffer.

        Measured here and printed by the box arm at the shipped numbers: the cue readout
        is exactly zero ``cue_tail_steps`` folds after the last sounding step and the clip
        readout at ``clip_tail_steps``, 7 against 14 at this fixture.
        """
        cue_curve, clip_curve, cue_zero, clip_zero = decay_curves(
            self.ir, self.clip, HOP, CLIP_TAIL_STEPS + 2
        )
        print(
            "\n  [tail] fixture decay, fraction of settled"
            "\n    cue  " + "  ".join("{:.4f}".format(v) for v in cue_curve[:9])
            + "\n    clip " + "  ".join("{:.4f}".format(v) for v in clip_curve[:15]),
            flush=True,
        )
        self.assertEqual(cue_zero, CUE_TAIL_STEPS)
        self.assertEqual(clip_zero, CLIP_TAIL_STEPS)
        self.assertLess(
            cue_zero, clip_zero,
            "the cue readout must empty strictly before the clip readout -- if it does "
            "not, the two windows are the same window and the split did nothing",
        )
        # measured 0.8213 0.1905 0.0651 ... against 0.9533 0.8876 0.8002 ...
        self.assertLess(cue_curve[1], 0.5 * clip_curve[1])

    def test_the_clip_readout_is_untouched_and_is_what_clap_is_handed(self):
        """ADR-0018's bank of record was measured on clip-length waveforms.

        ``heard_clip_window`` must return exactly what ``heard_step`` returned before the
        split, so the CLAP domain is provably unmoved: ``mix_bed(clip_readout, bed_clip)``,
        clip-length, once per episode.
        """
        state = sounded(
            open_tail(window=WINDOW, hop=HOP, headroom=IR_SAMPLES - 1),
            self.ir, self.clip, 4,
        )
        bed_clip = bed_signal(WINDOW, BED_RMS)
        heard = heard_clip_window(state, bed_clip=bed_clip)
        self.assertEqual(heard.shape, (2, WINDOW))
        np.testing.assert_allclose(heard, clip_readout(state) + bed_clip, atol=0.0)

    def test_handing_heard_step_the_clip_bed_dies_loudly(self):
        """THE FORCED-FAILURE ARM for the ``bed`` -> ``bed_cue`` rename.

        The rename is load-bearing rather than cosmetic: it is what makes ``mix_bed``'s
        shape refusal fire instead of a caller composing a signal of the wrong domain.
        """
        with self.assertRaises(ValueError) as caught:
            heard_step(
                open_tail(window=WINDOW, hop=HOP),
                ir=self.ir, clip=self.clip,
                bed_cue=bed_signal(WINDOW, BED_RMS), sounding=False,
            )
        message = str(caught.exception)
        self.assertIn(str(HOP), message)
        self.assertIn(str(WINDOW), message)
        self.assertIn("bed_cue", message)


class TestTheCueRamp(unittest.TestCase):
    """``CUE_RAMP_STEPS`` is 1, and FILL is not the same thing as LEVEL-SETTLE."""

    def setUp(self):
        self.ir = fakes.synthetic_ir(n_samples=IR_SAMPLES)
        self.clip = a_clip()

    def test_one_sounding_fold_writes_the_whole_cue_window(self):
        """Why ``onset_delay_steps`` carries no fill bias since the split.

        Fold 0 already reads at 0.8463 of the settled cue level, where the clip readout is
        at 0.2992 and needs ``CLIP_RAMP_STEPS`` folds to reach 0.9526. That is the whole
        of ``CUE_RAMP_STEPS = 1``: one fold emits ``hop`` samples and the cue window is
        ``hop`` wide, so there is nothing left to fill.
        """
        self.assertEqual(CUE_RAMP_STEPS, 1)
        state = advance_tail(
            open_tail(window=WINDOW, hop=HOP, headroom=IR_SAMPLES - 1),
            ir=self.ir, clip=self.clip, sounding=True,
        )
        settled = cue_level(steady_state_cue_rms(self.ir, self.clip, hop=HOP))
        first_cue = rms(cue_readout(state))
        first_clip = rms(clip_readout(state))
        self.assertGreater(first_cue, 0.0)
        self.assertGreater(first_cue, 0.8 * settled)  # measured 0.8463
        # ...and the CLIP readout on the same fold is nowhere near its own settled level
        self.assertLess(first_clip / rms(steady_state_render(self.ir, self.clip, hop=HOP)),
                        0.4)  # measured 0.2992

    def test_the_cue_level_still_settles_over_the_rooms_own_build_up(self):
        """FILL is 1; LEVEL-SETTLE is ``cue_tail_steps``, and they are different numbers.

        The cue window is WRITTEN whole by fold 1 and its level still RISES afterwards,
        because the room's reverberation from earlier folds keeps arriving for ``L``
        samples. For the CLIP readout the two nearly coincide (5 and 7 at the box) only
        because ``N >> L``; here they are 1 and 7.

        Compared at ONE loop phase, which is the only honest comparison: fold 1 and fold
        ``1 + PHASE_FOLDS`` emit from the same point in the clip, so a difference between
        them is the room and nothing else. Measured 0.8463 of the settled level against
        1.1850, a rise of 40%.
        """
        state = open_tail(window=WINDOW, hop=HOP, headroom=IR_SAMPLES - 1)
        readings = []
        for _ in range(2 * PHASE_FOLDS + 1):
            state = advance_tail(state, ir=self.ir, clip=self.clip, sounding=True)
            readings.append(rms(cue_readout(state)))
        self.assertEqual(state.cue_tail_steps, CUE_TAIL_STEPS)
        # fold 1 against fold 1 + PHASE_FOLDS: same phase, one still building up
        self.assertGreater(readings[PHASE_FOLDS], 1.2 * readings[0])
        # ...and by then it IS settled, so the next period repeats it exactly
        self.assertAlmostEqual(
            readings[PHASE_FOLDS], readings[2 * PHASE_FOLDS], places=12
        )


class TestTheAnechoicControl(unittest.TestCase):
    """**THE measurement that decides whether the tail is now really reverberation.**

    Run at the BOX's numbers (``N = 220500``, ``hop = 44100``, ``L = 72300`` at RT60
    0.8 s) because that is what the claim is about: ``L < N`` in every configuration this
    tree can produce, so the CLIP tail can never be reverb-dominated.
    """

    def setUp(self):
        self.clip = (
            np.random.default_rng(3).standard_normal(BOX_WINDOW) * 0.1
        ).astype(np.float32)
        self.room = a_room_ir()

    def test_the_clip_tail_is_reproduced_by_a_room_with_no_reverberation(self):
        """THE FORCED-FAILURE ARM, and it is the defect ADR-0019 corrects.

        A 1-sample IR has literally no reverberation and still reproduces the CLIP
        readout's decay to within 1.3 points over the first four steps -- measured
        0.9015 0.7829 0.6420 0.4602 for the room against 0.8953 0.7747 0.6329 0.4476
        anechoic. **``clip_tail_steps`` is therefore not evidence that the geometric
        acoustics did any work.**
        """
        room_cue, room_clip, _, room_clip_zero = decay_curves(
            self.room, self.clip, BOX_HOP, 8
        )
        dead_cue, dead_clip, dead_cue_zero, dead_clip_zero = decay_curves(
            anechoic_ir(), self.clip, BOX_HOP, 8
        )
        print(
            "\n  [tail] box anechoic control, fraction of settled"
            "\n    clip room     " + "  ".join("{:.4f}".format(v) for v in room_clip[:6])
            + "\n    clip anechoic " + "  ".join("{:.4f}".format(v) for v in dead_clip[:6])
            + "\n    cue  room     " + "  ".join("{:.4f}".format(v) for v in room_cue[:4])
            + "\n    cue  anechoic " + "  ".join("{:.4f}".format(v) for v in dead_cue[:4]),
            flush=True,
        )
        for fold, (room, dead) in enumerate(zip(room_clip[:4], dead_clip[:4])):
            self.assertLess(
                abs(room - dead), 0.02,
                "the clip readout's decay at fold {} is {:.4f} with a room and {:.4f} "
                "with none -- if these ever separate, the L < N argument has "
                "changed".format(fold, room, dead),
            )
        self.assertEqual(room_clip_zero, 7)
        self.assertEqual(dead_clip_zero, 5)
        self.assertEqual(dead_cue_zero, 1)

    def test_the_cue_tail_is_NOT_reproduced_and_that_is_the_repair(self):
        """THE HEALTHY ARM. The cue tail collapses when the room is taken away.

        With the room the cue readout is 0.2450 of settled on the offset step and reaches
        exactly zero at ``cue_tail_steps`` = 3. With a 1-sample IR it is exactly zero on
        the offset step itself and ``cue_tail_steps`` is 1 -- a 24.5 point separation
        against the clip readout's 0.6. **The cue tail IS reverberation.**
        """
        state = sounded(
            open_tail(window=BOX_WINDOW, hop=BOX_HOP, headroom=BOX_IR_SAMPLES - 1),
            self.room, self.clip, 8,
        )
        dead = sounded(
            open_tail(window=BOX_WINDOW, hop=BOX_HOP),
            anechoic_ir(), self.clip, 8,
        )
        self.assertEqual(state.cue_tail_steps, 3)
        self.assertEqual(dead.cue_tail_steps, 1)
        self.assertEqual(state.clip_tail_steps, 7)
        self.assertEqual(dead.clip_tail_steps, 5)

        room_cue, _, _, _ = decay_curves(self.room, self.clip, BOX_HOP, 3)
        dead_cue, _, _, _ = decay_curves(anechoic_ir(), self.clip, BOX_HOP, 3)
        self.assertEqual(dead_cue[0], 0.0)
        self.assertGreater(room_cue[0], 0.2)  # measured 0.2450
        self.assertGreater(
            room_cue[0] - dead_cue[0], 0.2,
            "the cue readout's first silent step is the same with and without a room, "
            "which would mean the split did NOT make the tail reverberation",
        )


class TestTheLoopPhase(unittest.TestCase):
    """The cue reading cycles with ``phase_folds``. Honest, and it has to be measurable."""

    def setUp(self):
        self.ir = fakes.synthetic_ir(n_samples=IR_SAMPLES)
        self.clip = a_clip()

    def test_the_period_is_phase_folds_and_is_not_one(self):
        """At a held pose the cue RMS repeats at a lag of ``PHASE_FOLDS`` and not at 1.

        Measured: lag-8 agreement is exact (0.000e+00 max difference) while lag-1 differs
        by 1.44e-01 on a level of 0.58. That is the intermittency; it bounds the onset's
        worst-case delay at ``phase_folds - 1`` steps and it cannot prevent a crossing,
        because ``observe_step`` is one-shot and monotone-latching.
        """
        self.assertEqual(phase_folds(window=WINDOW, hop=HOP), PHASE_FOLDS)
        state = sounded(
            open_tail(window=WINDOW, hop=HOP, headroom=IR_SAMPLES - 1),
            self.ir, self.clip, 30,
        )
        series = []
        for _ in range(2 * PHASE_FOLDS + 4):
            state = advance_tail(state, ir=self.ir, clip=self.clip, sounding=True)
            series.append(rms(cue_readout(state)))
        self.assertEqual(state.phase_folds, PHASE_FOLDS)
        for early, late in zip(series, series[PHASE_FOLDS:]):
            self.assertAlmostEqual(early, late, places=12)
        self.assertGreater(
            max(abs(a - b) for a, b in zip(series, series[1:])), 0.1,
            "the cue reading does not move fold to fold at a held pose, so this fixture "
            "cannot see the loop phase at all",
        )

    def test_the_crest_and_the_min_ratio_on_a_flat_clip_and_on_a_burst(self):
        """BOTH ARMS. A clip whose energy is flat over the hop, and one whose is not.

        Measured at this fixture: a constant clip gives crest 1.0000 and min_ratio
        1.000000 -- every fold identical -- while a 60-sample burst on an 800-sample loop
        gives crest 2.7451 and min_ratio 0.000000, one fold carrying the sound and one
        carrying nothing at all. The second is most of ESC-50.
        """
        flat = np.full(WINDOW, 0.1, dtype=np.float32)
        burst = np.zeros(WINDOW, dtype=np.float32)
        burst[:60] = 1.0

        flat_phases = steady_state_cue_rms(self.ir, flat, hop=HOP)
        burst_phases = steady_state_cue_rms(self.ir, burst, hop=HOP)
        print(
            "\n  [tail] cue crest / min_ratio"
            "\n    flat  {:.4f} / {:.6f}".format(
                cue_crest(flat_phases), cue_min_ratio(flat_phases))
            + "\n    burst {:.4f} / {:.6f}".format(
                cue_crest(burst_phases), cue_min_ratio(burst_phases)),
            flush=True,
        )
        self.assertAlmostEqual(cue_crest(flat_phases), 1.0, places=4)
        self.assertAlmostEqual(cue_min_ratio(flat_phases), 1.0, places=4)
        self.assertGreater(cue_crest(burst_phases), 2.0)
        self.assertLess(cue_min_ratio(burst_phases), 0.05)

    def test_the_crest_is_over_the_QUADRATIC_mean_and_a_ceiling_of_sqrt_folds_says_so(self):
        """WHICH denominator, pinned, because ``> 2.0`` above cannot tell two apart.

        ``cue_crest`` divides by ``cue_level`` -- the quadratic mean, the aggregation the
        threshold is placed with -- and swapping in the arithmetic mean was measured to
        leave the whole suite green: on a burst both denominators give a crest over 2 and
        the flat arm gives 1.0 either way. That is a silently different number on the
        calibration record and in every box print.

        The property that separates them is a CEILING. For non-negative phases the
        quadratic mean is at least ``max / sqrt(n)``, so ``crest <= sqrt(phase_folds)``
        always, with equality when one fold carries everything. Over the arithmetic mean
        the same burst reaches ``n`` -- 8 here against a ceiling of 2.83.
        """
        burst = np.zeros(WINDOW, dtype=np.float32)
        burst[:60] = 1.0
        phases = steady_state_cue_rms(self.ir, burst, hop=HOP)
        arithmetic = sum(phases) / len(phases)
        print(
            "\n  [tail] crest denominators: quadratic {:.4f}  arithmetic {:.4f}  "
            "ceiling sqrt({}) = {:.4f}".format(
                cue_crest(phases), max(phases) / arithmetic, PHASE_FOLDS,
                math.sqrt(PHASE_FOLDS)),
            flush=True,
        )
        self.assertAlmostEqual(
            cue_crest(phases), max(phases) / cue_level(phases), places=12
        )
        self.assertLessEqual(cue_crest(phases), math.sqrt(PHASE_FOLDS))
        # ...and the rejected denominator breaks that ceiling on this very fixture.
        self.assertGreater(max(phases) / arithmetic, math.sqrt(PHASE_FOLDS))

    def test_cue_level_refuses_an_empty_sweep_rather_than_returning_zero(self):
        """A level of zero and a measurement that did not happen are opposite facts."""
        with self.assertRaises(ValueError):
            cue_level([])
        self.assertAlmostEqual(cue_level([3.0, 4.0]), math.sqrt(12.5), places=12)

    def test_phase_folds_is_one_definition_read_in_two_places(self):
        for window, hop, expected in (
            (WINDOW, HOP, 8), (2205, 441, 5), (BOX_WINDOW, BOX_HOP, 5), (800, 300, 8)
        ):
            self.assertEqual(phase_folds(window=window, hop=hop), expected)
            self.assertEqual(
                open_tail(window=window, hop=hop).phase_folds, expected
            )
        with self.assertRaises(ValueError):
            phase_folds(window=0, hop=1)


class TestTheBufferGrows(unittest.TestCase):
    def test_a_wider_ir_grows_the_buffer_rather_than_truncating_the_tail(self):
        """A buffer sized on ticket 06's measured 72300 would truncate a tail on a more
        reverberant scene, quietly, and the signal would stay plausible.
        ``spec.py:81-86`` sets no ``maxIRLength``, so there is no width to size against.
        """
        clip = a_clip()
        narrow = fakes.synthetic_ir(n_samples=64)
        wide = fakes.synthetic_ir(n_samples=4096)

        state = open_tail(window=WINDOW, hop=HOP)
        state = advance_tail(state, ir=narrow, clip=clip, sounding=True)
        grows_after_narrow = state.n_grows
        state = advance_tail(state, ir=wide, clip=clip, sounding=True)
        self.assertEqual(state.n_grows, grows_after_narrow + 1)
        self.assertEqual(state.max_ir_samples, 4096)
        self.assertEqual(state.buffer.shape, (2, WINDOW + 4096 - 1))

        # the energy is conserved against an INDEPENDENT convolution (time domain)
        fresh = advance_tail(
            open_tail(window=WINDOW, hop=HOP), ir=wide, clip=clip, sounding=True
        )
        expected = sum(
            float(np.sum(np.square(np.convolve(clip[:HOP], wide[ear]))))
            for ear in (0, 1)
        )
        actual = float(np.sum(np.square(fresh.buffer)))
        self.assertLess(abs(actual - expected) / expected, 1e-5)

        # THE FORCED-FAILURE ARM: the same fold, clipped to the pre-grow width.
        truncated = np.zeros((2, WINDOW), dtype=np.float32)
        for ear in (0, 1):
            conv = np.convolve(clip[:HOP], wide[ear]).astype(np.float32)
            room = WINDOW - (WINDOW - HOP)
            truncated[ear, WINDOW - HOP :] = conv[:room]
        lost = float(np.sum(np.square(truncated)))
        self.assertLess(lost, 0.9 * expected)

    def test_both_tails_widen_with_the_ir_rather_than_with_a_constant(self):
        """The two tails are DERIVED from the widest IR seen, and they separate cleanly.

        This is what stops either of them from being writable as a literal. At this
        fixture a 64-sample IR gives cue 2 / clip 9 and a 4096-sample IR gives cue 42 /
        clip 49 -- both tails move by the same 40 folds, because both are the same ``L``
        divided by the same ``hop``, offset by ``hop`` against ``window``.
        """
        clip = a_clip()
        state = advance_tail(
            open_tail(window=WINDOW, hop=HOP),
            ir=fakes.synthetic_ir(n_samples=64), clip=clip, sounding=True,
        )
        self.assertEqual(state.cue_tail_steps, 2)
        self.assertEqual(state.clip_tail_steps, 9)
        wider = advance_tail(
            state, ir=fakes.synthetic_ir(n_samples=4096), clip=clip, sounding=True
        )
        self.assertEqual(wider.cue_tail_steps, 42)
        self.assertEqual(wider.clip_tail_steps, 49)
        self.assertEqual(wider.clip_ramp_steps, CLIP_RAMP_STEPS)
        self.assertEqual(wider.phase_folds, PHASE_FOLDS)

    def test_a_preallocated_headroom_means_the_buffer_never_reallocates(self):
        """ADR-0017's word is "preallocated"; this is preallocated-with-an-explicit-grow,
        and the hint is what makes the grow never fire on the ordinary path."""
        clip = a_clip()
        ir = fakes.synthetic_ir(n_samples=IR_SAMPLES)
        state = open_tail(window=WINDOW, hop=HOP, headroom=IR_SAMPLES - 1)
        state = sounded(state, ir, clip, 5)
        self.assertEqual(state.n_grows, 0)
        self.assertEqual(state.buffer.shape, (2, WINDOW + IR_SAMPLES - 1))


class TestRefusals(unittest.TestCase):
    def test_the_control_arm_is_the_real_fold_minus_one_if(self):
        """The NaN control has to BE ``advance_tail`` without the guard, and it was not.

        It read ``clip[:hop]`` where the real fold reads the looped chunk from ``phase``,
        so it agreed on fold 0 and drifted on every fold after -- a control that would
        not have tracked a change to the fold it is the control for. Measured here on a
        FINITE IR over four folds, which is the arm that makes the poisoned arm below
        evidence about the missing ``if``.
        """
        clip = a_clip()
        ir = fakes.synthetic_ir(n_samples=IR_SAMPLES)
        real = open_tail(window=WINDOW, hop=HOP)
        control = open_tail(window=WINDOW, hop=HOP)
        for fold in range(4):
            real = advance_tail(real, ir=ir, clip=clip, sounding=True)
            control = _fold_without_the_finite_check(control, ir=ir, clip=clip)
            np.testing.assert_array_equal(
                control.buffer,
                real.buffer,
                "the control diverged from advance_tail on fold {}".format(fold),
            )
            self.assertEqual(control, real, fold)
        # ...and the clip really did loop inside those folds, so the agreement is not
        # about a phase that never moved: 4 folds of 100 samples into an 800-sample clip.
        self.assertEqual(real.phase, 4 * HOP)
        self.assertGreater(real.phase, 0)

    def test_a_non_finite_ir_raises_rather_than_poisoning_every_later_step(self):
        """``render_through_ir`` is stateless, so a NaN cost exactly one step. An
        accumulator carries it forever, and ``NaN >= onset_rms`` is False, so the onset
        would never fire again and the episode would read as ordinary attrition."""
        clip = a_clip()
        ir = np.array(fakes.synthetic_ir(n_samples=IR_SAMPLES))
        ir[1, 17] = np.nan
        state = open_tail(window=WINDOW, hop=HOP)

        with self.assertRaises(TailError) as caught:
            advance_tail(state, ir=ir, clip=clip, sounding=True)
        self.assertIn("onset_rms", str(caught.exception))

        # THE CONTROL: the same fold with the check taken out. The NaN stays in the
        # buffer for the whole of the tail, and every reading in between compares False
        # against a threshold it can never cross again. Read on the CUE readout, which is
        # what `observe_step` sees -- it goes NaN sooner and clears sooner, and neither
        # helps: a NaN that clears after `cue_tail_steps` has already latched nothing.
        poisoned = _fold_without_the_finite_check(state, ir=ir, clip=clip)
        self.assertEqual(poisoned.clip_tail_steps, CLIP_TAIL_STEPS)
        self.assertEqual(poisoned.cue_tail_steps, CUE_TAIL_STEPS)
        for _ in range(poisoned.cue_tail_steps - 1):
            measured = rms(cue_readout(poisoned))
            self.assertTrue(math.isnan(measured))
            self.assertFalse(measured >= 0.003)
            poisoned = advance_tail(poisoned, ir=None, clip=clip, sounding=False)

    def test_a_step_that_outruns_the_read_window_is_refused(self):
        """Consecutive CLIP readouts would be DISJOINT -- a different sensor, one that
        drops the audio between two steps -- so it has to be asked for rather than fallen
        into. At ``hop == window`` there is a third reason: the cue readout IS the clip
        readout and ``phase_folds`` collapses to 1.
        """
        for hop in (WINDOW, WINDOW + 100):
            with self.assertRaises(ValueError) as caught:
                open_tail(window=WINDOW, hop=hop)
            message = str(caught.exception)
            self.assertIn(str(hop), message)
            self.assertIn(str(WINDOW), message)
            self.assertIn("step_seconds", message)

    def test_an_empty_window_or_hop_is_refused(self):
        with self.assertRaises(ValueError):
            open_tail(window=0, hop=1)
        with self.assertRaises(ValueError):
            open_tail(window=WINDOW, hop=0)

    def test_a_sounding_step_without_an_ir_names_what_is_missing(self):
        with self.assertRaises(ValueError) as caught:
            advance_tail(
                open_tail(window=WINDOW, hop=HOP),
                ir=None,
                clip=a_clip(),
                sounding=True,
            )
        self.assertIn("IR", str(caught.exception))

    def test_a_hop_of_zero_samples_is_refused_at_the_unit_conversion(self):
        with self.assertRaises(ValueError) as caught:
            hop_samples(step_seconds=0.0, sample_rate=44100)
        self.assertIn("44100", str(caught.exception))
        self.assertEqual(hop_samples(step_seconds=1.0, sample_rate=44100), 44100)
        self.assertEqual(hop_samples(step_seconds=0.25, sample_rate=44100), 11025)

    def test_the_hop_rounds_rather_than_truncating(self):
        """Both cases above land on exact products, so they cannot tell the two apart.

        ``0.7 * 44100`` is ``30869.999999999996`` in binary floating point: truncating
        loses a sample per step to representation error alone, and the loss is silent --
        the readout is a few parts per million quiet and every threshold moves with it.
        The two candidate ``step_seconds`` values named in ``audio/config.py``'s
        provenance box are 1.0 and 0.25, which is exactly why the exact cases were the
        ones written down.
        """
        self.assertEqual(hop_samples(step_seconds=0.7, sample_rate=44100), 30870)
        self.assertEqual(int(0.7 * 44100), 30869)
        self.assertEqual(hop_samples(step_seconds=0.03, sample_rate=44100), 1323)

    def test_an_empty_clip_is_refused_rather_than_emitting_silence(self):
        """A source that emits nothing is not a quiet episode, it is a broken one.

        ``load_anomaly_clip`` raises rather than substituting a synthetic burst, and this
        is the same rule one layer down: the accumulator folding an empty clip would run
        the whole window at the bed level, the onset would never fire, and the episode
        would read as ordinary §2.5 attrition.
        """
        with self.assertRaises(ValueError) as caught:
            advance_tail(
                open_tail(window=WINDOW, hop=HOP),
                ir=fakes.synthetic_ir(n_samples=IR_SAMPLES),
                clip=np.zeros(0, dtype=np.float32),
                sounding=True,
            )
        self.assertIn("empty", str(caught.exception))


class TestTheSteadyState(unittest.TestCase):
    def setUp(self):
        self.ir = fakes.synthetic_ir(n_samples=IR_SAMPLES)
        self.clip = a_clip()

    def test_the_identity_that_keeps_onset_rms_where_it_was(self):
        """**THE most important test in ADR-0019.**

        The ``phase_folds`` cue windows are disjoint, consecutive and tile the settled
        period an integer number of times, so the quadratic mean of their RMSs EQUALS the
        clip readout's RMS -- which is what ``steady_state_render`` returns and what the
        pre-split sweep placed ``onset_rms`` against. Measured ratio 1.000000000000 here
        for three IRs and, in the spec work, in all four configurations this tree ships.

        If this ever separates, the calibration sweep changed the threshold's LEVEL as
        well as its domain, and every gate number on disk becomes unpriceable.
        """
        for gain in (1.0, 0.5, 2.0):
            ir = fakes.synthetic_ir(n_samples=IR_SAMPLES, left=gain, right=gain)
            phases = steady_state_cue_rms(ir, self.clip, hop=HOP)
            self.assertEqual(len(phases), PHASE_FOLDS)
            reference = rms(steady_state_render(ir, self.clip, hop=HOP))
            self.assertAlmostEqual(
                cue_level(phases) / reference, 1.0, places=9,
                msg="gain {}: cue_level {:.10g} against steady_state_render {:.10g}"
                    .format(gain, cue_level(phases), reference),
            )

    def test_the_steady_state_render_is_what_the_clip_readout_converges_to(self):
        """The phase-independence is why the CLIP arm may take this number: the readout at
        a fixed pose is a rotation of one period of the looped clip, so its samples move
        and its RMS does not. The CUE readout does NOT have that property -- see
        ``TestTheLoopPhase`` -- which is why the sweep aggregates its phases instead.
        """
        settled = rms(steady_state_render(self.ir, self.clip, hop=HOP))

        state = sounded(
            open_tail(window=WINDOW, hop=HOP, headroom=IR_SAMPLES - 1),
            self.ir,
            self.clip,
            CLIP_TAIL_STEPS + 6,
        )
        self.assertAlmostEqual(rms(clip_readout(state)), settled, places=6)

        rotated = sounded(state, self.ir, self.clip, 7)
        self.assertAlmostEqual(rms(clip_readout(rotated)), settled, places=6)
        self.assertFalse(
            np.array_equal(clip_readout(rotated), clip_readout(state)),
            "the waveform is phase-dependent; only the RMS is not",
        )

    def test_the_accumulated_level_is_close_to_the_bare_render_but_not_equal_to_it(self):
        """The calibration domain shift, closed by the sweep sharing this code path
        rather than asserted away. ``onset.py`` names the failure and nothing raises."""
        bare = rms(render_through_ir(self.ir, self.clip))
        settled = rms(steady_state_render(self.ir, self.clip, hop=HOP))
        ratio = settled / bare
        # measured 1.0498 with this fixture; 1.0014 (RT60 0.2 s) to 1.0146 (RT60 2.0 s)
        # on a 5 s clip and a 72300-sample IR -- provenance: fake, synthetic IRs
        # throughout. It is a property of the IR's decay and the clip, not a constant.
        self.assertGreater(ratio, 0.75)
        self.assertLess(ratio, 1.25)
        self.assertNotAlmostEqual(ratio, 1.0, places=3)

    def test_the_settled_readout_is_the_clips_length(self):
        """``mix_bed`` refuses anything else, and ADR-0018's bank was measured on it."""
        rendered = steady_state_render(self.ir, self.clip, hop=HOP)
        self.assertEqual(rendered.shape, (2, WINDOW))
        self.assertEqual(rendered.dtype, np.float32)

    def test_the_cue_sweep_returns_the_loops_phases_whatever_the_settle(self):
        """The settle is the CLIP settle, and it is conservative rather than necessary.

        What must be true is that the answer does not depend on how long the settle ran,
        only that it ran long enough -- so a sweep settled three times as long returns the
        SAME MULTISET of phases, rotated, and the same ``cue_level``. Order rotates
        because a longer settle leaves the accumulator at a different point in the loop;
        that is why the identity is stated on the aggregate and never on phase 0.

        BOTH ARMS, because "long enough" needs a boundary to mean anything. Measured at
        this fixture: a settle of ``CUE_TAIL_STEPS - 1`` = 6 folds already reproduces the
        multiset exactly, and a settle of 4 does not -- its level is 0.0143% high, a
        threshold placed against a ramp. ``clip_tail_steps + 1`` = 15 is chosen anyway,
        because it is the expression ``steady_state_render`` uses and the two sides of the
        identity settling by one expression is a property a reader can check.
        """
        settled = steady_state_cue_rms(self.ir, self.clip, hop=HOP)

        def phases_after(settle):
            state = sounded(
                open_tail(window=WINDOW, hop=HOP, headroom=IR_SAMPLES - 1),
                self.ir, self.clip, settle,
            )
            out = []
            for _ in range(PHASE_FOLDS):
                state = advance_tail(state, ir=self.ir, clip=self.clip, sounding=True)
                out.append(rms(cue_readout(state)))
            return out

        longer = phases_after(3 * (CLIP_TAIL_STEPS + 1))
        for one, other in zip(sorted(settled), sorted(longer)):
            self.assertAlmostEqual(one, other, places=12)
        self.assertAlmostEqual(cue_level(settled), cue_level(longer), places=12)

        enough = phases_after(CUE_TAIL_STEPS - 1)
        self.assertAlmostEqual(cue_level(enough), cue_level(settled), places=12)

        # THE FORCED-FAILURE ARM: a settle inside the room's own build-up
        too_short = phases_after(4)
        self.assertNotAlmostEqual(cue_level(too_short), cue_level(settled), places=9)


class TestTheCallableSurface(unittest.TestCase):
    def test_the_public_signatures_are_what_the_runner_calls(self):
        """``test_box_call_arity.py`` exists because ``World(scene_path)`` passed a green
        Mac suite, a green lint and a merge, and cost a V100 round trip. The runner is the
        caller here and it lands in a different diff.

        ``heard_step``'s ``bed_cue`` is pinned here for a second reason: the rename from
        ``bed`` is what makes ``mix_bed``'s shape refusal fire on a caller that hands over
        the clip-length bed, so a silent revert to ``bed`` would restore the failure.
        """
        expected = {
            heard_step: (
                ["state"],
                ["ir", "clip", "bed_cue", "sounding"],
            ),
            heard_clip_window: (["state"], ["bed_clip"]),
            advance_tail: (["state"], ["ir", "clip", "sounding"]),
            open_tail: ([], ["window", "hop", "headroom"]),
            hop_samples: ([], ["step_seconds", "sample_rate"]),
            phase_folds: ([], ["window", "hop"]),
            cue_readout: (["state"], []),
            clip_readout: (["state"], []),
            cue_level: (["phases"], []),
            cue_crest: (["phases"], []),
            cue_min_ratio: (["phases"], []),
        }
        for function, (positional, keyword_only) in expected.items():
            parameters = inspect.signature(function).parameters
            self.assertEqual(
                [
                    name
                    for name, p in parameters.items()
                    if p.kind is not inspect.Parameter.KEYWORD_ONLY
                ],
                positional,
                function.__name__,
            )
            self.assertEqual(
                [
                    name
                    for name, p in parameters.items()
                    if p.kind is inspect.Parameter.KEYWORD_ONLY
                ],
                keyword_only,
                function.__name__,
            )

        for function in (steady_state_render, steady_state_cue_rms):
            rendered = inspect.signature(function).parameters
            self.assertEqual(list(rendered), ["ir", "clip", "hop"], function.__name__)
            self.assertIs(rendered["hop"].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_the_old_single_readout_name_is_gone(self):
        """No alias, deliberately. ``tail_readout`` is the name that says nothing about
        which of the two windows it returns, and after ADR-0019 "the tail" is the one word
        that means two different lengths. An alias would leave the swappable name alive at
        every call site, which is the swap this change exists to prevent.
        """
        self.assertFalse(hasattr(tail_module, "tail_readout"))
        self.assertNotIn("tail_readout", tail_module.__all__)
        self.assertFalse(hasattr(TailState, "tail_steps"))
        self.assertFalse(hasattr(TailState, "source_fill"))
        self.assertEqual(
            sorted(tail_module.__all__),
            sorted(
                [
                    "TailError", "TailState", "CUE_RAMP_STEPS", "hop_samples",
                    "phase_folds", "open_tail", "advance_tail", "cue_readout",
                    "clip_readout", "heard_step", "heard_clip_window",
                    "steady_state_render", "steady_state_cue_rms", "cue_level",
                    "cue_crest", "cue_min_ratio",
                ]
            ),
        )

    def test_two_states_compare_on_their_scalars_rather_than_raising_on_the_buffer(self):
        """A bare ``==`` on a frozen dataclass holding an ndarray raises "truth value of
        an array is ambiguous", and a 2.3 MB repr in a traceback is unreadable."""
        state = open_tail(window=WINDOW, hop=HOP)
        self.assertEqual(state, open_tail(window=WINDOW, hop=HOP))
        self.assertNotIn("array", repr(state))
        self.assertIsInstance(state, TailState)


if __name__ == "__main__":
    unittest.main(verbosity=2)
