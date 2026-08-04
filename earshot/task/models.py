"""The model connectors the wiring layer constructs, and the seam they satisfy.

``audio/clap.py`` is pure: it takes an object exposing ``encode_audio(waveform,
sample_rate)`` and ``encode_text(text)`` and does the cosines itself. That injection is
what keeps the whole audio layer Mac-testable and numpy-only, and it is also why the
concrete encoder has had **no home** until now — ADR-0013's tree names ``vlm.py`` for
the captioner and names nothing for CLAP, because ticket 22 dissolved the constructor
the assertion used to live in.

It lands here, in ``task/``, for the reason ADR-0013 gives for ``task/`` existing: it is
the only layer allowed to import everything, and a connector to an external system is
exactly what it wires. That is a **disclosed addition to the ADR's tree**, not a
correction to it — the ADR's layer graph is unchanged and the seam is unchanged; what
changes is that the seam now ships with both sides rather than one, which is the property
``audio/clap.py``'s own docstring argues for.

**torch and transformers are imported inside the constructor.** They are in the box's
``ss2`` env and nowhere near ``earshot/tools/mac-requirements.txt``, and every module in
this package has to stay importable on a Mac — the same discipline ``audio/clips.py``
applies to scipy. The import is at *construction*, which happens once at startup
(requirement 9), so nothing here is a lazy-loading seam inside the episode loop.

**There is no fallback encoder.** A stub that returned plausible vectors would let a run
produce anomaly classifications with no model behind them, which is the failure class
``clips.load_anomaly_clip`` refuses in the signal domain. ``load_clap_encoder`` raises.
"""

from __future__ import annotations

from typing import Any, Optional

__all__ = ["ClapEncoder", "load_clap_encoder", "CLAP_MODEL_ID"]

# provenance: box — ticket 13's known-good checkpoint, the one `env_check`'s
# `probe_clap_instantiable` constructs and the one the Gate-0b calibration that produced
# `ANOMALY_GATE_DELTA` / `ANOMALY_GATE_TAU` was measured against. Changing it invalidates
# those two constants, which is why it is named in one place.
CLAP_MODEL_ID = "laion/clap-htsat-unfused"


class ClapEncoder:
    """A connector: ``laion/clap-htsat-unfused`` behind the two calls ``clap.py`` needs.

    A class in a tree that is otherwise functions, on the rule this repo states for it —
    classes are for connectors to external systems, and this owns a loaded model with a
    lifecycle and a device.

    Both methods return a 512-d vector; ``audio/clap.py`` normalises and dots them, so
    nothing here decides anything. The waveform is the **heard clip** —
    ``clap.heard_clip_for_clap`` collapses the binaural signal the onset fired on into
    mono — rather than a re-render, because the gate and the onset must be answering
    questions about the same sound.
    """

    def __init__(self, model_id: str = CLAP_MODEL_ID, device: Optional[str] = None) -> None:
        import torch
        from transformers import ClapModel, ClapProcessor

        self._torch = torch
        self.model_id = str(model_id)
        # `cuda` when there is one, because the audio path already pays for the render
        # and 153.5 M params on the CPU would land inside the per-step budget criterion 7
        # audits. Ticket 15 measured CLAP at 0.713 GiB against 26.45 GiB of margin.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = ClapProcessor.from_pretrained(self.model_id)
        self._model = ClapModel.from_pretrained(self.model_id).to(self.device)
        self._model.eval()

    def encode_audio(self, waveform: Any, sample_rate: int) -> Any:
        """The audio embedding for one mono waveform.

        ``ClapProcessor`` resamples to the checkpoint's own 48 kHz when it has to, so the
        branch's 44100 Hz signal (ticket 22: ``sampleRate`` reads 44100.0 and ESC-50 is
        44.1 kHz, so nothing else in the tree resamples) is handed over with its rate
        rather than silently reinterpreted.
        """
        inputs = self._processor(
            audios=self._as_1d(waveform), sampling_rate=int(sample_rate), return_tensors="pt"
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            features = self._model.get_audio_features(**inputs)
        return features[0].detach().cpu().numpy()

    def encode_text(self, text: str) -> Any:
        """The text embedding for one prompt."""
        inputs = self._processor(text=[str(text)], return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            features = self._model.get_text_features(**inputs)
        return features[0].detach().cpu().numpy()

    @staticmethod
    def _as_1d(waveform: Any) -> Any:
        """Flatten to mono samples without importing numpy at module scope.

        ``heard_clip_for_clap`` already averages the ears, so this is a guard against a
        caller that passed the binaural signal straight through rather than a conversion
        anything is meant to rely on.
        """
        values = getattr(waveform, "reshape", None)
        return waveform if values is None else waveform.reshape(-1)


def load_clap_encoder(model_id: str = CLAP_MODEL_ID) -> ClapEncoder:
    """Construct the encoder, or raise with the diagnosis ticket 13 paid for.

    The failure this message exists for is not "transformers is missing": it is
    ``transformers`` gating its torch backend on ``torch >= 2.1`` and substituting a
    ``DummyObject`` that **imports cleanly and raises only when constructed**. So the
    error names the pair rather than the package, and ``env_check``'s
    ``probe_clap_instantiable`` asks the same question at startup by constructing too.
    """
    try:
        return ClapEncoder(model_id)
    except Exception as exc:
        raise RuntimeError(
            "could not construct CLAP ({}): {}. transformers disables its torch backend "
            "below torch 2.1 and substitutes a DummyObject that imports fine and raises "
            "on construction — check the torch/transformers pair, not just that both are "
            "installed (ticket 13). `python -m earshot.env_check --clap --strict` is the "
            "same probe.".format(model_id, exc)
        ) from exc
