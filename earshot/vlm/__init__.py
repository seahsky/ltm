"""The visual half of DREAM's observation, and nothing else.

`ICRA2027_Memory`'s §III.C defines the short-term memory entry as
``e_t = (z^v_t, z^u_t, p_t, a_{t-1})`` with ``z^v_t = f_v(v_t)`` (eq. 5-6). The tree had
``f_u`` — `audio/clap.py`'s `audio_embedding`, the one path a `SemanticStore` is both
written and queried through — and no ``f_v`` at all. `earshot/vlm/` was a name reserved in
ADR-0013's layer table with no package under it, and `audio/normality.py`'s `RoomLabeler`
had only `NullRoomLabeler`, which abstains. RGB reached `runner.py` and was consumed by
nothing that produced a vector.

Four of DREAM's equations are blocked on that absence: the STM entry (5), the fused
observation ``z^av_t = f_fuse(z^v_t, z^u_t)`` (7), the episodic row's audio-visual context
``h^av_i`` (15), and the retrieval query ``q_t = f_q(z^av_t, M^S_t)`` (19). So this package
is the first thing DREAM needs, not a convenience.

**A LEAF (ADR-0013: ``"vlm": ()``).** It imports nothing intra-package — not even
`earshot.types` — so it deals in numpy arrays and the encoder protocol alone. The model's
lifecycle, its device and where its checkpoint is staged belong to `task/models.py`, the
wiring layer, exactly as `ClapEncoder` does. That split is what keeps this importable on a
Mac with no torch.
"""
