"""
Habitat/torch-free unit tests for the trained-predictor (U head) wiring.

The consolidator's importance is I = αR + βU + γN. Run 13/14 closed the R
lever (trained head reaches but does not beat the heuristic). U (β=0.3) is
the proposal's *surprise* term — by default a weak heuristic (deviation of
the R score from its running mean). ``dialogue_memory.train_predictor``
already trains the real thing on embodied data (next-caption forward model:
history embedding → predicted next embedding; surprise = prediction error)
but nothing ever loaded it at inference. This suite covers the wiring,
mirroring ``test_scorer_wiring.py``:

  P1  ``train_predictor._infer_predictor_dims`` — recover (embed_dim,
      hidden_dim) from a PredictionMLP state_dict (net.0 is
      Linear(embed_dim, hidden_dim); checkpoints don't store the dims).

  P2  ``train_predictor._cosine_surprise`` — the bounded inference-time
      surprise (1 − cos(predicted, actual)) / 2 ∈ [0, 1]. Raw MSE on
      normalized SBERT embeddings is ~1e-3-scale and would vanish inside
      I = αR + βU + γN; cosine surprise is scale-free and bounded like the
      heuristic U it replaces.

  P3  ``DialogueConsolidation`` with an injected ``utility_predictor`` —
      ``_compute_uniqueness`` uses the trained head when one is set AND the
      segment carries a matching-dim embedding AND there is at least one
      PRIOR utterance this session (the predictor needs a history; the
      history string is the join of the last ``predictor_history_len``
      utterances, exactly how training pairs are built in
      ``EmbodiedPredictionDataset``). The history RESETS per
      ``consolidate_session`` call (training pairs never cross episodes).
      The dialogue path (no predictor) must be byte-for-byte unchanged.

  P4/P5  source-scan: ``predictor_ckpt`` threaded bridge → CLI with a LOUD
      dim-mismatch raise (a silent heuristic fallback would read as a false
      null result — the scorer arc's lesson).

Invoke with::

    python embodied_memory/scripts/test_predictor_wiring.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ----------------------------------------------------------------------
# stubs: torch (for train_predictor) and faiss (for consolidation -> ltm)
# (verbatim harness of test_scorer_wiring.py)
# ----------------------------------------------------------------------
def _install_stubs():
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        nn = types.ModuleType("torch.nn")

        class _Module:  # minimal nn.Module stand-in
            def __init__(self, *a, **k):
                pass

            def __call__(self, *a, **k):
                raise RuntimeError("stub torch.nn.Module is not callable")

        nn.Module = _Module
        nn.Linear = lambda *a, **k: _Module()
        nn.ReLU = lambda *a, **k: _Module()
        nn.Dropout = lambda *a, **k: _Module()
        nn.Sigmoid = lambda *a, **k: _Module()
        nn.Sequential = lambda *a, **k: _Module()
        nn.MSELoss = lambda *a, **k: _Module()
        nn.BCELoss = lambda *a, **k: _Module()
        torch.nn = nn
        utils = types.ModuleType("torch.utils")
        data = types.ModuleType("torch.utils.data")
        data.Dataset = object
        data.DataLoader = object
        utils.data = data
        torch.utils = utils
        optim = types.ModuleType("torch.optim")
        optim.Adam = lambda *a, **k: None
        torch.optim = optim
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch.Tensor = object
        torch.FloatTensor = lambda *a, **k: None
        torch.save = lambda *a, **k: None
        torch.load = lambda *a, **k: {}
        torch.no_grad = lambda: types.SimpleNamespace(
            __enter__=lambda *a: None, __exit__=lambda *a: None
        )
        sys.modules["torch"] = torch
        sys.modules["torch.nn"] = nn
        sys.modules["torch.utils"] = utils
        sys.modules["torch.utils.data"] = data
        sys.modules["torch.optim"] = optim

    if "faiss" not in sys.modules:
        faiss = types.ModuleType("faiss")
        faiss.IndexFlatL2 = lambda *a, **k: None
        faiss.IndexFlatIP = lambda *a, **k: None
        faiss.normalize_L2 = lambda *a, **k: None
        sys.modules["faiss"] = faiss

    for name in ("transformers", "sentence_transformers", "sklearn",
                 "sklearn.cluster", "sklearn.metrics"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
    sys.modules["transformers"].GPT2LMHeadModel = object
    sys.modules["transformers"].GPT2Tokenizer = object
    sys.modules["transformers"].AutoModel = object
    sys.modules["transformers"].AutoTokenizer = object
    sys.modules["sentence_transformers"].SentenceTransformer = object
    sys.modules["sklearn.cluster"].KMeans = object
    sys.modules["sklearn.cluster"].DBSCAN = object
    sys.modules["sklearn.metrics"].silhouette_score = lambda *a, **k: 0.0


_install_stubs()

from dialogue_memory.train_predictor import (  # noqa: E402
    _cosine_surprise,
    _infer_predictor_dims,
)
from dialogue_memory.consolidation import (  # noqa: E402
    DialogueConsolidation,
    DialogueSegment,
)


def _approx(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


# ----------------------------------------------------------------------
# P1 — _infer_predictor_dims
# ----------------------------------------------------------------------
def case_predictor_infer_dims_from_first_linear():
    sd = {"net.0.weight": np.zeros((1024, 512), dtype=np.float32)}
    embed_dim, hidden_dim = _infer_predictor_dims(sd)
    assert embed_dim == 512 and hidden_dim == 1024, (embed_dim, hidden_dim)


def case_predictor_infer_dims_nondefault_shapes():
    sd = {"net.0.weight": np.zeros((256, 384), dtype=np.float32)}
    embed_dim, hidden_dim = _infer_predictor_dims(sd)
    assert embed_dim == 384 and hidden_dim == 256, (embed_dim, hidden_dim)


def case_predictor_infer_dims_missing_key_raises():
    try:
        _infer_predictor_dims({"other.weight": np.zeros((2, 2))})
    except (KeyError, ValueError):
        return
    raise AssertionError("expected KeyError/ValueError when net.0.weight absent")


# ----------------------------------------------------------------------
# P2 — _cosine_surprise (pure, bounded [0,1])
# ----------------------------------------------------------------------
def case_cosine_surprise_anchors():
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert _approx(_cosine_surprise(v, v), 0.0)          # perfect prediction
    assert _approx(_cosine_surprise(v, -v), 1.0)         # maximally wrong
    w = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert _approx(_cosine_surprise(v, w), 0.5)          # orthogonal


def case_cosine_surprise_scale_invariant():
    # cosine, not MSE: a scaled prediction of the right direction is not a
    # surprise (raw MSE on normalized SBERT embeddings is ~1e-3-scale and
    # would vanish inside I = αR + βU + γN).
    v = np.array([0.3, 0.4, 0.5], dtype=np.float32)
    assert _approx(_cosine_surprise(2.0 * v, v), 0.0)


def case_cosine_surprise_zero_norm_guard():
    # Degenerate vectors -> neutral 0.5 (matches the heuristic's data-starved
    # default), never a ZeroDivision.
    z = np.zeros(3, dtype=np.float32)
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert _approx(_cosine_surprise(z, v), 0.5)
    assert _approx(_cosine_surprise(v, z), 0.5)


# ----------------------------------------------------------------------
# P3 — DialogueConsolidation utility_predictor wiring
# ----------------------------------------------------------------------
class _StubLayer:
    def get_all_embeddings(self):
        return None


class _StubLTM:
    def __init__(self):
        self.layers = {"fine": _StubLayer(), "mid": _StubLayer(),
                       "coarse": _StubLayer()}
        self.n = 0

    def insert(self, level, embedding, content, metadata):
        self.n += 1
        return f"{level}-{self.n}"


def _seg_u(utt, emb):
    return DialogueSegment(
        session_id=0, dialogue_id=0, speaker="agent",
        utterance=utt, response=None, embedding=emb, metadata={},
    )


def _recording_encoder(record):
    def enc(text):
        record.append(text)
        return np.ones(768, dtype=np.float32)
    return enc


def case_uniqueness_uses_predictor_when_history():
    # First segment of a session has NO prior utterance -> heuristic;
    # from the second on, U comes from the trained head.
    cons = DialogueConsolidation(
        ltm=_StubLTM(), utility_predictor=lambda h, e: 0.9,
        predictor_embed_dim=768,
    )
    enc = _recording_encoder([])
    emb = np.ones(768, dtype=np.float32)
    _, b1 = cons.compute_importance(_seg_u("a chair by a table", emb), enc)
    assert not _approx(b1["uniqueness"], 0.9), b1   # no history yet
    _, b2 = cons.compute_importance(_seg_u("a bed in a room", emb), enc)
    assert _approx(b2["uniqueness"], 0.9), b2


def case_uniqueness_history_is_prior_utterances_capped():
    # The history string fed to the encoder is the join of the last
    # predictor_history_len PRIOR utterances — exactly how training pairs
    # are built (EmbodiedPredictionDataset: " ".join of the last H captions
    # within the episode, target = the next caption).
    histories = []

    def pred(history_emb, emb):
        return 0.7

    record = []
    cons = DialogueConsolidation(
        ltm=_StubLTM(), utility_predictor=pred,
        predictor_embed_dim=768, predictor_history_len=5,
    )
    enc = _recording_encoder(record)
    emb = np.ones(768, dtype=np.float32)
    for i in range(1, 8):
        cons.compute_importance(_seg_u(f"u{i}", emb), enc)
    # On segment 7 the history must be the PRIOR 5 utterances u2..u6.
    assert record[-1] == "u2 u3 u4 u5 u6", record[-1]
    del histories


def case_uniqueness_falls_back_on_dim_mismatch():
    # predictor trained for 768-d; segment carries 512-d -> heuristic,
    # predictor never called.
    called = {"n": 0}

    def pred(h, e):
        called["n"] += 1
        return 0.9

    cons = DialogueConsolidation(
        ltm=_StubLTM(), utility_predictor=pred, predictor_embed_dim=768,
    )
    enc = _recording_encoder([])
    emb = np.ones(512, dtype=np.float32)
    cons.compute_importance(_seg_u("first", emb), enc)
    _, b2 = cons.compute_importance(_seg_u("second", emb), enc)
    assert called["n"] == 0, "predictor must not be called on a dim mismatch"
    assert not _approx(b2["uniqueness"], 0.9), b2


def case_uniqueness_heuristic_byte_identity_when_no_predictor():
    # Default (None) path must match the pre-existing heuristic exactly:
    # U_i = min(|R_i - mean(R_1..R_{i-1})| * 2, 1).
    cons = DialogueConsolidation(ltm=_StubLTM())
    enc = _recording_encoder([])
    emb = np.ones(768, dtype=np.float32)
    cons.compute_importance(_seg_u("i love my favorite chair", emb), enc)
    _, b2 = cons.compute_importance(_seg_u("a wall", emb), enc)
    r1, r2 = cons.info_richness_history[0], cons.info_richness_history[1]
    expected = min(abs(r2 - r1) * 2.0, 1.0)
    assert _approx(b2["uniqueness"], expected), (b2["uniqueness"], expected)


def case_history_resets_per_consolidate_session():
    # Training pairs never cross episodes; the inference history must not
    # either. The FIRST segment of session 2 sees an empty history even
    # though session 1 left utterances behind.
    calls = {"n": 0}

    def pred(h, e):
        calls["n"] += 1
        return 0.9

    cons = DialogueConsolidation(
        ltm=_StubLTM(), utility_predictor=pred, predictor_embed_dim=768,
    )
    enc = _recording_encoder([])
    emb = np.ones(768, dtype=np.float32)
    cons.consolidate_session(
        [_seg_u("s1-a", emb), _seg_u("s1-b", emb)], enc, dialogue_id=0)
    assert calls["n"] == 1, calls  # only s1-b had a prior utterance
    cons.consolidate_session([_seg_u("s2-a", emb)], enc, dialogue_id=1)
    assert calls["n"] == 1, calls  # s2-a saw a RESET (empty) history


# ----------------------------------------------------------------------
# P4 / P5 — source-scan: predictor_ckpt threaded bridge -> CLI (torch-free)
# ----------------------------------------------------------------------
def case_bridge_threads_predictor_ckpt():
    src = (REPO / "embodied_memory" / "memory_bridge.py").read_text()
    assert "predictor_ckpt: Optional[str] = None" in src, "bridge param missing"
    assert "from dialogue_memory.train_predictor import load_predictor" in src
    assert "utility_predictor=utility_predictor" in src, \
        "predictor not passed to consolidator"
    assert "predictor_embed_dim=predictor_embed_dim" in src, \
        "dim guard not passed"
    assert ".compute_surprise_norm" in src, "predictor inference hook not used"
    # Loud failure on dim mismatch (else a silent heuristic fallback would
    # read as a false null result — the scorer arc's lesson).
    assert "predictor_ckpt embed_dim" in src, "missing dim-mismatch guard"


def case_cli_exposes_predictor_ckpt():
    src = (REPO / "embodied_memory" / "run_hm3d_pol.py").read_text()
    assert '"--predictor-ckpt"' in src, "CLI flag missing"
    assert "predictor_ckpt=args.predictor_ckpt" in src, "flag not passed to bridge"
    assert '"predictor_ckpt": args.predictor_ckpt' in src, \
        "flag not recorded in run_config"


def main():
    cases = [
        case_predictor_infer_dims_from_first_linear,
        case_predictor_infer_dims_nondefault_shapes,
        case_predictor_infer_dims_missing_key_raises,
        case_cosine_surprise_anchors,
        case_cosine_surprise_scale_invariant,
        case_cosine_surprise_zero_norm_guard,
        case_uniqueness_uses_predictor_when_history,
        case_uniqueness_history_is_prior_utterances_capped,
        case_uniqueness_falls_back_on_dim_mismatch,
        case_uniqueness_heuristic_byte_identity_when_no_predictor,
        case_history_resets_per_consolidate_session,
        case_bridge_threads_predictor_ckpt,
        case_cli_exposes_predictor_ckpt,
    ]
    failed = 0
    for c in cases:
        try:
            c()
            print(f"  ok  {c.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {c.__name__}: {type(e).__name__}: {e}")
    if failed:
        print(f"{failed}/{len(cases)} cases FAILED")
        return 1
    print(f"All {len(cases)} cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
