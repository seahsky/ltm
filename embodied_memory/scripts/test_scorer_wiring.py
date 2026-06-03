"""
Habitat/torch-free unit tests for the trained-scorer (R head) wiring.

Covers two pure seams of the "train the LTM importance head" work:

  D1  ``dialogue_memory.train_scorer._infer_scorer_dims`` — recover
      (embed_dim, hidden_dim) from a checkpoint state_dict by reading the
      first Linear's weight shape (checkpoints don't store the dims).

  D2  ``dialogue_memory.consolidation.DialogueConsolidation`` with an injected
      ``relevance_scorer`` — ``_compute_relevance`` uses the trained scorer
      when one is set AND the segment carries a matching-dim embedding, else
      falls back to the existing length/keyword heuristic. The dialogue path
      (no scorer) must be byte-for-byte unchanged.

We stub ``torch`` (so train_scorer imports without the CUDA stack) and ``faiss``
(so consolidation imports without the native lib). Neither stub is exercised by
the code under test — the dim inference reads ``.shape`` only, and
``_compute_relevance`` is pure numpy + an injected callable.

Invoke with::

    python embodied_memory/scripts/test_scorer_wiring.py
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
# stubs: torch (for train_scorer) and faiss (for consolidation -> ltm)
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

    # The dialogue_memory package __init__ eagerly imports modules that pull
    # transformers / sentence-transformers / sklearn. None are touched by the
    # code under test; stub them so the import resolves.
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

from dialogue_memory.train_scorer import _infer_scorer_dims  # noqa: E402
from dialogue_memory.consolidation import (  # noqa: E402
    DialogueConsolidation,
    DialogueSegment,
)


def _approx(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


# ----------------------------------------------------------------------
# D1 — _infer_scorer_dims
# ----------------------------------------------------------------------
def case_infer_dims_from_first_linear():
    # ImportanceScorer.net[0] is Linear(embed_dim, hidden_dim) whose weight is
    # stored [hidden_dim, embed_dim]. 768-d SBERT, 512 hidden is the default.
    sd = {
        "net.0.weight": np.zeros((512, 768)),
        "net.0.bias": np.zeros((512,)),
        "net.3.weight": np.zeros((256, 512)),
        "net.6.weight": np.zeros((1, 256)),
    }
    embed_dim, hidden_dim = _infer_scorer_dims(sd)
    assert embed_dim == 768, embed_dim
    assert hidden_dim == 512, hidden_dim


def case_infer_dims_nondefault_shapes():
    sd = {"net.0.weight": np.zeros((128, 384))}
    embed_dim, hidden_dim = _infer_scorer_dims(sd)
    assert embed_dim == 384, embed_dim
    assert hidden_dim == 128, hidden_dim


def case_sbert_training_encoder_normalizes():
    # The embodied scorer MUST train in the same L2-normalized SBERT space the
    # bridge scores in. Stub the SBERT model with a non-unit vector and assert
    # the "sbert" training encoder returns a unit vector.
    import dialogue_memory.encoder as enc_mod
    from dialogue_memory.train_scorer import _build_text_encoder

    class _FakeSBERT:
        embed_dim = 2

        def encode(self, text):
            return np.array([3.0, 4.0], dtype=np.float32)  # norm 5

    saved = getattr(enc_mod, "SentenceTransformerEncoder", None)
    enc_mod.SentenceTransformerEncoder = _FakeSBERT
    try:
        adapter = _build_text_encoder("sbert")
        v = adapter.encode("anything")
    finally:
        if saved is not None:
            enc_mod.SentenceTransformerEncoder = saved
    assert _approx(float(np.linalg.norm(v)), 1.0), np.linalg.norm(v)
    assert adapter.embed_dim == 2, adapter.embed_dim


def case_infer_dims_missing_key_raises():
    try:
        _infer_scorer_dims({"unexpected": np.zeros((1, 1))})
    except (KeyError, ValueError):
        return
    raise AssertionError("expected KeyError/ValueError when net.0.weight absent")


# ----------------------------------------------------------------------
# D2 — DialogueConsolidation relevance_scorer wiring
# ----------------------------------------------------------------------
def _seg(emb, success=None):
    meta = {}
    if success is not None:
        meta["episode_success"] = success
    return DialogueSegment(
        session_id=0,
        dialogue_id=0,
        speaker="agent",
        utterance="there is a chair next to a table",
        response=None,
        embedding=emb,
        metadata=meta,
    )


def case_relevance_uses_scorer_when_set():
    # scorer returns a fixed 0.9; with no episode_success meta the multiplier is
    # 1.0, so R must equal the scorer output, NOT the heuristic.
    cons = DialogueConsolidation(ltm=None, relevance_scorer=lambda e: 0.9)
    emb = np.ones(768, dtype=np.float32)
    r = cons._compute_relevance(_seg(emb))
    assert _approx(r, 0.9), r


def case_relevance_heuristic_when_no_scorer():
    # Default (None) path must match the pre-existing heuristic exactly.
    cons_none = DialogueConsolidation(ltm=None)
    emb = np.ones(768, dtype=np.float32)
    seg = _seg(emb)
    r_none = cons_none._compute_relevance(seg)
    # Recompute the heuristic independently.
    text = "there is a chair next to a table "
    length_score = min(len(text) / 200.0, 1.0)
    personal = ["like", "love", "hate", "favorite", "hobby", "enjoy",
                "my", "i am", "i have", "i want", "i need",
                "喜欢", "爱", "讨厌", "爱好", "我的"]
    personal_score = min(sum(1 for kw in personal if kw.lower() in text.lower()) * 0.2, 1.0)
    qkw = ["what", "how", "why", "when", "where", "do you", "are you", "?"]
    question_score = min(sum(1 for kw in qkw if kw.lower() in text.lower()) * 0.1, 0.5)
    expected = 0.4 * length_score + 0.4 * personal_score + 0.2 * question_score
    assert _approx(r_none, expected), (r_none, expected)


def case_relevance_falls_back_when_no_embedding():
    # scorer set but segment has no embedding -> heuristic, scorer never called.
    called = {"n": 0}

    def _scorer(e):
        called["n"] += 1
        return 0.9

    cons = DialogueConsolidation(ltm=None, relevance_scorer=_scorer)
    seg = _seg(None)
    r = cons._compute_relevance(seg)
    assert called["n"] == 0, "scorer must not be called without an embedding"
    assert not _approx(r, 0.9), r


def case_relevance_falls_back_on_dim_mismatch():
    # scorer trained for 768-d; segment carries a 512-d embedding -> heuristic.
    def _scorer(e):
        if e.shape[-1] != 768:
            raise ValueError("dim mismatch must be guarded before calling scorer")
        return 0.9

    cons = DialogueConsolidation(
        ltm=None, relevance_scorer=_scorer, scorer_embed_dim=768
    )
    seg = _seg(np.ones(512, dtype=np.float32))
    r = cons._compute_relevance(seg)  # must not raise
    assert not _approx(r, 0.9), r


def case_relevance_scorer_failed_episode_multiplier():
    # The failed-episode multiplier still applies on top of the scorer base.
    cons = DialogueConsolidation(ltm=None, relevance_scorer=lambda e: 0.8)
    emb = np.ones(768, dtype=np.float32)
    r = cons._compute_relevance(_seg(emb, success=False))
    assert _approx(r, 0.8 * DialogueConsolidation.FAILED_EPISODE_RELEVANCE_WEIGHT), r


# ----------------------------------------------------------------------
# D3 / D4 — source-scan: scorer_ckpt threaded bridge -> CLI (torch-free)
# ----------------------------------------------------------------------
def case_bridge_threads_scorer_ckpt():
    src = (REPO / "embodied_memory" / "memory_bridge.py").read_text()
    assert "scorer_ckpt: Optional[str] = None" in src, "bridge param missing"
    assert "from dialogue_memory.train_scorer import load_scorer" in src
    assert "relevance_scorer=relevance_scorer" in src, "scorer not passed to consolidator"
    assert "scorer_embed_dim=scorer_embed_dim" in src, "dim guard not passed"
    assert ".compute_importance" in src, "scorer inference hook not used"
    # Loud failure on dim mismatch (else a silent heuristic fallback would read
    # as a false null result).
    assert "would silently" in src, "missing dim-mismatch guard"


def case_cli_exposes_scorer_ckpt():
    src = (REPO / "embodied_memory" / "run_hm3d_pol.py").read_text()
    assert '"--scorer-ckpt"' in src, "CLI flag missing"
    assert "scorer_ckpt=args.scorer_ckpt" in src, "flag not passed to bridge"
    assert '"scorer_ckpt": args.scorer_ckpt' in src, "flag not recorded in run_config"


def main():
    cases = [
        case_infer_dims_from_first_linear,
        case_infer_dims_nondefault_shapes,
        case_sbert_training_encoder_normalizes,
        case_infer_dims_missing_key_raises,
        case_relevance_uses_scorer_when_set,
        case_relevance_heuristic_when_no_scorer,
        case_relevance_falls_back_when_no_embedding,
        case_relevance_falls_back_on_dim_mismatch,
        case_relevance_scorer_failed_episode_multiplier,
        case_bridge_threads_scorer_ckpt,
        case_cli_exposes_scorer_ckpt,
    ]
    failed = 0
    for c in cases:
        try:
            c()
            print(f"  ok  {c.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {c.__name__}: {e}")
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
