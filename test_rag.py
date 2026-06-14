"""
pytest suite for the RAG pipeline.

Covers the logic that can regress silently: chunking math, loader dispatch,
index save/load roundtrip, and end-to-end retrieval quality. generate() is
NOT tested here because it calls an external LLM (Ollama); that is an
integration concern, not a unit test.

Note: importing rag loads the embedder (and, for search tests, the reranker).
The first run is slow while models download. Run: pytest test_rag.py -v
"""

import json
import numpy as np
import pytest

import rag


# ---------- chunking (pure logic, no models) ----------

def test_chunk_overlap_slides_by_size_minus_overlap():
    text = " ".join(str(i) for i in range(100))  # 100 distinct words
    chunks = rag.chunk_text(text, size=10, overlap=2)
    # step = 8, so chunk 0 starts at word 0, chunk 1 at word 8
    assert chunks[0].split()[0] == "0"
    assert chunks[1].split()[0] == "8"


def test_chunk_overlap_shares_words_at_boundary():
    text = " ".join(str(i) for i in range(100))
    chunks = rag.chunk_text(text, size=10, overlap=2)
    # last 2 words of chunk 0 should reappear as first 2 of chunk 1
    assert chunks[0].split()[-2:] == chunks[1].split()[:2]


def test_chunk_short_text_one_chunk():
    chunks = rag.chunk_text("only a few words here", size=180, overlap=20)
    assert len(chunks) == 1


# ---------- loader dispatch (pure logic) ----------

def test_unsupported_extension_raises():
    with pytest.raises(ValueError):
        rag.load_documents("notes.xyz")


def test_txt_loader_roundtrip(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("hello maya", encoding="utf-8")
    assert rag.load_documents(str(f)) == "hello maya"


# ---------- index persistence (no models) ----------

def test_save_load_index_roundtrip(tmp_path):
    chunks = ["chunk a", "chunk b"]
    vectors = np.array([[0.1, 0.2], [0.3, 0.4]], dtype="float32")
    folder = str(tmp_path / "idx")
    rag.save_index(chunks, vectors, folder=folder)
    loaded_chunks, loaded_vectors = rag.load_index(folder=folder)
    assert loaded_chunks == chunks
    np.testing.assert_allclose(loaded_vectors, vectors)


def test_load_index_missing_returns_none(tmp_path):
    assert rag.load_index(folder=str(tmp_path / "nope")) is None


# ---------- faiss + retrieval (loads embedder) ----------

def test_build_faiss_dimension_matches_vectors():
    vecs = rag.embed(["a cake with almonds", "the shop opening hours"])
    index = rag.build_faiss(vecs)
    assert index.d == vecs.shape[1]      # index dim == embedding dim
    assert index.ntotal == 2             # both vectors added


def test_retrieve_returns_k_chunks():
    chunks = ["almond cake has nuts", "vanilla sponge is nut free", "shop closed monday"]
    index = rag.build_faiss(rag.embed(chunks))
    top = rag.retrieve("nut allergy safe cake", chunks, index, k=2)
    assert len(top) == 2


def test_retrieve_finds_relevant_chunk():
    chunks = ["almond cake has nuts", "vanilla sponge is nut free", "shop closed monday"]
    index = rag.build_faiss(rag.embed(chunks))
    top = rag.retrieve("which cake is safe for a nut allergy", chunks, index, k=1)
    assert "vanilla" in top[0].lower() or "nut free" in top[0].lower()


# ---------- two-stage search (loads embedder + reranker) ----------

def test_search_reranks_to_k():
    chunks = [f"filler chunk number {i}" for i in range(20)]
    chunks.append("vanilla sponge is safe for a nut allergy")
    index = rag.build_faiss(rag.embed(chunks))
    top = rag.search("nut allergy safe cake", chunks, index, k=3, fetch_k=10)
    assert len(top) == 3
    # the clearly-relevant chunk should survive reranking into the top 3
    assert any("vanilla" in c.lower() for c in top)
