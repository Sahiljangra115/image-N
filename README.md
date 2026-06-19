---
title: image-N
emoji: 🖼️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# image-N 🖼️

**A multimodal Retrieval-Augmented Generation system, built from scratch in plain Python.** No LangChain, no framework glue. Every stage is readable top to bottom, every parameter is defensible, and the whole thing runs locally with no API keys.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/vector%20search-FAISS-orange.svg)](https://github.com/facebookresearch/faiss)
[![Ollama](https://img.shields.io/badge/LLM-Ollama%20(local)-black.svg)](https://ollama.com/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](https://docs.pytest.org/)

---

## What it is

A private-document question-answering system. Point it at a knowledge base (the demo uses a bakery's recipes, allergy notes, and supplier records), and it answers questions grounded in those documents without retraining any model. It also indexes images and answers questions about them, in the same project.

The implementation is the point. The core text pipeline is roughly 300 lines of readable Python, so every retrieval decision, every `k`, and every threshold is something you can explain and defend.

## Features

- **Text RAG** over `.txt`, `.md`, `.pdf`, `.docx`, and `.xlsx` via an extension dispatch table.
- **Two-stage retrieval.** FAISS casts a wide net (`fetch_k=10`), then a cross-encoder reranks to the top `k=3`. Recall from the vector search, precision from the reranker, which is the production-standard pattern.
- **Grounded generation.** The prompt instructs the model to answer using only the retrieved context, which suppresses hallucination.
- **Multimodal.** CLIP ViT-L/14 embeds text and images into a shared space, a vision LLM answers questions about a retrieved image, and Stable Diffusion Turbo runs img2img generation conditioned on it.
- **Persistence.** Chunks and vectors are cached to disk, so re-runs skip the expensive embedding step.
- **Memory-aware.** The multimodal stack fits inside a 4 GB GPU budget via sequential CPU offload and VAE slicing.
- **Four entry points.** CLI REPL, FastAPI REST endpoint with a single-page chat UI, a Gradio app for Hugging Face Spaces, and Docker Compose for a self-contained stack.

## Architecture

Text pipeline (`rag.py`):

```
Documents -> Load (dispatch by extension) -> Chunk (180 words, 20 overlap)
          -> Embed (BGE-small, normalized) -> Persist (chunks.json + vectors.npy)

Query -> Embed (BGE + search prefix) -> FAISS retrieve (fetch_k=10)
      -> Cross-encoder rerank (k=3) -> Build grounded prompt -> Generate
```

Multimodal pipeline (`image_rag.py`, `multimodal.py`):

```
Images -> Embed (CLIP ViT-L/14, normalized) -> Persist (vectors + paths)

Query -> Embed query (CLIP text encoder, same space) -> Retrieve top image (FAISS)
      -> Vision LLM grounded answer -> (optional) free CLIP -> SD-Turbo img2img
```

The text index (BGE, 384-dim) and image index (CLIP, 768-dim) live in separate spaces and are queried independently, then merged as "best of each." That is how you handle a modality mismatch honestly, instead of pretending one embedder covers both.

## Models

| Stage | Model | Notes |
|-------|-------|-------|
| Text embedding | `BAAI/bge-small-en-v1.5` | 384-dim, normalized so dot product equals cosine |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | scores query + chunk pairs |
| Image embedding | `CLIP ViT-L/14` | 768-dim shared text/image space |
| Vision LLM | `HuggingFaceTB/SmolVLM-500M-Instruct` | ~1 GB VRAM |
| Image generation | `stabilityai/sd-turbo` | 1 to 4 steps, img2img |
| Generation LLM | `qwen3.5:4b` via Ollama | local, no API key, configurable |

## Quickstart

```bash
pip install -r requirements.txt
ollama pull qwen3.5:4b
ollama serve            # daemon required for all modes
```

CLI:

```bash
python rag.py check                    # retrieval self-test, no LLM
python rag.py chat                     # interactive REPL
python rag.py chat data/folder         # REPL over a folder of docs
python eval.py                         # text retrieval benchmark
python multimodal.py "your question"   # text + image RAG
python multimodal.py "q" --generate "edit prompt"   # img2img generation
pytest test_rag.py -v
```

Web (FastAPI):

```bash
uvicorn app:app --reload               # http://127.0.0.1:8000
```

Docker:

```bash
docker compose up -d                   # bundles Ollama + app
docker compose exec ollama ollama pull qwen3.5:4b
```

## Evaluation

Retrieval is measured, not asserted by vibe. The image evaluation uses each caption's own image as ground truth, so no hand-labeling is required.

Image retrieval (`image_eval.py`): upgrading the embedder from CLIP ViT-B/32 to ViT-L/14 lifted recall consistently. The cost is paid once, at index time.

| recall@k | ViT-B/32 | ViT-L/14 | delta |
|----------|----------|----------|-------|
| @1 | 0.561 | 0.606 | +4.5% |
| @3 | 0.753 | 0.787 | +3.4% |
| @5 | 0.809 | 0.851 | +4.2% |
| @10 | 0.887 | 0.921 | +3.4% |

Text retrieval (`eval.py`): hit-rate@3 on a labeled query set, with a regression assertion to catch drops.

## Design notes

- **Normalized embeddings** mean a dot product equals cosine similarity, so `IndexFlatIP` is correct and fast.
- **Index auto-swap.** Below ~10k vectors the system uses exact flat search; above that it switches to HNSW for speed.
- **Lazy model loading** keeps unused models off the GPU until needed.
- **Source tags** are stored per chunk so the UI can show provenance.

## Project layout

```
.
├── rag.py            # core text RAG pipeline
├── image_rag.py      # CLIP image indexing + retrieval
├── multimodal.py     # unified text + image RAG
├── app.py            # FastAPI server + single-page UI
├── app_gradio.py     # Hugging Face Spaces app
├── eval.py           # text retrieval benchmark
├── image_eval.py     # recall@k for image retrieval
├── test_rag.py       # unit + retrieval tests
├── docker-compose.yml
└── static/           # chat UI
```

## Limitations

- Generation quality is bounded by the local model you pull.
- The image and text indexes are separate, so cross-modal ranking is "best of each," not a single fused score.
- Tests cover retrieval and indexing math; LLM generation is skipped in CI because it depends on an external daemon.

## License

MIT.
