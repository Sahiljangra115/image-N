"""
FastAPI server for the RAG pipeline.

The index is built ONCE at startup (state held in app.state), then every
/ask request reuses it. This is the web version of the REPL: load once,
query many times.

Run:  uvicorn app:app --reload
Then open http://127.0.0.1:8000
"""

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import rag

# Where the documents live. Override with DATA_PATH env var (Docker, deploy).
DATA_PATH = os.environ.get("DATA_PATH", "data/maya_binder.txt")


@asynccontextmanager
async def lifespan(app: FastAPI):
    chunks, index = rag.build_index(DATA_PATH)
    app.state.chunks = chunks
    app.state.index = index
    print(f"Index ready: {len(chunks)} chunks from {DATA_PATH}")
    yield


app = FastAPI(title="Maya RAG", lifespan=lifespan)


class Question(BaseModel):
    query: str


class Answer(BaseModel):
    answer: str
    sources: list[str]


@app.post("/ask", response_model=Answer)
def ask(q: Question) -> Answer:
    """Retrieve, rerank, generate. Returns the answer plus the source chunks
    so the UI can show what grounded the answer."""
    top = rag.search(q.query, app.state.chunks, app.state.index, k=3)
    prompt = rag.build_prompt(q.query, top)
    answer = rag.generate(prompt)
    return Answer(answer=answer, sources=top)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "chunks": len(app.state.chunks)}


# serve the single-page UI at the root
@app.get("/")
def home() -> FileResponse:
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
