"""
Unified RAG over BOTH text documents and images.

The constraint: BGE (text, 384-dim) and CLIP (image, 768-dim) live in DIFFERENT
vector spaces. A BGE vector and a CLIP vector are not comparable, so we cannot
put them in one FAISS index and sort by score. Instead we keep two indexes,
query each with its own embedder, and merge "best of each" at the top.

  text  -> rag.py        (BGE, good at long documents)
  image -> image_rag.py  (CLIP, good at pictures)

Then one vision-capable LLM call sees the text chunks AND the best image and
writes a single grounded answer.

Run:
  python multimodal.py check                         (retrieval only, no LLM)
  python multimodal.py ask "your question"

ponytail: scores across the two indexes are on different scales, so we do NOT
globally rank a mixed list. We return top-n text + top-1 image. Add a router or
score-normalization only if "always show both" proves wrong.
"""

# rag and image_rag are imported lazily inside each function: importing them
# loads their embedders (BGE / CLIP-L) on the GPU. On 4GB we only want to pay
# for the models a given mode actually uses (-t needs BGE, -i needs CLIP).

TEXT_PATH = "data/maya_binder.txt"   # or a folder of docs
IMAGE_FOLDER = "data/images"


def retrieve_both(query: str, n_text: int = 3, n_img: int = 1):
    """Query both indexes independently. Returns (text_chunks, image_paths).
    Each side uses its own embedder; no cross-space comparison happens."""
    import rag
    import image_rag
    chunks, tindex = rag.build_index(TEXT_PATH)
    text_hits = rag.search(query, chunks, tindex, k=n_text)   # BGE + reranker

    paths, iindex = image_rag.build_image_index(IMAGE_FOLDER)
    image_hits = image_rag.retrieve_images(query, paths, iindex, k=n_img)  # CLIP

    return text_hits, image_hits


def generate_multimodal(query: str, text_chunks: list[str], image_paths: list[str]) -> str:
    """One SmolVLM call grounded in BOTH the text chunks and the image.
    Text goes in the instruction, image goes in as a vision token."""
    import image_rag

    context = "\n\n".join(text_chunks) if text_chunks else "(no text context)"
    instruction = (
        "Answer using ONLY the text context below and the attached image. "
        "If neither contains the answer, say you do not know.\n\n"
        f"Text context:\n{context}\n\n"
        f"Question: {query}"
    )
    return image_rag.vlm_answer(image_paths, instruction)


def answer_multimodal(query: str):
    """Full pipeline: retrieve from both, answer with one grounded call.
    Returns (answer, text_chunks, image_paths) for provenance."""
    text_hits, image_hits = retrieve_both(query)
    answer = generate_multimodal(query, text_hits, image_hits)
    return answer, text_hits, image_hits


def answer_text(query: str):
    """Text-only: retrieve doc chunks, answer with the plain text LLM.
    Returns (answer, text_chunks)."""
    import rag
    chunks, tindex = rag.build_index(TEXT_PATH)
    top = rag.search(query, chunks, tindex, k=3)
    return rag.generate(rag.build_prompt(query, top)), top


def _selfcheck() -> None:
    """Retrieval only, no LLM. Confirms both indexes answer the same query."""
    text_hits, image_hits = retrieve_both("a child playing")
    assert text_hits, "text index returned nothing"
    assert image_hits, "image index returned nothing"
    print(f"text top: {text_hits[0][:80]}")
    print(f"image top: {image_hits[0]}")
    print("OK. Both indexes responded.")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Multimodal RAG over text docs + images.")
    p.add_argument("query", nargs="?", help="the question / search query")
    p.add_argument("-t", action="store_true", help="text documents only")
    p.add_argument("-i", action="store_true", help="images only")
    p.add_argument("--generate", metavar="PROMPT",
                   help="find the image for QUERY, then generate an edited image from PROMPT")
    p.add_argument("--check", action="store_true", help="retrieval self-test, no LLM")
    args = p.parse_args()

    if args.check:
        _selfcheck()
    elif not args.query:
        p.error("a query is required (unless --check)")
    elif args.generate:
        import image_rag
        out, src = image_rag.retrieve_and_make(IMAGE_FOLDER, args.query, args.generate)
        print(f"\nSource image: {src}\nGenerated: {out}")
    elif args.t:
        ans, texts = answer_text(args.query)
        print(f"\nText sources: {len(texts)} chunks\n\nRAG: {ans}")
    elif args.i:
        import image_rag
        ans, img = image_rag.answer_with_image(IMAGE_FOLDER, args.query)
        print(f"\nImage used: {img}\n\nRAG: {ans}")
    else:
        ans, texts, imgs = answer_multimodal(args.query)
        print(f"\nText sources: {len(texts)} chunks\nImage used: {imgs}\n\nRAG: {ans}")
