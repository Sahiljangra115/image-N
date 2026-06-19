"""
Multimodal RAG: find a relevant IMAGE for a text query, then let a vision
LLM answer grounded in that image.

Same skeleton as rag.py. The only modality-specific parts live here:
  - load_images   : read image files (not utf-8 text)
  - embed_images  : CLIP image encoder  (shared text+image space)
  - embed_query   : CLIP text encoder   (SAME space, so comparable)
  - generate_vision: send query + image to a local vision model (Ollama)

Everything else is reused from rag.py unchanged:
  build_faiss, save_index, load_index   (a vector is a vector; a path is a string)

Flow:  load images -> embed once -> save -> [query] -> retrieve image -> vision LLM
                                     ^
                            next run: load from disk, skip embedding

Why no chunking: an image is one atomic unit. One image, one vector.
Why no rerank: ms-marco cross-encoder is text-text only. v1 runs on the
bi-encoder (CLIP) alone. See lessons for the upgrade path.
"""

from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
import torch
import os

import rag

DEVICE = rag.DEVICE

CLIP = SentenceTransformer("clip-ViT-L-14", device=DEVICE)

IMAGE_INDEX = "data/image_index"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def load_images(folder: str) -> list[str]:
    """Return sorted paths of every image in the folder.
    Sorted so index position is stable across runs (vector row i == this path)."""
    paths = []
    for name in sorted(os.listdir(folder)):
        if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
            paths.append(os.path.join(folder, name))
    return paths


def embed_images(paths: list[str]) -> np.ndarray:
    """One vector per image. normalize=True so dot product == cosine,
    which is what rag.build_faiss (IndexFlatIP) assumes."""
    imgs = [Image.open(p).convert("RGB") for p in paths]
    return CLIP.encode(imgs, normalize_embeddings=True, show_progress_bar=True)


def embed_query(query: str) -> np.ndarray:
    """Encode the text query into the SAME space as the images.
    No BGE-style prefix: CLIP text encoder needs none. Shape (1, 512)."""
    return CLIP.encode([query], normalize_embeddings=True)


def retrieve_images(query: str, paths: list[str], index, k: int = 3) -> list[str]:
    """Top-k image paths whose vectors are closest to the query vector.
    Same as rag.retrieve, but encodes the query with CLIP instead of BGE."""
    q = np.ascontiguousarray(embed_query(query), dtype="float32")
    scores, top_idx = index.search(q, k)
    return [paths[i] for i in top_idx[0]]


def build_image_index(folder: str):
    """Load cached index, embed ONLY images not already in it, append, save.
    Returns (paths, faiss_index). Incremental: re-running after adding images
    embeds just the new ones, not the whole folder.
    ponytail: handles additions only. A deleted file stays stale in the index;
    delete data/image_index to force a clean rebuild if that matters."""
    current = load_images(folder)
    if not current:
        raise ValueError(f"No images found in {folder} (exts: {sorted(IMAGE_EXTS)})")

    cached = rag.load_index(folder=IMAGE_INDEX)
    if cached:
        paths, vectors = cached
        new = [p for p in current if p not in set(paths)]
        if new:
            print(f"Embedding {len(new)} new images (skipping {len(paths)} cached)...")
            vectors = np.vstack([vectors, embed_images(new)])
            paths = paths + new
            rag.save_index(paths, vectors, folder=IMAGE_INDEX)
        else:
            print(f"(all {len(paths)} images already indexed, skipped embedding)")
    else:
        print(f"Embedding {len(current)} images...")
        paths = current
        vectors = embed_images(paths)
        rag.save_index(paths, vectors, folder=IMAGE_INDEX)
        print(f"(built and saved image index: {len(paths)} images)")
    return paths, rag.build_faiss(vectors)


_VLM = None
_VLM_PROC = None
_VLM_NAME = "HuggingFaceTB/SmolVLM-500M-Instruct"


def _get_vlm():
    global _VLM, _VLM_PROC
    if _VLM is None:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        _VLM_PROC = AutoProcessor.from_pretrained(_VLM_NAME)
        _VLM = AutoModelForImageTextToText.from_pretrained(
            _VLM_NAME, dtype=dtype
        ).to(DEVICE)
    return _VLM, _VLM_PROC


def vlm_answer(image_paths: list[str], instruction: str, max_new_tokens: int = 256) -> str:
    """Ground SmolVLM on the given image(s) + a text instruction.
    Decodes only the generated tokens (strips the echoed prompt)."""
    model, proc = _get_vlm()
    images = [Image.open(p).convert("RGB") for p in image_paths]
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": instruction}]
    text = proc.apply_chat_template([{"role": "user", "content": content}],
                                    add_generation_prompt=True)
    inputs = proc(text=text, images=images, return_tensors="pt").to(DEVICE)
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = gen[:, inputs["input_ids"].shape[1]:]
    return proc.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def generate_vision(query: str, image_path: str) -> str:
    """Answer the query grounded in the single retrieved image (SmolVLM)."""
    instruction = (
        "Answer the question using ONLY the attached image. "
        "If the image does not show the answer, say you do not know.\n\n"
        f"Question: {query}"
    )
    return vlm_answer([image_path], instruction)


def answer_with_image(folder: str, query: str) -> tuple[str, str]:
    """Retrieve the best image, answer grounded in it.
    Returns (answer, image_path) for provenance."""
    paths, index = build_image_index(folder)
    best = retrieve_images(query, paths, index, k=1)[0]
    return generate_vision(query, best), best


# CLIP-L + SD Turbo together is ~3.4GB on 4GB. If this OOMs, set
# CLIP device="cpu" (top of file) or run generation as a separate process.
_SD = None


def _get_sd():
    global _SD
    if _SD is None:
        from diffusers import AutoPipelineForImage2Image
        if DEVICE == "cuda":
            _SD = AutoPipelineForImage2Image.from_pretrained(
                "stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16"
            )
            _SD.enable_sequential_cpu_offload()
            _SD.enable_vae_slicing()
        else:
            _SD = AutoPipelineForImage2Image.from_pretrained(
                "stabilityai/sd-turbo", torch_dtype=torch.float32
            ).to("cpu")   # offload is CUDA-only
    return _SD


def _free_clip() -> None:
    """On GPU, move CLIP off-device to free ~1.7GB before loading SD (its
    query-encode job is already done by retrieval time). No-op on CPU."""
    if DEVICE == "cuda":
        CLIP.to("cpu")
        torch.cuda.empty_cache()


def modify_image(image_path: str, prompt: str, strength: float = 0.6,
                 steps: int = 4, seed: int | None = None,
                 out_dir: str = "outputs") -> str:
    """Generate a new image FROM the retrieved one, edited per the prompt.
    This is the real RAG link: the retrieved image conditions generation
    (img2img), so what we found shapes what we make.

    Quality knobs (SD Turbo img2img):
      strength: 0=identical to source, 1=ignore it. Main edit-strength dial.
      steps:    1-4 for Turbo. effective steps = steps*strength must be >= 1.
      seed:     fix for reproducibility / reroll a bad result.
    guidance_scale stays 0.0: Turbo is distilled without CFG, raising it hurts.
    Returns the saved path."""
    import time
    import torch

    _free_clip()
    pipe = _get_sd()
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    init = Image.open(image_path).convert("RGB").resize((512, 512))
    result = pipe(prompt=prompt, image=init, num_inference_steps=steps,
                  strength=strength, guidance_scale=0.0, generator=gen).images[0]
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"make_{time.strftime('%Y%m%d_%H%M%S')}.png")
    result.save(out)
    return out


def retrieve_and_make(folder: str, query: str, edit: str) -> tuple[str, str]:
    """Full generative pipeline: find the image for `query`, then edit it
    per `edit`. Returns (generated_path, source_image_path)."""
    paths, index = build_image_index(folder)
    src = retrieve_images(query, paths, index, k=1)[0]
    return modify_image(src, edit), src


def _selfcheck(folder: str) -> None:
    """Retrieval-only check, no LLM. Run: python image_rag.py check <folder>
    Verifies the shared-space claim: a text query lands near a sane image."""
    paths, index = build_image_index(folder)
    assert index.d == 768, "CLIP ViT-L-14 must give 768-dim vectors"
    assert index.ntotal == len(paths), "every image must be in the index"
    top = retrieve_images("a photo", paths, index, k=1)
    assert top and os.path.exists(top[0]), "retrieve must return a real image path"
    print(f"OK. {index.ntotal} images indexed. Top hit for 'a photo':\n{top[0]}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2 and sys.argv[1] == "check":
        _selfcheck(sys.argv[2])
    elif len(sys.argv) > 2 and sys.argv[1] == "ask":
        folder, q = sys.argv[2], " ".join(sys.argv[3:])
        ans, img = answer_with_image(folder, q)
        print(f"\nImage used: {img}\n\nRAG: {ans}")
    elif len(sys.argv) > 4 and sys.argv[1] == "make":
        # make <folder> "<query>" "<edit prompt>"
        folder, query, edit = sys.argv[2], sys.argv[3], sys.argv[4]
        out, src = retrieve_and_make(folder, query, edit)
        print(f"\nSource image: {src}\nGenerated: {out}")
    else:
        print("Usage:")
        print("  python image_rag.py check <image_folder>")
        print('  python image_rag.py ask <image_folder> "your question"')
        print('  python image_rag.py make <image_folder> "<query>" "<edit prompt>"')
