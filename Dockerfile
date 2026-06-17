# Multimodal RAG demo image. Targets HuggingFace Spaces (port 7860).
# ponytail: CPU base. Models (CLIP, SmolVLM, SD Turbo) download at runtime and
# run on CPU here, slow for generation. On a GPU host, swap to an
# nvidia/cuda:12.1-runtime base and install GPU torch for real speed.
FROM python:3.11-slim

# libgomp1: OpenMP runtime needed by faiss / torch
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# deps first so this layer caches when only code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces sets HOME=/app; keep model cache inside the writable workdir
ENV HF_HOME=/app/.cache/huggingface
ENV GRADIO_SERVER_NAME=0.0.0.0

EXPOSE 7860
CMD ["python", "app_gradio.py"]
