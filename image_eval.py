"""
Free retrieval eval for the image RAG. No hand-labeling.

Each caption is a query whose CORRECT answer is its own image. So for every
caption we retrieve top-k and check: did the caption's own image come back?
That is recall@k, averaged over ~1000 real queries.

This is the honest metric the text eval.py faked with 4 keyword pairs. Here
the ground truth is built into the dataset.

Run: python image_eval.py
"""

import json
import numpy as np
import os

import image_rag

KS = (1, 3, 5, 10)


def main() -> None:
    paths, index = image_rag.build_image_index("data/images")
    caps = json.load(open("data/captions.json"))

    # row position of each image, so we can check "did MY image come back"
    pos = {os.path.basename(p): i for i, p in enumerate(paths)}

    # one query per image (caption_0). Batch-encode all at once: fast.
    names = list(caps.keys())
    queries = [caps[n][0] for n in names]
    qv = np.ascontiguousarray(
        image_rag.CLIP.encode(queries, normalize_embeddings=True), dtype="float32"
    )

    maxk = max(KS)
    _, idx = index.search(qv, maxk)   # (N, maxk) retrieved row positions

    hits = {k: 0 for k in KS}
    for row, name in enumerate(names):
        gold = pos[name]
        retrieved = idx[row]
        for k in KS:
            if gold in retrieved[:k]:
                hits[k] += 1

    n = len(names)
    print(f"queries: {n}  images: {len(paths)}")
    for k in KS:
        print(f"recall@{k}: {hits[k] / n:.3f}")


if __name__ == "__main__":
    main()
