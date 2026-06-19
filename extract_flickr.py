"""
One-time: unpack the Flickr8k parquet into loose image files + a caption map.

Why: image_rag.py loads a FOLDER of image files, and Ollama's vision model
needs an image path. The HF dataset ships images as bytes inside parquet, so
we extract once. The captions become a free recall@k eval set later
(each caption is a query whose correct answer is its own image).

Run: python extract_flickr.py
Output:
  data/images/img_0000.jpg ...        (the images)
  data/captions.json                  ({filename: [5 captions]})
"""

import pyarrow.parquet as pq
import glob
import json
import os

# Swap to train-* to scale up.
SRC = glob.glob(
    "/home/ladliju/.cache/huggingface/hub/datasets--jxie--flickr8k/"
    "snapshots/*/data/validation-*.parquet"
)[0]
OUT_DIR = "data/images"
CAP_FILE = "data/captions.json"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    table = pq.read_table(SRC)
    cap_cols = [c for c in table.column_names if c.startswith("caption_")]

    captions = {}
    images = table.column("image")
    for i in range(table.num_rows):
        # zero-padded so sorted filename order == row order (eval alignment)
        name = f"img_{i:04d}.jpg"
        img_bytes = images[i]["bytes"].as_py()
        with open(os.path.join(OUT_DIR, name), "wb") as f:
            f.write(img_bytes)
        captions[name] = [table.column(c)[i].as_py() for c in cap_cols]

    with open(CAP_FILE, "w") as f:
        json.dump(captions, f)

    assert len(captions) == table.num_rows
    assert len(os.listdir(OUT_DIR)) == table.num_rows
    print(f"OK. {table.num_rows} images -> {OUT_DIR}/, captions -> {CAP_FILE}")


if __name__ == "__main__":
    main()
