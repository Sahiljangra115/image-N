# Persistence and Multi-Document Lessons

## Question 1: Why Cache Vectors (P1)
* Question: Maya's bakery binder has 500 pages. Every time a customer asks a question, the app re-embeds all 500 pages before answering. What specifically makes this wasteful? What changes between one question and the next?
* Answer: we should use the vector database to store vectors so we do not waste processing power and time doing this again and again.
* Verdict: Half right.
* Explanation: The conclusion (store vectors, don't recompute) is correct, but the reason is more specific. The document does not change between questions. The chunks do not change. The vectors do not change. Only the query changes. So re-embedding produces byte-for-byte identical output every run. You are redoing work whose result never changed. One line to remember: embed once per document, not once per question.

## Question 2: Two Files (P2)
* Question: We saved two files: chunks.json and vectors.npy. Why two separate files? What does each hold and why are they different types?
* Answer: chunks.json stores the chunks and vectors.npy stores the embeddings. We do that so we can pick the answer from the vector file and then take the corresponding chunk from the chunk file and pass it to the LLM.
* Verdict: Correct.
* Explanation: Text lives in JSON (human-readable strings), numbers live in .npy (numpy binary, fast and compact for float arrays). Index position links them: chunk 7 in the JSON is the text for vector row 7 in the .npy. The relationship between them is what makes retrieval work.

## Question 3: Caller Decides on None (P3)
* Question: load_index() returns None when no files exist. Why put that decision in the caller (answer_question) instead of inside load_index itself?
* Answer: don't know.
* Explanation: load_index has one job: load files if they exist, return None if they do not. It does not know what should happen next. Maybe the caller wants to embed. Maybe it wants to throw an error. Maybe it wants to download from S3. Keeping that decision in the caller means load_index stays reusable. You could call it from ten different places, each doing something different on None. One line to remember: a function that loads should not decide what to do when loading fails.

## Re-Test: Overwrite vs Append (RT1)
* Question: Maya adds a second binder (desserts). You call save_index with the new chunks and vectors. What happens to the old binder's data?
* Answer: they will be there, new binder data will be appended and embeddings appended with a new index along with the old binder.
* Verdict: Misconception.
* Explanation: save_index uses open(..., "w") which overwrites the file, and np.save also overwrites. The old binder is gone. The current code supports exactly one document at a time. To support multiple binders you would either merge all chunks before saving (one big index) or give each document its own index folder. We handle this now with load_folder and the answer_question folder-detection logic.

## Question 4: Source Tagging (P4)
* Question: We prepend [source: filename] to every chunk before embedding. Why does this text end up inside the chunk itself rather than stored separately? What could go wrong if you forgot to tag chunks with their source?
* Answer: we will lose the vectors and their chunk relation so we will not be able to send the chunks to the llm for answer and the source citation is necessary for surity of answer in this case i think so.
* Verdict: Half right.
* Explanation: Prepending the tag inline ensures the source metadata is embedded as part of the text. When the chunk is retrieved and passed to the LLM inside the prompt, the LLM reads this inline metadata directly to cite the source. If it was stored separately in database columns or metadata fields, you would have to maintain a parallel index mapping; if that mapping is lost or not added to the prompt, the LLM would have no way of knowing where each chunk came from, leading to inaccurate citations or missing source attribution.

## Question 5: Overwrite Is a Feature (P5)
* Question: Right now save_index always overwrites the index. If Maya updates her binder with a corrected recipe, is the overwrite behavior a bug or the correct thing to do? Explain why.
* Answer: I think it is necesary as old binder can give the wrong info and that can lead to answers overall. we need to replace the old vectors and use the new chunks with new vectors
* Verdict: Correct.
* Explanation: Yes, overwriting is correct because when files change (e.g. updating a recipe), the old chunks and vectors become outdated. If we appended instead of overwriting, the index would contain duplicate/conflicting information, and retrieval might fetch the old version alongside the new one, confusing the LLM. Replacing the old index ensures only the latest data is used.

## Question 6: Folder vs File Detection (P6 - Stretch)
* Question: answer_question now checks os.path.isdir(path) to decide whether to load one file or a whole folder. What would break if you passed a folder path but the code only called load_documents (the single-file function)?
* Answer: we should add a try-catch block there for safety and also the error would be in case that the bug might be we would not get any chunks and llm may generate soome random answers.
* Verdict: Half right.
* Explanation: Passing a directory path to `load_documents` (which uses python's `open()`) immediately raises an `IsADirectoryError` (on Unix-like systems) or `PermissionError` (on Windows). The program crashes on line 30 of `rag.py` before any chunking or LLM querying ever happens. While a try-catch block could handle the exception, checking `os.path.isdir` acts as a guard that routes the folder path to `load_folder` instead of crashing.

---

# Vector DB, Reranking, and Web Layer Lessons

## Question 7: O(N) Flat Scan (V1)
* Question: In `retrieve()`, the original code was `scores = chunk_vectors @ q`. If `chunk_vectors` has 5 million rows, that runs 5 million dot products per query. Is that O(1), O(log N), or O(N)? What does it mean for query speed as Maya's collection grows from 10 chunks to 5 million?
* Answer:
* Verdict:
* Explanation:

## Question 8: What an Index Buys (V2)
* Question: `build_faiss()` wraps the vectors in `faiss.IndexFlatIP`. The "Flat" type still checks every vector, like the numpy scan did. So what does FAISS actually buy us at this scale, and what changes if we later swap to `IndexIVFFlat`?
* Answer:
* Verdict:
* Explanation:

## Question 9: Why IndexFlatIP not IndexFlatL2 (V3)
* Question: `build_faiss()` uses `faiss.IndexFlatIP(vectors.shape[1])`. IP means inner product (dot product), L2 means Euclidean distance. Given that `embed()` calls `EMBEDDER.encode(..., normalize_embeddings=True)`, why is IP the right choice here? What is the relationship between dot product and cosine similarity for normalized vectors?
* Answer:
* Verdict:
* Explanation:

## Question 10: The Dimension Argument (V4)
* Question: `faiss.IndexFlatIP(vectors.shape[1])` is given `384` because BGE-small produces 384-dim vectors. What happens at `index.add(vectors)` if you swapped to an embedder that produces 768-dim vectors but left the index built for 384?
* Answer:
* Verdict:
* Explanation:

## Question 11: The Dropped [0] (V5)
* Question: The old numpy `retrieve()` ended with `embed([query])[0]` to get shape `(384,)`. The FAISS version keeps `EMBEDDER.encode([...])` WITHOUT the `[0]`, so `q` stays shape `(1, 384)`. Why does `index.search(q, k)` need the 2D shape, and what does `top_idx[0]` on the next line pull out?
* Answer:
* Verdict:
* Explanation:

## Question 12: Bi-encoder vs Cross-encoder (V6)
* Question: `search()` does two stages: `retrieve()` (FAISS, fetch_k=10) then `rerank()` (cross-encoder, k=3). The embedder (bi-encoder) already scored similarity. Why run a second, slower model on top? What does the cross-encoder in `rerank()` see that the FAISS stage cannot?
* Answer:
* Verdict:
* Explanation:

## Question 13: Why Fetch 10 to Keep 3 (V7)
* Question: In `search()`, `fetch_k=10` but `k=3`. Why fetch more candidates than you keep? What goes wrong if you set `fetch_k=3` (fetch exactly what you return)?
* Answer:
* Verdict:
* Explanation:

## Question 14: Lazy Reranker Load (V8 - Stretch)
* Question: `_RERANKER` starts as `None` and `_get_reranker()` only loads the model on first call. `python rag.py check` and `eval.py` use `retrieve()` directly, not `search()`. Why does this lazy pattern matter for those two entry points? What is the cost of loading the cross-encoder at import time instead?
* Answer:
* Verdict:
* Explanation:

## Question 15: Index Built Once (W1)
* Question: In `app.py`, the index is built inside `lifespan()` at startup and stored on `app.state.chunks` / `app.state.index`. Each `/ask` request reads from `app.state`. Why build it there instead of inside the `ask()` request handler? What would happen to response time if `build_index()` ran on every request?
* Answer:
* Verdict:
* Explanation:

## Question 16: Returning Sources (W2)
* Question: The `/ask` endpoint returns both `answer` and `sources` (the retrieved chunks). The CLI version only printed the answer. Why is returning the source chunks to the UI valuable, and how does it connect to the `[source: filename]` tag from Question 4?
* Answer:
* Verdict:
* Explanation:

## Question 17: The Loader Dispatch Table (D1)
* Question: `load_documents()` does `loader = _LOADERS.get(ext)` then calls it, instead of a long `if ext == ".pdf": ... elif ext == ".docx": ...` chain. What does the dict-dispatch pattern buy you when you want to add support for a new file type like `.html`?
* Answer:
* Verdict:
* Explanation:
