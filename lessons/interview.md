# RAG Interview Questions

Broader than the per-line code lessons. These cover what a one-hour RAG
interview (L4 target) asks beyond your specific pipeline. Grounded in your
code where possible, so you defend real decisions, not textbook abstractions.

Format: answer in your own words, leave Verdict/Explanation for grading.

---

## Section A: Chunking Strategy

### A1: Fixed-size vs Recursive vs Semantic
* Question: Your `chunk_text()` splits on word count (180 words, 20 overlap). Name two other chunking strategies (recursive/structural, semantic) and explain when each beats naive word-count splitting. For Maya's bakery binder, which would you actually pick and why?
* Answer:
* Verdict:
* Explanation:

### A2: Chunk Size vs Embedder Limit
* Question: You picked 180 words because BGE-small caps near 256 tokens. Walk through the tradeoff: what gets worse if chunks are too big (say 1000 words), and what gets worse if they are too small (say 20 words)? Two failure directions.
* Answer:
* Verdict:
* Explanation:

### A3: Chunking by Structure
* Question: A markdown file has headers, a CSV has rows, a PDF has pages. Your loaders flatten all of these into one big string before chunking. When would chunking by the document's own structure (per-header, per-row) beat a flat word-window, and what do you lose by flattening?
* Answer:
* Verdict:
* Explanation:

---

## Section B: Embeddings

### B1: Bi-encoder vs Cross-encoder (concept)
* Question: Your pipeline uses a bi-encoder (BGE) for retrieval and a cross-encoder for reranking. Explain the core architectural difference: why can the bi-encoder be precomputed and cached, while the cross-encoder cannot? Why is the cross-encoder more accurate but unusable for scanning millions of chunks?
* Answer:
* Verdict:
* Explanation:

### B2: What 384 Dimensions Means
* Question: BGE-small outputs 384-dim vectors. A bigger model might output 1024. What does adding dimensions buy you, and what does it cost (storage, speed, overfitting risk)? When is a smaller embedding model the right call?
* Answer:
* Verdict:
* Explanation:

### B3: When to Fine-tune Embeddings
* Question: Off-the-shelf BGE works for Maya's bakery. Describe a domain where a general embedding model retrieves poorly and fine-tuning embeddings would help. What signal tells you retrieval is failing because of the embedder specifically?
* Answer:
* Verdict:
* Explanation:

---

## Section C: Retrieval & Vector Search

### C1: Flat vs ANN (HNSW, IVF)
* Question: Your `IndexFlatIP` does exact search (checks every vector). HNSW and IVF are approximate (ANN). Explain the tradeoff ANN makes and the one number that decides when you must switch from flat to ANN. Why is approximate acceptable for retrieval?
* Answer:
* Verdict:
* Explanation:

### C2: Hybrid Search
* Question: Your retrieval is pure dense (vector similarity). A customer searches "qwen3.5" (an exact model name / SKU). Why might dense vector search miss it, and how does adding keyword search (BM25) fix this? What is hybrid search?
* Answer:
* Verdict:
* Explanation:

### C3: Metadata Filtering
* Question: Maya now has binders from 3 store locations. A customer asks about the Delhi store only. Pure vector search over all chunks could return Mumbai chunks. How would metadata filtering solve this, and where in your current pipeline (which embeds `[source: filename]` into the text) does this approach fall short?
* Answer:
* Verdict:
* Explanation:

### C4: Managed DB vs FAISS
* Question: You use FAISS (a library, in-memory). Pinecone / Weaviate / pgvector are managed vector databases. Name two things a managed DB gives you that your FAISS-in-memory setup does not. When is FAISS the right choice anyway?
* Answer:
* Verdict:
* Explanation:

---

## Section D: Evaluation

### D1: Retrieval Metrics
* Question: Your `eval.py` measures keyword hit-rate@3. Define precision@k, recall@k, and MRR. Why is hit-rate (did any of k chunks contain the keyword) a weak metric compared to these? Give one case where hit-rate looks fine but retrieval is actually bad.
* Answer:
* Verdict:
* Explanation:

### D2: Generation Metrics (RAGAS)
* Question: Retrieval can be perfect while the final answer is still wrong. Name the RAGAS-style metrics for the generation half (faithfulness, answer relevance, context precision) and explain what each one catches that a retrieval metric cannot.
* Answer:
* Verdict:
* Explanation:

### D3: Building a Test Set
* Question: Your `GOLD` list in `eval.py` is 4 hand-written question/keyword pairs. How would you build a realistic eval set of 100+ questions for Maya's RAG without hand-writing all of them? What is the risk of a test set that is too small or written by the same person who built the system?
* Answer:
* Verdict:
* Explanation:

---

## Section E: Failure Modes & Debugging

### E1: Retrieval vs Generation Failure
* Question: Maya's RAG returns a wrong answer. Describe the exact debugging flowchart: how do you tell whether the wrong answer came from retrieval (wrong chunks fetched) or generation (right chunks, model ignored them)? Which part of your code would you print to diagnose each?
* Answer:
* Verdict:
* Explanation:

### E2: Lost in the Middle
* Question: Even with good retrieval, stuffing many chunks into `build_prompt()` can degrade the answer. Explain "lost in the middle." How does your reranker (which orders best-first) partly help, and where would you place the single most relevant chunk in the prompt?
* Answer:
* Verdict:
* Explanation:

### E3: Hallucination Despite Context
* Question: Your `build_prompt()` says "use ONLY the context." The model still invents an answer not in the chunks. Why does the instruction not fully prevent this? Name two mitigations beyond the prompt instruction.
* Answer:
* Verdict:
* Explanation:

### E4: Multi-hop Questions
* Question: "Which cake is cheaper, the one with almonds or the nut-free one?" needs facts from two different chunks combined. Why does single-shot retrieval struggle with multi-hop questions, and what technique (query decomposition, iterative retrieval) addresses it?
* Answer:
* Verdict:
* Explanation:

### E5: The "I Don't Know" Path
* Question: A customer asks something not in Maya's binder at all. Your prompt tells the model to say it does not know. At the retrieval layer, what signal (think FAISS scores) could you use to detect "nothing relevant was found" BEFORE even calling the LLM? Why is a similarity threshold tricky to set?
* Answer:
* Verdict:
* Explanation:

---

## Section F: System Design & Production

### F1: Scaling to Millions of Docs
* Question: Maya's system is licensed to a hospital: 5 million documents. Walk through what breaks in your current pipeline (embedding time, index type, memory, the build-once-at-startup model in `app.py`) and what you change at each layer.
* Answer:
* Verdict:
* Explanation:

### F2: Keeping the Index Fresh
* Question: Hospital records update hourly. Your `build_index()` embeds everything once at startup and overwrites the whole index. How would you handle incremental updates (new and changed documents) without re-embedding all 5 million chunks every hour?
* Answer:
* Verdict:
* Explanation:

### F3: Latency Budget
* Question: A query flows: embed query, FAISS search, rerank 10 candidates, LLM generate. Which stage dominates latency, and which is negligible? If a user complains the app is slow, where do you look first and what do you cut?
* Answer:
* Verdict:
* Explanation:

### F4: Cost
* Question: You run Ollama locally (free). In production with a hosted LLM, every generate() call costs tokens. Given retrieval is cheap and generation is expensive, name two ways to cut LLM cost without hurting answer quality much.
* Answer:
* Verdict:
* Explanation:

---

## Section G: The "Why" Questions (architecture defense)

### G1: Why RAG over Fine-tuning
* Question: Why not just fine-tune the LLM on Maya's binder instead of building RAG? Give the cases where RAG wins (freshness, cost, citations, hallucination control) and the cases where fine-tuning wins.
* Answer:
* Verdict:
* Explanation:

### G2: Why Not Just a Bigger Context Window
* Question: Modern LLMs fit 100k+ tokens. Why not skip retrieval entirely and stuff Maya's whole binder into the prompt every time? Name the cost, latency, and quality reasons retrieval still wins.
* Answer:
* Verdict:
* Explanation:

### G3: Why No LangChain
* Question: You built raw, no LangChain. An interviewer asks "why not use LangChain, isn't that the standard?" Give the honest answer that shows you understand the tradeoff, not just that you avoided a dependency.
* Answer:
* Verdict:
* Explanation:
