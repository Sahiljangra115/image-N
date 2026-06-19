"""
Tiny retrieval eval. Answers the interview question: "how do you know
retrieval works?" with a number instead of a vibe.

We hand-label a few questions with a keyword that MUST appear in a correct
chunk. Then we measure hit-rate@k: of the k chunks search() returns, did
any contain the expected keyword?

This is the cheap, dependency-free cousin of RAGAS. RAGAS adds LLM-judged
faithfulness and answer-relevance on top; the idea (label, retrieve, score)
is the same. Run: python eval.py
"""

import rag

# (question, keyword that a correct chunk must contain)
GOLD = [
    ("Which cake is safe for a nut allergy?", "vanilla"),
    ("What cake contains almonds?", "almond"),
    ("When is the shop closed?", "monday"),
    ("Where does the flour come from?", "greenfield"),
]

DATA_PATH = "data/maya_binder.txt"


def hit_rate(k: int = 3) -> float:
    chunks, index = rag.build_index(DATA_PATH)
    hits = 0
    for question, keyword in GOLD:
        top = rag.search(question, chunks, index, k=k)
        joined = " ".join(top).lower()
        ok = keyword in joined
        hits += ok
        print(f"[{'HIT ' if ok else 'MISS'}] {question}  (expected '{keyword}')")
    rate = hits / len(GOLD)
    print(f"\nhit-rate@{k}: {hits}/{len(GOLD)} = {rate:.0%}")
    return rate


if __name__ == "__main__":
    score = hit_rate(k=3)
    assert score >= 0.75, "retrieval quality dropped below 75%"
