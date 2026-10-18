"""Record LLM answers grounded in retrieved products (W-48), for chapter 10.

    python recordings/rag_answers.py

For every test query, the LLM (recordings/llm.py) answers a shopper from five products, cited by ID.
The five products come from three contexts: semantic retrieval's top 5, BM25's top 5, and five products
drawn at random with a fixed seed. The same prompt and model see each context, so differences in the
answers come from retrieval.
Outputs, in examples/data/recorded/rag-answers/: answers.csv (query_id, context, context_ids, answer,
cited), with IDs space-separated, and calls.jsonl.
"""
import numpy as np
import pandas as pd

from common import write_manifest
from lib.lexical import LexicalSearch
from lib.semantic import SemanticSearch
from lib.store import load_products, load_queries
from llm import LICENCE, MODEL, Recorder, describe, settings

NAME = "rag-answers"
K, SEED = 5, 2026
PROMPT = ("You help shoppers at an online furniture and home-goods store. Using only the products listed below, "
          "answer the shopper. Recommend up to three products that fit, cite each one by its ID, and say plainly if "
          "none of them fit. Don't mention products that aren't listed.\n\nShopper: I'm looking for {query}.\n\n"
          "Products:\n{products}\n\nAnswer in JSON: the answer to show the shopper, and the IDs you cited.")
SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}, "cited": {"type": "array", "items": {"type": "integer"}}},
          "required": ["answer", "cited"]}


def main() -> None:
    products, test = load_products(), load_queries("test")
    contexts = {"semantic": SemanticSearch().run(test, k=K), "bm25": LexicalSearch(products).run(test, k=K)}
    rng = np.random.default_rng(SEED)
    contexts["random"] = {qid: rng.choice(products.index.to_numpy(), size=K, replace=False).tolist() for qid in test.index}
    recorder = Recorder(NAME)
    rows = []
    for qid, query in test["query"].items():
        for context, run in contexts.items():
            ids = [int(p) for p in run[qid]]
            listing = "\n\n".join(describe(products.loc[p], p) for p in ids)
            answer = recorder.json(f"{context}:{qid}", [{"role": "user", "content": PROMPT.format(query=query, products=listing)}],
                                   SCHEMA) or {}
            rows.append({"query_id": qid, "context": context, "context_ids": " ".join(map(str, ids)),
                         "answer": answer.get("answer", ""), "cited": " ".join(str(int(c)) for c in answer.get("cited", []))})
    pd.DataFrame(rows).to_csv(recorder.path.parent / "answers.csv", index=False)
    write_manifest(NAME, "rag_answers.py", MODEL, recorder.digest, PROMPT, ["answers.csv", "calls.jsonl"], LICENCE,
                   **settings({"answer": SCHEMA}),
                   contexts={"semantic": f"lib.semantic top {K}", "bm25": f"lib.lexical top {K}", "random": f"{K} products, seed {SEED}"})
    print(f"answers: {len(rows)}")


if __name__ == "__main__":
    main()
