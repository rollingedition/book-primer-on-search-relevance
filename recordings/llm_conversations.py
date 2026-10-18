"""Record standalone rewrites of follow-up messages (W-48), for chapter 8.

    python recordings/llm_conversations.py

The follow-ups are written for the book, not taken from shoppers. Each refines a sampled test query by
one attribute the catalog records (colour, shape, size or finish), so a result's attributes show whether
the refinement was kept. The LLM (recordings/llm.py) rewrites each follow-up as one standalone query.
Three forms of the second turn are embedded with the pinned embedding model, as queries: the follow-up
alone, the previous query and the follow-up joined, and the LLM's standalone query.
Outputs, in examples/data/recorded/llm-conversations/: conversations.csv (conversation_id, query_id,
follow_up, attribute, value, joined, standalone), follow_up.npy, joined.npy and standalone.npy
(float16, unit length, in conversations.csv order), and calls.jsonl.
"""
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from common import out_dir, write_manifest
from embeddings import MODEL as EMBEDDING_MODEL, REVISION as EMBEDDING_REVISION
from lib.store import load_queries
from llm import LICENCE, MODEL, Recorder, settings

NAME = "llm-conversations"
# (query_id of the first search, follow-up written for the book, attribute it refines, value to look for)
FOLLOW_UPS = [
    (255, "do you have it in white?", "color", "white"), (244, "in gray please", "color", "gray"),
    (249, "only round tables", "shape", "round"), (292, "a square one", "shape", "square"),
    (312, "something in black", "color", "black"), (46, "it's for a queen bed", "size", "queen"),
    (459, "king size", "size", "king"), (293, "round, please", "shape", "round"),
    (436, "in white", "color", "white"), (58, "with a gold finish", "finish", "gold"),
    (243, "in black", "color", "black"), (346, "round ones", "shape", "round"),
]
PROMPT = ("A shopper is searching an online furniture and home-goods store.\n\nTheir previous search: {query}\n"
          "Their next message: {follow_up}\n\nRewrite the next message as one standalone search query that keeps "
          "everything the shopper still wants. Answer in JSON.")
SCHEMA = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}


def main() -> None:
    queries = load_queries()
    recorder = Recorder(NAME)
    rows = []
    for n, (qid, follow_up, attribute, value) in enumerate(FOLLOW_UPS):
        query = queries.at[qid, "query"]
        answer = recorder.json(f"standalone:{n}", [{"role": "user", "content": PROMPT.format(query=query, follow_up=follow_up)}],
                               SCHEMA)
        rows.append({"conversation_id": n, "query_id": qid, "follow_up": follow_up, "attribute": attribute, "value": value,
                     "joined": f"{query} {follow_up}", "standalone": (answer or {}).get("query", f"{query} {follow_up}")})
    table = pd.DataFrame(rows)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION, device=device)
    out = out_dir(NAME)
    table.to_csv(out / "conversations.csv", index=False)
    for column in ("follow_up", "joined", "standalone"):
        vectors = model.encode_query(table[column].tolist(), normalize_embeddings=True, show_progress_bar=False)
        np.save(out / f"{column}.npy", vectors.astype(np.float16))
    write_manifest(NAME, "llm_conversations.py", MODEL, recorder.digest, {"standalone": PROMPT, "follow_ups": FOLLOW_UPS},
                   ["conversations.csv", "follow_up.npy", "joined.npy", "standalone.npy", "calls.jsonl"], LICENCE,
                   **settings({"standalone": SCHEMA}),
                   embedding_model=EMBEDDING_MODEL, embedding_revision=EMBEDDING_REVISION,
                   note="Follow-ups are written for the book, not taken from shoppers.")
    print(f"conversations: {len(table)}")


if __name__ == "__main__":
    main()
