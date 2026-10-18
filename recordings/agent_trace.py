"""Record an LLM agent searching the catalog with a tool (W-48), for chapter 10.

    python recordings/agent_trace.py      (after recordings/embeddings.py)

For every test query, the LLM (recordings/llm.py) gets a shopper's request and a search tool, may search
up to three times with any wording, and returns the three products it picks. It runs twice, with the same
model and prompt: once with BM25 behind the tool (lib.lexical) and once with semantic retrieval over the
recorded product embeddings, embedding the agent's own search text with the pinned embedding model. The
tool returns the top 5 products with ID, name, class and attributes.
Output, in examples/data/recorded/agent-trace/: traces.jsonl, one line per (retriever, query) with every
search the agent made, what the tool returned, the three IDs it picked, and every raw reply the model gave
(its text and tool calls), in order.
"""
import json
import re

import numpy as np
import ollama
import torch
from sentence_transformers import SentenceTransformer

from common import out_dir, ollama_version, write_manifest
from embeddings import MODEL as EMBEDDING_MODEL, REVISION as EMBEDDING_REVISION
from lib.lexical import LexicalSearch
from lib.recorded import embeddings
from lib.store import load_products, load_queries
from llm import LICENCE, MODEL, OPTIONS, settings

NAME = "agent-trace"
MAX_SEARCHES, SHOWN = 3, 5
TASK = ("You help a shopper at an online furniture and home-goods store. The shopper wants: {query}\n\n"
        "Use the search tool to find the three products that best match what the shopper wants. You may search up to "
        "three times, with any wording. When you're done, reply with only JSON: {{\"product_ids\": [three IDs]}}.")
NO_MORE = "No more searches. Reply with only the JSON now."   # sent when the agent asks for a fourth search
TOOL = {"type": "function", "function": {
    "name": "search", "description": "Search the store's catalog. Returns the top 5 products: ID, name, class and attributes.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "the search text"}},
                   "required": ["query"]}}}


def main() -> None:
    products, test = load_products(), load_queries("test")
    lexical = LexicalSearch(products)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION, device=device)
    product_ids, product_vectors, _, _ = embeddings()

    def semantic(text: str) -> list[int]:
        vector = embedder.encode_query([text], normalize_embeddings=True, show_progress_bar=False)[0]
        return product_ids[np.argsort(-(product_vectors @ vector), kind="stable")[:SHOWN]].tolist()

    retrievers = {"bm25": lambda text: [int(p) for p in lexical.search(text, SHOWN).index], "semantic": semantic}
    path = out_dir(NAME) / "traces.jsonl"
    done = {(t["retriever"], t["query_id"]) for t in map(json.loads, path.read_text().splitlines())} if path.exists() else set()
    for retriever, search in retrievers.items():
        for qid, query in test["query"].items():
            if (retriever, qid) in done:
                continue
            messages = [{"role": "user", "content": TASK.format(query=query)}]
            searches, picked, replies = [], [], []
            for _ in range(MAX_SEARCHES + 2):
                reply = ollama.chat(model=MODEL, messages=messages, tools=[TOOL], think=False, options=OPTIONS).message
                messages.append(reply)
                replies.append({"content": reply.content or "",
                                "tool_calls": [{"name": c.function.name, "arguments": dict(c.function.arguments)}
                                               for c in reply.tool_calls or []]})
                if reply.tool_calls and len(searches) < MAX_SEARCHES:
                    for call in reply.tool_calls:
                        text = str(call.function.arguments.get("query", ""))
                        found = search(text)
                        listing = [{"product_id": p, "name": products.at[p, "name"], "class": products.at[p, "product_class"],
                                    "attributes": products.at[p, "attributes"]} for p in found]
                        searches.append({"query": text, "results": found})
                        messages.append({"role": "tool", "content": json.dumps(listing), "tool_name": "search"})
                    continue
                if reply.tool_calls:
                    messages.append({"role": "user", "content": NO_MORE})
                    continue
                match = re.search(r"\{.*\}", reply.content or "", re.S)
                try:
                    picked = [int(p) for p in json.loads(match.group(0))["product_ids"]][:3] if match else []
                except (ValueError, KeyError, TypeError):
                    picked = []
                break
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"retriever": retriever, "query_id": int(qid), "searches": searches, "picked": picked,
                                    "replies": replies}) + "\n")
    write_manifest(NAME, "agent_trace.py", MODEL, ollama_version(MODEL), {"task": TASK, "no_more_searches": NO_MORE, "tool": TOOL}, ["traces.jsonl"], LICENCE,
                   **settings({"tool": TOOL}), max_searches=MAX_SEARCHES, results_shown=SHOWN,
                   embedding_model=EMBEDDING_MODEL, embedding_revision=EMBEDDING_REVISION,
                   retrievers={"bm25": "lib.lexical", "semantic": "recorded product embeddings; tool text embedded live"})
    print(f"traces: {len(path.read_text().splitlines())}")


if __name__ == "__main__":
    main()
