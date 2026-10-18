"""Record LLM query rewrites and hypothetical products, with their embeddings (W-48), for chapter 8.

    python recordings/llm_rewrites.py

For every query, the LLM (recordings/llm.py) writes:
- a rewrite: spelling fixed, abbreviations expanded, product-type synonyms added, requirements kept
- a hypothetical product (HyDE): the listing it imagines would match, as name, class and description
Both are embedded with the same pinned model as recordings/embeddings.py: rewrites as queries (with the
model's query prompt), hypothetical products as documents, so they can be matched against the recorded
product embeddings.
Outputs, in examples/data/recorded/llm-rewrites/: rewrites.csv (query_id, rewrite, hypothetical),
rewrites.npy and hypothetical.npy (float16, unit length, in rewrites.csv order), and calls.jsonl.
"""
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from common import out_dir, write_manifest
from embeddings import MODEL as EMBEDDING_MODEL, REVISION as EMBEDDING_REVISION
from lib.store import load_queries
from llm import LICENCE, MODEL, Recorder, settings

NAME = "llm-rewrites"
PROMPTS = {
    "rewrite": ("Rewrite this search query from an online furniture and home-goods store so that a search engine finds "
                "what the shopper wants. Fix spelling, expand abbreviations and add common synonyms for the product "
                "type, but keep every requirement the shopper stated and add none of your own. Return a single line.\n\n"
                "Query: {query}\n\nAnswer in JSON."),
    "hypothetical": ("A shopper searched an online furniture and home-goods store for: {query}\n\nWrite the product "
                     "listing that would match it best, as the store would list it: a product name, the product class "
                     "and a description of two or three sentences. Answer in JSON."),
}
SCHEMAS = {
    "rewrite": {"type": "object", "properties": {"rewrite": {"type": "string"}}, "required": ["rewrite"]},
    "hypothetical": {"type": "object", "properties": {"name": {"type": "string"}, "product_class": {"type": "string"},
                                                      "description": {"type": "string"}},
                     "required": ["name", "product_class", "description"]},
}


def main() -> None:
    queries = load_queries()
    recorder = Recorder(NAME)
    rows = []
    for qid, query in queries["query"].items():
        rewrite = recorder.json(f"rewrite:{qid}", [{"role": "user", "content": PROMPTS["rewrite"].format(query=query)}],
                                SCHEMAS["rewrite"])
        hypo = recorder.json(f"hypothetical:{qid}",
                             [{"role": "user", "content": PROMPTS["hypothetical"].format(query=query)}], SCHEMAS["hypothetical"])
        rows.append({"query_id": qid, "rewrite": (rewrite or {}).get("rewrite", query),
                     "hypothetical": ". ".join(hypo[k] for k in ("name", "product_class", "description")) if hypo else query})
    table = pd.DataFrame(rows)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION, device=device)
    rewrites = model.encode_query(table["rewrite"].tolist(), normalize_embeddings=True, show_progress_bar=False)
    hypothetical = model.encode_document(table["hypothetical"].tolist(), normalize_embeddings=True, show_progress_bar=False)
    out = out_dir(NAME)
    table.to_csv(out / "rewrites.csv", index=False)
    np.save(out / "rewrites.npy", rewrites.astype(np.float16))
    np.save(out / "hypothetical.npy", hypothetical.astype(np.float16))
    write_manifest(NAME, "llm_rewrites.py", MODEL, recorder.digest, PROMPTS,
                   ["rewrites.csv", "rewrites.npy", "hypothetical.npy", "calls.jsonl"], LICENCE,
                   **settings(SCHEMAS),
                   embedding_model=EMBEDDING_MODEL, embedding_revision=EMBEDDING_REVISION,
                   embedded_as={"rewrite": "query (model's query prompt)", "hypothetical": "document (no prompt)"})
    print(f"rewrites: {len(table)}")


if __name__ == "__main__":
    main()
