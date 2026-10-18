"""Record cross-encoder scores for each query's candidates (writing standards W-48).

    python recordings/cross_encoder.py      (after recordings/embeddings.py)

Model: Alibaba-NLP/gte-reranker-modernbert-base (Apache-2.0), pinned to a commit. Candidates for every
query are the union of BM25's top 100 (lib.lexical, default parameters) and the recorded embeddings' top
100, so any reranking of those two lists, or of a fusion of them, finds a score for every product.
Output, in examples/data/recorded/cross-encoder/: scores.csv with query_id, product_id and score, the
model's raw output (higher is more relevant; no activation applied). Rows are sorted by score within each
query, so never rely on their order: sort by product_id first (the plan's rule on order leaks).

A known quirk, kept so the recording matches what was run: the semantic candidates were chosen from the
stored 16-bit vectors without rescaling them to unit length, while lib.semantic rescales. For queries 277 and
406 the two top-100 lists differ at the margin, so lib.semantic's 100th product has no score here.
"""
import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder

from common import RECORDED, out_dir, write_manifest
from lib.lexical import LexicalSearch
from lib.store import load_products, load_queries, product_text

MODEL = "Alibaba-NLP/gte-reranker-modernbert-base"
REVISION = "f7481e6055501a30fb19d090657df9ec1f79ab2c"
NAME = "cross-encoder"
DEPTH = 100


def main() -> None:
    products, queries = load_products(), load_queries()
    text = product_text(products)
    lexical = LexicalSearch(products)
    emb = RECORDED / "embeddings"
    docs, doc_ids = np.load(emb / "products.npy").astype(np.float32), np.load(emb / "product_ids.npy")
    qs, q_ids = np.load(emb / "queries.npy").astype(np.float32), np.load(emb / "query_ids.npy")
    q_row = {qid: i for i, qid in enumerate(q_ids)}

    pairs = []
    for qid, query in queries["query"].items():
        semantic = doc_ids[np.argsort(-(docs @ qs[q_row[qid]]), kind="stable")[:DEPTH]]
        candidates = list(dict.fromkeys(list(lexical.search(query, DEPTH).index) + list(semantic)))
        pairs += [(qid, pid) for pid in candidates]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(MODEL, revision=REVISION, device=device)
    scores = model.predict([(queries.at[q, "query"], text.at[p]) for q, p in pairs], batch_size=64,
                           activation_fn=torch.nn.Identity(), show_progress_bar=False)
    out = pd.DataFrame(pairs, columns=["query_id", "product_id"]).assign(score=np.round(scores.astype(float), 5))
    out.sort_values(["query_id", "score"], ascending=[True, False]).to_csv(out_dir(NAME) / "scores.csv", index=False)
    write_manifest(NAME, "cross_encoder.py", MODEL, REVISION,
                   {"pair": "(query text, lib.store.product_text)", "activation": "none: raw scores"},
                   ["scores.csv"], "Apache-2.0", candidates=f"union of BM25 top {DEPTH} and embeddings top {DEPTH}",
                   pairs=len(out), device=device)
    print(f"pairs scored: {len(out):,}")


if __name__ == "__main__":
    main()
