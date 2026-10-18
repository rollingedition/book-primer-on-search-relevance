"""Record LLM relevance judgments (writing standards W-48), for chapter 4.

    python recordings/llm_judgments.py

Two pools of (query, product) pairs:
- test: for each test query, every product in the top 10 of chapter 1's four rankers (BM25, semantic,
  reciprocal rank fusion of the two, and that fusion's top 50 reranked by the recorded cross-encoder).
  Some of these pairs have human judgments and some don't, so chapter 4 can compare system rankings
  scored with human and with LLM judgments.
- train: for each train query, up to 4 human-judged pairs of each label, drawn with a fixed seed, for
  measuring agreement with the human judgments.
Each pair is judged twice: with a short prompt (the three label definitions from WANDS) and with a
longer one (guidelines in the spirit of WANDS's annotation guidelines). The model sees the text in
lib.store plus the attributes column, never product images.
Output, in examples/data/recorded/llm-judgments/: judgments.csv (query_id, product_id, pool, prompt,
label, grade), and calls.jsonl, every raw model reply.
"""
import numpy as np
import pandas as pd

from common import write_manifest
from lib.fusion import reciprocal_rank_fusion
from lib.lexical import LexicalSearch
from lib.recorded import cross_encoder_scores
from lib.semantic import SemanticSearch
from lib.store import load_labels, load_products, load_queries
from llm import LICENCE, MODEL, Recorder, describe, settings

NAME = "llm-judgments"
GRADES = {"exact": 2, "partial": 1, "irrelevant": 0}
SCHEMA = {"type": "object", "properties": {"label": {"type": "string", "enum": list(GRADES)}}, "required": ["label"]}
DEFINITIONS = ('Exact: the product fully matches the search query.\n'
               'Partial: the product only matches the target entity of the query, but does not satisfy the modifiers for the query.\n'
               'Irrelevant: the product is not relevant to the query.')
PROMPTS = {
    "short": ("You judge product search results for an online furniture and home-goods store.\n\n" + DEFINITIONS +
              "\n\nQuery: {query}\n\nProduct:\n{product}\n\nAnswer with the label in JSON."),
    "guidelines": ("You judge product search results for an online furniture and home-goods store. A shopper typed the "
                   "query and was shown the product. Decide how well the product matches what the shopper asked for.\n\n"
                   + DEFINITIONS +
                   "\n\nGuidelines:\n"
                   "- First find the target entity: the kind of product the query asks for (a sofa, a lamp, a rug).\n"
                   "- Then find the modifiers: everything else the query asks for, such as colour, size, material, style, "
                   "brand or a feature.\n"
                   "- Exact needs the right kind of product and every modifier. Partial is the right kind of product with "
                   "at least one modifier missing or contradicted. Irrelevant is the wrong kind of product, even when it "
                   "shares words with the query, such as an accessory for the product asked for.\n"
                   "- Read misspelled queries as the shopper meant them.\n"
                   "- Judge only from the product's text and attributes. If a modifier isn't mentioned, decide from what "
                   "is written, and don't assume it is met.\n\n"
                   "Query: {query}\n\nProduct:\n{product}\n\nAnswer with the label in JSON."),
}
TOP, PER_LABEL, SEED = 10, 4, 2026


def test_pool(products: pd.DataFrame, test: pd.DataFrame) -> list[tuple[int, int]]:
    bm25 = LexicalSearch(products).run(test)
    semantic = SemanticSearch().run(test)
    hybrid = reciprocal_rank_fusion([bm25, semantic])
    ce = cross_encoder_scores().set_index(["query_id", "product_id"])["score"]
    reranked = {q: sorted(r[:50], key=lambda p: -ce.get((q, p), float("-inf"))) for q, r in hybrid.items()}
    pairs = []
    for qid in test.index:
        seen = dict.fromkeys(p for run in (bm25, semantic, hybrid, reranked) for p in run[qid][:TOP])
        pairs += [(qid, p) for p in seen]
    return pairs


def train_pool(train: pd.DataFrame) -> list[tuple[int, int]]:
    labels = load_labels()
    labels = labels[labels["query_id"].isin(train.index)]
    rng = np.random.default_rng(SEED)
    pairs = []
    for (qid, _), group in labels.groupby(["query_id", "label"]):
        take = group.iloc[rng.permutation(len(group))[:PER_LABEL]]
        pairs += [(qid, p) for p in take["product_id"]]
    return sorted(pairs)


def main() -> None:
    products, queries = load_products(), load_queries()
    recorder = Recorder(NAME)
    pools = [("test", test_pool(products, load_queries("test"))), ("train", train_pool(load_queries("train")))]
    rows = []
    for pool, pairs in pools:
        for qid, pid in pairs:
            for prompt, template in PROMPTS.items():
                text = template.format(query=queries.at[qid, "query"], product=describe(products.loc[pid]))
                answer = recorder.json(f"{prompt}:{qid}:{pid}", [{"role": "user", "content": text}], SCHEMA)
                label = answer.get("label") if answer else None
                rows.append({"query_id": qid, "product_id": pid, "pool": pool, "prompt": prompt,
                             "label": label, "grade": GRADES.get(label)})
        print(f"{pool}: {len(pairs)} pairs judged")
    out = pd.DataFrame(rows).astype({"grade": "Int64"})
    out.to_csv(recorder.path.parent / "judgments.csv", index=False)
    write_manifest(NAME, "llm_judgments.py", MODEL, recorder.digest, PROMPTS, ["judgments.csv", "calls.jsonl"], LICENCE,
                   **settings({"label": SCHEMA}), pairs=int(len(out) / len(PROMPTS)),
                   invalid=int(out["label"].isna().sum()), sees="lib.store text plus attributes; no images")
    print(f"judgments: {len(out):,}  invalid: {int(out['label'].isna().sum())}")


if __name__ == "__main__":
    main()
