"""Record LLM reranking of semantic retrieval's top 20 (W-48), for chapter 9.

    python recordings/llm_rerank.py

For every test query, the LLM (recordings/llm.py) reranks the top 20 products of lib.semantic, the
store's ranker after chapter 7, in two ways. Chapter 9 scores LLM rerankers on the test queries' human
judgments and tunes nothing with them, so train queries aren't recorded:
- listwise: all 20 products in one prompt, numbered, and the model returns their order. It runs twice,
  with the products in semantic order and in reversed order, to show whether the order shown changes the
  answer. An answer that isn't an ordering of all 20 is repaired: repeats are dropped and missing
  products follow in the order shown; `valid` records whether the answer needed repair.
- pointwise: each product alone, with a label and a score from 0 to 100.
Outputs, in examples/data/recorded/llm-rerank/: listwise.csv (query_id, order, position, product_id,
valid), pointwise.csv (query_id, product_id, label, score), and calls.jsonl.
"""
import pandas as pd

from common import write_manifest
from lib.semantic import SemanticSearch
from lib.store import load_products, load_queries
from llm import LICENCE, MODEL, Recorder, describe, settings

NAME = "llm-rerank"
DEPTH = 20
LISTWISE = ("You rank product search results for an online furniture and home-goods store.\n\nQuery: {query}\n\n"
            "Products:\n{products}\n\nOrder all {n} products from the best match for the query to the worst. Return the "
            "product numbers, each exactly once, in JSON.")
POINTWISE = ("You judge a product search result for an online furniture and home-goods store.\n\nQuery: {query}\n\n"
             "Product:\n{product}\n\nLabel it exact (matches the query fully), partial (the right kind of product, but "
             "a requirement isn't met) or irrelevant, and give a score from 0 to 100 for how well it matches. Answer in JSON.")
LIST_SCHEMA = {"type": "object", "properties": {"ranking": {"type": "array", "items": {"type": "integer"},
                                                            "minItems": DEPTH, "maxItems": DEPTH}}, "required": ["ranking"]}
POINT_SCHEMA = {"type": "object", "properties": {"label": {"type": "string", "enum": ["exact", "partial", "irrelevant"]},
                                                 "score": {"type": "integer", "minimum": 0, "maximum": 100}},
                "required": ["label", "score"]}


def brief(product: pd.Series) -> str:
    """A shorter description for the listwise prompt, which holds 20 products."""
    text = f"{product['name']} ({product['product_class']})"
    if product["attributes"]:
        text += f"; {product['attributes']}"
    return text + f"; {product['description'][:160]}"


def repair(answer: list[int] | None, n: int) -> tuple[list[int], bool]:
    """1-based positions of the products in the model's order, and whether the answer was already valid."""
    picked = [i for i in dict.fromkeys(answer or []) if 1 <= i <= n]
    valid = len(picked) == n and len(answer or []) == n
    return picked + [i for i in range(1, n + 1) if i not in picked], valid


def main() -> None:
    products, queries = load_products(), load_queries("test")
    run = SemanticSearch().run(queries, k=DEPTH)
    recorder = Recorder(NAME)
    listwise, pointwise = [], []
    for qid, query in queries["query"].items():
        for order in ("semantic", "reversed"):
            shown = run[qid] if order == "semantic" else run[qid][::-1]
            lines = "\n".join(f"[{i}] {brief(products.loc[p])}" for i, p in enumerate(shown, start=1))
            answer = recorder.json(f"listwise:{order}:{qid}",
                                   [{"role": "user", "content": LISTWISE.format(query=query, products=lines, n=len(shown))}],
                                   LIST_SCHEMA)
            positions, valid = repair((answer or {}).get("ranking"), len(shown))
            listwise += [{"query_id": qid, "order": order, "position": rank, "product_id": shown[i - 1], "valid": valid}
                         for rank, i in enumerate(positions, start=1)]
        for pid in run[qid]:
            answer = recorder.json(f"pointwise:{qid}:{pid}",
                                   [{"role": "user", "content": POINTWISE.format(query=query, product=describe(products.loc[pid]))}],
                                   POINT_SCHEMA) or {}
            pointwise.append({"query_id": qid, "product_id": pid, "label": answer.get("label"), "score": answer.get("score")})
    out = recorder.path.parent
    pd.DataFrame(listwise).to_csv(out / "listwise.csv", index=False)
    pd.DataFrame(pointwise).astype({"score": "Int64"}).to_csv(out / "pointwise.csv", index=False)
    invalid = pd.DataFrame(listwise).drop_duplicates(["query_id", "order"])["valid"].eq(False).sum()
    write_manifest(NAME, "llm_rerank.py", MODEL, recorder.digest, {"listwise": LISTWISE, "pointwise": POINTWISE},
                   ["listwise.csv", "pointwise.csv", "calls.jsonl"], LICENCE,
                   **settings({"listwise": LIST_SCHEMA, "pointwise": POINT_SCHEMA}), candidates=f"lib.semantic top {DEPTH}, test queries",
                   listwise_answers_repaired=int(invalid))
    print(f"listwise answers repaired: {invalid} of {2 * len(queries)}")


if __name__ == "__main__":
    main()
