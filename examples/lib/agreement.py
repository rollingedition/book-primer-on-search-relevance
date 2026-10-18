"""LLM judgments next to the human ones, for chapter 4.

The LLM grades come from lib.recorded.llm_judgments(): one open model, two prompts ("short" and "guidelines"),
and two pools of (query, product) pairs. The "train" pool holds up to 4 human-judged pairs per label per train
query, for measuring agreement. The "test" pool holds every product in the top 10 of chapter 1's four runs for
each test query, judged by people or not. The manifest in data/recorded/llm-judgments/ holds both prompts.
"""
import pandas as pd

from .fusion import reciprocal_rank_fusion
from .lexical import LexicalSearch
from .recorded import cross_encoder_scores, llm_judgments
from .semantic import SemanticSearch
from .significance import compare, verdict
from .store import load_labels

PROMPTS = ["short", "guidelines"]
LABELS = {2: "exact", 1: "partial", 0: "irrelevant"}
PLANNED = [("semantic", "bm25"), ("hybrid", "semantic"), ("reranked", "hybrid")]   # chapter 3's: each stage against the last


def graded_pairs(pool: str) -> pd.DataFrame:
    """One row per pair in the pool: query_id, product_id, human (the human grade, missing when nobody judged
    the pair) and one column of LLM grades per prompt."""
    recorded = llm_judgments()
    recorded = recorded[recorded["pool"] == pool]
    pairs = recorded.pivot(index=["query_id", "product_id"], columns="prompt", values="grade")[PROMPTS]
    pairs.columns.name = None
    human = load_labels().set_index(["query_id", "product_id"])["grade"]
    pairs.insert(0, "human", human.reindex(pairs.index).astype("Int64"))
    return pairs.reset_index()


def llm_qrels(prompt: str, pool: str = "test") -> dict:
    """The LLM's grades for one prompt as {query_id: {product_id: grade}}, the form lib.metrics reads."""
    pairs = graded_pairs(pool)
    return {query_id: dict(zip(group["product_id"], group[prompt].astype(int)))
            for query_id, group in pairs.groupby("query_id")}


def fill_unjudged(human: dict, llm: dict) -> dict:
    """Human grades wherever people judged a pair, and the LLM's grade for the pairs nobody judged."""
    return {query_id: {**llm.get(query_id, {}), **judged} for query_id, judged in human.items()}


def chapter_one_runs(products: pd.DataFrame, queries: pd.DataFrame) -> dict:
    """Chapter 1's four runs, in the order chapter 1 builds them: BM25, semantic retrieval, their fusion with
    reciprocal rank fusion, and that fusion's top 50 reranked by the recorded cross-encoder."""
    bm25 = LexicalSearch(products).run(queries, k=100)
    semantic = SemanticSearch().run(queries, k=100)
    hybrid = reciprocal_rank_fusion([bm25, semantic])
    score = {(query_id, product_id): value for query_id, product_id, value in cross_encoder_scores().itertuples(index=False)}
    reranked = {query_id: sorted(ranking[:50], key=lambda product_id: score[(query_id, product_id)], reverse=True)
                + ranking[50:] for query_id, ranking in hybrid.items()}
    return {"bm25": bm25, "semantic": semantic, "hybrid": hybrid, "reranked": reranked}


def as_the_llm_saw(product: pd.Series) -> str:
    """A product as the recording showed it to the LLM (describe in recordings/llm.py): the text that search
    reads, plus the attributes. No image and no product page."""
    return "\n".join([f"Name: {product['name']}", f"Class: {product['product_class']}",
                      f"Category: {product['category']}", f"Description: {product['description']}",
                      f"Attributes: {product['attributes'] or 'none listed'}"])


def planned_verdicts(ndcg: dict, threshold: float) -> dict:
    """For each of chapter 3's planned comparisons, whether the book's rule establishes the difference. ndcg maps
    each ranker to its per-query scores; the threshold is 0.05 divided by the number of planned comparisons."""
    return {(after, before): verdict(compare(ndcg[after], ndcg[before], confidence=1 - threshold), threshold)["established"]
            for after, before in PLANNED}
