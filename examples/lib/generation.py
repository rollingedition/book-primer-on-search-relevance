"""Retrieval for LLM applications, as chapter 10 builds it: answers and agent picks scored with the human
judgments, citations checked against the context, and BM25 over other retrievable units.

The answers and agent traces come from lib.recorded.rag_answers() and agent_traces(): one open model, run once
by recordings/rag_answers.py and recordings/agent_trace.py, whose manifests hold the prompts and the tool.
Every score here uses the human judgments; a product nobody judged counts as not exact. Chapter 10 fixes its
planned test comparisons in PLANNED_TEST_COMPARISONS before reading any test result, and reads them with
report_planned.
"""
import re

import bm25s
import numpy as np
import pandas as pd

from .lexical import best_first, bm25_scores
from .significance import compare, report, verdict

# Chapter 10's planned comparisons on the test queries: semantic retrieval against BM25 behind the recorded
# answers (the exact share of what each answer cites) and behind the agent's search tool (precision@3 of its picks).
PLANNED_TEST_COMPARISONS = ["answers, semantic over bm25", "agent picks, semantic over bm25"]
THRESHOLD = 0.05 / len(PLANNED_TEST_COMPARISONS)        # Bonferroni: 0.025
EXACT = 2


# ---------------------------------------------------------------- answers
def exact_share(cited: list, judgments: dict) -> float:
    """The share of an answer's cited products that people judged exact. An answer that cites nothing scores 0:
    it gives the shopper no product that fits, whether or not that was the right call."""
    return sum(judgments.get(p, 0) == EXACT for p in cited) / len(cited) if cited else 0.0


def score_answers(answers: pd.DataFrame, qrels: dict) -> pd.DataFrame:
    """One row per recorded answer: query_id, context, the context's exact and unjudged products, how many
    products the answer cites, how many of those are exact, partial or unjudged, and its exact share."""
    rows = []
    for answer in answers.itertuples():
        judged = qrels[answer.query_id]
        rows.append({"query_id": answer.query_id, "context": answer.context,
                     "exact in context": sum(judged.get(p, 0) == EXACT for p in answer.context_ids),
                     "unjudged in context": sum(p not in judged for p in answer.context_ids),
                     "cited": len(answer.cited), "exact cited": sum(judged.get(p, 0) == EXACT for p in answer.cited),
                     "partial cited": sum(judged.get(p) == 1 for p in answer.cited),
                     "unjudged cited": sum(p not in judged for p in answer.cited),
                     "exact share": exact_share(answer.cited, judged)})
    return pd.DataFrame(rows)


def ids_in_text(text: str, candidates: list) -> set:
    """The candidate product IDs that an answer's text writes out as whole numbers."""
    return {p for p in candidates if re.search(rf"(?<!\d){p}(?!\d)", text)}


# ---------------------------------------------------------------- the context's measures
def context_precision(context: list, judged: dict, k: int = 5) -> float:
    """Ragas's context precision with the judges as the reference: the mean of precision@i over the positions i
    in the top k that hold an exact product; 0 when none does."""
    hits = [i for i, p in enumerate(context[:k], start=1) if judged.get(p, 0) == EXACT]
    return sum(sum(judged.get(q, 0) == EXACT for q in context[:i]) / i for i in hits) / len(hits) if hits else 0.0


def context_recall(context: list, judged: dict, k: int = 5) -> float:
    """Ragas's ID-based context recall: the share of the reference IDs, here the exact products, in the top k."""
    reference = {p for p, grade in judged.items() if grade == EXACT}
    return len(reference & set(context[:k])) / len(reference)


# ---------------------------------------------------------------- retrievable units
def sentences(text: str) -> list[str]:
    """A description cut after every full stop, question mark or exclamation mark that a space follows, keeping
    pieces with at least one letter or digit. Crude: an abbreviation such as "gal ." ends a sentence too, and the
    last piece of a description the sample cut short (data/wands/README.md) is a fragment."""
    return [piece for piece in re.split(r"(?<=[.!?])\s+", text) if re.search(r"\w", piece)]


class UnitSearch:
    """BM25 at bm25s's defaults, as lib.lexical, over any retrievable units, each belonging to one product. A
    product takes the place of its best-scoring unit, so a run still lists product IDs."""

    def __init__(self, texts: list[str], owners: list[int]):
        self.owners = np.asarray(owners)
        self.retriever = bm25s.BM25()
        self.retriever.index(bm25s.tokenize(list(texts), stopwords="en", show_progress=False), show_progress=False)

    def search(self, query: str, k: int = 5) -> list[int]:
        """The top k products, each scored by its best unit; equal scores by product_id."""
        scores = pd.Series(bm25_scores(self.retriever, query)).groupby(self.owners).max()
        top = best_first(scores.to_numpy(), scores.index.to_numpy(), k)
        return [int(p) for p in scores.index.to_numpy()[top]]

    def run(self, queries: pd.DataFrame, k: int = 5) -> dict:
        """Rankings for every query: {query_id: [product_id, ...]}."""
        return {query_id: self.search(query, k) for query_id, query in queries["query"].items()}


# ---------------------------------------------------------------- agent traces
def words(text: str) -> set[str]:
    """The lowercase words and numbers of a text."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def shown(product_ids: list, products: pd.DataFrame) -> str:
    """What the search tool returned for these products: name, class and attributes, and no description."""
    return " ".join(f"{products.at[p, 'name']} {products.at[p, 'product_class']} {products.at[p, 'attributes']}"
                    for p in product_ids)


# ---------------------------------------------------------------- the planned comparisons
def report_planned(name: str, after: pd.Series, before: pd.Series) -> dict:
    """One of this chapter's planned test comparisons, printed by lib.significance.report at its threshold."""
    return report(after, before, THRESHOLD, name=name, planned=PLANNED_TEST_COMPARISONS)
