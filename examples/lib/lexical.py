"""BM25 search over the store's products with the bm25s library. Chapter 5 builds the same scoring by hand.

Every product is scored, and products with equal scores are ordered by product_id, so a ranking never
depends on the order a library's top-k selection happens to leave ties in (the plan's rule on order leaks).
"""
import bm25s
import numpy as np
import pandas as pd

from .store import product_text


def bm25_scores(retriever: bm25s.BM25, query: str) -> np.ndarray:
    """The BM25 score of every indexed item for a query, in index order. Words the index never saw score 0."""
    words = bm25s.tokenize([query], stopwords="en", return_ids=False, show_progress=False)[0]
    known = [w for w in words if w in retriever.vocab_dict]
    return retriever.get_scores(known) if known else np.zeros(retriever.scores["num_docs"])


def best_first(scores: np.ndarray, ids: np.ndarray, k: int) -> np.ndarray:
    """Positions of the top k positive scores, highest first, equal scores in ascending id order."""
    order = np.lexsort((ids, -scores))
    return order[scores[order] > 0][:k]


class LexicalSearch:
    def __init__(self, products: pd.DataFrame, k1: float = 1.5, b: float = 0.75):
        self.ids = products.index.to_numpy()
        self.retriever = bm25s.BM25(k1=k1, b=b)
        self.retriever.index(bm25s.tokenize(product_text(products).tolist(), stopwords="en", show_progress=False),
                             show_progress=False)

    def search(self, query: str, k: int = 100) -> pd.Series:
        """Scores of the top k products for a query, best first, indexed by product_id; ties by product_id."""
        scores = bm25_scores(self.retriever, query)
        top = best_first(scores, self.ids, k)
        return pd.Series(scores[top], index=self.ids[top], name="bm25")

    def run(self, queries: pd.DataFrame, k: int = 100) -> dict:
        """Rankings for every query, in the form the metrics read: {query_id: [product_id, ...]}."""
        return {qid: list(self.search(q, k).index) for qid, q in queries["query"].items()}
