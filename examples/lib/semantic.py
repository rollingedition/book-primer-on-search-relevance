"""Semantic search over the recorded embeddings, by exact dot product. Chapter 6 builds it step by step."""
import numpy as np
import pandas as pd

from .recorded import embeddings


class SemanticSearch:
    def __init__(self, dims: int | None = None):
        self.product_ids, self.products, query_ids, queries = embeddings(dims)
        self.query_vectors = dict(zip(query_ids.tolist(), queries))

    def search_vector(self, vector: np.ndarray, k: int = 100) -> pd.Series:
        """Similarity of the top k products to a vector, best first, indexed by product_id."""
        scores = self.products @ vector
        top = np.argsort(-scores, kind="stable")[:k]
        return pd.Series(scores[top], index=self.product_ids[top], name="similarity")

    def search(self, query_id: int, k: int = 100) -> pd.Series:
        """The same for a query in the data, using its recorded embedding."""
        return self.search_vector(self.query_vectors[query_id], k)

    def run(self, queries: pd.DataFrame, k: int = 100) -> dict:
        """Rankings for every query: {query_id: [product_id, ...]}."""
        return {qid: list(self.search(qid, k).index) for qid in queries.index}
