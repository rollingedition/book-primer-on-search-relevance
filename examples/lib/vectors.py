"""Nearest-neighbour search over embeddings, as chapter 6 builds it.

Exact search is one matrix product and a partial sort, in NumPy. Approximate search uses faiss's HNSW
index, and its quality is the share of the exact top k it finds. All vectors are unit length, so the
inner product is the cosine similarity.
"""
import faiss
import numpy as np

EF_SEARCH_GRID = list(range(16, 257, 16))   # the efSearch values chapter 6 sweeps on train queries
WORST_RECALL = 0.9                          # chapter 6's rule: every query keeps 9 of its exact top 10


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Column indices of the k highest scores in each row, best first.
    argpartition finds the top k without ordering them; only those k are then sorted."""
    candidates = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, candidates, axis=1), axis=1, kind="stable")
    return np.take_along_axis(candidates, order, axis=1)


def exact_search(query_vectors: np.ndarray, product_vectors: np.ndarray, k: int) -> np.ndarray:
    """Row numbers of each query's k most similar products, by comparing it with every product."""
    return top_k(query_vectors @ product_vectors.T, k)


def as_run(found: np.ndarray, query_ids: np.ndarray, product_ids: np.ndarray) -> dict:
    """Row numbers from a search, one row per query, as a run: {query_id: [product_id, ...]}."""
    return {query_id: list(product_ids[rows]) for query_id, rows in zip(query_ids, found)}


def hnsw_index(product_vectors: np.ndarray, links: int = 32) -> faiss.IndexHNSWFlat:
    """An HNSW graph over the products, searched by inner product. links is what faiss calls M."""
    index = faiss.IndexHNSWFlat(product_vectors.shape[1], links, faiss.METRIC_INNER_PRODUCT)
    index.add(product_vectors)
    return index


def hnsw_search(index: faiss.IndexHNSWFlat, query_vectors: np.ndarray, k: int, ef_search: int) -> tuple[np.ndarray, float]:
    """(row numbers of the top k per query, similarities computed per query) at the given efSearch.
    The count comes from faiss's search statistics, so it is the same on every machine."""
    index.hnsw.efSearch = ef_search
    faiss.cvar.hnsw_stats.reset()
    _, rows = index.search(query_vectors, k)
    return rows, faiss.cvar.hnsw_stats.ndis / len(query_vectors)


def recall_against_exact(found: np.ndarray, exact: np.ndarray) -> np.ndarray:
    """Per query, the share of the exact top k that an approximate search also returned.
    This compares two searches; it says nothing about the judgments."""
    k = exact.shape[1]
    return np.array([len(set(approximate) & set(nearest)) / k for approximate, nearest in zip(found, exact)])


def choose_ef_search(index: faiss.IndexHNSWFlat, query_vectors: np.ndarray, product_vectors: np.ndarray,
                     k: int = 10, worst_recall: float = WORST_RECALL, grid: list = EF_SEARCH_GRID) -> int | None:
    """The smallest efSearch in grid at which every query finds at least worst_recall of its exact top k.
    Pass train queries: this is tuning."""
    exact = exact_search(query_vectors, product_vectors, k)
    for ef_search in grid:
        found, _ = hnsw_search(index, query_vectors, k, ef_search)
        if recall_against_exact(found, exact).min() >= worst_recall:
            return ef_search
    return None
