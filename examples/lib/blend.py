"""Hybrid retrieval: blending BM25 and semantic retrieval, as chapter 7 builds it.

Both blends take one weight, `alpha`, the share given to semantic retrieval: 0 is BM25 alone, 1 is
semantic retrieval alone. The weighted sum adds normalized scores over the candidate tables that
`candidates` builds: the union of each retriever's top `depth`, with both scores for every product.
Weighted reciprocal rank fusion adds 1 / (k + rank) over whatever runs it is given. In both, `depth`
is the length of the fused ranking returned.
"""
import pandas as pd

from .metrics import evaluate

ALPHAS = [round(0.1 * step, 1) for step in range(11)]    # the grid tuned on train queries


def candidates(lexical, semantic, queries: pd.DataFrame, depth: int = 100) -> dict[int, pd.DataFrame]:
    """For each query, the union of both retrievers' top `depth` products, with both scores for each.
    `bm25` is 0 for a product that shares no word with the query; `similarity` is its cosine."""
    tables = {}
    for query_id, query in queries["query"].items():
        bm25 = lexical.search(query, k=depth)
        similarity = semantic.search(query_id, k=depth)
        union = list(dict.fromkeys([*bm25.index, *similarity.index]))
        every_bm25 = lexical.search(query, k=len(semantic.product_ids))     # scores beyond the top depth
        every_similarity = semantic.search(query_id, k=len(semantic.product_ids))
        tables[query_id] = pd.DataFrame({"bm25": every_bm25.reindex(union, fill_value=0.0),
                                         "similarity": every_similarity.reindex(union)})
    return tables


def min_max(scores: pd.Series, floor: float | None = None) -> pd.Series:
    """(score - lowest) / (highest - lowest) over one query's candidates. With `floor`, the lowest
    possible score replaces the lowest seen (theoretical min-max): 0 for BM25, -1 for cosine."""
    lowest = scores.min() if floor is None else floor
    spread = scores.max() - lowest
    return (scores - lowest) / spread if spread > 0 else scores * 0.0


def z_score(scores: pd.Series) -> pd.Series:
    """(score - mean) / standard deviation over one query's candidates."""
    spread = scores.std()
    return (scores - scores.mean()) / spread if spread > 0 else scores * 0.0


NORMALIZATIONS = {
    "min-max": lambda scores, floor: min_max(scores),
    "theoretical min-max": lambda scores, floor: min_max(scores, floor),
    "z-score": lambda scores, floor: z_score(scores),
}
FLOORS = {"bm25": 0.0, "similarity": -1.0}               # the lowest score each retriever can give


def weighted_sum(tables: dict, alpha: float, normalization: str = "min-max", depth: int = 100) -> dict:
    """A run ranked by alpha * normalized similarity + (1 - alpha) * normalized BM25."""
    normalize = NORMALIZATIONS[normalization]
    run = {}
    for query_id, table in tables.items():
        blended = (alpha * normalize(table["similarity"], FLOORS["similarity"])
                   + (1 - alpha) * normalize(table["bm25"], FLOORS["bm25"]))
        ordered = blended.sort_index().sort_values(ascending=False, kind="stable")   # equal scores by product_id
        run[query_id] = list(ordered.index[:depth])
    return run


def weighted_rrf(bm25: dict, semantic: dict, alpha: float, k: int = 60, depth: int = 100) -> dict:
    """Reciprocal rank fusion with a weight: (1 - alpha) / (k + BM25 rank) + alpha / (k + semantic rank).
    alpha = 0.5 ranks exactly as unweighted RRF does (lib.fusion). Equal scores are ordered by product_id."""
    fused = {}
    for query_id in bm25:
        scores = {}
        for weight, run in [(1 - alpha, bm25), (alpha, semantic)]:
            for rank, product_id in enumerate(run[query_id], start=1):
                scores[product_id] = scores.get(product_id, 0) + weight / (k + rank)
        fused[query_id] = sorted(scores, key=lambda product_id: (-scores[product_id], product_id))[:depth]
    return fused


def scores_by_alpha(make_run, judgments: dict, alphas: list = ALPHAS) -> pd.DataFrame:
    """nDCG@10 per query (rows) for each alpha (columns). make_run(alpha) returns a run."""
    return pd.DataFrame({alpha: evaluate(make_run(alpha), judgments)["ndcg@10"] for alpha in alphas})


def best_alpha(per_query: pd.DataFrame) -> float:
    """The alpha with the highest mean nDCG@10. A tie goes to the smaller alpha."""
    return per_query.mean().idxmax()
