"""Reciprocal rank fusion (Cormack, Clarke and Büttcher, SIGIR 2009), as chapter 1 writes it by hand.

Chapter 7 compares it with weighted sums of scores and tunes it on train queries.
"""


def rrf_scores(rankings: list[list], k: int = 60) -> dict:
    """Each product scores the sum of 1 / (k + rank) over the rankings that contain it."""
    scores = {}
    for ranking in rankings:
        for rank, product_id in enumerate(ranking, start=1):
            scores[product_id] = scores.get(product_id, 0) + 1 / (k + rank)
    return scores


def reciprocal_rank_fusion(runs: list[dict], k: int = 60, depth: int = 100) -> dict:
    """One run from several: for each query, products in order of their RRF scores, best first.
    Equal scores are ordered by product_id, so no input run is favoured on ties (the plan's rule on order leaks)."""
    fused = {}
    for query_id in runs[0]:
        scores = rrf_scores([run[query_id] for run in runs], k)
        fused[query_id] = sorted(scores, key=lambda product_id: (-scores[product_id], product_id))[:depth]
    return fused
