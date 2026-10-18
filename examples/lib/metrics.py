"""Ranking metrics, as chapter 3 builds them.

A ranking is a list of product IDs, best first. Judgments map product IDs to grades: 2 exact, 1 partial,
0 irrelevant. A product with no judgment counts as grade 0. Binary metrics count a product as relevant
when its grade is at least `relevant`, which is 2 (exact) unless you say otherwise.
"""
import math

import pandas as pd


def precision_at_k(ranking: list, judgments: dict, k: int, relevant: int = 2) -> float:
    return sum(judgments.get(p, 0) >= relevant for p in ranking[:k]) / k


def recall_at_k(ranking: list, judgments: dict, k: int, relevant: int = 2) -> float:
    total = sum(g >= relevant for g in judgments.values())
    return sum(judgments.get(p, 0) >= relevant for p in ranking[:k]) / total if total else 0.0


def reciprocal_rank(ranking: list, judgments: dict, relevant: int = 2) -> float:
    for position, p in enumerate(ranking, start=1):
        if judgments.get(p, 0) >= relevant:
            return 1 / position
    return 0.0


def dcg_at_k(grades: list, k: int) -> float:
    """Discounted cumulative gain: each grade divided by log2 of its position plus one."""
    return sum(g / math.log2(position + 1) for position, g in enumerate(grades[:k], start=1))


def ndcg_at_k(ranking: list, judgments: dict, k: int) -> float:
    """DCG of the ranking, divided by the DCG of the best possible ranking of the judged products."""
    ideal = dcg_at_k(sorted(judgments.values(), reverse=True), k)
    return dcg_at_k([judgments.get(p, 0) for p in ranking], k) / ideal if ideal else 0.0


def judged_at_k(ranking: list, judgments: dict, k: int) -> float:
    """The share of the top k that has any judgment. Low values mean the other metrics are guessing."""
    return sum(p in judgments for p in ranking[:k]) / k


def evaluate(run: dict, qrels: dict, k: int = 10) -> pd.DataFrame:
    """One row per query: precision, recall, reciprocal rank, nDCG and judged at k. run maps query IDs to rankings."""
    rows = []
    for qid, judgments in qrels.items():
        ranking = list(run.get(qid, []))
        rows.append({"query_id": qid, f"precision@{k}": precision_at_k(ranking, judgments, k),
                     f"recall@{k}": recall_at_k(ranking, judgments, k), "rr": reciprocal_rank(ranking, judgments),
                     f"ndcg@{k}": ndcg_at_k(ranking, judgments, k), f"judged@{k}": judged_at_k(ranking, judgments, k)})
    return pd.DataFrame(rows).set_index("query_id")
