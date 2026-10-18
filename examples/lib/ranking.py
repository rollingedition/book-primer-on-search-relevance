"""Ranking, as chapter 9 builds it: features for each candidate, rankers learned with LightGBM, reranking a run
with recorded scores, and the size of what each reranker reads.

A query's candidates are the pairs that recordings/cross_encoder.py scored: the union of BM25's top 100
(lib.lexical) and the recorded embeddings' top 100, so every candidate has a cross-encoder score. The recording
ranked by the stored vectors before scaling them to length 1, so for two queries its 100th semantic product
differs from lib.semantic's, a near-tie. Every learned ranker is deterministic: one thread, LightGBM's
deterministic mode and a fixed seed. Chapter 9 fixes its planned test comparisons in PLANNED_TEST_COMPARISONS
before reading any test result, and reads them with report_planned.
"""
import lightgbm
import numpy as np
import pandas as pd

from .lexical import LexicalSearch
from .lexical_by_hand import tokenize
from .metrics import evaluate
from .recorded import cross_encoder_scores
from .semantic import SemanticSearch
from .significance import compare, report, verdict
from .store import load_labels

# Chapter 9's planned comparisons on the test queries, each against semantic retrieval's run.
PLANNED_TEST_COMPARISONS = ["lambdamart", "cross-encoder", "llm pointwise", "llm listwise", "distilled"]
THRESHOLD = 0.05 / len(PLANNED_TEST_COMPARISONS)        # Bonferroni: 0.01
SEED = 2026
DEPTH = 20                                              # rerank depth: the 20 products the LLM recording saw

FEATURES = ["bm25", "similarity", "cross_encoder", "name_match", "attribute_match",
            "rating_count", "average_rating", "review_count"]
CHEAP_FEATURES = [name for name in FEATURES if name != "cross_encoder"]     # no model call at query time

# Tree size and learning rate, chosen from TREE_GRID by cross-validation on the train queries (chapter 9). Early
# stopping decides how many of the 500 trees to keep.
TREE_GRID = [{"num_leaves": leaves, "learning_rate": rate} for leaves in (3, 7, 15, 31) for rate in (0.05, 0.1)]
TREES = {"n_estimators": 500, "num_leaves": 3, "learning_rate": 0.1}
DETERMINISTIC = {"random_state": SEED, "n_jobs": 1, "deterministic": True, "force_row_wise": True, "verbose": -1}
LAMBDARANK = {"objective": "lambdarank", "label_gain": [0, 1, 2]}           # linear gains, as lib.metrics uses
STOPPING_ROUNDS = 30                    # stop when nDCG@10 (or l2) on held-out train queries stops improving
STOPPING_QUERIES = 10                   # train queries held out of fitting, for early stopping


def feature_table(products: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    """One row per candidate: query_id, product_id, the FEATURES, and grade, the human grade (missing when nobody
    judged the pair). Rows are sorted by query, then by product_id: the recording lists each query's candidates by
    descending cross-encoder score, and that order must not reach a model, which reads its first ranking from it. bm25 is 0 for a product that shares no term with the query; name_match and attribute_match
    are the shares of the query's terms (lib.lexical_by_hand.tokenize) found in the name and in the attributes."""
    lexical, semantic = LexicalSearch(products), SemanticSearch()
    pairs = cross_encoder_scores().rename(columns={"score": "cross_encoder"})
    tables = []
    for query_id, table in pairs[pairs["query_id"].isin(queries.index)].groupby("query_id"):
        query, ids = queries.at[query_id, "query"], table["product_id"]
        terms = set(tokenize(query))
        share = lambda texts: [len(terms & set(tokenize(text))) / len(terms) for text in texts]
        tables.append(table.assign(
            bm25=lexical.search(query, k=len(products)).reindex(ids, fill_value=0.0).to_numpy(),
            similarity=semantic.search(query_id, k=len(products)).reindex(ids).to_numpy(),
            name_match=share(products.loc[ids, "name"]), attribute_match=share(products.loc[ids, "attributes"]),
            **{column: products.loc[ids, column].to_numpy() for column in FEATURES[5:]}))
    table = pd.concat(tables).sort_values(["query_id", "product_id"], ignore_index=True)   # an order that says nothing
    grades = load_labels().set_index(["query_id", "product_id"])["grade"]
    table["grade"] = grades.reindex(pd.MultiIndex.from_frame(table[["query_id", "product_id"]])).to_numpy()
    return table[["query_id", "product_id", *FEATURES, "grade"]]


def stopping_split(query_ids, size: int = STOPPING_QUERIES, seed: int = SEED) -> tuple[list, list]:
    """(fit, stop): the query IDs to fit on, and `size` drawn at random to hold out for early stopping."""
    stop = set(np.random.default_rng(seed).choice(sorted(query_ids), size=size, replace=False).tolist())
    return [q for q in sorted(query_ids) if q not in stop], sorted(stop)


def fit(model, table: pd.DataFrame, target: str, features: list, fit_ids, stop_ids):
    """Fit a LightGBM model on the rows of fit_ids, stopping early on the rows of stop_ids. Rows are sorted by
    query, and a ranker gets `group`, the number of rows of each query in that order."""
    parts = [table[table["query_id"].isin(ids)].sort_values("query_id", kind="stable") for ids in (fit_ids, stop_ids)]
    extra = {}
    if isinstance(model, lightgbm.LGBMRanker):
        sizes = [part.groupby("query_id", sort=True).size().to_numpy() for part in parts]
        extra = {"group": sizes[0], "eval_group": [sizes[1]], "eval_at": [10]}
    return model.fit(parts[0][features], parts[0][target], eval_X=parts[1][features], eval_y=parts[1][target],
                     callbacks=[lightgbm.early_stopping(STOPPING_ROUNDS, verbose=False)], **extra)


def cross_validate(make_model, training: pd.DataFrame, candidates: pd.DataFrame, target: str, features: list,
                   judgments: dict, folds: int = 5, seed: int = SEED) -> pd.Series:
    """nDCG@10 per query, each query ranked by a model that never saw it. The queries are split into `folds` folds
    at random; a fold's candidates are ranked by a model fitted on the rows of `training` from all but two folds,
    and stopped early on the next fold. `training` may hold fewer rows than `candidates`, such as judged ones only.
    The result's attrs["trees"] lists the trees early stopping kept in each fold."""
    split = np.array_split(np.random.default_rng(seed).permutation(sorted(candidates["query_id"].unique())), folds)
    scores, trees = [], []
    for number, scored in enumerate(split):
        stop = split[(number + 1) % folds]
        fit_ids = [q for part in split for q in part if q not in set(scored) | set(stop)]
        model = fit(make_model(), training, target, features, fit_ids, stop)
        trees.append(model.best_iteration_)
        part = candidates[candidates["query_id"].isin(scored)]
        run = ranked(part, model.predict(part[features]))
        scores.append(evaluate(run, {q: judgments[q] for q in scored})["ndcg@10"])
    result = pd.concat(scores).sort_index()
    result.attrs["trees"] = trees
    return result


def ranked(table: pd.DataFrame, scores) -> dict:
    """A run from one score per row of table: each query's candidates, highest score first, ties by product_id."""
    scored = table[["query_id", "product_id"]].assign(score=np.asarray(scores))
    return {query_id: list(group.sort_values(["score", "product_id"], ascending=[False, True])["product_id"])
            for query_id, group in scored.groupby("query_id")}


def rerank(run: dict, score, depth: int = DEPTH) -> dict:
    """Each ranking's top `depth` reordered by score[(query_id, product_id)], highest first, the rest kept in place.
    Equal scores keep the run's order."""
    return {query_id: sorted(ranking[:depth], key=lambda product_id: -score[(query_id, product_id)]) + ranking[depth:]
            for query_id, ranking in run.items()}


def listwise_line(product: pd.Series) -> str:
    """A product as the listwise prompt showed it (brief in recordings/llm_rerank.py): name, class, attributes and
    the first 160 characters of the description."""
    text = f"{product['name']} ({product['product_class']})"
    if product["attributes"]:
        text += f"; {product['attributes']}"
    return text + f"; {product['description'][:160]}"


def report_planned(name: str, after: pd.Series, before: pd.Series) -> dict:
    """One of this chapter's planned test comparisons, printed by lib.significance.report at its threshold."""
    return report(after, before, THRESHOLD, name=name, planned=PLANNED_TEST_COMPARISONS)
