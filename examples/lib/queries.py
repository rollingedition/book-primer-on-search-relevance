"""Query understanding, as chapter 8 builds it: spelling, units, synonyms, query classes and follow-ups.

Every change here rewrites or scores the query before retrieval; the retrievers themselves are lib.lexical
(BM25, unstemmed, the book's baseline) and lib.semantic. Chapter 8 fixes its planned test comparisons in
PLANNED_TEST_COMPARISONS before reading any test result, and reads them with report_planned.
"""
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from .recorded import llm_rewrites
from .significance import compare, report, verdict
from .store import product_text

# Chapter 8's planned comparisons on the test queries, each against the same retriever on the query as typed.
PLANNED_TEST_COMPARISONS = ["synonyms, bm25", "class boost, semantic", "llm rewrite, bm25",
                            "llm rewrite, semantic", "hyde, semantic"]
THRESHOLD = 0.05 / len(PLANNED_TEST_COMPARISONS)        # Bonferroni: 0.01

# ---------------------------------------------------------------- spelling
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def word_counts(texts: pd.Series) -> Counter:
    """How often each lowercase word occurs in the texts: the corrector's lexicon and its P(c)."""
    return Counter(re.findall(r"[a-z]+", " ".join(texts).lower()))


def edits1(word: str) -> set[str]:
    """Every string one edit away: a deletion, a transposition, a replacement or an insertion."""
    splits = [(word[:cut], word[cut:]) for cut in range(len(word) + 1)]
    return ({left + right[1:] for left, right in splits if right}
            | {left + right[1] + right[0] + right[2:] for left, right in splits if len(right) > 1}
            | {left + letter + right[1:] for left, right in splits if right for letter in LETTERS}
            | {left + letter + right for left, right in splits for letter in LETTERS})


def correct_word(word: str, counts: Counter) -> str:
    """A known word stays. Otherwise the most frequent known word one edit away, else two, else the word."""
    if not word.isalpha() or word in counts:
        return word
    near = edits1(word)
    candidates = ({edit for edit in near if edit in counts}
                  or {two for edit in near for two in edits1(edit) if two in counts})
    return max(sorted(candidates), key=counts.get) if candidates else word


def correct_query(query: str, counts: Counter) -> str:
    return " ".join(correct_word(word, counts) for word in query.lower().split())


# ---------------------------------------------------------------- units
NUMBER = r"(\d+(?:\.\d+)?)"
INCH_SPELLINGS = ("inches", "inch", "in", "''", '"')


def unit_rules(inch_spellings=INCH_SPELLINGS) -> list[tuple[re.Pattern, str]]:
    """(pattern, unit) pairs: a number, then a spelling of quarts or of inches. "in" counts as inches only when
    no number or "one" follows it, so "3 in 1" stays as it is."""
    inches = "|".join(r"in\b(?!\s*(?:\d|one\b))" if spelling == "in" else re.escape(spelling)
                      for spelling in inch_spellings)
    return [(re.compile(NUMBER + r"\s*-?\s*(?:quarts|quart|qt)\b"), "qt"),
            (re.compile(NUMBER + r"\s*-?\s*(?:" + inches + ")"), "in")]


UNITS = unit_rules()


def normalize_units(text: str, rules=UNITS) -> str:
    """Lowercase, and join each number to its unit as one term: 7 qt -> 7qt, 48 '' -> 48in. A decimal point
    becomes an underscore, 3.7-quart -> 3_7qt, because the tokenizer splits words at a point but not at an
    underscore, and would otherwise read 3.7qt as 7qt."""
    text = text.lower()
    for pattern, unit in rules:
        text = pattern.sub(lambda match: match.group(1).replace(".", "_") + unit, text)
    return text


# ---------------------------------------------------------------- synonyms
# Built by hand from the train queries' exact products that miss a query word; never from test results.
SYNONYMS = [("sofa", "couch"), ("armoire", "wardrobe"), ("gray", "grey"), ("hamper", "basket"),
            ("jar", "canister"), ("baby", "infant"), ("kids", "toddler"), ("pad", "topper"), ("stand", "unit"),
            ("box", "chest"), ("bar stool", "barstool"), ("mid century", "midcentury"),
            ("free standing", "freestanding"), ("wall light", "sconce")]


def expand(query: str, groups=SYNONYMS) -> str:
    """The query followed by the other members of every synonym group that one of its words or phrases is in."""
    text, added = f" {query.lower()} ", []
    for group in groups:
        if any(f" {term} " in text for term in group):
            added += [term for term in group if f" {term} " not in text]
    return " ".join([query, *added])


# ---------------------------------------------------------------- query classes
def class_rows(products: pd.DataFrame) -> pd.DataFrame:
    """One row per (product, class) for products that have a class: a product can have several, joined by |."""
    rows = products[products["product_class"] != ""]
    return rows.assign(label=rows["product_class"].str.split("|")).explode("label")


def train_classifier(products: pd.DataFrame) -> tuple[CountVectorizer, MultinomialNB]:
    """Naive Bayes on word counts of product text, predicting product_class. Chapter 8 builds it by hand first."""
    rows = class_rows(products)
    vectorizer = CountVectorizer()
    return vectorizer, MultinomialNB().fit(vectorizer.fit_transform(product_text(rows)), rows["label"])


def in_class(products: pd.DataFrame, label: str) -> pd.Series:
    """True for each product whose classes include label."""
    return products["product_class"].str.split("|").apply(lambda labels: label in labels)


def class_boost_run(semantic, query_ids, predicted: dict, products: pd.DataFrame, weight: float | None,
                    k: int = 100) -> dict:
    """Semantic retrieval with weight added to the similarity of every product in the query's predicted class.
    weight=None keeps only those products: a class filter."""
    labels, run = products["product_class"].str.split("|"), {}
    for query_id in query_ids:
        scores = semantic.search(query_id, k=len(semantic.product_ids))
        member = labels.apply(lambda names: predicted[query_id] in names).reindex(scores.index)
        scores = scores[member.to_numpy()] if weight is None else scores + weight * member.astype(float)
        run[query_id] = list(scores.sort_values(ascending=False, kind="stable").index[:k])
    return run


# ---------------------------------------------------------------- follow-ups
def mentions(product_ids, value: str, texts: pd.Series) -> np.ndarray:
    """For each product, whether its text in texts contains the value as a word, such as "white" in
    "color: white". Pass products["attributes"] or product_text(products)."""
    pattern = re.compile(rf"\b{re.escape(value)}\b")
    return np.array([bool(pattern.search(texts[product_id].lower())) for product_id in product_ids], dtype=bool)


# ---------------------------------------------------------------- LLM rewrites
def rewrite_runs(queries: pd.DataFrame, lexical, semantic, k: int = 100) -> dict:
    """For "bm25" and "semantic": (the run on the LLM's recorded rewrite, the run on the query as typed).
    BM25 searches the rewrite's text; semantic retrieval searches its recorded embedding."""
    table, rewrite_vectors, _ = llm_rewrites()
    rewrites, vectors = table.set_index("query_id")["rewrite"], dict(zip(table["query_id"], rewrite_vectors))
    return {"bm25": (lexical.run(queries.assign(query=rewrites[queries.index].to_numpy()), k), lexical.run(queries, k)),
            "semantic": ({query_id: list(semantic.search_vector(vectors[query_id], k).index) for query_id in queries.index},
                         semantic.run(queries, k))}


def condensed(run: dict, qrels: dict) -> dict:
    """The run with unjudged products dropped, so the judged ones move up: chapter 2's diagnostic."""
    return {query_id: [product_id for product_id in ranking if product_id in qrels.get(query_id, {})]
            for query_id, ranking in run.items()}


# ---------------------------------------------------------------- the planned comparisons
def report_planned(name: str, after: pd.Series, before: pd.Series) -> dict:
    """One of this chapter's planned test comparisons, printed by lib.significance.report at its threshold."""
    return report(after, before, THRESHOLD, name=name, planned=PLANNED_TEST_COMPARISONS)
