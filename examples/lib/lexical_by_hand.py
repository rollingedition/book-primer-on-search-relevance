"""Lexical retrieval built by hand, as chapter 5 builds it: terms, an inverted index, BM25 and term positions.

BM25 here is the variant bm25s uses by default (method="lucene"). Chapter 5 checks that the scores match
bm25s's to float32 rounding. Equal scores are ordered by product_id, so tied products can come out in a
different order from bm25s's. The book's shared baseline stays lib.lexical, unstemmed; stemmed BM25 is
InvertedIndex(text, stem=True).
"""
import math
import re
from collections import Counter

import pandas as pd
import Stemmer
from bm25s.stopwords import STOPWORDS_EN

from .significance import compare, report, verdict

# Chapter 5's planned comparisons on the test queries, fixed before any test result is read. A difference is
# established when its paired permutation test's p-value is below THRESHOLD (the book's rule, chapter 3).
PLANNED_TEST_COMPARISONS = ["stemming", "k1 and b", "name weight", "attributes field", "phrase bonus"]
THRESHOLD = 0.05 / len(PLANNED_TEST_COMPARISONS)       # Bonferroni: 0.01

TOKEN = re.compile(r"(?u)\b\w\w+\b")      # two or more letters or digits: bm25s's default pattern
STOPWORDS = set(STOPWORDS_EN)               # the 33 English stop words bm25s removes by default
STEMMER = Stemmer.Stemmer("english")        # Snowball English (Porter2), from PyStemmer


def tokenize(text: str, stem: bool = False) -> list[str]:
    """The terms of a text: lowercased, split into tokens, stop words removed, and stemmed if asked."""
    tokens = [token for token in TOKEN.findall(text.lower()) if token not in STOPWORDS]
    return STEMMER.stemWords(tokens) if stem else tokens


class InvertedIndex:
    """For each term, the products that contain it and how often: postings[term][product_id] = count."""

    def __init__(self, texts: pd.Series, stem: bool = False):
        self.stem = stem
        self.postings, self.lengths = {}, {}          # lengths: {product_id: number of terms}
        for product_id, text in texts.items():
            terms = tokenize(text, stem)
            self.lengths[product_id] = len(terms)
            for term, count in Counter(terms).items():
                self.postings.setdefault(term, {})[product_id] = count
        self.average_length = sum(self.lengths.values()) / len(self.lengths)

    def bm25(self, query: str, k1: float = 1.5, b: float = 0.75) -> dict:
        """{product_id: BM25 score} for every product that contains at least one of the query's terms."""
        n, scores = len(self.lengths), {}
        for term in tokenize(query, self.stem):
            postings = self.postings.get(term, {})
            idf = math.log(1 + (n - len(postings) + 0.5) / (len(postings) + 0.5))
            for product_id, tf in postings.items():
                length = 1 - b + b * self.lengths[product_id] / self.average_length
                scores[product_id] = scores.get(product_id, 0.0) + idf * tf / (tf + k1 * length)
        return scores

    def search(self, query: str, k: int = 100, k1: float = 1.5, b: float = 0.75) -> pd.Series:
        """Scores of the top k products, best first, indexed by product_id."""
        scores = self.bm25(query, k1, b)
        top = sorted(scores, key=lambda product_id: (-scores[product_id], product_id))[:k]
        return pd.Series([scores[product_id] for product_id in top], index=top, dtype=float, name="bm25")

    def run(self, queries: pd.DataFrame, k: int = 100, k1: float = 1.5, b: float = 0.75) -> dict:
        """Rankings for every query: {query_id: [product_id, ...]}."""
        return {query_id: list(self.search(query, k, k1, b).index) for query_id, query in queries["query"].items()}


class PositionalIndex:
    """Where each term occurs in each text: positions[term][product_id] = [position, ...], counting from 0.
    Positions count terms after stop words are removed, so "door for laundry" holds the adjacent pair door, laundry."""

    def __init__(self, texts: pd.Series, stem: bool = False):
        self.positions = {}
        for product_id, text in texts.items():
            for place, term in enumerate(tokenize(text, stem)):
                self.positions.setdefault(term, {}).setdefault(product_id, []).append(place)

    def has_phrase(self, product_id, terms: list[str]) -> bool:
        """True if the terms occur next to each other, in this order, in the product's text."""
        def places(term):
            return self.positions.get(term, {}).get(product_id, [])
        return any(all(start + offset in places(term) for offset, term in enumerate(terms)) for start in places(terms[0]))


def report_test_comparison(after: pd.Series, before: pd.Series) -> dict:
    """One planned test comparison, printed by lib.significance.report at this chapter's threshold."""
    return report(after, before, THRESHOLD)
