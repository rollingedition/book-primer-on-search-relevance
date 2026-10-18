"""The furniture store's data: products, queries and relevance judgments from the WANDS sample.

See data/wands/README.md for where the data comes from and how the sample was drawn.
"""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "wands"
TEXT_COLUMNS = ["name", "product_class", "category", "description", "attributes"]


def load_products() -> pd.DataFrame:
    """One row per product, indexed by product_id. Missing text is an empty string."""
    products = pd.read_csv(DATA / "products.csv")
    products[TEXT_COLUMNS] = products[TEXT_COLUMNS].fillna("")
    return products.set_index("product_id")


def product_text(products: pd.DataFrame) -> pd.Series:
    """The text search reads for each product: name, class, category and description. The attributes column
    isn't part of it; chapter 5 adds attributes as a field of their own."""
    return (products["name"] + ". " + products["product_class"] + ". "
            + products["category"] + ". " + products["description"])


def load_queries(split: str | None = None) -> pd.DataFrame:
    """One row per query, indexed by query_id. split is "train", "test" or None for both."""
    queries = pd.read_csv(DATA / "queries.csv").set_index("query_id")
    return queries if split is None else queries[queries["split"] == split]


def load_labels() -> pd.DataFrame:
    """One row per judged (query, product) pair: label is exact, partial or irrelevant; grade is 2, 1 or 0."""
    return pd.read_csv(DATA / "labels.csv")


def qrels(split: str | None = None) -> dict[int, dict[int, int]]:
    """Judgments as {query_id: {product_id: grade}}, the form the metrics read."""
    labels = load_labels()
    if split is not None:
        labels = labels[labels["query_id"].isin(load_queries(split).index)]
    return {qid: dict(zip(group["product_id"], group["grade"])) for qid, group in labels.groupby("query_id")}
