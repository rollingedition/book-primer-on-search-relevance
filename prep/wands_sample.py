"""Build the book's sample of WANDS, the Wayfair ANnotation DataSet, from a pinned commit.

    python prep/wands_sample.py [--source DIR]

Reads product.csv, query.csv and label.csv from WANDS (MIT licence) at the commit below, downloading them
unless --source points at a local copy, and writes examples/data/wands/: products.csv, queries.csv,
labels.csv and README.md, the data card. The same commit and seed always give the same files.

How the sample is drawn:
- queries: those with 60 to 300 judged products, at least 3 judged exact and 20 judged partial; at most
  one query per query class, so the set spans many kinds of product; 80 of them, chosen with a fixed seed.
  Queries that name the store, its own brands or a person are swapped for the next query in the same
  seeded order, which takes the swapped query's split
- split: 50 training queries (for tuning and learning to rank) and 30 test queries (for reporting)
- products: every product judged for a chosen query, plus randomly chosen unjudged products, up to 12,000
- text: descriptions are cut to about 400 characters at a word boundary, to keep the files small
- attributes: from WANDS's product_features (thousands of keys, mostly warranties and compliance), a short
  list of shopping attributes: colour, material, style, finish, shape, size and overall dimensions
- judgments: WANDS repeats some (query, product) pairs. Repeats with the same label become one row; a
  pair judged twice with different labels keeps the lower grade, and label_conflicts.csv lists both labels
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

WANDS_REPO = "https://github.com/wayfair/WANDS"
WANDS_COMMIT = "3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5"
RAW = f"https://raw.githubusercontent.com/wayfair/WANDS/{WANDS_COMMIT}/dataset/{{name}}"
SEED = 2026
N_QUERIES, N_TRAIN, N_PRODUCTS = 80, 50, 12_000
LABELS_PER_QUERY = (60, 300)
MIN_EXACT, MIN_PARTIAL = 3, 20
DESCRIPTION_CHARS = 400
GRADES = {"exact": 2, "partial": 1, "irrelevant": 0}
# (attribute, the product_features keys it comes from, first one present wins)
ATTRIBUTES = [("color", ["color"]), ("material", ["primarymaterial", "material"]), ("style", ["style", "dsprimaryproductstyle"]),
              ("finish", ["finish"]), ("shape", ["shape"]), ("size", ["size", "mattresssize", "rugsize"]),
              ("width", ["overallwidth-sidetoside"]), ("height", ["overallheight-toptobottom"]), ("depth", ["overalldepth-fronttoback"])]
# Queries the book never uses: the book doesn't name the store or people (plan, running case).
EXCLUDE = {54: "names the store", 108: "names a person", 137: "one of the store's own brands", 233: "names a person"}
OUT = Path(__file__).resolve().parent.parent / "examples" / "data" / "wands"


def read(name: str, source: Path | None) -> pd.DataFrame:
    if source is None:
        with urllib.request.urlopen(RAW.format(name=name), timeout=120) as r:
            return pd.read_csv(r, sep="\t")
    return pd.read_csv(source / name, sep="\t")


def trim(text: object, limit: int = DESCRIPTION_CHARS) -> str:
    text = " ".join(str(text).split()) if isinstance(text, str) else ""
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + " ..."


def attributes(features: object) -> str:
    """'color: clear; material: polyester; width: 70' from WANDS's 'key : value|key : value' features."""
    pairs = {}
    for item in str(features).split("|") if isinstance(features, str) else []:
        key, sep, value = item.partition(":")
        if sep and key.strip() and value.strip():
            pairs.setdefault(key.strip(), " ".join(value.split()))
    found = []
    for name, keys in ATTRIBUTES:
        value = next((pairs[k] for k in keys if k in pairs), None)
        if value:
            found.append(f"{name}: {value}")
    return "; ".join(found)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, help="a local folder holding WANDS's product.csv, query.csv and label.csv")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    products, queries, labels = (read(n, args.source) for n in ("product.csv", "query.csv", "label.csv"))
    labels["label"] = labels["label"].str.lower()

    counts = labels.pivot_table(index="query_id", columns="label", values="product_id", aggfunc="count", fill_value=0)
    counts["total"] = counts.sum(axis=1)
    lo, hi = LABELS_PER_QUERY
    eligible = counts[(counts.total.between(lo, hi)) & (counts.exact >= MIN_EXACT) & (counts.partial >= MIN_PARTIAL)]
    pool = queries[queries.query_id.isin(eligible.index)]
    # one query per query class: shuffle with the seed, keep each class's first query, then draw N_QUERIES
    pool = pool.sample(frac=1, random_state=SEED).drop_duplicates("query_class")
    order = pool.sample(frac=1, random_state=SEED)            # the order sample(n=N_QUERIES) draws in
    first = order.head(N_QUERIES).sort_values("query_id")
    train_ids = set(rng.choice(first.query_id.to_numpy(), size=N_TRAIN, replace=False).tolist())
    split = {qid: ("train" if qid in train_ids else "test") for qid in first.query_id}
    spare = [qid for qid in order.query_id.iloc[N_QUERIES:] if qid not in EXCLUDE]
    for qid in [q for q in first.query_id if q in EXCLUDE]:
        split[spare.pop(0)] = split.pop(qid)                  # the replacement keeps the split
    chosen = order[order.query_id.isin(split)].assign(split=lambda d: d.query_id.map(split)).sort_values("query_id")

    judged = labels[labels.query_id.isin(chosen.query_id)].drop_duplicates(["query_id", "product_id", "label"]).copy()
    judged["grade"] = judged.label.map(GRADES)
    seen = judged.groupby(["query_id", "product_id"]).label.agg(lambda x: "|".join(sorted(set(x))))
    conflicts = seen[seen.str.contains("|", regex=False)].rename("labels").reset_index()
    judged = judged.sort_values("grade").drop_duplicates(["query_id", "product_id"], keep="first")   # the lower grade
    judged_products = set(judged.product_id)
    others = products.loc[~products.product_id.isin(judged_products), "product_id"].to_numpy()
    extra = rng.choice(others, size=max(0, N_PRODUCTS - len(judged_products)), replace=False)
    keep = judged_products | set(extra.tolist())

    out_products = (products[products.product_id.isin(keep)]
                    .rename(columns={"product_name": "name", "category hierarchy": "category", "product_description": "description"})
                    .assign(description=lambda d: d.description.map(trim), category=lambda d: d.category.fillna(""),
                            attributes=lambda d: d.product_features.map(attributes))
                    [["product_id", "name", "product_class", "category", "description", "attributes",
                      "rating_count", "average_rating", "review_count"]]
                    .sort_values("product_id"))
    out_queries = chosen[["query_id", "query", "query_class", "split"]]
    out_labels = judged[["query_id", "product_id", "label", "grade"]].sort_values(["query_id", "product_id"])

    OUT.mkdir(parents=True, exist_ok=True)
    out_products.to_csv(OUT / "products.csv", index=False)
    out_queries.to_csv(OUT / "queries.csv", index=False)
    out_labels.to_csv(OUT / "labels.csv", index=False)
    conflicts.to_csv(OUT / "label_conflicts.csv", index=False)

    by_label = out_labels.label.value_counts()
    per_query = out_labels.groupby("query_id").size()
    (OUT / "README.md").write_text(f"""# WANDS sample

The data every example in this book reads: a sample of WANDS, the Wayfair ANnotation DataSet, a public
product-search dataset with human relevance judgments. Generated by `prep/wands_sample.py`; never edit
these files by hand.

**Source:** {WANDS_REPO}, commit `{WANDS_COMMIT}`.
**Licence:** MIT, Copyright (c) 2021 ecir2022. See `LICENSE` in the source repository.
**Paper:** Chen, Y., Liu, S., Liu, Z., Sun, W., Baltrunas, L. and Schroeder, B. (2022). WANDS: Dataset for
Product Search Relevance Assessment. ECIR 2022.

| File | Rows | What |
| --- | --- | --- |
| `products.csv` | {len(out_products):,} | `product_id`, `name`, `product_class`, `category` (a path such as "{out_products.category[out_products.category != ""].mode().iat[0]}"), `description` (cut to about {DESCRIPTION_CHARS} characters), `attributes` (see below), `rating_count`, `average_rating`, `review_count` |
| `queries.csv` | {len(out_queries)} | `query_id`, `query`, `query_class`, `split`: {N_TRAIN} `train` queries for tuning, {N_QUERIES - N_TRAIN} `test` queries for reporting |
| `labels.csv` | {len(out_labels):,} | `query_id`, `product_id`, `label` (`exact`, `partial` or `irrelevant`) and `grade` (2, 1 or 0): one row per pair |
| `label_conflicts.csv` | {len(conflicts)} | Pairs WANDS judged twice with different labels: `query_id`, `product_id`, `labels` (both, joined by `|`). `labels.csv` keeps the lower grade |

Labels: {by_label.get('exact', 0):,} exact, {by_label.get('partial', 0):,} partial, {by_label.get('irrelevant', 0):,} irrelevant. Judged products per query: {per_query.min()} to {per_query.max()}, median {int(per_query.median())}.

**Attributes:** WANDS's `product_features` holds thousands of keys, mostly warranties and compliance. `attributes`
keeps a few shopping attributes, as "name: value" pairs separated by "; ": {", ".join(a for a, _ in ATTRIBUTES)}. Values are as WANDS gives them;
it doesn't state units for the dimensions. {(out_products.attributes != "").sum():,} products have at least one attribute.

**How it was drawn** (seed {SEED}): queries with {lo} to {hi} judged products, at least {MIN_EXACT} exact and
{MIN_PARTIAL} partial judgments, at most one per query class; {N_QUERIES} of them, split {N_TRAIN} train and
{N_QUERIES - N_TRAIN} test. {len(EXCLUDE)} queries that name the store, its own brands or a person were swapped
for the next in the same order. Products: every product judged for a chosen query, plus randomly chosen unjudged
products to make {N_PRODUCTS:,}. A product that wasn't judged for a query is unjudged for it, not irrelevant.
""", encoding="utf-8")
    print(f"products: {len(out_products):,}  queries: {len(out_queries)} ({N_TRAIN} train)  labels: {len(out_labels):,}")


if __name__ == "__main__":
    main()
