"""Draws each query's share of exact, partial and irrelevant judgments as one stacked column, queries sorted by exact share.
The prose's point rests on the `lowest exact share`, `median exact share` and `highest exact share` lines, which are the
same numbers count-labels-per-query prints."""
import pandas as pd

from lib.figures import ACCENT, INK, MUTED, new_figure, save
from lib.store import load_labels

labels = load_labels()
counts = pd.crosstab(labels["query_id"], labels["label"])[["exact", "partial", "irrelevant"]]
shares = counts.div(counts.sum(axis=1), axis=0).sort_values("exact")

print(f"queries drawn: {len(shares)}")
print(f"median exact share: {shares['exact'].median():.3f}")
print(f"lowest exact share: {shares['exact'].min():.3f}")
print(f"highest exact share: {shares['exact'].max():.3f}")

fig, ax = new_figure(height=3.4)
columns = range(len(shares))
bottom = pd.Series(0.0, index=shares.index)
for label, color, hatch in [("exact", ACCENT, None), ("partial", MUTED, None), ("irrelevant", "none", "///")]:
    ax.bar(columns, shares[label], bottom=bottom, width=0.85, color=color, edgecolor=INK if hatch else "none",
           hatch=hatch, linewidth=0, label=label)
    bottom = bottom + shares[label]
ax.set_xticks([])
ax.set_xlim(-0.6, len(shares) - 0.4)
ax.set_ylim(0, 1)
ax.set_ylabel("share of the query's judgments")
ax.set_xlabel(f"each column is one query, sorted by its exact share ({len(shares)} queries)")
ax.legend(title=None, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
save(fig, "label-mix-per-query")
