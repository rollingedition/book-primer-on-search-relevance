"""Draws the hybrid's nDCG@10 minus BM25's on each test query as one bar, sorted from the largest loss to the largest gain. The prose's point rests on the mean and median difference lines, the win, loss and tie counts, and the largest-gain line, which per-query-hybrid-versus-bm25 prints the same."""
import numpy as np
from matplotlib.patches import Patch

from lib.figures import ACCENT, INK, MUTED, new_figure, save
from lib.fusion import reciprocal_rank_fusion
from lib.lexical import LexicalSearch
from lib.metrics import evaluate
from lib.semantic import SemanticSearch
from lib.store import load_products, load_queries, qrels

queries, judgments = load_queries("test"), qrels("test")
bm25 = LexicalSearch(load_products()).run(queries, k=100)
hybrid = reciprocal_rank_fusion([bm25, SemanticSearch().run(queries, k=100)])
before, after = evaluate(bm25, judgments), evaluate(hybrid, judgments)

# rounded to three decimals first, so a difference that rounds to 0 counts as a tie; sorted from the largest loss up
difference = (after["ndcg@10"] - before["ndcg@10"]).round(3).sort_values()
print(f"mean difference: {difference.mean():.3f}")
print(f"median difference: {difference.median():.3f}")
print("hybrid better:", (difference > 0).sum())
print("bm25 better:", (difference < 0).sum())
print("tied:", (difference == 0).sum())
print("largest gain:", queries.at[difference.index[-1], "query"])

values = difference.to_numpy()
colors = np.where(values > 0, ACCENT, MUTED)
fig, ax = new_figure(height=3.2)
ax.bar(range(len(values)), values, color=colors, width=0.8, linewidth=0)
ax.axhline(0, color=INK, linewidth=0.9)
ax.axhline(difference.mean(), color=INK, linestyle="--", linewidth=1.1,
           label=f"mean difference, {difference.mean():.3f}")
ax.legend(handles=[Patch(color=ACCENT, label="gain: hybrid better"), Patch(color=MUTED, label="loss: bm25 better"),
                   ax.lines[-1]], loc="upper left", frameon=False)
ax.set_xlim(-0.8, len(values) - 0.2)
ax.set_xticks([])
ax.set_xlabel(f"one bar per test query, sorted from loss to gain ({len(values)} queries)")
ax.set_ylabel("hybrid minus BM25, nDCG@10")
save(fig, "per-query-differences")
