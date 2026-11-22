"""Draws the train nDCG@10 of semantic retrieval's top k, reranked by the recorded cross-encoder, for k from 5 to 75, with
semantic retrieval alone as a dashed line. The prose rests on each depth's printed nDCG@10 and the best depth with its value;
the printed values match rerank-depth-on-train."""
from lib.figures import ACCENT, MUTED, new_figure, save
from lib.metrics import evaluate
from lib.ranking import rerank
from lib.recorded import cross_encoder_scores
from lib.semantic import SemanticSearch
from lib.store import load_queries, qrels

queries, judgments = load_queries("train"), qrels("train")      # choosing a depth is tuning: train queries
semantic = SemanticSearch().run(queries, k=100)
recorded = cross_encoder_scores()
# the recorded cross-encoder score of each (query, product) pair
score = dict(zip(zip(recorded["query_id"], recorded["product_id"]), recorded["score"]))

depths = [5, 10, 20, 30, 50, 75]
alone = round(evaluate(semantic, judgments)["ndcg@10"].mean(), 3)
means = [round(evaluate(rerank(semantic, score, depth), judgments)["ndcg@10"].mean(), 3) for depth in depths]
best = depths[means.index(max(means))]

print(f"depth 0 (semantic alone), train ndcg@10: {alone:.3f}")
for depth, mean in zip(depths, means):
    print(f"depth {depth}, train ndcg@10: {mean:.3f}")
print(f"best depth: {best}")
print(f"best depth, train ndcg@10: {max(means):.3f}")

fig, ax = new_figure(height=2.8)
ax.plot(depths, means, color=ACCENT, marker="o", markersize=3.5, label="cross-encoder reranks semantic retrieval's top k")
ax.axhline(alone, color=MUTED, linestyle="--", label=f"semantic retrieval alone (depth 0), {alone:.3f}")
ax.set_xticks(depths)
ax.set_xlabel("products reranked")
ax.set_ylabel("train nDCG@10")
ax.legend(loc="lower right", frameon=False)
save(fig, "rerank-depth-curve")
