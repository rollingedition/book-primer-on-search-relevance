"""Tests the hybrid's gap to semantic retrieval by flipping each query's sign at random and counting how often the flipped
mean lands as far from 0 as the observed one, then checks the count against SciPy. The prose rests on the `p-value by hand`, `p-value from scipy` and `difference at the 0.05 level` lines."""
import numpy as np
from scipy.stats import permutation_test

from lib.fusion import reciprocal_rank_fusion
from lib.lexical import LexicalSearch
from lib.metrics import evaluate
from lib.semantic import SemanticSearch
from lib.store import load_products, load_queries, qrels

queries, judgments = load_queries("test"), qrels("test")
semantic = SemanticSearch().run(queries, k=100)
hybrid = reciprocal_rank_fusion([LexicalSearch(load_products()).run(queries, k=100), semantic])
after = evaluate(hybrid, judgments)["ndcg@10"].to_numpy()
before = evaluate(semantic, judgments)["ndcg@10"].to_numpy()
difference = after - before                                       # hybrid minus semantic, per test query
print("test queries:", len(difference))
print(f"sign patterns for the test queries: {2 ** len(difference):,}")
print(f"observed mean difference: {difference.mean():.3f}")

# if the two rankers were interchangeable, each query's difference would be as likely to have the other sign
rng = np.random.default_rng(2026)                                 # fixed seed: the same p-value on every run
signs = rng.choice([-1, 1], size=(9999, len(difference)))
flipped = (signs * difference).mean(axis=1)                       # 9,999 mean differences under that assumption
print(f"95% of flipped means are smaller than: {np.quantile(np.abs(flipped), 0.95):.3f}")
print(f"p-value by hand: {np.mean(np.abs(flipped) >= abs(difference.mean())):.3f}")   # two-sided: as far from 0 in either direction

result = permutation_test((after, before), lambda after_scores, before_scores, axis: np.mean(after_scores - before_scores, axis=axis),   # SciPy passes the arrays and an axis
                          permutation_type="samples", rng=np.random.default_rng(2026))   # "samples" flips within each pair: the paired setting; SciPy draws its own patterns
print(f"p-value from scipy: {result.pvalue:.3f}")
print("difference at the 0.05 level:", "larger than chance" if result.pvalue < 0.05 else "within chance")
