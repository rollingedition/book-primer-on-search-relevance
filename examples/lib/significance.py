"""Paired comparisons of two runs, as chapter 3 builds them by hand, and the book's rule for reading them.

Both arguments are per-query scores for the same queries, such as one column of lib.metrics.evaluate: `after`
for the ranker you want to adopt and `before` for the one it must beat. Every difference is after minus before.
A difference is established when its permutation-test p-value is below the chapter's Bonferroni threshold;
the t-test and a bootstrap interval at the matching level are reported beside it (chapter 3). Resampling is
seeded, so the same scores always give the same result.
"""
import numpy as np
import pandas as pd
from scipy import stats

SEED = 2026


def mean_difference(after, before, axis=-1):
    """The statistic every test here uses: the mean of the per-query differences."""
    return np.mean(after - before, axis=axis)


def compare(after: pd.Series, before: pd.Series, seed: int = SEED, confidence: float = 0.95) -> dict:
    """Mean difference, its BCa bootstrap interval, the permutation and t-test p-values, the paired effect size,
    and per-query wins, losses and ties (differences rounded to 3 decimals)."""
    if not after.index.sort_values().equals(before.index.sort_values()):
        raise ValueError("after and before must score the same queries")
    before = before.reindex(after.index)                  # pair the scores by query_id
    after_scores, before_scores = after.to_numpy(float), before.to_numpy(float)
    difference = after_scores - before_scores
    rounded = difference.round(3)
    counts = {"wins": int((rounded > 0).sum()), "losses": int((rounded < 0).sum()), "ties": int((rounded == 0).sum())}
    if not difference.any():                              # identical scores: nothing to resample
        return {"mean difference": 0.0, "ci low": 0.0, "ci high": 0.0, "t-test p": 1.0, "permutation p": 1.0,
                "effect size": 0.0, **counts}
    spread = difference.std(ddof=1)
    interval = stats.bootstrap((difference,), np.mean, confidence_level=confidence,
                               rng=np.random.default_rng(seed)).confidence_interval
    permutation = stats.permutation_test((after_scores, before_scores), mean_difference,
                                         permutation_type="samples", rng=np.random.default_rng(seed))
    return {"mean difference": difference.mean(), "ci low": interval.low, "ci high": interval.high,
            "t-test p": stats.ttest_rel(after_scores, before_scores).pvalue, "permutation p": permutation.pvalue,
            "effect size": difference.mean() / spread if spread else 0.0, **counts}


def verdict(result: dict, threshold: float) -> dict:
    """The book's rule, applied to one result of compare, with intervals at the level 1 - threshold: whether the
    permutation test establishes the difference, and whether the t-test and the interval agree with it.
    Call it only for planned comparisons on test queries; anything else is exploratory, never established."""
    established = bool(result["permutation p"] < threshold)
    return {"established": established,
            "t-test agrees": bool(result["t-test p"] < threshold) == established,
            "interval agrees": bool(not result["ci low"] <= 0 <= result["ci high"]) == established}


def _shortest(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def report(after: pd.Series, before: pd.Series, threshold: float, name: str | None = None,
           planned: list | None = None) -> dict:
    """Print one planned test comparison by the book's rule, one labelled line per value, and return compare's
    result. name prefixes each label; planned, when given, is the chapter's list, and name must be in it."""
    if planned is not None and name not in planned:
        raise ValueError(f"{name} is not a planned comparison")
    result = compare(after, before, confidence=1 - threshold)
    checks, answer, prefix = verdict(result, threshold), {True: "yes", False: "no"}, f"{name}, " if name else ""
    level = _shortest((1 - threshold) * 100, 1) + "%"
    print(f"{prefix}mean difference: {result['mean difference']:.3f}")
    print(f"{prefix}wins, losses, ties: {result['wins']}, {result['losses']}, {result['ties']}")
    print(f"{prefix}permutation p: {result['permutation p']:.3f}")
    print(f"{prefix}established at {_shortest(threshold)}: {answer[checks['established']]}")
    print(f"{prefix}t-test p: {result['t-test p']:.3f}")
    print(f"{prefix}t-test agrees: {answer[checks['t-test agrees']]}")
    print(f"{prefix}{level} interval: {result['ci low']:.3f} to {result['ci high']:.3f}")
    print(f"{prefix}interval agrees: {answer[checks['interval agrees']]}")
    return result

