"""Online tests rehearsed offline: team-draft interleaving and searches by simulated shoppers.

Chapter 11 builds team draft and one simulated shopper in examples first, then uses this module.
The simulated shoppers' probabilities are assumptions, not measurements, and they are kept here, in one place.
"""
import numpy as np
import pandas as pd

# Simulated shoppers follow a cascade click model: they read the results from the top, click a product with
# probability click[grade], and after a click leave with probability stop[grade]. Each simulated search is for
# a query drawn uniformly at random, as in the standard setup. Oosterhuis and de Rijke (CIKM 2018, Table 2) give
# three users for five grades, 0 to 4, with 10 results shown:
#   navigational: click 0.05, 0.3, 0.5, 0.7, 0.95; stop 0.2, 0.3, 0.5, 0.7, 0.9
#   perfect:      click 0.0, 0.2, 0.4, 0.8, 1.0;   stop 0 at every grade
# WANDS has three grades, so we assume that irrelevant, partial and exact products behave like those users'
# grades 0, 2 and 4. Unjudged products count as grade 0, as they do in the metrics.
SHOPPERS = {
    "navigational": {"click": {0: 0.05, 1: 0.5, 2: 0.95}, "stop": {0: 0.2, 1: 0.5, 2: 0.9}},
    "perfect": {"click": {0: 0.0, 1: 0.4, 2: 1.0}, "stop": {0: 0.0, 1: 0.0, 2: 0.0}},
}
NAVIGATIONAL = SHOPPERS["navigational"]      # the default shopper in every simulation
RESULTS_SHOWN = 10


def team_draft(ranking_a: list, ranking_b: list, rng: np.random.Generator, length: int = RESULTS_SHOWN) -> tuple[list, list]:
    """Team-draft interleaving (Radlinski, Kurup and Joachims, CIKM 2008, Algorithm 2): the interleaved list,
    and for each position the team that picked it, "a" or "b"."""
    shown, teams = [], []
    while len(shown) < length:
        picks_a, picks_b = teams.count("a"), teams.count("b")
        a_picks = picks_a < picks_b or (picks_a == picks_b and rng.random() < 0.5)    # coin flip when level
        ranking, team = (ranking_a, "a") if a_picks else (ranking_b, "b")
        product_id = next((p for p in ranking if p not in shown), None)           # the team's best unshown product
        if product_id is None:
            break
        shown.append(product_id)
        teams.append(team)
    return shown, teams


def cascade_clicks(ranking: list, judgments: dict, rng: np.random.Generator, shopper: dict = NAVIGATIONAL) -> list[int]:
    """Positions, from 0, that one simulated shopper clicks in the top RESULTS_SHOWN of a ranking."""
    clicks = []
    for position, product_id in enumerate(ranking[:RESULTS_SHOWN]):
        grade = judgments.get(product_id, 0)
        if rng.random() < shopper["click"][grade]:
            clicks.append(position)
            if rng.random() < shopper["stop"][grade]:
                break
    return clicks


def interleaving_outcome(ranking_a: list, ranking_b: list, judgments: dict, rng: np.random.Generator,
                         shopper: dict = NAVIGATIONAL) -> int:
    """One interleaved search: 1 if ranker a's picks get more clicks, -1 if ranker b's do, 0 for a tie."""
    shown, teams = team_draft(ranking_a, ranking_b, rng)
    credited = [teams[position] for position in cascade_clicks(shown, judgments, rng, shopper)]
    return int(np.sign(credited.count("a") - credited.count("b")))


def simulate_interleaving(run_a: dict, run_b: dict, qrels: dict, searches: int, rng: np.random.Generator,
                          shopper: dict = NAVIGATIONAL) -> pd.Series:
    """The outcome of each of `searches` interleaved searches, each for a query drawn uniformly from qrels."""
    query_ids = rng.choice(sorted(qrels), size=searches)
    return pd.Series([interleaving_outcome(run_a[q], run_b[q], qrels[q], rng, shopper) for q in query_ids], name="outcome")


def simulate_searches(run: dict, qrels: dict, searches: int, rng: np.random.Generator,
                      shopper: dict = NAVIGATIONAL) -> pd.DataFrame:
    """One row per simulated search on one ranker, as in one group of an A/B test: the query, the number of
    clicks, and whether the shopper clicked an exact product. Queries are drawn uniformly from qrels."""
    rows = []
    for query_id in rng.choice(sorted(qrels), size=searches):
        ranking, judgments = run[query_id], qrels[query_id]
        clicks = cascade_clicks(ranking, judgments, rng, shopper)
        exact = any(judgments.get(ranking[position], 0) == 2 for position in clicks)
        rows.append({"query_id": query_id, "clicks": len(clicks), "exact_click": exact})
    return pd.DataFrame(rows)
