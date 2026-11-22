"""Draws the log discount nDCG applies to each position from 1 to 10, as points joined by a line. The prose's point rests on the discount lines at ranks 1, 2, 3 and 10, which use the same formula as ndcg-from-scratch."""
import math

import numpy as np

from lib.figures import ACCENT, MUTED, new_figure, save


def discount(position):
    """The factor each grade is multiplied by at its position, as in ndcg-from-scratch's dcg."""
    return 1 / math.log2(position + 1)


positions = np.arange(1, 11)
discounts = [discount(position) for position in positions]

# the same formula as dcg in ndcg-from-scratch, read at four positions
for position in (1, 2, 3, 10):
    print(f"discount at rank {position}: {discount(position):.3f}")

fig, ax = new_figure(height=2.8)
ax.plot(positions, discounts, color=ACCENT, marker="o", markersize=4, linewidth=1.4,
        label="discount 1 / log2(rank + 1)")
ax.axhline(1, color=MUTED, linestyle="--", linewidth=0.9, label="factor 1, what rank 1 keeps")
ax.set_xticks(positions)
ax.set_xlabel("position in the ranking, 1 is the top result")
ax.set_ylabel("factor on the grade")
ax.set_ylim(0, 1.1)
ax.legend(loc="lower left", frameon=False)
save(fig, "ndcg-discount")
