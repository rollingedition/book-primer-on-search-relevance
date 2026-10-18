"""Record embeddings for probe queries (writing standards W-48).

    python recordings/probes.py

Probe queries are written for the book to test known weak spots of embeddings: negation, numbers and
sizes, product codes and spelling. They aren't shopper queries and have no judgments of their own; chapters
use them to show a behaviour. The one exception is a spelling probe: it is the correctly spelled form of a
sampled query (of_query_id), so it shares that query's need and may be scored with its judgments, saying so.
Same model and revision as recordings/embeddings.py. New probes are added at the end, so IDs never change.
Outputs, in examples/data/recorded/probes/: probes.csv (probe_id, pair, kind, query, of_query_id) and
queries.npy (float16, unit length, one row per probe, in probe_id order).
"""
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from common import out_dir, write_manifest
from embeddings import MODEL, REVISION

NAME = "probes"
# (pair, kind, query): each pair differs in one way an embedding may miss
PROBES = [
    ("arms", "negation", "sofa with arms"), ("arms", "negation", "sofa without arms"),
    ("glass", "negation", "coffee table with glass top"), ("glass", "negation", "coffee table not glass"),
    ("round", "negation", "round rug"), ("round", "negation", "rug that is not round"),
    ("hood", "number", "36 inch kitchen hood"), ("hood", "number", "48 inch kitchen hood"),
    ("cooker", "number", "4 qt slow cooker"), ("cooker", "number", "7 qt slow cooker"),
    ("bed", "number", "twin bed"), ("bed", "number", "king bed"),
    ("seats", "number", "2 seat sofa"), ("seats", "number", "4 seat sofa"),
    ("code", "code", "shag rug w003096809"), ("code", "code", "shag rug w003096810"),
    # a resolution string that one product's text contains, and one that no product contains
    ("resolution", "code", "1920x1080p projector"), ("resolution", "code", "2560x1440p projector"),
]
# correct spellings of sampled queries that contain typos: (pair, query, the sampled query's ID)
SPELLINGS = [("ligth", "light bulb", 305), ("foutains", "fountains with brick look", 20),
             ("desins", "twin over full bunk beds cool designs", 224), ("ice-an", "refrigerator with ice and water in door", 406)]


def main() -> None:
    rows = [(pair, kind, query, None) for pair, kind, query in PROBES]
    rows += [(pair, "spelling", query, of) for pair, query, of in SPELLINGS]
    probes = pd.DataFrame(rows, columns=["pair", "kind", "query", "of_query_id"]).astype({"of_query_id": "Int64"})
    probes = probes.rename_axis("probe_id").reset_index()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL, revision=REVISION, device=device)
    vectors = model.encode_query(probes["query"].tolist(), normalize_embeddings=True, show_progress_bar=False)
    out = out_dir(NAME)
    probes.to_csv(out / "probes.csv", index=False)
    np.save(out / "queries.npy", vectors.astype(np.float16))
    write_manifest(NAME, "probes.py", MODEL, REVISION, {"query": model.prompts.get("query", ""), "probes": PROBES, "spellings": SPELLINGS},
                   ["probes.csv", "queries.npy"], "Apache-2.0", dtype="float16", normalized=True, device=device,
                   note="Probe queries written for the book; not shopper queries, no judgments.")
    print(f"probes: {len(probes)}")


if __name__ == "__main__":
    main()
