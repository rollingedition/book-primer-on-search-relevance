"""Record product and query embeddings (writing standards W-48).

    python recordings/embeddings.py

Model: Snowflake/snowflake-arctic-embed-m-v1.5 (Apache-2.0), pinned to a commit. It gives 768 dimensions
trained so the first 256 still work on their own (Matryoshka), and adds its query prompt itself.
Outputs, in examples/data/recorded/embeddings/: products.npy and queries.npy (float16, unit length,
one row per ID), with product_ids.npy and query_ids.npy giving each row's ID.
"""
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from common import out_dir, write_manifest
from lib.store import load_products, load_queries, product_text

MODEL = "Snowflake/snowflake-arctic-embed-m-v1.5"
REVISION = "e58a8f756156a1293d763f17e3aae643474e9b8a"
NAME = "embeddings"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL, revision=REVISION, device=device)
    products, queries = load_products(), load_queries()
    texts = product_text(products).tolist()
    lengths = [len(ids) for ids in model.tokenizer(texts)["input_ids"]]
    print(f"longest product text: {max(lengths)} tokens (the model reads {model.max_seq_length})")

    docs = model.encode_document(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    qs = model.encode_query(queries["query"].tolist(), batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    out = out_dir(NAME)
    np.save(out / "products.npy", docs.astype(np.float16))
    np.save(out / "product_ids.npy", products.index.to_numpy(dtype=np.int64))
    np.save(out / "queries.npy", qs.astype(np.float16))
    np.save(out / "query_ids.npy", queries.index.to_numpy(dtype=np.int64))
    write_manifest(NAME, "embeddings.py", MODEL, REVISION,
                   {"document": "lib.store.product_text: name. class. category. description",
                    "query": model.prompts.get("query", "")},
                   ["products.npy", "product_ids.npy", "queries.npy", "query_ids.npy"], "Apache-2.0",
                   dimensions=int(docs.shape[1]), dtype="float16", normalized=True, device=device,
                   longest_product_tokens=max(lengths), max_seq_length=model.max_seq_length)
    print(f"products: {docs.shape}  queries: {qs.shape}")


if __name__ == "__main__":
    main()
