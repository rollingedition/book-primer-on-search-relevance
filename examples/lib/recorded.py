"""Model outputs recorded once by the scripts in recordings/, read here offline (writing standards W-48).

Each folder in data/recorded/ has a manifest.json naming the model, its exact version and the prompts.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RECORDED = Path(__file__).resolve().parent.parent / "data" / "recorded"


def manifest(name: str) -> dict:
    return json.loads((RECORDED / name / "manifest.json").read_text())


def _unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def embeddings(dims: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(product_ids, product_vectors, query_ids, query_vectors), unit length, one row per ID.
    dims keeps only the first dims dimensions and rescales to unit length; the model was trained for 256."""
    folder = RECORDED / "embeddings"
    products = np.load(folder / "products.npy").astype(np.float32)
    queries = np.load(folder / "queries.npy").astype(np.float32)
    if dims is not None:
        products, queries = products[:, :dims], queries[:, :dims]
    return (np.load(folder / "product_ids.npy"), _unit(products),
            np.load(folder / "query_ids.npy"), _unit(queries))


def cross_encoder_scores() -> pd.DataFrame:
    """query_id, product_id, score: the cross-encoder's raw score for each query's candidates."""
    return pd.read_csv(RECORDED / "cross-encoder" / "scores.csv")


def probes() -> tuple[pd.DataFrame, np.ndarray]:
    """(probes, vectors): probe queries written to test weak spots of embeddings (negation, numbers,
    codes, spelling), with their embeddings from the same model. Use them to show a behaviour. A spelling
    probe is the correct spelling of a sampled query (of_query_id) and may be scored with that query's
    judgments, saying so; no other probe has judgments."""
    folder = RECORDED / "probes"
    probes = pd.read_csv(folder / "probes.csv", dtype={"of_query_id": "Int64"})
    return probes, _unit(np.load(folder / "queries.npy").astype(np.float32))


# ---------------------------------------------------------------- LLM recordings
# One open model, gemma4:12b-it-q4_K_M, run locally with temperature 0 and a fixed seed; each folder's
# manifest.json holds the model digest and the exact prompts.


def _ids(text: object) -> list[int]:
    return [int(x) for x in str(text).split()] if isinstance(text, str) and text.strip() else []


def llm_judgments() -> pd.DataFrame:
    """query_id, product_id, pool ("test" top-10 pairs or a "train" sample of human-judged pairs), prompt
    ("short" or "guidelines"), label and grade (2, 1, 0; missing if the model's answer wasn't valid)."""
    return pd.read_csv(RECORDED / "llm-judgments" / "judgments.csv", dtype={"grade": "Int64"})


def llm_rewrites() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """(table, rewrite_vectors, hypothetical_vectors): each query's LLM rewrite and hypothetical product
    (HyDE), and their embeddings from the same model as the products, in table order."""
    folder = RECORDED / "llm-rewrites"
    return (pd.read_csv(folder / "rewrites.csv"), _unit(np.load(folder / "rewrites.npy").astype(np.float32)),
            _unit(np.load(folder / "hypothetical.npy").astype(np.float32)))


def llm_conversations() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """(table, vectors): follow-ups written for the book, each refining a test query by one attribute, and
    embeddings of the follow-up alone, the two turns joined, and the LLM's standalone query."""
    folder = RECORDED / "llm-conversations"
    vectors = {name: _unit(np.load(folder / f"{name}.npy").astype(np.float32)) for name in ("follow_up", "joined", "standalone")}
    return pd.read_csv(folder / "conversations.csv"), vectors


def llm_listwise() -> pd.DataFrame:
    """For each test query: query_id, order ("semantic" or "reversed": the order the 20 products were shown
    in), position, product_id and valid (False when the model's answer had to be repaired)."""
    return pd.read_csv(RECORDED / "llm-rerank" / "listwise.csv")


def llm_pointwise() -> pd.DataFrame:
    """query_id, product_id, label and score (0 to 100) for each product of semantic retrieval's top 20, for
    each test query."""
    return pd.read_csv(RECORDED / "llm-rerank" / "pointwise.csv", dtype={"score": "Int64"})


def rag_answers() -> pd.DataFrame:
    """query_id, context ("semantic", "bm25" or "random"), context_ids and cited (lists of product IDs),
    and the answer text."""
    answers = pd.read_csv(RECORDED / "rag-answers" / "answers.csv", keep_default_na=False)
    answers["context_ids"] = answers["context_ids"].map(_ids)
    answers["cited"] = answers["cited"].map(_ids)
    return answers


def agent_traces() -> list[dict]:
    """One dict per (retriever, test query): retriever ("bm25" or "semantic"), query_id, searches (each
    with the agent's search text and the product IDs returned) and picked (the three IDs it chose)."""
    path = RECORDED / "agent-trace" / "traces.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
