"""Shared code for the LLM recordings: one open model, run locally through Ollama, with fixed settings.

Model: gemma4:12b-it-q4_K_M (Gemma 4, 12B, instruction-tuned, 4-bit; Apache 2.0). Every call uses
temperature 0, a fixed seed and thinking switched off, and asks for JSON that matches a schema, so a
rerun on the same model digest gives the same outputs. Calls are cached in an append-only JSON Lines
file beside the outputs, so an interrupted recording resumes where it stopped.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import ollama
import pandas as pd

from common import ollama_version, out_dir

MODEL = "gemma4:12b-it-q4_K_M"
OPTIONS = {"temperature": 0, "seed": 2026, "num_ctx": 8192}
LICENCE = "Apache-2.0"


def settings(schemas: dict | None = None) -> dict:
    """Manifest fields that pin how the model was called: every option, and the JSON schemas answers had to match."""
    return {"options": {**OPTIONS, "think": False}, "schemas": schemas or {}}


def describe(product: pd.Series, product_id: int | None = None) -> str:
    """The product as the LLM sees it: the same text search reads, plus the attributes."""
    lines = [f"ID: {product_id}"] if product_id is not None else []
    lines += [f"Name: {product['name']}", f"Class: {product['product_class']}", f"Category: {product['category']}",
              f"Description: {product['description']}", f"Attributes: {product['attributes'] or 'none listed'}"]
    return "\n".join(lines)


class Recorder:
    """Calls the model, caching each answer by key in {name}/calls.jsonl."""

    def __init__(self, name: str):
        self.name, self.path = name, out_dir(name) / "calls.jsonl"
        self.cache = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue                    # a line cut off by an interrupted run: that call is made again
                self.cache[row["key"]] = (row["content"], row.get("prompt_sha256"))
        self.digest = ollama_version(MODEL)

    def chat(self, key: str, messages: list[dict], schema: dict | None = None) -> str:
        """The model's answer, from the cache when this key was asked with the same prompt. A cached answer
        recorded before prompts were hashed is reused as is."""
        digest = hashlib.sha256(json.dumps([messages, schema], sort_keys=True).encode("utf-8")).hexdigest()
        cached = self.cache.get(key)
        if cached is None or (cached[1] is not None and cached[1] != digest):
            reply = ollama.chat(model=MODEL, messages=messages, format=schema, think=False, options=OPTIONS)
            cached = (reply.message.content, digest)
            self.cache[key] = cached
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "prompt_sha256": digest, "content": cached[0]}) + "\n")
        return cached[0]

    def json(self, key: str, messages: list[dict], schema: dict) -> dict | None:
        """The parsed JSON answer, or None when the model didn't return valid JSON."""
        try:
            return json.loads(self.chat(key, messages, schema))
        except json.JSONDecodeError:
            return None
