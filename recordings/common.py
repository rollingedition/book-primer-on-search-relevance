"""Shared code for the recording scripts (writing standards W-48).

A recording runs a model once, with a GPU and network, and saves its outputs under
examples/data/recorded/{name}/ with a manifest. Examples then read the outputs offline. Run a recording
from the book's root, in the environment recordings/requirements.txt pins:

    python recordings/{name}.py

Every recording uses an open model anyone can download, and records the exact version it used.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECORDED = ROOT / "examples" / "data" / "recorded"
sys.path.insert(0, str(ROOT / "examples"))   # so recordings use the same lib code as the examples


def out_dir(name: str) -> Path:
    path = RECORDED / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_manifest(name: str, script: str, model: str, model_version: str, prompts: dict[str, str] | str,
                   outputs: list[str], licence: str, **extra) -> None:
    """The manifest re-check style requires: what ran, on which model version, with which prompts."""
    prompt_text = prompts if isinstance(prompts, str) else json.dumps(prompts, sort_keys=True)
    manifest = {
        "script": f"recordings/{script}",
        "model": model,
        "model_version": model_version,
        "model_licence": licence,
        "prompt_sha256": sha256(prompt_text),
        "prompts": prompts,
        "created": datetime.now(timezone.utc).date().isoformat(),
        "outputs": outputs,
        **extra,
    }
    (out_dir(name) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def hf_revision(model_id: str) -> str:
    """The Hugging Face commit a model was downloaded at."""
    from huggingface_hub import model_info
    return model_info(model_id).sha


def ollama_version(tag: str) -> str:
    """The digest of a local Ollama model, which pins its exact weights."""
    import ollama
    for m in ollama.list().models:
        if m.model == tag:
            return m.digest
    raise SystemExit(f"{tag} isn't pulled; run: ollama pull {tag}")
