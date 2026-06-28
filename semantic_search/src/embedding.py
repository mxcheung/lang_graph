"""Embedding provider abstraction — selects OpenAI or Ollama via EMBEDDING_PROVIDER env var."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL_ALIASES = {
    "text-embedding-3-small": "openai_text_small",
    "text-embedding-3-large": "openai_text_large",
    "qwen3-embedding:latest": "qwen3",
    "all-minilm":             "all_minilm",
}

def alias_for(model: str) -> str:

    if model in MODEL_ALIASES:
        return MODEL_ALIASES[model]
    
    return model.replace(":", "_").replace("/", "_")


def embedding_alias(provider: str | None = None) -> str:

    if provider is None:
        provider = provider_name()

    if provider == "openai":
        model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    elif provider == "ollama":
        model = os.environ.get("OLLAMA_EMBED_MODEL", "qwen3-embedding:latest")
    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER={provider!r}.")
    
    return alias_for(model)


def bundle_path(filename: str, embedding_alias: str | None = None) -> Path:

    if embedding_alias:
        return Path(f"data/embeddings/{filename}_{embedding_alias}.json")

    return Path(f"data/{filename}.json")


def provider_name() -> str:
    return os.environ.get("EMBEDDING_PROVIDER", "openai").lower()


def embed_texts(texts: Sequence[str], provider: str | None = None, model: str | None = None) -> list[list[float]]:
    texts = list(texts)
    if not texts:
        raise ValueError("embed_texts received an empty list of texts.")

    if provider is None:
        provider = provider_name()

    if provider == "openai":
        embeddings = _embed_openai(texts, model)

    elif provider == "ollama":
        embeddings = _embed_ollama(texts, model)

    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER={provider!r}. "
            f"Set it to 'openai' or 'ollama' in your .env file."
        )

    return embeddings


def embed_query(text: str, provider: str | None = None, model: str | None = None) -> list[float]:
    return embed_texts([text], provider, model)[0]


def _openai_client() -> OpenAI:
    
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(base_url=base_url)

    return OpenAI()


def _embed_openai(texts: list[str], model: str | None = None):

    client = _openai_client()

    if model is None:
        model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    # User task: Call the API with all texts in one batch
    response = client.embeddings.create(model=model, input=texts)

    
    # Pull the embedding vector out of each item, e.g. response.data[0].embedding
    embeddings = [item.embedding for item in response.data]

    return embeddings


def _embed_ollama(texts: list[str], model: str | None = None):

    try:
        import ollama
    except ImportError as e:
        raise ImportError(
            "The 'ollama' package is required for EMBEDDING_PROVIDER=ollama. "
            "Install it with: pip install ollama"
        ) from e

    host = os.environ.get("OLLAMA_HOST")
    client = ollama.Client(host=host) if host else ollama

    if model is None:
        model = os.environ.get("OLLAMA_EMBED_MODEL", "qwen3-embedding:latest")

    # Call the API with all texts in one batch
    response = client.embed(model=model, input=texts)

    return response["embeddings"]
