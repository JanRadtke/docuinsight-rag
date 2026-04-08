"""
LLM Provider Factory
=====================
Switches between OpenAI and Ollama based on LLM_PROVIDER env variable.

Usage:
    from llm_provider import get_llm_client, get_embedding_client

    client = get_llm_client()          # Use for chat completions
    embed_client = get_embedding_client()  # Use for embeddings

Configuration (.env):
    LLM_PROVIDER=openai    # default
    LLM_PROVIDER=ollama

    OPENAI_API_KEY=sk-...
    OLLAMA_BASE_URL=http://localhost:11434/v1   # optional
    OLLAMA_MODEL=llama3.1                        # optional
"""

import os
import logging
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
from dotenv import load_dotenv

# Central exception tuple — import this instead of openai directly
LLM_ERRORS = (APIError, APIConnectionError, APITimeoutError)

logger = logging.getLogger("docuinsight")

load_dotenv()


def get_llm_client() -> OpenAI:
    """
    Returns an OpenAI-compatible client based on LLM_PROVIDER.
    Works for both chat completions and embeddings (same endpoint).
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        # Non-blocking health check — warn in background, don't delay startup
        import threading
        import urllib.request
        import urllib.error
        def _check(url=base_url.replace("/v1", "")):
            try:
                urllib.request.urlopen(url, timeout=3)
            except (ConnectionError, OSError, urllib.error.URLError, ValueError):
                logger.warning("Ollama not reachable at %s — is it running? (ollama serve)", url)
        threading.Thread(target=_check, daemon=True).start()
        return OpenAI(base_url=base_url, api_key="ollama")

    # Default: OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No OpenAI API key found. Set OPENAI_API_KEY in your .env file.")
    return OpenAI(api_key=api_key)


# Alias — same client, kept for semantic clarity at call sites
get_embedding_client = get_llm_client


def get_model_name() -> str:
    """Returns the configured LLM model name."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "ollama":
        return os.getenv("OLLAMA_MODEL", "llama3.1")
    return os.getenv("LLM_MODEL", "gpt-4o-mini")


def get_embedding_model() -> str:
    """Returns the configured embedding model name."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "ollama":
        return os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    return os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def get_embedding_dim() -> int:
    """Returns the embedding dimension for the configured provider."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "ollama":
        return int(os.getenv("OLLAMA_EMBEDDING_DIM", "768"))
    return int(os.getenv("EMBEDDING_DIM", "1536"))


def get_embedding(text: str, client=None) -> list:
    """Generates an embedding vector for a text.

    Central function used by both ingest.py and retriever.py to ensure
    consistent embedding parameters (model, dimensions) across the pipeline.
    """
    dim = get_embedding_dim()
    if not text:
        return [0.0] * dim
    clean_text = text.replace("\n", " ")
    model = get_embedding_model()
    kwargs: dict = {"input": [clean_text], "model": model}
    if os.getenv("LLM_PROVIDER", "openai").lower() != "ollama":
        kwargs["dimensions"] = dim
    if client is None:
        client = get_embedding_client()
    return client.embeddings.create(**kwargs).data[0].embedding
