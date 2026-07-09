"""Centralized embedding provider resolution.

Eliminates the duplicated provider-selection logic that was previously
copy-pasted in run_ingestion_pipeline, run_batch_ingestion_pipeline,
and the chat endpoint.
"""

import os


def resolve_embedding_provider(provider, gemini_key=None, openai_key=None, openrouter_key=None):
    """Determines the embedding provider and corresponding API key.

    Args:
        provider: The user-selected provider hint (e.g. "gemini", "openai", "local", "local-ai", "openrouter", "groq", "deepseek").
        gemini_key: Optional user-provided Gemini API key.
        openai_key: Optional user-provided OpenAI API key.
        openrouter_key: Optional user-provided OpenRouter API key.

    Returns:
        Tuple of (emb_provider: str, emb_key: str | None).
    """
    if provider in ("local", "local-ai"):
        return provider, None

    # If the provider directly supports embeddings, use it
    if provider in ("gemini", "openai", "openrouter"):
        emb_provider = provider
    else:
        # For providers that don't have their own embedding API (groq, deepseek, llm7),
        # fall back to the first available key
        if gemini_key or os.getenv("GEMINI_API_KEY"):
            emb_provider = "gemini"
        elif openai_key or os.getenv("OPENAI_API_KEY"):
            emb_provider = "openai"
        elif openrouter_key or os.getenv("OPENROUTER_API_KEY"):
            emb_provider = "openrouter"
        else:
            emb_provider = "gemini"  # Default

    # Resolve the API key for the chosen provider
    if emb_provider == "gemini":
        emb_key = gemini_key
    elif emb_provider == "openai":
        emb_key = openai_key
    else:
        emb_key = openrouter_key

    return emb_provider, emb_key
