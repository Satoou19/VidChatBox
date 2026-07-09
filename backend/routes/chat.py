"""Chat endpoint with RAG similarity search and LLM streaming."""

import os

from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.pipeline.project_manager import get_project_dir
from backend.rag.vector_store import LocalVectorStore
from backend.rag.chatbot import RAGChatbot
from backend.services.embedding import resolve_embedding_provider

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    query: str
    provider: str = "gemini"
    model: str = ""
    persona_mode: str = "streamer"
    project_id: str = "default"


@router.post("/chat")
def run_chat(
    req: ChatRequest,
    x_groq_key: str = Header(None, alias="X-Groq-Key"),
    x_openai_key: str = Header(None, alias="X-Openai-Key"),
    x_gemini_key: str = Header(None, alias="X-Gemini-Key"),
    x_deepseek_key: str = Header(None, alias="X-Deepseek-Key"),
    x_openrouter_key: str = Header(None, alias="X-Openrouter-Key"),
    x_llm7_key: str = Header(None, alias="X-Llm7-Key"),
):
    """Executes RAG similarity search and streams LLM response."""
    proj_vector_store = LocalVectorStore(data_dir=get_project_dir(req.project_id))

    # Determine embedding provider based on available keys and provider selection
    gemini_key = x_gemini_key or os.getenv("GEMINI_API_KEY")
    openai_key = x_openai_key or os.getenv("OPENAI_API_KEY")
    openrouter_key = x_openrouter_key or os.getenv("OPENROUTER_API_KEY")

    g_key_valid = bool(gemini_key and gemini_key.strip())
    o_key_valid = bool(openai_key and openai_key.strip())
    or_key_valid = bool(openrouter_key and openrouter_key.strip())

    if req.provider in ("gemini", "openai", "openrouter"):
        embedding_provider = req.provider
    else:
        if or_key_valid:
            embedding_provider = "openrouter"
        elif g_key_valid:
            embedding_provider = "gemini"
        elif o_key_valid:
            embedding_provider = "openai"
        else:
            g_env = os.getenv("GEMINI_API_KEY")
            o_env = os.getenv("OPENAI_API_KEY")
            or_env = os.getenv("OPENROUTER_API_KEY")
            if or_env and or_env.strip():
                embedding_provider = "openrouter"
            elif g_env and g_env.strip():
                embedding_provider = "gemini"
            else:
                embedding_provider = "openai"

    # Auto-align embedding provider with stored embeddings to prevent dimension mismatch
    stored_dim = proj_vector_store.get_stored_embedding_dim()
    if stored_dim is not None:
        if stored_dim == 0:
            embedding_provider = "local"
        elif stored_dim == 384:
            embedding_provider = "local-ai"
        elif stored_dim == 768:
            embedding_provider = "gemini"
        elif stored_dim == 3072 and embedding_provider != "gemini":
            embedding_provider = "gemini"
        elif stored_dim == 1536 and embedding_provider == "gemini":
            if o_key_valid:
                embedding_provider = "openai"
            elif or_key_valid:
                embedding_provider = "openrouter"
            else:
                o_env = os.getenv("OPENAI_API_KEY")
                or_env = os.getenv("OPENROUTER_API_KEY")
                if o_env and o_env.strip():
                    embedding_provider = "openai"
                elif or_env and or_env.strip():
                    embedding_provider = "openrouter"
                else:
                    embedding_provider = "openai"

    if embedding_provider == "gemini":
        emb_key = gemini_key
    elif embedding_provider == "openai":
        emb_key = openai_key
    elif embedding_provider in ("local", "local-ai"):
        emb_key = None
    else:
        emb_key = openrouter_key

    try:
        search_results = proj_vector_store.search(req.query, top_k=5, provider=embedding_provider, api_key=emb_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    if not search_results:
        def empty_generator():
            yield "I don't have any streamer VID data indexed in this group yet! Please ingest some video URLs first using the Ingest VIDs tab."
        return StreamingResponse(empty_generator(), media_type="text/plain")

    # Generate and stream Chat response with user custom keys
    chatbot = RAGChatbot(
        gemini_api_key=x_gemini_key,
        openai_api_key=x_openai_key,
        groq_api_key=x_groq_key,
        deepseek_api_key=x_deepseek_key,
        openrouter_api_key=x_openrouter_key,
        llm7_api_key=x_llm7_key,
    )

    def event_generator():
        try:
            for text in chatbot.chat_stream(
                query=req.query,
                search_results=search_results,
                provider=req.provider,
                model=req.model,
                persona_mode=req.persona_mode,
            ):
                yield text
        except Exception as chat_err:
            yield f"\n[Error generating response: {str(chat_err)}]"

    return StreamingResponse(event_generator(), media_type="text/plain")
