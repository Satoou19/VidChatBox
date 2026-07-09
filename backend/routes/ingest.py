"""Video ingestion endpoints and background pipelines."""

import os
import re
import hashlib
import traceback
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from backend.pipeline.downloader import VideoDownloader
from backend.pipeline.processor import TranscriptProcessor
from backend.pipeline.project_manager import get_project_dir
from backend.rag.vector_store import LocalVectorStore
from backend.services.embedding import resolve_embedding_provider
from backend.task_store import TaskStore

router = APIRouter(prefix="/api", tags=["ingest"])

# Persistent task stores
DATA_DIR = os.getenv("DATA_DIR", "./backend/data")
ingestion_tasks = TaskStore(os.path.join(DATA_DIR, "tasks_ingestion.json"))
batch_tasks = TaskStore(os.path.join(DATA_DIR, "tasks_batch.json"))


# ------------------------------------------------------------------
# Request models
# ------------------------------------------------------------------

class IngestRequest(BaseModel):
    url: str
    provider: str = "gemini"
    project_id: str = "default"
    openai_key: Optional[str] = None
    gemini_key: Optional[str] = None
    openrouter_key: Optional[str] = None


class BatchIngestRequest(BaseModel):
    urls: List[str]
    provider: str = "gemini"
    project_id: str = "default"
    openai_key: Optional[str] = None
    gemini_key: Optional[str] = None
    openrouter_key: Optional[str] = None


# ------------------------------------------------------------------
# Background pipeline functions
# ------------------------------------------------------------------

def run_ingestion_pipeline(
    task_id: str,
    url: str,
    provider: str = "gemini",
    project_id: str = "default",
    openai_key: str = None,
    gemini_key: str = None,
    openrouter_key: str = None,
):
    """Executes ingestion using subtitles directly extracted from YouTube (no AI transcription used)."""
    try:
        project_dir = get_project_dir(project_id)
        proj_downloader = VideoDownloader(data_dir=project_dir)
        proj_processor = TranscriptProcessor(data_dir=project_dir)
        proj_vector_store = LocalVectorStore(data_dir=project_dir)

        # Step 1: Extract info and download subtitles directly
        ingestion_tasks.update(task_id, {
            "status": "extracting_metadata",
            "percent": 20.0,
            "title": "Extracting metadata and subtitles...",
            "current_video_title": url,
            "current_video_status": "extracting_metadata",
            "current_video_percent": 20.0,
            "overall_percent": 20.0,
            "error": None,
        })
        ingestion_tasks.update_nested(task_id, ["videos", 0], {
            "status": "extracting_metadata",
            "percent": 20.0,
        })

        def sub_progress(status, pct):
            ingestion_tasks.update(task_id, {
                "status": status,
                "percent": pct,
                "current_video_status": status,
                "current_video_percent": pct,
                "overall_percent": pct,
            })
            ingestion_tasks.update_nested(task_id, ["videos", 0], {
                "status": status,
                "percent": pct,
            })

        sub_info = proj_downloader.download_subtitles(url, progress_callback=sub_progress)
        video_id = sub_info["id"]
        title = sub_info["title"]
        segments = sub_info["segments"]

        ingestion_tasks.update(task_id, {
            "title": title,
            "status": "processing",
            "percent": 80.0,
            "current_video_title": title,
            "current_video_status": "processing",
            "current_video_percent": 80.0,
            "overall_percent": 80.0,
        })
        ingestion_tasks.update_nested(task_id, ["videos", 0], {
            "title": title,
            "status": "processing",
            "percent": 80.0,
        })

        # Step 2: Chunk & Process
        _, chunks = proj_processor.save_knowledge_package(sub_info, segments)

        # Step 3: Embed & Index in Vector DB
        ingestion_tasks.update(task_id, {
            "status": "indexing",
            "percent": 90.0,
            "title": f"{title} - Indexing embeddings in search database...",
            "current_video_status": "indexing",
            "current_video_percent": 90.0,
            "overall_percent": 90.0,
        })
        ingestion_tasks.update_nested(task_id, ["videos", 0], {
            "status": "indexing",
            "percent": 90.0,
        })

        emb_provider, emb_key = resolve_embedding_provider(
            provider, gemini_key, openai_key, openrouter_key
        )
        proj_vector_store.add_video_chunks(video_id, chunks, provider=emb_provider, api_key=emb_key)

        # Complete
        ingestion_tasks.update(task_id, {
            "status": "completed",
            "percent": 100.0,
            "title": title,
            "completed_videos": 1,
            "current_video_status": "completed",
            "current_video_percent": 100.0,
            "overall_percent": 100.0,
        })
        ingestion_tasks.update_nested(task_id, ["videos", 0], {
            "status": "completed",
            "percent": 100.0,
        })

    except Exception as e:
        traceback.print_exc()
        ingestion_tasks.update(task_id, {
            "status": "failed",
            "percent": 0.0,
            "current_video_status": "failed",
            "current_video_percent": 0.0,
            "overall_percent": 0.0,
            "error": str(e),
        })
        ingestion_tasks.update_nested(task_id, ["videos", 0], {
            "status": "failed",
            "error": str(e),
            "percent": 0.0,
        })


def run_batch_ingestion_pipeline(
    batch_task_id: str,
    urls: List[str],
    provider: str,
    project_id: str,
    openai_key: str = None,
    gemini_key: str = None,
    openrouter_key: str = None,
):
    """Executes sequential batch ingestion by extracting subtitles directly (no AI used)."""
    batch_tasks.update(batch_task_id, {"status": "processing"})

    project_dir = get_project_dir(project_id)
    proj_downloader = VideoDownloader(data_dir=project_dir)
    proj_processor = TranscriptProcessor(data_dir=project_dir)
    proj_vector_store = LocalVectorStore(data_dir=project_dir)

    completed_count = 0

    for idx, url in enumerate(urls):
        current_idx = idx + 1
        batch_tasks.update(batch_task_id, {
            "current_index": current_idx,
            "current_video_status": "extracting_metadata",
            "current_video_percent": 10.0,
            "current_video_title": url,
            "overall_percent": (idx / len(urls)) * 100,
        })
        batch_tasks.update_nested(batch_task_id, ["videos", idx], {
            "status": "extracting_metadata",
            "percent": 10.0,
        })

        try:
            # Download subtitles
            def sub_progress(status, pct, _idx=idx):
                batch_tasks.update(batch_task_id, {
                    "current_video_status": status,
                    "current_video_percent": pct,
                    "overall_percent": ((_idx + (pct / 100.0 * 0.7)) / len(urls)) * 100,
                })
                batch_tasks.update_nested(batch_task_id, ["videos", _idx], {
                    "status": status,
                    "percent": pct,
                })

            sub_info = proj_downloader.download_subtitles(url, progress_callback=sub_progress)
            video_id = sub_info["id"]
            title = sub_info["title"]
            segments = sub_info["segments"]

            batch_tasks.update(batch_task_id, {
                "current_video_title": title,
                "current_video_status": "processing_package",
                "current_video_percent": 80.0,
                "overall_percent": ((idx + 0.8) / len(urls)) * 100,
            })
            batch_tasks.update_nested(batch_task_id, ["videos", idx], {
                "title": title,
                "status": "processing_package",
                "percent": 80.0,
            })

            # Chunk & Save package
            _, chunks = proj_processor.save_knowledge_package(sub_info, segments)

            # Embed & Save
            batch_tasks.update(batch_task_id, {
                "current_video_status": "indexing",
                "current_video_percent": 95.0,
                "overall_percent": ((idx + 0.95) / len(urls)) * 100,
            })
            batch_tasks.update_nested(batch_task_id, ["videos", idx], {
                "status": "indexing",
                "percent": 95.0,
            })

            emb_provider, emb_key = resolve_embedding_provider(
                provider, gemini_key, openai_key, openrouter_key
            )
            proj_vector_store.add_video_chunks(video_id, chunks, provider=emb_provider, api_key=emb_key)

            completed_count += 1
            batch_tasks.update(batch_task_id, {
                "completed_videos": completed_count,
                "current_video_status": "completed",
                "current_video_percent": 100.0,
                "overall_percent": (current_idx / len(urls)) * 100,
            })
            batch_tasks.update_nested(batch_task_id, ["videos", idx], {
                "status": "completed",
                "percent": 100.0,
            })

        except Exception as e:
            traceback.print_exc()
            batch_tasks.update(batch_task_id, {
                "current_video_status": "failed",
                "current_video_percent": 0.0,
                "overall_percent": (current_idx / len(urls)) * 100,
            })
            batch_tasks.update_nested(batch_task_id, ["videos", idx], {
                "status": "failed",
                "error": str(e),
                "percent": 0.0,
            })

    # Finalize batch task status
    if completed_count == len(urls):
        batch_tasks.update(batch_task_id, {"status": "completed"})
    elif completed_count == 0:
        batch_tasks.update(batch_task_id, {
            "status": "failed",
            "error": "All videos in the batch failed to process.",
        })
    else:
        batch_tasks.update(batch_task_id, {"status": "completed_with_errors"})


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@router.post("/ingest")
def start_ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    """Enqueues VID ingestion in the background."""
    if not req.url:
        raise HTTPException(status_code=400, detail="URL cannot be empty")

    clean_url = req.url.strip(" \t\n\r\"'[]")
    if not clean_url:
        raise HTTPException(status_code=400, detail="URL cannot be empty")

    # Generate stable task ID from URL hash to avoid duplicate ingestion tasks
    url_hash = hashlib.md5(clean_url.encode("utf-8")).hexdigest()
    task_id = f"task_{url_hash}"

    # Check if already running or completed
    if task_id in ingestion_tasks:
        task = ingestion_tasks.get(task_id, {})
        if task.get("status") not in ("failed", "completed"):
            return {"task_id": task_id, "status": task.get("status"), "message": "Task already running"}

    # Initialize task status
    ingestion_tasks.set(task_id, {
        "status": "pending",
        "percent": 0.0,
        "title": clean_url,
        "error": None,
        "total_videos": 1,
        "completed_videos": 0,
        "current_index": 1,
        "current_video_title": clean_url,
        "current_video_status": "pending",
        "current_video_percent": 0.0,
        "overall_percent": 0.0,
        "videos": [{"url": clean_url, "title": clean_url, "status": "pending", "percent": 0.0, "error": None}],
    })

    background_tasks.add_task(
        run_ingestion_pipeline, task_id, clean_url, req.provider, req.project_id,
        req.openai_key, req.gemini_key, req.openrouter_key,
    )
    return {"task_id": task_id, "status": "pending", "message": "Ingestion started in background"}


@router.post("/ingest-batch")
def start_batch_ingest(req: BatchIngestRequest, background_tasks: BackgroundTasks):
    """Starts sequential batch ingestion in the background."""
    if not req.urls:
        raise HTTPException(status_code=400, detail="URL list cannot be empty")

    urls = [url.strip(" \t\n\r\"'[]") for url in req.urls if url.strip(" \t\n\r\"'[]")]
    if not urls:
        raise HTTPException(status_code=400, detail="No valid URLs provided")

    # Generate batch task ID
    urls_str = ",".join(urls)
    batch_hash = hashlib.md5(urls_str.encode("utf-8")).hexdigest()
    batch_task_id = f"batch_{batch_hash}"

    if batch_task_id in batch_tasks:
        task = batch_tasks.get(batch_task_id, {})
        if task.get("status") in ("pending", "processing"):
            return {"batch_task_id": batch_task_id, "status": task.get("status"), "message": "Batch task already running"}

    batch_tasks.set(batch_task_id, {
        "status": "pending",
        "total_videos": len(urls),
        "completed_videos": 0,
        "current_index": 0,
        "current_video_title": "",
        "current_video_status": "pending",
        "current_video_percent": 0.0,
        "overall_percent": 0.0,
        "videos": [{"url": url, "title": url, "status": "pending", "percent": 0.0, "error": None} for url in urls],
        "error": None,
    })

    background_tasks.add_task(
        run_batch_ingestion_pipeline, batch_task_id, urls, req.provider, req.project_id,
        req.openai_key, req.gemini_key, req.openrouter_key,
    )
    return {"batch_task_id": batch_task_id, "status": "pending", "message": "Batch ingestion started"}


@router.get("/status/{task_id}")
def get_status(task_id: str):
    """Returns the progress of a specific ingestion task."""
    if task_id not in ingestion_tasks:
        raise HTTPException(status_code=404, detail="Ingestion task not found")
    return ingestion_tasks.get(task_id)


@router.get("/batch-status/{batch_task_id}")
def get_batch_status(batch_task_id: str):
    """Returns the progress of a batch ingestion task."""
    if batch_task_id not in batch_tasks:
        raise HTTPException(status_code=404, detail="Batch task not found")
    return batch_tasks.get(batch_task_id)


@router.get("/videos")
def list_videos(project_id: str = "default"):
    """Returns a list of all successfully ingested videos with metadata within a project."""
    proj_vector_store = LocalVectorStore(data_dir=get_project_dir(project_id))
    unique_videos = []
    seen_ids = set()

    for chunk in proj_vector_store.get_all_chunks():
        v_id = chunk["video_id"]
        if v_id not in seen_ids:
            seen_ids.add(v_id)
            unique_videos.append({
                "id": v_id,
                "title": chunk["video_title"],
                "url": chunk["video_url"],
            })

    return unique_videos


@router.delete("/videos/{video_id}")
def api_delete_video(video_id: str, project_id: str = "default"):
    """Deletes a specific video and all its search indexes/markdown files."""
    project_dir = get_project_dir(project_id)
    safe_video_id = re.sub(r'[^\w\-_]', '', video_id)

    # 1. Remove from vector store
    proj_vector_store = LocalVectorStore(data_dir=project_dir)
    proj_vector_store.remove_video(safe_video_id)

    # 2. Delete JSON package in knowledge folder
    json_path = os.path.join(project_dir, "knowledge", f"video_{safe_video_id}.json")
    if os.path.exists(json_path):
        try:
            os.remove(json_path)
        except Exception as e:
            print(f"Error removing json file: {e}")

    # 3. Delete markdown files in markdown folder
    md_path = os.path.join(project_dir, "markdown", f"video_{safe_video_id}.md")
    if os.path.exists(md_path):
        try:
            os.remove(md_path)
        except Exception as e:
            print(f"Error removing md file: {e}")

    md_polished_path = os.path.join(project_dir, "markdown", f"video_{safe_video_id}_polished.md")
    if os.path.exists(md_polished_path):
        try:
            os.remove(md_polished_path)
        except Exception as e:
            print(f"Error removing polished md file: {e}")

    return {"video_id": safe_video_id, "message": "Video successfully deleted"}


@router.delete("/videos")
def api_delete_all_videos(project_id: str = "default"):
    """Deletes all videos in the project knowledge base, resetting the search index."""
    project_dir = get_project_dir(project_id)

    # 1. Reset vector store
    proj_vector_store = LocalVectorStore(data_dir=project_dir)
    proj_vector_store.clear_all()

    # 2. Clear knowledge folder
    knowledge_dir = os.path.join(project_dir, "knowledge")
    if os.path.exists(knowledge_dir):
        for f in os.listdir(knowledge_dir):
            if f.endswith(".json"):
                try:
                    os.remove(os.path.join(knowledge_dir, f))
                except Exception as e:
                    print(f"Error removing file: {e}")

    # 3. Clear markdown folder
    markdown_dir = os.path.join(project_dir, "markdown")
    if os.path.exists(markdown_dir):
        for f in os.listdir(markdown_dir):
            if f.endswith(".md"):
                try:
                    os.remove(os.path.join(markdown_dir, f))
                except Exception as e:
                    print(f"Error removing file: {e}")

    return {"message": "All videos successfully deleted"}
