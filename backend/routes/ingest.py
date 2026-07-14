"""Video ingestion endpoints and background pipelines."""

import os
import re
import json
import hashlib
import traceback
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from backend.pipeline.downloader import VideoDownloader
from backend.pipeline.processor import TranscriptProcessor
from backend.pipeline.project_manager import get_project_dir
from backend.pipeline.transcriber import AudioTranscriber
from backend.rag.vector_store import LocalVectorStore
from backend.services.embedding import resolve_embedding_provider
from backend.task_store import TaskStore

router = APIRouter(prefix="/api", tags=["ingest"])

# Persistent task stores
DATA_DIR = os.getenv("DATA_DIR", "./backend/data")
ingestion_tasks = TaskStore(os.path.join(DATA_DIR, "tasks_ingestion.json"))
batch_tasks = TaskStore(os.path.join(DATA_DIR, "tasks_batch.json"))


# ------------------------------------------------------------------
# Google Drive Ingestion Helpers
# ------------------------------------------------------------------

def get_google_drive_file_id(url: str) -> Optional[str]:
    """Extracts Google Drive file ID from a shareable link."""
    patterns = [
        r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)',
        r'drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)',
        r'drive\.google\.com/uc\?id=([a-zA-Z0-9_-]+)',
        r'docs\.google\.com/file/d/([a-zA-Z0-9_-]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def download_google_drive_file(file_id: str, destination: str, progress_callback=None) -> str:
    """Downloads a public Google Drive file (supporting files > 100MB by bypassing warning page)"""
    import requests
    url = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    if progress_callback:
        progress_callback("connecting_to_drive", 10)

    # Step 1: Send request to get confirmation token if it is a large file
    response = session.get(url, params={'id': file_id}, stream=True)
    
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
            
    # Step 2: Re-request with confirm token if present
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(url, params=params, stream=True)
        
    if response.status_code != 200:
        raise Exception(f"Failed to download file from Google Drive. HTTP Status: {response.status_code}")

    # Extract filename from headers if possible
    disposition = response.headers.get("Content-Disposition", "")
    filename = "downloaded_file.mp4"
    if "filename=" in disposition:
        matches = re.findall(r'filename="?([^";\n]+)"?', disposition)
        if matches:
            filename = matches[0]

    # Content length for progress calculation
    total_length = response.headers.get('content-length')
    
    if progress_callback:
        progress_callback("downloading_from_drive", 20)

    # Save to file
    with open(destination, "wb") as f:
        if total_length is None:
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for chunk in response.iter_content(chunk_size=40960):
                dl += len(chunk)
                f.write(chunk)
                if progress_callback:
                    # Let downloading take up to 60% of progress
                    percent = int(20 + (dl / total_length) * 40)
                    progress_callback(f"downloading_from_drive ({percent}%)", percent)
                    
    return filename


def process_google_drive_ingestion(
    url: str,
    drive_id: str,
    project_dir: str,
    gemini_key: str = None,
    openai_key: str = None,
    progress_callback=None
) -> dict:
    """Handles the complete Google Drive download and ASR transcription pipeline."""
    # Setup paths
    temp_dir = os.path.join(project_dir, "temp_downloads")
    os.makedirs(temp_dir, exist_ok=True)
    temp_filepath = os.path.join(temp_dir, f"drive_{drive_id}")
    real_filepath = temp_filepath

    try:
        # Download
        drive_filename = download_google_drive_file(drive_id, temp_filepath, progress_callback=progress_callback)
        
        # Ensure we have a valid media extension for Google Generative AI to detect the MIME type
        supported_exts = {'.mp4', '.mp3', '.wav', '.mov', '.avi', '.mkv', '.webm', '.m4a', '.aac', '.flac', '.ogg'}
        _, ext = os.path.splitext(drive_filename)
        ext = ext.lower()
        if ext not in supported_exts:
            ext = ".mp4"  # Default to mp4 if unsupported or missing

        real_filepath = temp_filepath + ext
        if os.path.exists(temp_filepath):
            if os.path.exists(real_filepath):
                try:
                    os.remove(real_filepath)
                except:
                    pass
            os.rename(temp_filepath, real_filepath)

        clean_title = os.path.splitext(drive_filename)[0]
        clean_title = re.sub(r'[_\-\s]+', ' ', clean_title).strip()
        if not clean_title:
            clean_title = f"Drive Video {drive_id}"

        # Transcription provider auto-resolution
        t_provider = None
        if gemini_key and gemini_key.strip():
            t_provider = "gemini"
        elif openai_key and openai_key.strip():
            t_provider = "openai"
        else:
            g_env = os.getenv("GEMINI_API_KEY")
            o_env = os.getenv("OPENAI_API_KEY")
            if g_env and g_env.strip():
                t_provider = "gemini"
            elif o_env and o_env.strip():
                t_provider = "openai"
            else:
                raise ValueError(
                    "No transcription API Key provided. Google Drive videos require a "
                    "Gemini or OpenAI API Key to generate subtitles."
                )

        transcriber = AudioTranscriber(
            gemini_api_key=gemini_key,
            openai_api_key=openai_key
        )
        
        def trans_progress_cb(status, pct):
            # Map transcription progress (from 60% to 80%)
            mapped_pct = int(60 + (pct / 100.0) * 20)
            if progress_callback:
                progress_callback(f"transcribing ({status})", mapped_pct)

        if progress_callback:
            progress_callback("transcribing", 60)
            
        segments = transcriber.transcribe(
            audio_path=real_filepath,
            provider=t_provider,
            progress_callback=trans_progress_cb
        )

        sub_info = {
            "id": f"drive_{drive_id}",
            "title": clean_title,
            "duration": segments[-1]["end"] if segments else 0.0,
            "uploader": "Google Drive",
            "webpage_url": url,
            "extractor": "google_drive",
            "segments": segments
        }
        return sub_info

    finally:
        # Clean up local files
        for path in (temp_filepath, real_filepath):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                print(f"Warning: Failed to delete file {path}: {e}")


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

        # Step 1: Download and extract transcript segments
        drive_id = get_google_drive_file_id(url)
        if drive_id:
            # Check for API key before downloading
            has_api_key = (
                (gemini_key and gemini_key.strip())
                or (openai_key and openai_key.strip())
                or os.getenv("GEMINI_API_KEY")
                or os.getenv("OPENAI_API_KEY")
            )
            if not has_api_key:
                raise ValueError("API Key for Gemini or OpenAI is required to transcribe Google Drive files.")

            ingestion_tasks.update(task_id, {
                "status": "downloading_from_drive",
                "percent": 10.0,
                "title": "Connecting to Google Drive...",
                "current_video_title": url,
                "current_video_status": "downloading_from_drive",
                "current_video_percent": 10.0,
                "overall_percent": 10.0,
                "error": None,
            })
            ingestion_tasks.update_nested(task_id, ["videos", 0], {
                "status": "downloading_from_drive",
                "percent": 10.0,
            })

            def drive_progress(status, pct):
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

            sub_info = process_google_drive_ingestion(
                url=url,
                drive_id=drive_id,
                project_dir=project_dir,
                gemini_key=gemini_key,
                openai_key=openai_key,
                progress_callback=drive_progress
            )
        else:
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
            drive_id = get_google_drive_file_id(url)
            if drive_id:
                # Double-check API key in batch loop too
                has_api_key = (
                    (gemini_key and gemini_key.strip())
                    or (openai_key and openai_key.strip())
                    or os.getenv("GEMINI_API_KEY")
                    or os.getenv("OPENAI_API_KEY")
                )
                if not has_api_key:
                    raise ValueError("API Key for Gemini or OpenAI is required to transcribe Google Drive files.")

                def drive_progress(status, pct, _idx=idx):
                    batch_tasks.update(batch_task_id, {
                        "current_video_status": status,
                        "current_video_percent": pct,
                        "overall_percent": ((_idx + (pct / 100.0 * 0.7)) / len(urls)) * 100,
                    })
                    batch_tasks.update_nested(batch_task_id, ["videos", _idx], {
                        "status": status,
                        "percent": pct,
                    })

                sub_info = process_google_drive_ingestion(
                    url=url,
                    drive_id=drive_id,
                    project_dir=project_dir,
                    gemini_key=gemini_key,
                    openai_key=openai_key,
                    progress_callback=drive_progress
                )
            else:
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

    # Validate API key for Google Drive videos
    drive_id = get_google_drive_file_id(clean_url)
    if drive_id:
        has_api_key = (
            (req.gemini_key and req.gemini_key.strip())
            or (req.openai_key and req.openai_key.strip())
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not has_api_key:
            raise HTTPException(
                status_code=400,
                detail="Google Drive video ingestion requires a Gemini or OpenAI API Key to generate subtitles. Please configure an API Key in the Settings modal first."
            )

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

    # Validate API key for Google Drive videos in batch
    has_drive_links = any(get_google_drive_file_id(url) is not None for url in urls)
    if has_drive_links:
        has_api_key = (
            (req.gemini_key and req.gemini_key.strip())
            or (req.openai_key and req.openai_key.strip())
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not has_api_key:
            raise HTTPException(
                status_code=400,
                detail="One or more Google Drive videos require a Gemini or OpenAI API Key to generate subtitles. Please configure an API Key in the Settings modal first."
            )

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
