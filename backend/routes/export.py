"""Export and markdown download endpoints."""

import os
import re
import json
import unicodedata
from urllib.parse import quote
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Response, Header
from fastapi.responses import StreamingResponse, JSONResponse

from backend.pipeline.project_manager import get_project_dir, generate_markdown, format_timestamp
from backend.services.polish import polish_transcript_background
from backend.task_store import TaskStore

router = APIRouter(prefix="/api", tags=["export"])

# Persistent polishing task store
DATA_DIR = os.getenv("DATA_DIR", "./backend/data")
polishing_tasks = TaskStore(os.path.join(DATA_DIR, "tasks_polishing.json"))


@router.get("/videos/{video_id}/export")
def export_video_markdown(
    video_id: str,
    project_id: str = "default",
    use_ai: bool = False,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    background_tasks: BackgroundTasks = None,
    x_groq_key: str = Header(None, alias="X-Groq-Key"),
    x_openai_key: str = Header(None, alias="X-Openai-Key"),
    x_gemini_key: str = Header(None, alias="X-Gemini-Key"),
    x_deepseek_key: str = Header(None, alias="X-Deepseek-Key"),
    x_openrouter_key: str = Header(None, alias="X-Openrouter-Key"),
    x_llm7_key: str = Header(None, alias="X-Llm7-Key"),
):
    """Generates and serves a formatted Markdown transcription file.

    If use_ai is True, uses AI to format, summarize, and polish the transcript.
    """
    project_dir = get_project_dir(project_id)
    safe_video_id = re.sub(r'[^\w\-_]', '', video_id)
    knowledge_file = os.path.join(project_dir, "knowledge", f"video_{safe_video_id}.json")

    if not os.path.exists(knowledge_file):
        raise HTTPException(status_code=404, detail=f"Transcript package not found for video: {video_id}")

    try:
        with open(knowledge_file, "r", encoding="utf-8") as f:
            package = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read transcript package: {str(e)}")

    metadata = package.get("metadata", {})
    segments = package.get("segments", [])

    if not segments:
        # Fallback to chunk data if segments key is missing
        segments = []
        for chunk in package.get("chunks", []):
            segments.append({
                "start": chunk["start"],
                "end": chunk["end"],
                "text": chunk["text"],
            })

    markdown_content = generate_markdown(metadata, segments)

    if use_ai:
        polished_path = os.path.join(project_dir, "markdown", f"video_{safe_video_id}_polished.md")
        if os.path.exists(polished_path):
            with open(polished_path, "r", encoding="utf-8") as md_f:
                markdown_content = md_f.read()
        else:
            task_id = f"polish_{safe_video_id}"
            existing = polishing_tasks.get(task_id)
            if existing is None or existing.get("status") == "failed":
                polishing_tasks.set(task_id, {
                    "status": "pending",
                    "percent": 0.0,
                    "error": None,
                })
                keys = {
                    "gemini_key": x_gemini_key,
                    "openai_key": x_openai_key,
                    "groq_key": x_groq_key,
                    "deepseek_key": x_deepseek_key,
                    "openrouter_key": x_openrouter_key,
                    "llm7_key": x_llm7_key,
                }
                background_tasks.add_task(
                    polish_transcript_background,
                    task_id=task_id,
                    project_dir=project_dir,
                    safe_video_id=safe_video_id,
                    metadata=metadata,
                    segments=segments,
                    provider=provider,
                    model=model,
                    keys=keys,
                    polishing_tasks=polishing_tasks,
                )

            current_status = polishing_tasks.get(task_id, {})
            return JSONResponse(
                status_code=202,
                content={
                    "status": current_status.get("status", "pending"),
                    "percent": current_status.get("percent", 0.0),
                    "task_id": task_id,
                    "message": "Polishing task started in background.",
                },
            )
    else:
        # Save a copy of the raw markdown in the project's markdown directory
        md_dir = os.path.join(project_dir, "markdown")
        os.makedirs(md_dir, exist_ok=True)
        md_filepath = os.path.join(md_dir, f"video_{safe_video_id}.md")
        with open(md_filepath, "w", encoding="utf-8") as md_f:
            md_f.write(markdown_content)

    # Serve as file attachment
    raw_title = metadata.get("title", video_id)
    suffix = "_polished" if use_ai else ""

    # UTF-8 filename for modern browsers (preserves Vietnamese diacritics)
    utf8_title = re.sub(r'[\\/:*?"<>|]', '_', raw_title)
    utf8_title = re.sub(r'_+', '_', utf8_title).strip('_')
    if not utf8_title:
        utf8_title = "video_" + re.sub(r'[^\w\-_]', '_', video_id)
    utf8_filename = f"{utf8_title}{suffix}.md"

    # ASCII fallback: transliterate diacritics (e.g. "GIẢI" → "GIAI") instead of replacing with "_"
    nfkd = unicodedata.normalize('NFKD', utf8_title)
    ascii_title = nfkd.encode('ascii', 'ignore').decode('ascii')
    ascii_title = re.sub(r'[^\w\-_]', '_', ascii_title)
    ascii_title = re.sub(r'_+', '_', ascii_title).strip('_')
    if not ascii_title:
        ascii_title = "video_" + re.sub(r'[^\w\-_]', '_', video_id)
    ascii_filename = f"{ascii_title}{suffix}.md"

    return Response(
        content=markdown_content,
        media_type="text/markdown",
        headers={
            "Content-Disposition": f"attachment; filename=\"{ascii_filename}\"; filename*=UTF-8''{quote(utf8_filename)}"
        },
    )


@router.get("/videos/{video_id}/polish-status")
def get_polish_status(video_id: str, project_id: str = "default"):
    project_dir = get_project_dir(project_id)
    safe_video_id = re.sub(r'[^\w\-_]', '', video_id)
    polished_path = os.path.join(project_dir, "markdown", f"video_{safe_video_id}_polished.md")

    if os.path.exists(polished_path):
        return {
            "status": "completed",
            "percent": 100.0,
            "error": None,
        }

    task_id = f"polish_{safe_video_id}"
    if task_id not in polishing_tasks:
        return {
            "status": "not_started",
            "percent": 0.0,
            "error": None,
        }

    return polishing_tasks.get(task_id)


@router.get("/projects/{project_id}/export-batch")
def export_project_batch(project_id: str):
    """Generates a ZIP archive containing all VID transcripts in the project formatted as Markdown."""
    import io
    import zipfile

    project_dir = get_project_dir(project_id)
    knowledge_dir = os.path.join(project_dir, "knowledge")

    if not os.path.exists(knowledge_dir):
        raise HTTPException(status_code=404, detail="No knowledge base found for this project.")

    json_files = [f for f in os.listdir(knowledge_dir) if f.endswith(".json")]
    if not json_files:
        raise HTTPException(status_code=404, detail="No ingested videos found in this project.")

    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for f_name in json_files:
            file_path = os.path.join(knowledge_dir, f_name)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    package = json.load(f)

                metadata = package.get("metadata", {})
                segments = package.get("segments", [])

                if not segments:
                    segments = []
                    for chunk in package.get("chunks", []):
                        segments.append({
                            "start": chunk["start"],
                            "end": chunk["end"],
                            "text": chunk["text"],
                        })

                markdown_content = generate_markdown(metadata, segments)

                # Make a safe filename (preserve Vietnamese diacritics)
                vid_id = package.get("video_id", f_name.replace("video_", "").replace(".json", ""))
                raw_title = metadata.get("title", vid_id)
                safe_title = re.sub(r'[\\/:*?"<>|]', '_', raw_title)
                safe_title = re.sub(r'_+', '_', safe_title).strip('_')
                if not safe_title:
                    safe_title = "video_" + re.sub(r'[^\w\-_]', '_', vid_id)
                filename = f"{safe_title}.md"

                zip_file.writestr(filename, markdown_content.encode('utf-8'))
            except Exception as e:
                print(f"Error adding {f_name} to batch ZIP: {e}")

    zip_buffer.seek(0)

    # UTF-8 filename for ZIP
    utf8_project_id = re.sub(r'[\\/:*?"<>|]', '_', project_id)
    utf8_project_id = re.sub(r'_+', '_', utf8_project_id).strip('_')
    if not utf8_project_id:
        utf8_project_id = "default"
    utf8_zip_name = f"transcripts_{utf8_project_id}.zip"

    # ASCII fallback
    nfkd = unicodedata.normalize('NFKD', utf8_project_id)
    ascii_project_id = nfkd.encode('ascii', 'ignore').decode('ascii')
    ascii_project_id = re.sub(r'[^\w\-_]', '_', ascii_project_id)
    ascii_project_id = re.sub(r'_+', '_', ascii_project_id).strip('_')
    if not ascii_project_id:
        ascii_project_id = "default"
    ascii_zip_name = f"transcripts_{ascii_project_id}.zip"

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=\"{ascii_zip_name}\"; filename*=UTF-8''{quote(utf8_zip_name)}"
        },
    )
