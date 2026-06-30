import os
import uuid
import hashlib
import re
import json
from typing import List, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException, Response, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Import our custom modules
from backend.pipeline.downloader import VideoDownloader
from backend.pipeline.processor import TranscriptProcessor
from backend.rag.vector_store import LocalVectorStore
from backend.rag.chatbot import RAGChatbot
from backend.pipeline.project_manager import (
    get_project_dir,
    list_projects,
    create_project,
    delete_project,
    generate_markdown
)

# Load environment variables
load_dotenv(override=True)

app = FastAPI(title="VidChatBox API", version="1.0.0")

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global status tracking dicts
# task_id -> { "status": str, "percent": float, "title": str, "error": str }
ingestion_tasks = {}

# batch_task_id -> { "status": str, "total_videos": int, "completed_videos": int, "current_index": int, ... }
batch_tasks = {}

# Initialize configurations
DATA_DIR = os.getenv("DATA_DIR", "./backend/data")

# Create default project folders on startup to ensure system is ready
create_project("default")

class IngestRequest(BaseModel):
    url: str
    provider: str = "gemini" # "gemini" or "openai"
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

class CreateProjectRequest(BaseModel):
    name: str

class ChatRequest(BaseModel):
    query: str
    provider: str = "gemini" # or "openai"
    model: str = ""
    persona_mode: str = "streamer" # or "assistant"
    project_id: str = "default"


def run_ingestion_pipeline(task_id: str, url: str, provider: str = "gemini", project_id: str = "default", openai_key: str = None, gemini_key: str = None, openrouter_key: str = None):
    """Executes ingestion using subtitles directly extracted from YouTube (no AI transcription used)."""
    try:
        project_dir = get_project_dir(project_id)
        proj_downloader = VideoDownloader(data_dir=project_dir)
        proj_processor = TranscriptProcessor(data_dir=project_dir)
        proj_vector_store = LocalVectorStore(data_dir=project_dir)

        # Step 1: Extract info and download subtitles directly
        ingestion_tasks[task_id].update({
            "status": "extracting_metadata",
            "percent": 20.0,
            "title": "Extracting metadata and subtitles...",
            "current_video_title": url,
            "current_video_status": "extracting_metadata",
            "current_video_percent": 20.0,
            "overall_percent": 20.0,
            "error": None
        })
        ingestion_tasks[task_id]["videos"][0].update({
            "status": "extracting_metadata",
            "percent": 20.0
        })
        
        def sub_progress(status, pct):
            ingestion_tasks[task_id].update({
                "status": "downloading_subtitles",
                "percent": pct,
                "current_video_status": "downloading_subtitles",
                "current_video_percent": pct,
                "overall_percent": pct
            })
            ingestion_tasks[task_id]["videos"][0].update({
                "status": "downloading_subtitles",
                "percent": pct
            })
            
        sub_info = proj_downloader.download_subtitles(url, progress_callback=sub_progress)
        video_id = sub_info["id"]
        title = sub_info["title"]
        segments = sub_info["segments"]
        
        ingestion_tasks[task_id].update({
            "title": title,
            "status": "processing",
            "percent": 80.0,
            "current_video_title": title,
            "current_video_status": "processing",
            "current_video_percent": 80.0,
            "overall_percent": 80.0
        })
        ingestion_tasks[task_id]["videos"][0].update({
            "title": title,
            "status": "processing",
            "percent": 80.0
        })
        
        # Step 2: Chunk & Process
        _, chunks = proj_processor.save_knowledge_package(sub_info, segments)
        
        # Step 3: Embed & Index in Vector DB
        ingestion_tasks[task_id].update({
            "status": "indexing",
            "percent": 90.0,
            "title": f"{title} - Indexing embeddings in search database...",
            "current_video_status": "indexing",
            "current_video_percent": 90.0,
            "overall_percent": 90.0
        })
        ingestion_tasks[task_id]["videos"][0].update({
            "status": "indexing",
            "percent": 90.0
        })
        
        # Resolve valid embedding provider (gemini, openai, or openrouter)
        emb_provider = "gemini"
        if provider in ["gemini", "openai", "openrouter"]:
            emb_provider = provider
        else:
            if gemini_key or os.getenv("GEMINI_API_KEY"):
                emb_provider = "gemini"
            elif openai_key or os.getenv("OPENAI_API_KEY"):
                emb_provider = "openai"
            else:
                emb_provider = "openrouter"

        if emb_provider == "gemini":
            emb_key = gemini_key
        elif emb_provider == "openai":
            emb_key = openai_key
        else:
            emb_key = openrouter_key
        proj_vector_store.add_video_chunks(video_id, chunks, provider=emb_provider, api_key=emb_key)
        
        # Complete
        ingestion_tasks[task_id].update({
            "status": "completed",
            "percent": 100.0,
            "title": title,
            "completed_videos": 1,
            "current_video_status": "completed",
            "current_video_percent": 100.0,
            "overall_percent": 100.0
        })
        ingestion_tasks[task_id]["videos"][0].update({
            "status": "completed",
            "percent": 100.0
        })
                
    except Exception as e:
        import traceback
        traceback.print_exc()
        ingestion_tasks[task_id].update({
            "status": "failed",
            "percent": 0.0,
            "current_video_status": "failed",
            "current_video_percent": 0.0,
            "overall_percent": 0.0,
            "error": str(e)
        })
        ingestion_tasks[task_id]["videos"][0].update({
            "status": "failed",
            "error": str(e),
            "percent": 0.0
        })


def run_batch_ingestion_pipeline(batch_task_id: str, urls: List[str], provider: str, project_id: str, openai_key: str = None, gemini_key: str = None, openrouter_key: str = None):
    """Executes sequential batch ingestion by extracting subtitles directly (no AI used)."""
    batch = batch_tasks[batch_task_id]
    batch["status"] = "processing"
    
    project_dir = get_project_dir(project_id)
    proj_downloader = VideoDownloader(data_dir=project_dir)
    proj_processor = TranscriptProcessor(data_dir=project_dir)
    proj_vector_store = LocalVectorStore(data_dir=project_dir)
    
    for idx, url in enumerate(urls):
        current_idx = idx + 1
        batch["current_index"] = current_idx
        
        video_entry = batch["videos"][idx]
        video_entry["status"] = "extracting_metadata"
        video_entry["percent"] = 10.0
        
        batch["current_video_status"] = "extracting_metadata"
        batch["current_video_percent"] = 10.0
        batch["current_video_title"] = url
        batch["overall_percent"] = (idx / len(urls)) * 100
        
        try:
            # Download subtitles
            def sub_progress(status, pct):
                batch["current_video_status"] = "downloading_subtitles"
                batch["current_video_percent"] = pct
                batch["overall_percent"] = ((idx + (pct / 100.0 * 0.7)) / len(urls)) * 100
                video_entry["status"] = "downloading_subtitles"
                video_entry["percent"] = pct
                
            sub_info = proj_downloader.download_subtitles(url, progress_callback=sub_progress)
            video_id = sub_info["id"]
            title = sub_info["title"]
            segments = sub_info["segments"]
            
            batch["current_video_title"] = title
            video_entry["title"] = title
            
            # Chunk & Save package
            batch["current_video_status"] = "processing_package"
            batch["current_video_percent"] = 80.0
            batch["overall_percent"] = ((idx + 0.8) / len(urls)) * 100
            
            video_entry["status"] = "processing_package"
            video_entry["percent"] = 80.0
            
            _, chunks = proj_processor.save_knowledge_package(sub_info, segments)
            
            # Embed & Save
            batch["current_video_status"] = "indexing"
            batch["current_video_percent"] = 95.0
            batch["overall_percent"] = ((idx + 0.95) / len(urls)) * 100
            
            video_entry["status"] = "indexing"
            video_entry["percent"] = 95.0
            
            # Resolve valid embedding provider (gemini, openai, or openrouter)
            emb_provider = "gemini"
            if provider in ["gemini", "openai", "openrouter"]:
                emb_provider = provider
            else:
                if gemini_key or os.getenv("GEMINI_API_KEY"):
                    emb_provider = "gemini"
                elif openai_key or os.getenv("OPENAI_API_KEY"):
                    emb_provider = "openai"
                else:
                    emb_provider = "openrouter"

            if emb_provider == "gemini":
                emb_key = gemini_key
            elif emb_provider == "openai":
                emb_key = openai_key
            else:
                emb_key = openrouter_key
            proj_vector_store.add_video_chunks(video_id, chunks, provider=emb_provider, api_key=emb_key)
            
            batch["completed_videos"] += 1
            batch["current_video_status"] = "completed"
            batch["current_video_percent"] = 100.0
            batch["overall_percent"] = (current_idx / len(urls)) * 100
            
            video_entry["status"] = "completed"
            video_entry["percent"] = 100.0
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            video_entry["status"] = "failed"
            video_entry["error"] = str(e)
            video_entry["percent"] = 0.0
            
            batch["current_video_status"] = "failed"
            batch["current_video_percent"] = 0.0
            batch["overall_percent"] = (current_idx / len(urls)) * 100
            
    # Finalize batch task status
    if batch["completed_videos"] == len(urls):
        batch["status"] = "completed"
    elif batch["completed_videos"] == 0:
        batch["status"] = "failed"
        batch["error"] = "All videos in the batch failed to process."
    else:
        batch["status"] = "completed_with_errors"


# Project Management endpoints
@app.get("/api/projects")
def api_list_projects():
    """Lists all projects."""
    return list_projects()

@app.post("/api/projects")
def api_create_project(req: CreateProjectRequest):
    """Creates a new project directory."""
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Project name cannot be empty")
    safe_name = create_project(name)
    return {"project_id": safe_name, "message": f"Project '{safe_name}' created successfully"}

@app.delete("/api/projects/{project_id}")
def api_delete_project(project_id: str):
    """Deletes a project recursively."""
    if project_id == "default":
        delete_project("default")
        return {"project_id": "default", "message": "Default project cleared"}
    delete_project(project_id)
    return {"project_id": project_id, "message": f"Project '{project_id}' deleted successfully"}


# Ingestion endpoints
@app.post("/api/ingest")
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
        task = ingestion_tasks[task_id]
        if task["status"] not in ["failed", "completed"]:
            return {"task_id": task_id, "status": task["status"], "message": "Task already running"}
            
    # Initialize task status
    ingestion_tasks[task_id] = {
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
        "videos": [{"url": clean_url, "title": clean_url, "status": "pending", "percent": 0.0, "error": None}]
    }
    
    background_tasks.add_task(run_ingestion_pipeline, task_id, clean_url, req.provider, req.project_id, req.openai_key, req.gemini_key, req.openrouter_key)
    return {"task_id": task_id, "status": "pending", "message": "Ingestion started in background"}


@app.post("/api/ingest-batch")
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
        task = batch_tasks[batch_task_id]
        if task["status"] in ["pending", "processing"]:
            return {"batch_task_id": batch_task_id, "status": task["status"], "message": "Batch task already running"}
            
    batch_tasks[batch_task_id] = {
        "status": "pending",
        "total_videos": len(urls),
        "completed_videos": 0,
        "current_index": 0,
        "current_video_title": "",
        "current_video_status": "pending",
        "current_video_percent": 0.0,
        "overall_percent": 0.0,
        "videos": [{"url": url, "title": url, "status": "pending", "percent": 0.0, "error": None} for url in urls],
        "error": None
    }
    
    background_tasks.add_task(run_batch_ingestion_pipeline, batch_task_id, urls, req.provider, req.project_id, req.openai_key, req.gemini_key, req.openrouter_key)
    return {"batch_task_id": batch_task_id, "status": "pending", "message": "Batch ingestion started"}


@app.get("/api/status/{task_id}")
def get_status(task_id: str):
    """Returns the progress of a specific ingestion task."""
    if task_id not in ingestion_tasks:
        raise HTTPException(status_code=404, detail="Ingestion task not found")
    return ingestion_tasks[task_id]


@app.get("/api/batch-status/{batch_task_id}")
def get_batch_status(batch_task_id: str):
    """Returns the progress of a batch ingestion task."""
    if batch_task_id not in batch_tasks:
        raise HTTPException(status_code=404, detail="Batch task not found")
    return batch_tasks[batch_task_id]


@app.get("/api/videos")
def list_videos(project_id: str = "default"):
    """Returns a list of all successfully ingested videos with metadata within a project."""
    proj_vector_store = LocalVectorStore(data_dir=get_project_dir(project_id))
    unique_videos = []
    seen_ids = set()
    
    for chunk in proj_vector_store.data.get("chunks", []):
        v_id = chunk["video_id"]
        if v_id not in seen_ids:
            seen_ids.add(v_id)
            unique_videos.append({
                "id": v_id,
                "title": chunk["video_title"],
                "url": chunk["video_url"]
            })
            
    return unique_videos


@app.delete("/api/videos/{video_id}")
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


@app.delete("/api/videos")
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


@app.get("/api/videos/{video_id}/export")
def export_video_markdown(video_id: str, project_id: str = "default", use_ai: bool = False):
    """Generates and serves a formatted Markdown transcription file.
    
    If use_ai is True, uses Gemini to format, summarize, and polish the transcript.
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
                "text": chunk["text"]
            })
            
    markdown_content = generate_markdown(metadata, segments)
    
    if use_ai:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=400, detail="GEMINI_API_KEY is not set. Cannot run AI Markdown polishing.")
            
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            system_prompt = (
                "You are an expert technical editor and content summarizer. "
                "Your task is to take a raw timestamped video transcript in Markdown and format it into a premium, readable, structured document.\n\n"
                "CRITICAL RULES:\n"
                "1. You MUST preserve the exact chronological order of the statements.\n"
                "2. You MUST keep the exact markdown timestamp citation links, e.g., `[HH:MM:SS](URL_WITH_TIMESTAMP)`. Never change the URLs or the link texts.\n"
                "3. Correct grammar, spelling, and sentence flow, removing filler words (like 'um', 'uh', stuttering) while maintaining the original meaning and tone.\n"
                "4. Group segments into logical chapters/sections with descriptive H2 (`##`) headings.\n"
                "5. Under each heading, provide a brief 1-2 sentence bulleted summary of what was discussed in that section, followed by the polished chronological timestamped bullet points.\n"
                "6. Return ONLY the polished Markdown content. Do not add conversational intro/outro text or markdown block wrappers (e.g. ```markdown)."
            )
            
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=system_prompt
            )
            
            response = model.generate_content(markdown_content)
            polished_markdown = response.text.strip()
            
            if polished_markdown.startswith("```"):
                polished_markdown = re.sub(r"^```(?:markdown)?\n", "", polished_markdown)
                polished_markdown = re.sub(r"\n```$", "", polished_markdown)
                polished_markdown = polished_markdown.strip()
                
            markdown_content = polished_markdown
        except Exception as ai_err:
            print(f"Warning: AI polishing failed, falling back to raw markdown. Error: {ai_err}")
            
    # Save a copy in the project's markdown directory
    md_dir = os.path.join(project_dir, "markdown")
    os.makedirs(md_dir, exist_ok=True)
    suffix = "_polished.md" if use_ai else ".md"
    md_filepath = os.path.join(md_dir, f"video_{safe_video_id}{suffix}")
    with open(md_filepath, "w", encoding="utf-8") as md_f:
        md_f.write(markdown_content)
        
    # Serve as file attachment
    raw_title = metadata.get("title", video_id)
    # Clean non-ASCII to prevent latin-1 HTTP header encoding error
    ascii_title = re.sub(r'[^\x00-\x7F]', '_', raw_title)
    safe_title = re.sub(r'[^\w\-_]', '_', ascii_title)
    safe_title = re.sub(r'_+', '_', safe_title).strip('_')
    if not safe_title:
        safe_title = "video_" + re.sub(r'[^\w\-_]', '_', video_id)
    filename = f"{safe_title}_polished.md" if use_ai else f"{safe_title}.md"
    
    return Response(
        content=markdown_content,
        media_type="text/markdown",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


@app.get("/api/projects/{project_id}/export-batch")
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
                            "text": chunk["text"]
                        })
                
                markdown_content = generate_markdown(metadata, segments)
                
                # Make a safe filename
                video_id = package.get("video_id", f_name.replace("video_", "").replace(".json", ""))
                raw_title = metadata.get("title", video_id)
                # Clean non-ASCII to prevent ZipFile file-system decoding issues
                ascii_title = re.sub(r'[^\x00-\x7F]', '_', raw_title)
                safe_title = re.sub(r'[^\w\-_]', '_', ascii_title)
                safe_title = re.sub(r'_+', '_', safe_title).strip('_')
                if not safe_title:
                    safe_title = "video_" + re.sub(r'[^\w\-_]', '_', video_id)
                filename = f"{safe_title}.md"
                
                zip_file.writestr(filename, markdown_content)
            except Exception as e:
                print(f"Error adding {f_name} to batch ZIP: {e}")
                
    zip_buffer.seek(0)
    
    ascii_project_id = re.sub(r'[^\x00-\x7F]', '_', project_id)
    safe_project_id = re.sub(r'[^\w\-_]', '_', ascii_project_id)
    safe_project_id = re.sub(r'_+', '_', safe_project_id).strip('_')
    if not safe_project_id:
        safe_project_id = "default"
        
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="transcripts_{safe_project_id}.zip"'
        }
    )


@app.post("/api/chat")
def run_chat(
    req: ChatRequest,
    x_groq_key: str = Header(None, alias="X-Groq-Key"),
    x_openai_key: str = Header(None, alias="X-Openai-Key"),
    x_gemini_key: str = Header(None, alias="X-Gemini-Key"),
    x_deepseek_key: str = Header(None, alias="X-Deepseek-Key"),
    x_openrouter_key: str = Header(None, alias="X-Openrouter-Key")
):
    """Executes RAG similarity search and streams LLM response."""
    # Instantiate project-specific vector store first to read stored embedding dimension
    proj_vector_store = LocalVectorStore(data_dir=get_project_dir(req.project_id))
    
    # Determine embedding provider based on available keys and provider selection
    gemini_key = x_gemini_key or os.getenv("GEMINI_API_KEY")
    openai_key = x_openai_key or os.getenv("OPENAI_API_KEY")
    openrouter_key = x_openrouter_key or os.getenv("OPENROUTER_API_KEY")
    
    g_key_valid = bool(gemini_key and gemini_key.strip())
    o_key_valid = bool(openai_key and openai_key.strip())
    or_key_valid = bool(openrouter_key and openrouter_key.strip())
    
    if req.provider in ["gemini", "openai", "openrouter"]:
        embedding_provider = req.provider
    else:
        if g_key_valid:
            embedding_provider = "gemini"
        elif o_key_valid:
            embedding_provider = "openai"
        elif or_key_valid:
            embedding_provider = "openrouter"
        else:
            g_env = os.getenv("GEMINI_API_KEY")
            o_env = os.getenv("OPENAI_API_KEY")
            or_env = os.getenv("OPENROUTER_API_KEY")
            if g_env and g_env.strip():
                embedding_provider = "gemini"
            elif o_env and o_env.strip():
                embedding_provider = "openai"
            else:
                embedding_provider = "openrouter"

    # Auto-align embedding provider with stored embeddings in the database to prevent alignment errors
    if proj_vector_store.data.get("chunks"):
        stored_dim = len(proj_vector_store.data["chunks"][0]["embedding"])
        if stored_dim == 3072 and embedding_provider != "gemini":
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
    else:
        emb_key = openrouter_key

    try:
        search_results = proj_vector_store.search(req.query, top_k=5, provider=embedding_provider, api_key=emb_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
        
    if not search_results:
        # Return static streaming response indicating no data found
        def empty_generator():
            yield "I don't have any streamer VID data indexed in this group yet! Please ingest some video URLs first using the Ingest VIDs tab."
        return StreamingResponse(empty_generator(), media_type="text/plain")

    # Generate and stream Chat response with user custom keys
    chatbot = RAGChatbot(
        gemini_api_key=x_gemini_key,
        openai_api_key=x_openai_key,
        groq_api_key=x_groq_key,
        deepseek_api_key=x_deepseek_key,
        openrouter_api_key=x_openrouter_key
    )
    
    def event_generator():
        try:
            for text in chatbot.chat_stream(
                query=req.query,
                search_results=search_results,
                provider=req.provider,
                model=req.model,
                persona_mode=req.persona_mode
            ):
                yield text
        except Exception as chat_err:
            yield f"\n[Error generating response: {str(chat_err)}]"

    return StreamingResponse(event_generator(), media_type="text/plain")

# Mount frontend files at root
os.makedirs("./frontend", exist_ok=True)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
