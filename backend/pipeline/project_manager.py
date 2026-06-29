import os
import re
import shutil
from typing import List, Dict, Any

DATA_DIR = os.getenv("DATA_DIR", "./backend/data")
PROJECTS_DIR = os.path.join(DATA_DIR, "projects")

def get_safe_project_id(project_id: str) -> str:
    """Cleans project ID to avoid folder traversal and invalid characters."""
    if not project_id:
        return "default"
    safe = re.sub(r'[^\w\-_]', '', project_id)
    return safe if safe else "default"

def get_project_dir(project_id: str) -> str:
    """Returns the absolute directory path of a project."""
    safe_id = get_safe_project_id(project_id)
    return os.path.abspath(os.path.join(PROJECTS_DIR, safe_id))

def list_projects() -> List[str]:
    """Lists all available projects. Creates default if none exist."""
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    projects = []
    for item in os.listdir(PROJECTS_DIR):
        item_path = os.path.join(PROJECTS_DIR, item)
        if os.path.isdir(item_path):
            projects.append(item)
            
    if not projects:
        create_project("default")
        projects = ["default"]
    return sorted(projects)

def create_project(project_id: str) -> str:
    """Creates a new project directory structure."""
    safe_id = get_safe_project_id(project_id)
    project_path = get_project_dir(safe_id)
    os.makedirs(project_path, exist_ok=True)
    os.makedirs(os.path.join(project_path, "audio"), exist_ok=True)
    os.makedirs(os.path.join(project_path, "knowledge"), exist_ok=True)
    os.makedirs(os.path.join(project_path, "markdown"), exist_ok=True)
    return safe_id

def delete_project(project_id: str):
    """Deletes a project directory recursively."""
    safe_id = get_safe_project_id(project_id)
    # Prevent deleting the root PROJECTS_DIR or empty
    if not safe_id or safe_id == "default":
        project_path = get_project_dir("default")
        if os.path.exists(project_path):
            shutil.rmtree(project_path)
        create_project("default")
        return
        
    project_path = get_project_dir(safe_id)
    if os.path.exists(project_path):
        shutil.rmtree(project_path)

def format_timestamp(seconds: float) -> str:
    """Converts seconds float to HH:MM:SS or MM:SS format."""
    s = int(seconds)
    hours = s // 3600
    minutes = (s % 3600) // 60
    secs = s % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

def generate_markdown(video_metadata: Dict[str, Any], segments: List[Dict[str, Any]]) -> str:
    """Generates a structured markdown file of the transcript with links."""
    title = video_metadata.get("title", "Unknown Video")
    url = video_metadata.get("url") or video_metadata.get("webpage_url") or ""
    duration = video_metadata.get("duration")
    uploader = video_metadata.get("uploader", "Unknown")
    
    duration_str = format_timestamp(duration) if duration else "Unknown"
    
    md = []
    md.append(f"# {title}")
    md.append("")
    md.append(f"- **Uploader**: {uploader}")
    if url:
        md.append(f"- **Original VID**: [Watch on YouTube]({url})")
    md.append(f"- **Duration**: {duration_str}")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Transcript")
    md.append("")
    
    for seg in segments:
        start_time = seg.get("start", 0.0)
        time_str = format_timestamp(start_time)
        text = seg.get("text", "").strip()
        
        # Build timestamped link
        if url:
            if "youtube.com" in url or "youtu.be" in url:
                link = f"{url}&t={int(start_time)}" if "?" in url else f"{url}?t={int(start_time)}"
            else:
                link = f"{url}#t={int(start_time)}"
            md.append(f"- **[{time_str}]({link})**: {text}")
        else:
            md.append(f"- **[{time_str}]**: {text}")
            
    return "\n".join(md)
