"""Project CRUD endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.pipeline.project_manager import (
    list_projects,
    create_project,
    delete_project,
)

router = APIRouter(prefix="/api", tags=["projects"])


class CreateProjectRequest(BaseModel):
    name: str


@router.get("/projects")
def api_list_projects():
    """Lists all projects."""
    return list_projects()


@router.post("/projects")
def api_create_project(req: CreateProjectRequest):
    """Creates a new project directory."""
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Project name cannot be empty")
    safe_name = create_project(name)
    return {"project_id": safe_name, "message": f"Project '{safe_name}' created successfully"}


@router.delete("/projects/{project_id}")
def api_delete_project(project_id: str):
    """Deletes a project recursively."""
    if project_id == "default":
        delete_project("default")
        return {"project_id": "default", "message": "Default project cleared"}
    delete_project(project_id)
    return {"project_id": project_id, "message": f"Project '{project_id}' deleted successfully"}
