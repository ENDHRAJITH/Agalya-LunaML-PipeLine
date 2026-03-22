from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import os

from app.pipeline import run_pipeline

app = FastAPI(title="Luna ML Pipeline", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request schema ────────────────────────────────────────────────
class BuildRequest(BaseModel):
    problem_description: str
    model_id: Optional[str] = None
    user_id:  Optional[str] = None
    dataset_path: Optional[str] = None
    dataset_filename: Optional[str] = None

# ── POST /build-model ─────────────────────────────────────────────
@app.post("/build-model")
async def build_model(req: BuildRequest):
    try:
        result = run_pipeline(req.problem_description, req.model_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── GET /download/:file_name ──────────────────────────────────────
@app.get("/download/{file_name}")
async def download_model(file_name: str):
    file_path = os.path.join("outputs", file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        path=file_path,
        media_type="application/zip",
        filename=file_name
    )

# ── GET /health ───────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "Luna ML Pipeline",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    return {"message": "🌙 Luna ML Pipeline is running!", "docs": "/docs"}
