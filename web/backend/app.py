from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import (
    ALLOWED_VIDEO_EXTENSIONS,
    DEFAULT_PARAMS,
    DEVICES,
    MODES,
    PROJECT_ROOT,
    REQUIRED_MODELS,
    RESULTS_DIR,
    RUNTIME_DIR,
    UPLOADS_DIR,
)
from .jobs import JobManager, ProcessingParams, SubprocessJobRunner


app = FastAPI(title="FootAR API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOB_MANAGER = JobManager()
JOB_RUNNER = SubprocessJobRunner(JOB_MANAGER)


def model_statuses() -> list[dict[str, Any]]:
    statuses = []
    for key, path in REQUIRED_MODELS.items():
        exists = path.exists()
        statuses.append(
            {
                "key": key,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "absolute_path": str(path),
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else None,
            }
        )
    return statuses


def missing_required_models() -> list[dict[str, Any]]:
    return [status for status in model_statuses() if not status["exists"]]


def cuda_status() -> dict[str, Any]:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        return {
            "available": available,
            "device_name": torch.cuda.get_device_name(0) if available else None,
            "torch_version": getattr(torch, "__version__", None),
            "cuda_version": getattr(torch.version, "cuda", None),
            "error": None,
        }
    except Exception as exc:
        return {
            "available": False,
            "device_name": None,
            "torch_version": None,
            "cuda_version": None,
            "error": str(exc),
        }


def safe_filename(filename: str | None) -> str:
    candidate = Path(filename or "upload.mp4").name
    return candidate or "upload.mp4"


@app.get("/api/system")
def get_system() -> dict[str, Any]:
    models = model_statuses()
    return {
        "ready": all(model["exists"] for model in models),
        "project_root": str(PROJECT_ROOT),
        "runtime_dir": str(RUNTIME_DIR),
        "models": models,
        "cuda": cuda_status(),
        "defaults": DEFAULT_PARAMS,
        "modes": MODES,
        "devices": DEVICES,
    }


@app.post("/api/jobs")
async def create_job(
    video: UploadFile = File(...),
    mode: str = Form(DEFAULT_PARAMS["mode"]),
    device: str = Form(DEFAULT_PARAMS["device"]),
    debug: bool = Form(DEFAULT_PARAMS["debug"]),
    player_track_imgsz: int = Form(DEFAULT_PARAMS["player_track_imgsz"]),
    pitch_every_n_frames: int = Form(DEFAULT_PARAMS["pitch_every_n_frames"]),
    ball_track_imgsz: int = Form(DEFAULT_PARAMS["ball_track_imgsz"]),
    ball_track_every_n_frames: int = Form(DEFAULT_PARAMS["ball_track_every_n_frames"]),
    ball_track_conf: float = Form(DEFAULT_PARAMS["ball_track_conf"]),
    ball_max_hold_frames: int = Form(DEFAULT_PARAMS["ball_max_hold_frames"]),
) -> dict[str, Any]:
    filename = safe_filename(video.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    try:
        params = ProcessingParams.from_raw(
            mode=mode,
            device=device,
            debug=debug,
            player_track_imgsz=player_track_imgsz,
            pitch_every_n_frames=pitch_every_n_frames,
            ball_track_imgsz=ball_track_imgsz,
            ball_track_every_n_frames=ball_track_every_n_frames,
            ball_track_conf=ball_track_conf,
            ball_max_hold_frames=ball_max_hold_frames,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    missing = missing_required_models()
    if missing:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Required model files are missing",
                "missing_models": missing,
            },
        )

    job_id = uuid.uuid4().hex
    upload_dir = UPLOADS_DIR / job_id
    result_dir = RESULTS_DIR / job_id
    debug_dir = result_dir / "debug"
    upload_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    input_path = upload_dir / f"input{suffix}"
    output_path = result_dir / "output.mp4"

    with input_path.open("wb") as target:
        while True:
            chunk = await video.read(1024 * 1024)
            if not chunk:
                break
            target.write(chunk)

    JOB_MANAGER.create_job(
        input_filename=filename,
        input_path=input_path,
        output_path=output_path,
        debug_output_dir=debug_dir,
        params=params,
        job_id=job_id,
    )

    JOB_RUNNER.start(job_id)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    snapshot = JOB_MANAGER.snapshot(job_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return snapshot


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    cancelled = JOB_MANAGER.cancel(job_id)
    if not cancelled and JOB_MANAGER.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "cancelled": cancelled}


@app.get("/api/jobs/{job_id}/output")
def get_job_output(job_id: str) -> FileResponse:
    job = JOB_MANAGER.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "succeeded" or not job.output_path.exists():
        raise HTTPException(status_code=404, detail="Output video is not available")
    return FileResponse(job.output_path, media_type="video/mp4", filename=f"footar-{job_id}.mp4")


@app.get("/api/jobs/{job_id}/preview")
def get_job_preview(job_id: str) -> FileResponse:
    job = JOB_MANAGER.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "succeeded" or job.preview_path is None or not job.preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview video is not available")
    return FileResponse(job.preview_path, media_type="video/mp4", filename=f"footar-{job_id}-preview.mp4")


frontend_dist = PROJECT_ROOT / "web" / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
