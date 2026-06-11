from __future__ import annotations

import json
import subprocess
import uuid
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import (
    ALLOWED_VIDEO_EXTENSIONS,
    DEFAULT_PARAMS,
    DEVICES,
    MODES,
    PROCESSOR_REQUIRED_MODULES,
    PROJECT_ROOT,
    PYTHON_EXECUTABLE,
    PYTHON_EXECUTABLE_SOURCE,
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


@lru_cache(maxsize=1)
def processor_status() -> dict[str, Any]:
    status = {
        "executable": str(PYTHON_EXECUTABLE),
        "source": PYTHON_EXECUTABLE_SOURCE,
        "exists": PYTHON_EXECUTABLE.exists(),
        "required_modules": list(PROCESSOR_REQUIRED_MODULES),
        "modules": {},
        "missing_modules": list(PROCESSOR_REQUIRED_MODULES),
        "ready": False,
        "error": None,
    }
    if not PYTHON_EXECUTABLE.exists():
        status["error"] = "Processor Python executable was not found"
        return status

    command = [
        str(PYTHON_EXECUTABLE),
        "-c",
        (
            "import importlib.util, json; "
            f"modules={{name: importlib.util.find_spec(name) is not None for name in {repr(PROCESSOR_REQUIRED_MODULES)}}}; "
            "print(json.dumps(modules))"
        ),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as exc:
        status["error"] = str(exc)
        return status

    if result.returncode != 0:
        status["error"] = (result.stderr or result.stdout or f"Module check exited with code {result.returncode}").strip()
        return status

    try:
        modules = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        status["error"] = f"Could not parse processor module status: {exc}"
        return status

    status["modules"] = modules
    status["missing_modules"] = [name for name, exists in modules.items() if not exists]
    status["ready"] = len(status["missing_modules"]) == 0
    return status


@lru_cache(maxsize=1)
def cuda_status() -> dict[str, Any]:
    if not PYTHON_EXECUTABLE.exists():
        return {
            "available": False,
            "device_name": None,
            "torch_version": None,
            "cuda_version": None,
            "error": "Processor Python executable was not found",
        }

    command = [
        str(PYTHON_EXECUTABLE),
        "-c",
        (
            "import json; "
            "payload={'available': False, 'device_name': None, 'torch_version': None, 'cuda_version': None, 'error': None}; "
            "\ntry:\n"
            " import torch\n"
            " available=bool(torch.cuda.is_available())\n"
            " payload.update({'available': available, 'device_name': torch.cuda.get_device_name(0) if available else None, "
            "'torch_version': getattr(torch, '__version__', None), 'cuda_version': getattr(torch.version, 'cuda', None)})\n"
            "except Exception as exc:\n"
            " payload['error']=str(exc)\n"
            "print(json.dumps(payload))"
        ),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=75,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as exc:
        return {
            "available": False,
            "device_name": None,
            "torch_version": None,
            "cuda_version": None,
            "error": str(exc),
        }
    if result.returncode != 0:
        return {
            "available": False,
            "device_name": None,
            "torch_version": None,
            "cuda_version": None,
            "error": (result.stderr or result.stdout or f"CUDA check exited with code {result.returncode}").strip(),
        }
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {
            "available": False,
            "device_name": None,
            "torch_version": None,
            "cuda_version": None,
            "error": f"Could not parse CUDA status: {exc}",
        }
    return payload


def safe_filename(filename: str | None) -> str:
    candidate = Path(filename or "upload.mp4").name
    return candidate or "upload.mp4"


def allowed_heatmap_files(stats_path: Path) -> set[str]:
    try:
        data = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()

    heatmaps = data.get("heatmaps") or {}
    names: set[str] = set()
    for key in ("global", "ball"):
        if heatmaps.get(key):
            names.add(heatmaps[key])
    for value in (heatmaps.get("team") or {}).values():
        if value:
            names.add(value)
    for player in heatmaps.get("players") or []:
        if player.get("file"):
            names.add(player["file"])
    return names


@app.get("/api/system")
def get_system() -> dict[str, Any]:
    models = model_statuses()
    processor = processor_status()
    return {
        "ready": all(model["exists"] for model in models) and bool(processor["ready"]),
        "project_root": str(PROJECT_ROOT),
        "runtime_dir": str(RUNTIME_DIR),
        "processor_python": str(PYTHON_EXECUTABLE),
        "processor": processor,
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
    live: bool = Form(False),
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
    processor = processor_status()
    if not processor["ready"]:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Processor Python environment is missing required modules",
                "processor": processor,
            },
        )

    job_id = uuid.uuid4().hex
    upload_dir = UPLOADS_DIR / job_id
    result_dir = RESULTS_DIR / job_id
    debug_dir = result_dir / "debug"
    stats_dir = result_dir / "stats"
    upload_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    input_path = upload_dir / f"input{suffix}"
    output_path = result_dir / "output.mp4"
    live_frame_dir = result_dir / "live" if live else None
    if live_frame_dir is not None:
        live_frame_dir.mkdir(parents=True, exist_ok=True)

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
        live_frame_dir=live_frame_dir,
        stats_dir=stats_dir,
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


@app.get("/api/jobs/{job_id}/stats")
def get_job_stats(job_id: str) -> FileResponse:
    job = JOB_MANAGER.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    stats_path = job.stats_json_path
    if stats_path is None or not stats_path.exists():
        raise HTTPException(status_code=404, detail="Statistics are not available")
    return FileResponse(stats_path, media_type="application/json")


@app.get("/api/jobs/{job_id}/positions")
def get_job_positions(job_id: str) -> FileResponse:
    job = JOB_MANAGER.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    positions_path = job.positions_json_path
    if positions_path is None or not positions_path.exists():
        raise HTTPException(status_code=404, detail="Positions are not available")
    return FileResponse(positions_path, media_type="application/json")


@app.get("/api/jobs/{job_id}/heatmap/{name}")
def get_job_heatmap(job_id: str, name: str) -> FileResponse:
    job = JOB_MANAGER.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    stats_path = job.stats_json_path
    if stats_path is None or not stats_path.exists():
        raise HTTPException(status_code=404, detail="Statistics are not available")
    if name not in allowed_heatmap_files(stats_path):
        raise HTTPException(status_code=404, detail="Heatmap not found")
    image_path = job.stats_dir / name
    if not image_path.resolve().is_relative_to(job.stats_dir.resolve()):
        raise HTTPException(status_code=404, detail="Heatmap not found")
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Heatmap image is missing")
    return FileResponse(image_path, media_type="image/png")


@app.get("/api/jobs/{job_id}/preview")
def get_job_preview(job_id: str) -> FileResponse:
    job = JOB_MANAGER.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "succeeded" or job.preview_path is None or not job.preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview video is not available")
    return FileResponse(job.preview_path, media_type="video/mp4", filename=f"footar-{job_id}-preview.mp4")


@app.get("/api/jobs/{job_id}/live-frame")
def get_job_live_frame(job_id: str) -> FileResponse:
    job = JOB_MANAGER.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    frame_path = job.live_frame_path
    if frame_path is None:
        raise HTTPException(status_code=404, detail="Live preview is not enabled for this job")
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Live preview frame is not available yet")

    return FileResponse(
        frame_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, max-age=0"},
        filename=f"footar-{job_id}-live.jpg",
    )


def stream_live_frames(job_id: str):
    last_token: tuple[int, int] | None = None
    boundary = b"--footar-live-frame\r\n"

    while True:
        job = JOB_MANAGER.get(job_id)
        if job is None:
            return

        frame_path = job.live_frame_path
        if frame_path is not None and frame_path.exists():
            try:
                stat = frame_path.stat()
                token = (stat.st_mtime_ns, stat.st_size)
                if token != last_token:
                    frame = frame_path.read_bytes()
                    if frame:
                        last_token = token
                        yield (
                            boundary
                            + b"Content-Type: image/jpeg\r\n"
                            + b"Cache-Control: no-store\r\n"
                            + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                            + frame
                            + b"\r\n"
                        )
            except OSError:
                pass

        if job.status in {"succeeded", "failed", "cancelled"}:
            return

        time.sleep(0.15)


@app.get("/api/jobs/{job_id}/live-stream")
def get_job_live_stream(job_id: str) -> StreamingResponse:
    job = JOB_MANAGER.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.live_frame_dir is None:
        raise HTTPException(status_code=404, detail="Live preview is not enabled for this job")

    return StreamingResponse(
        stream_live_frames(job_id),
        media_type="multipart/x-mixed-replace; boundary=footar-live-frame",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


frontend_dist = PROJECT_ROOT / "web" / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
