from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_PARAMS,
    DEVICES,
    EVENT_PREFIX,
    MAIN_SEG_PATH,
    MODES,
    PROJECT_ROOT,
    PYTHON_EXECUTABLE,
)


TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
PROGRESS_RE = re.compile(r"Processed\s+(\d+)\s*/\s*(\d+)\s+frames", re.IGNORECASE)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


@dataclass(frozen=True)
class ProcessingParams:
    mode: str = DEFAULT_PARAMS["mode"]
    device: str = DEFAULT_PARAMS["device"]
    debug: bool = DEFAULT_PARAMS["debug"]
    player_track_imgsz: int = DEFAULT_PARAMS["player_track_imgsz"]
    pitch_every_n_frames: int = DEFAULT_PARAMS["pitch_every_n_frames"]
    ball_track_imgsz: int = DEFAULT_PARAMS["ball_track_imgsz"]
    ball_track_every_n_frames: int = DEFAULT_PARAMS["ball_track_every_n_frames"]
    ball_track_conf: float = DEFAULT_PARAMS["ball_track_conf"]
    ball_max_hold_frames: int = DEFAULT_PARAMS["ball_max_hold_frames"]

    @classmethod
    def from_raw(cls, **raw: Any) -> "ProcessingParams":
        values = dict(DEFAULT_PARAMS)
        for key, value in raw.items():
            if value is not None:
                values[key] = value

        mode = str(values["mode"]).strip().upper()
        device = str(values["device"]).strip().lower()
        if mode not in MODES:
            raise ValueError(f"Unsupported mode: {mode}")
        if device not in DEVICES:
            raise ValueError(f"Unsupported device: {device}")

        try:
            player_track_imgsz = int(values["player_track_imgsz"])
            pitch_every_n_frames = int(values["pitch_every_n_frames"])
            ball_track_imgsz = int(values["ball_track_imgsz"])
            ball_track_every_n_frames = int(values["ball_track_every_n_frames"])
            ball_track_conf = float(values["ball_track_conf"])
            ball_max_hold_frames = int(values["ball_max_hold_frames"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid numeric processing parameter") from exc

        if player_track_imgsz < 320 or player_track_imgsz > 1920:
            raise ValueError("player_track_imgsz must be between 320 and 1920")
        if ball_track_imgsz < 320 or ball_track_imgsz > 1920:
            raise ValueError("ball_track_imgsz must be between 320 and 1920")
        if pitch_every_n_frames < 1 or pitch_every_n_frames > 120:
            raise ValueError("pitch_every_n_frames must be between 1 and 120")
        if ball_track_every_n_frames < 1 or ball_track_every_n_frames > 120:
            raise ValueError("ball_track_every_n_frames must be between 1 and 120")
        if ball_max_hold_frames < 0 or ball_max_hold_frames > 300:
            raise ValueError("ball_max_hold_frames must be between 0 and 300")
        if not 0.01 <= ball_track_conf <= 1:
            raise ValueError("ball_track_conf must be between 0.01 and 1")

        return cls(
            mode=mode,
            device=device,
            debug=parse_bool(values["debug"]),
            player_track_imgsz=player_track_imgsz,
            pitch_every_n_frames=pitch_every_n_frames,
            ball_track_imgsz=ball_track_imgsz,
            ball_track_every_n_frames=ball_track_every_n_frames,
            ball_track_conf=ball_track_conf,
            ball_max_hold_frames=ball_max_hold_frames,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "device": self.device,
            "debug": self.debug,
            "player_track_imgsz": self.player_track_imgsz,
            "pitch_every_n_frames": self.pitch_every_n_frames,
            "ball_track_imgsz": self.ball_track_imgsz,
            "ball_track_every_n_frames": self.ball_track_every_n_frames,
            "ball_track_conf": self.ball_track_conf,
            "ball_max_hold_frames": self.ball_max_hold_frames,
        }


@dataclass
class JobRecord:
    id: str
    input_filename: str
    input_path: Path
    output_path: Path
    debug_output_dir: Path
    params: ProcessingParams
    preview_path: Path | None = None
    live_frame_dir: Path | None = None
    stats_dir: Path | None = None
    status: str = "queued"
    progress: float = 0.0
    processed_frames: int | None = None
    total_frames: int | None = None
    logs: list[str] = field(default_factory=list)
    error: str | None = None
    return_code: int | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    process: subprocess.Popen[str] | None = field(default=None, repr=False, compare=False)
    cancel_requested: bool = False

    def to_public_dict(self) -> dict[str, Any]:
        output_url = f"/api/jobs/{self.id}/output" if self.status == "succeeded" else None
        preview_ready = bool(self.preview_path and self.preview_path.exists())
        preview_url = f"/api/jobs/{self.id}/preview" if self.status == "succeeded" and preview_ready else output_url
        live_enabled = self.live_frame_dir is not None
        stats_ready = bool(self.stats_json_path and self.stats_json_path.exists())
        return {
            "job_id": self.id,
            "input_filename": self.input_filename,
            "status": self.status,
            "progress": self.progress,
            "processed_frames": self.processed_frames,
            "total_frames": self.total_frames,
            "logs": self.logs[-250:],
            "error": self.error,
            "return_code": self.return_code,
            "params": self.params.as_dict(),
            "output_url": output_url,
            "preview_url": preview_url,
            "preview_ready": preview_ready,
            "live_enabled": live_enabled,
            "live_frame_url": f"/api/jobs/{self.id}/live-frame" if live_enabled else None,
            "live_stream_url": f"/api/jobs/{self.id}/live-stream" if live_enabled else None,
            "stats_ready": stats_ready,
            "stats_url": f"/api/jobs/{self.id}/stats" if stats_ready else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @property
    def live_frame_path(self) -> Path | None:
        return self.live_frame_dir / "latest.jpg" if self.live_frame_dir is not None else None

    @property
    def stats_json_path(self) -> Path | None:
        return self.stats_dir / "stats.json" if self.stats_dir is not None else None


def build_processing_command(job: JobRecord) -> list[str]:
    params = job.params
    command = [
        str(PYTHON_EXECUTABLE),
        "-u",
        str(MAIN_SEG_PATH),
        "--source_video_path",
        str(job.input_path),
        "--target_video_path",
        str(job.output_path),
        "--device",
        params.device,
        "--mode",
        params.mode,
        "--debug_output_dir",
        str(job.debug_output_dir),
        "--player_track_imgsz",
        str(params.player_track_imgsz),
        "--pitch_every_n_frames",
        str(params.pitch_every_n_frames),
        "--ball_track_imgsz",
        str(params.ball_track_imgsz),
        "--ball_track_every_n_frames",
        str(params.ball_track_every_n_frames),
        "--ball_track_conf",
        str(params.ball_track_conf),
        "--ball_max_hold_frames",
        str(params.ball_max_hold_frames),
        "--no_preview",
        "--structured_logs",
    ]
    if job.live_frame_dir is not None:
        command.extend(
            [
                "--live_frame_dir",
                str(job.live_frame_dir),
                "--live_frame_every",
                "1",
            ]
        )
    if job.stats_dir is not None:
        command.extend(["--stats_output_dir", str(job.stats_dir)])
    if params.debug:
        command.append("--debug")
    return command


def parse_progress_line(line: str) -> dict[str, Any] | None:
    if line.startswith(EVENT_PREFIX):
        try:
            payload = json.loads(line[len(EVENT_PREFIX) :])
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    match = PROGRESS_RE.search(line)
    if not match:
        return None
    processed = int(match.group(1))
    total = int(match.group(2))
    progress = processed / total if total else 0
    return {
        "event": "progress",
        "processed_frames": processed,
        "total_frames": total,
        "progress": progress,
    }


def summarize_failure(logs: list[str], fallback: str) -> str:
    for line in reversed(logs):
        clean = line.strip()
        if not clean:
            continue
        if clean.startswith(("ModuleNotFoundError:", "ImportError:", "RuntimeError:", "ValueError:", "FileNotFoundError:")):
            return clean
    for line in reversed(logs):
        clean = line.strip()
        if clean and not clean.startswith(("File ", "Traceback ")):
            return clean
    return fallback


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def create_job(
        self,
        input_filename: str,
        input_path: Path,
        output_path: Path,
        debug_output_dir: Path,
        params: ProcessingParams,
        preview_path: Path | None = None,
        live_frame_dir: Path | None = None,
        stats_dir: Path | None = None,
        job_id: str | None = None,
    ) -> JobRecord:
        job = JobRecord(
            id=job_id or uuid.uuid4().hex,
            input_filename=input_filename,
            input_path=input_path,
            output_path=output_path,
            debug_output_dir=debug_output_dir,
            params=params,
            preview_path=preview_path or output_path.with_name("preview.mp4"),
            live_frame_dir=live_frame_dir,
            stats_dir=stats_dir,
        )
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_public_dict() if job else None

    def mark_running(self, job_id: str, process: subprocess.Popen[str]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.process = process
            job.updated_at = time.time()

    def append_log(self, job_id: str, line: str) -> None:
        clean = line.rstrip()
        if not clean:
            return
        with self._lock:
            job = self._jobs[job_id]
            job.logs.append(clean)
            if len(job.logs) > 500:
                job.logs = job.logs[-500:]
            job.updated_at = time.time()

    def apply_event(self, job_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            event_name = event.get("event") or event.get("type")
            if event_name == "progress":
                processed = event.get("processed_frames")
                total = event.get("total_frames")
                progress = event.get("progress")
                if processed is not None:
                    job.processed_frames = int(processed)
                if total is not None:
                    job.total_frames = int(total)
                if progress is not None:
                    job.progress = max(0.0, min(1.0, float(progress)))
                elif job.processed_frames is not None and job.total_frames:
                    job.progress = max(0.0, min(1.0, job.processed_frames / job.total_frames))
            elif event_name == "error":
                job.error = str(event.get("message") or "Processing failed")
            job.updated_at = time.time()

    def finish(self, job_id: str, status: str, return_code: int | None = None, error: str | None = None) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = status
            job.return_code = return_code
            job.error = error
            job.process = None
            if status == "succeeded":
                job.progress = 1.0
            job.updated_at = time.time()

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in TERMINAL_STATUSES:
                return False
            job.cancel_requested = True
            process = job.process
            job.status = "cancelled" if process is None else job.status
            job.updated_at = time.time()
        if process is not None and process.poll() is None:
            process.terminate()
        return True

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs[job_id]
            return job.cancel_requested


class SubprocessJobRunner:
    def __init__(self, manager: JobManager) -> None:
        self.manager = manager

    def start(self, job_id: str) -> None:
        thread = threading.Thread(target=self.run, args=(job_id,), daemon=True)
        thread.start()

    def run(self, job_id: str) -> None:
        job = self.manager.get(job_id)
        if job is None:
            return

        job.output_path.parent.mkdir(parents=True, exist_ok=True)
        job.debug_output_dir.mkdir(parents=True, exist_ok=True)
        command = build_processing_command(job)
        self.manager.append_log(job_id, "Starting FootAR processor")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        try:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                creationflags=creationflags,
            )
        except Exception as exc:
            self.manager.finish(job_id, "failed", error=str(exc))
            return

        self.manager.mark_running(job_id, process)

        try:
            if process.stdout is not None:
                for raw_line in process.stdout:
                    self.manager.append_log(job_id, raw_line)
                    event = parse_progress_line(raw_line.rstrip())
                    if event is not None:
                        self.manager.apply_event(job_id, event)
            return_code = process.wait()
        except Exception as exc:
            process.kill()
            self.manager.finish(job_id, "failed", error=str(exc))
            return

        if self.manager.is_cancel_requested(job_id):
            self.manager.finish(job_id, "cancelled", return_code=return_code)
        elif return_code == 0 and job.output_path.exists():
            self.manager.append_log(job_id, "Preparing browser preview")
            self.create_browser_preview(job)
            self.manager.finish(job_id, "succeeded", return_code=return_code)
        else:
            message = f"Processor exited with code {return_code}"
            if return_code == 0 and not job.output_path.exists():
                message = "Processor finished but did not create an output video"
            else:
                latest_job = self.manager.get(job_id)
                if latest_job is not None:
                    message = summarize_failure(latest_job.logs, message)
            self.manager.finish(job_id, "failed", return_code=return_code, error=message)

    def create_browser_preview(self, job: JobRecord) -> bool:
        if job.preview_path is None:
            return False
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self.manager.append_log(job.id, "ffmpeg not found; using original output for preview")
            return False

        command = [
            ffmpeg,
            "-y",
            "-i",
            str(job.output_path),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(job.preview_path),
        ]
        try:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=600,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception as exc:
            self.manager.append_log(job.id, f"Preview conversion failed: {exc}")
            return False

        if result.returncode == 0 and job.preview_path.exists():
            self.manager.append_log(job.id, "Browser preview ready (H.264)")
            return True

        tail = (result.stdout or "").splitlines()[-3:]
        detail = " | ".join(tail) if tail else f"ffmpeg exited with code {result.returncode}"
        self.manager.append_log(job.id, f"Preview conversion failed: {detail}")
        return False
