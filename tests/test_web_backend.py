from pathlib import Path

from fastapi.testclient import TestClient

import web.backend.app as api
from web.backend.config import MAIN_SEG_PATH
from web.backend.jobs import (
    JobManager,
    ProcessingParams,
    build_processing_command,
    parse_progress_line,
)


def test_processing_params_defaults_and_validation():
    params = ProcessingParams.from_raw(mode="radar", device="CPU", debug="false")

    assert params.mode == "RADAR"
    assert params.device == "cpu"
    assert params.debug is False
    assert params.player_track_imgsz == 1024
    assert params.pitch_every_n_frames == 5


def test_build_processing_command_includes_backend_flags(tmp_path):
    manager = JobManager()
    job = manager.create_job(
        input_filename="clip.mp4",
        input_path=tmp_path / "input.mp4",
        output_path=tmp_path / "output.mp4",
        debug_output_dir=tmp_path / "debug",
        params=ProcessingParams.from_raw(device="cpu", ball_track_conf=0.4),
        job_id="job-1",
    )

    command = build_processing_command(job)

    assert str(MAIN_SEG_PATH) in command
    assert "--no_preview" in command
    assert "--structured_logs" in command
    assert command[command.index("--mode") + 1] == "RADAR"
    assert command[command.index("--ball_track_conf") + 1] == "0.4"


def test_parse_structured_and_plain_progress_lines():
    structured = parse_progress_line(
        'FOOTAR_EVENT {"event":"progress","processed_frames":30,"total_frames":120,"progress":0.25}'
    )
    plain = parse_progress_line("Processed 60/120 frames")

    assert structured["progress"] == 0.25
    assert plain["processed_frames"] == 60
    assert plain["progress"] == 0.5


def test_job_cancel_marks_request_and_terminates_process(tmp_path):
    class DummyProcess:
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

    manager = JobManager()
    job = manager.create_job(
        input_filename="clip.mp4",
        input_path=tmp_path / "input.mp4",
        output_path=tmp_path / "output.mp4",
        debug_output_dir=tmp_path / "debug",
        params=ProcessingParams(),
        job_id="job-2",
    )
    process = DummyProcess()
    manager.mark_running(job.id, process)

    assert manager.cancel(job.id) is True
    assert manager.is_cancel_requested(job.id) is True
    assert process.terminated is True


def test_upload_rejects_non_video(monkeypatch):
    client = TestClient(api.app)

    response = client.post(
        "/api/jobs",
        files={"video": ("clip.txt", b"not a video", "text/plain")},
    )

    assert response.status_code == 400


def test_create_job_stores_upload_and_queues_runner(monkeypatch, tmp_path):
    manager = JobManager()
    started_jobs = []

    class StubRunner:
        def start(self, job_id):
            started_jobs.append(job_id)

    monkeypatch.setattr(api, "JOB_MANAGER", manager)
    monkeypatch.setattr(api, "JOB_RUNNER", StubRunner())
    monkeypatch.setattr(api, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(api, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(api, "missing_required_models", lambda: [])

    client = TestClient(api.app)
    response = client.post(
        "/api/jobs",
        files={"video": ("clip.mp4", b"video bytes", "video/mp4")},
        data={"mode": "RADAR", "device": "cpu", "ball_track_conf": "0.35"},
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    job = manager.get(job_id)
    assert job is not None
    assert Path(job.input_path).read_bytes() == b"video bytes"
    assert job.params.ball_track_conf == 0.35
    assert started_jobs == [job_id]
