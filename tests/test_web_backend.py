from pathlib import Path

from fastapi.testclient import TestClient

import web.backend.app as api
from web.backend.config import MAIN_SEG_PATH, PYTHON_EXECUTABLE
from web.backend.jobs import (
    JobManager,
    ProcessingParams,
    build_processing_command,
    parse_progress_line,
    summarize_failure,
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

    assert command[0] == str(PYTHON_EXECUTABLE)
    assert str(MAIN_SEG_PATH) in command
    assert "--no_preview" in command
    assert "--structured_logs" in command
    assert command[command.index("--mode") + 1] == "RADAR"
    assert command[command.index("--ball_track_conf") + 1] == "0.4"


def test_build_processing_command_includes_live_frame_flags(tmp_path):
    manager = JobManager()
    live_dir = tmp_path / "live"
    job = manager.create_job(
        input_filename="clip.mp4",
        input_path=tmp_path / "input.mp4",
        output_path=tmp_path / "output.mp4",
        debug_output_dir=tmp_path / "debug",
        params=ProcessingParams.from_raw(device="cpu"),
        live_frame_dir=live_dir,
        job_id="job-live",
    )

    command = build_processing_command(job)

    assert command[command.index("--live_frame_dir") + 1] == str(live_dir)
    assert command[command.index("--live_frame_every") + 1] == "1"
    assert manager.snapshot(job.id)["live_stream_url"] == "/api/jobs/job-live/live-stream"


def test_parse_structured_and_plain_progress_lines():
    structured = parse_progress_line(
        'FOOTAR_EVENT {"event":"progress","processed_frames":30,"total_frames":120,"progress":0.25}'
    )
    plain = parse_progress_line("Processed 60/120 frames")

    assert structured["progress"] == 0.25
    assert plain["processed_frames"] == 60
    assert plain["progress"] == 0.5


def test_summarize_failure_prefers_python_exception():
    logs = [
        "Traceback (most recent call last):",
        '  File "src/main_seg.py", line 11, in <module>',
        "ModuleNotFoundError: No module named 'supervision'",
    ]

    assert summarize_failure(logs, "Processor exited with code 1") == "ModuleNotFoundError: No module named 'supervision'"


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
    monkeypatch.setattr(api, "processor_status", lambda: {"ready": True})

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


def test_live_frame_endpoint_serves_latest_frame(monkeypatch, tmp_path):
    manager = JobManager()
    live_dir = tmp_path / "live"
    live_dir.mkdir()
    (live_dir / "latest.jpg").write_bytes(b"jpeg bytes")
    job = manager.create_job(
        input_filename="clip.mp4",
        input_path=tmp_path / "input.mp4",
        output_path=tmp_path / "output.mp4",
        debug_output_dir=tmp_path / "debug",
        params=ProcessingParams(),
        live_frame_dir=live_dir,
        job_id="job-frame",
    )

    monkeypatch.setattr(api, "JOB_MANAGER", manager)

    client = TestClient(api.app)
    response = client.get(f"/api/jobs/{job.id}/live-frame")

    assert response.status_code == 200
    assert response.content == b"jpeg bytes"
    assert response.headers["content-type"].startswith("image/jpeg")


def test_build_processing_command_includes_stats_dir(tmp_path):
    manager = JobManager()
    stats_dir = tmp_path / "stats"
    job = manager.create_job(
        input_filename="clip.mp4",
        input_path=tmp_path / "input.mp4",
        output_path=tmp_path / "output.mp4",
        debug_output_dir=tmp_path / "debug",
        params=ProcessingParams.from_raw(device="cpu"),
        stats_dir=stats_dir,
        job_id="job-stats",
    )

    command = build_processing_command(job)

    assert command[command.index("--stats_output_dir") + 1] == str(stats_dir)


def test_snapshot_reports_stats_ready(tmp_path):
    manager = JobManager()
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()
    (stats_dir / "stats.json").write_text('{"fps": 25.0, "players": []}', encoding="utf-8")
    job = manager.create_job(
        input_filename="clip.mp4",
        input_path=tmp_path / "input.mp4",
        output_path=tmp_path / "output.mp4",
        debug_output_dir=tmp_path / "debug",
        params=ProcessingParams(),
        stats_dir=stats_dir,
        job_id="job-snap",
    )
    job.status = "succeeded"

    snapshot = manager.snapshot("job-snap")
    assert snapshot["stats_ready"] is True
    assert snapshot["stats_url"] == "/api/jobs/job-snap/stats"
