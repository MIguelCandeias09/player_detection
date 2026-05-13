from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAIN_SEG_PATH = PROJECT_ROOT / "src" / "main_seg.py"
RUNTIME_DIR = PROJECT_ROOT / ".footar_runtime"
UPLOADS_DIR = RUNTIME_DIR / "uploads"
RESULTS_DIR = RUNTIME_DIR / "results"
DEBUG_DIR = RUNTIME_DIR / "debug"

EVENT_PREFIX = "FOOTAR_EVENT "

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

MODES = [
    "PITCH_DETECTION",
    "PLAYER_DETECTION",
    "BALL_DETECTION",
    "PLAYER_TRACKING",
    "TEAM_CLASSIFICATION",
    "RADAR",
]

DEVICES = ["cuda", "cpu"]

DEFAULT_PARAMS = {
    "mode": "RADAR",
    "device": "cuda",
    "debug": False,
    "player_track_imgsz": 1024,
    "pitch_every_n_frames": 5,
    "ball_track_imgsz": 960,
    "ball_track_every_n_frames": 2,
    "ball_track_conf": 0.25,
    "ball_max_hold_frames": 3,
}

REQUIRED_MODELS = {
    "player_segmentation": PROJECT_ROOT / "models" / "active" / "yolo11m_seg_players.pt",
    "pitch_keypoints": PROJECT_ROOT / "models" / "active" / "pitch_v11m_640_footar_best.pt",
    "ball_detection": PROJECT_ROOT / "models" / "active" / "ball_y11m_1280_footar_best.pt",
}

PYTHON_EXECUTABLE = Path(sys.executable)

