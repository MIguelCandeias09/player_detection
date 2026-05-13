from __future__ import annotations

import os
import subprocess
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

PROCESSOR_REQUIRED_MODULES = ("supervision", "torch", "ultralytics")


def processor_python_candidates() -> list[tuple[Path, str]]:
    candidates = [
        (Path(sys.executable), "backend Python"),
        (PROJECT_ROOT / ".venv" / "Scripts" / "python.exe", "project .venv"),
        (PROJECT_ROOT.parent / ".venv" / "Scripts" / "python.exe", "FootAR_V2 .venv"),
        (PROJECT_ROOT.parent.parent / "FootAR_old" / ".venv" / "Scripts" / "python.exe", "FootAR_old .venv"),
    ]

    seen = set()
    unique = []
    for path, source in candidates:
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            unique.append((path, source))
    return unique


def python_has_required_modules(python_path: Path) -> bool:
    if not python_path.exists():
        return False

    module_list = repr(PROCESSOR_REQUIRED_MODULES)
    command = [
        str(python_path),
        "-c",
        (
            "import importlib.util, sys; "
            f"missing=[name for name in {module_list} if importlib.util.find_spec(name) is None]; "
            "sys.exit(1 if missing else 0)"
        ),
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=12,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        return False
    return result.returncode == 0


def resolve_processor_python() -> tuple[Path, str]:
    configured = os.environ.get("FOOTAR_PYTHON")
    if configured:
        return Path(configured), "FOOTAR_PYTHON"

    for path, source in processor_python_candidates():
        if python_has_required_modules(path):
            return path, source

    return Path(sys.executable), "backend Python"


PYTHON_EXECUTABLE, PYTHON_EXECUTABLE_SOURCE = resolve_processor_python()
