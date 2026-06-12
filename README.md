# FootAR — Automatic Football Video Analysis

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![YOLO](https://img.shields.io/badge/Ultralytics-YOLOv11%2F12-purple.svg)](https://docs.ultralytics.com/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-red.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Liga Portugal](https://img.shields.io/badge/Liga%20Portugal-2024%2F25-green.svg)](#)

Real-time computer vision system for analyzing professional football match footage from **Liga Portugal 2024/25**. Detects players, goalkeepers, referees, and the ball; tracks persistent identities across frames; classifies teams from segmentation masks; and projects tactical positions onto a 2D pitch radar.

**~25 FPS** on NVIDIA RTX 4070 | **6 operational modes** | **Web UI (React + FastAPI)** | **Single-pass ball interpolation**

---

## Features

| Capability | Method | Model |
|---|---|---|
| **Player Detection** | YOLOv11-Medium **Segmentation** (4 classes: ball, GK, player, referee) | `yolo11m_seg_players.pt` |
| **Ball Detection** | YOLOv11-Medium @ 1280px | `ball_y11m_1280_footar_best.pt` |
| **Pitch Keypoints** | YOLOv11-Medium Pose (32 keypoints, FIFA standard) | `pitch_v11m_640_footar_best.pt` |
| **Multi-Object Tracking** | BoT-SORT with GMC (Sparse Optical Flow, 60-frame buffer) | Native YOLO `.track()` |
| **Team Classification** | Segmentation masks → 3D HSV Histogram → K-Means → Temporal Voting + Lock | Unsupervised (no model) |
| **Ball Interpolation** | Real-time single-pass linear interpolation (30-frame buffer) | Algorithmic |
| **Tactical Radar** | Homography projection (cv2.findHomography → 2D pitch) | Geometric |
| **Match Stats** | Possession, distance, heatmaps, 2D positions export (`stats.json` / `positions.json`) | Algorithmic |
 
### Processing Modes

```
PITCH_DETECTION      →  Visualize detected pitch keypoints with confidence colors
PLAYER_DETECTION     →  Bounding box detection of all players/GK/referees
BALL_DETECTION       →  Ball tracking with real-time interpolation for missed frames
PLAYER_TRACKING      →  Persistent IDs via BoT-SORT with camera motion compensation
TEAM_CLASSIFICATION  →  Mask-based team assignment with temporal voting
RADAR                →  Full pipeline: detection + tracking + teams + 2D pitch radar + stats
```

---

## Architecture

```
player_detection/
├── src/                        # Application source code
│   ├── main_seg.py             # ★ ACTIVE entry point (segmentation pipeline, all 6 modes)
│   ├── main.py                 # Legacy (bbox pipeline) — kept for reference only
│   ├── main_1.py               # Legacy variant — kept for reference only
│   ├── main_miguel.py          # Legacy variant — kept for reference only
│   └── sports/                 # Core library
│       ├── annotators/         # Pitch drawing, point projection
│       ├── common/             # Ball tracker, interpolator, team classifiers, view transformer
│       ├── configs/            # Soccer pitch geometry (32 vertices, FIFA dimensions)
│       ├── distance_tracker.py # Per-player distance covered
│       ├── possession_tracker.py # Team possession estimation
│       ├── heatmap_tracker.py  # Position heatmaps
│       ├── stats_export.py     # stats.json writer
│       └── positions_export.py # positions.json writer (2D pitch coordinates)
│
├── web/                        # Local Web UI
│   ├── backend/                # FastAPI wrapper (jobs, live preview, stats API)
│   └── frontend/               # React + Vite interface
│
├── configs/                    # BoT-SORT tracker parameters (YAML)
├── models/
│   ├── active/                 # Production models (loaded by main_seg.py)
│   │   ├── yolo11m_seg_players.pt
│   │   ├── ball_y11m_1280_footar_best.pt
│   │   └── pitch_v11m_640_footar_best.pt
│   └── archive/                # Previous experiments (kept for reproducibility)
│
├── training/                   # Model training notebooks + datasets (gitignored)
├── tests/                      # Test suite (pytest)
├── docs/                       # Technical documentation
├── scripts/                    # Legacy roboflow/sports setup (not used by the pipeline)
├── videos/                     # Input/output video data (gitignored)
├── requirements.txt            # Python dependencies (single source of truth)
└── check_system.py             # Sanity check script
```

> **Note:** `src/main.py`, `src/main_1.py`, and `src/main_miguel.py` are earlier
> iterations of the pipeline kept for historical reference. All current
> development happens in `src/main_seg.py` — the web backend and docs target it.

---

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment with Python 3.11
conda create -n footar python=3.11 -y
conda activate footar

# Install PyTorch with CUDA 12.1
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python check_system.py
```

### 3. Run the Pipeline (CLI)

```bash
# Full RADAR mode (detection + tracking + teams + pitch projection + stats)
python src/main_seg.py \
    --source_video_path videos/input/lp1/round1/goal_01.mp4 \
    --target_video_path output.mp4 \
    --device cuda \
    --mode RADAR

# Real-time display (no output file)
python src/main_seg.py \
    --source_video_path videos/input/lp1/round1/goal_01.mp4 \
    --device cuda \
    --mode TEAM_CLASSIFICATION

# Headless with stats export
python src/main_seg.py \
    --source_video_path video.mp4 \
    --target_video_path out.mp4 \
    --stats_output_dir stats_out \
    --no_preview \
    --mode RADAR
```

### 4. Run the Web UI (recommended)

```powershell
# Backend (from player_detection/) — use the inference environment
python -m web.backend

# Frontend (from player_detection/web/frontend/)
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. See [web/README.md](web/README.md) for processor
Python selection (`FOOTAR_PYTHON` / `.env`) and the API reference.

### CLI Arguments (`src/main_seg.py`)

| Argument | Type | Default | Description |
|---|---|---|---|
| `--source_video_path` | `str` | *required* | Path to video file or directory |
| `--target_video_path` | `str` | `None` | Output path (omit for real-time display) |
| `--device` | `str` | `cuda` | `cuda` or `cpu` |
| `--mode` | `Mode` | `RADAR` | Processing mode (see above) |
| `--debug` | `flag` | `False` | Save team classification debug images |
| `--debug_output_dir` | `str` | `None` | Debug image output directory |
| `--stats_output_dir` | `str` | `None` | Directory for `stats.json`, `positions.json`, heatmap PNGs |
| `--player_track_imgsz` | `int` | `1024` | Player tracking inference size (RADAR) |
| `--pitch_every_n_frames` | `int` | `5` | Run pitch detection every N frames (RADAR) |
| `--ball_track_imgsz` | `int` | `960` | Ball tracking inference size (RADAR) |
| `--ball_track_every_n_frames` | `int` | `2` | Run ball tracking every N frames (RADAR) |
| `--ball_track_conf` | `float` | `0.25` | Ball tracking confidence threshold (RADAR) |
| `--ball_max_hold_frames` | `int` | `3` | Keep last ball detection for N missed frames |
| `--no_preview` | `flag` | `False` | Disable OpenCV preview windows (headless) |
| `--structured_logs` | `flag` | `False` | Emit machine-readable `FOOTAR_EVENT` JSON progress |
| `--live_frame_dir` | `str` | `None` | Publish latest processed frame as `latest.jpg` |
| `--live_frame_every` | `int` | `1` | Publish one live frame every N processed frames |

---

## Team Classification Pipeline

The team classifier (`sports/common/team_seg.py`) uses an unsupervised,
segmentation-based approach — no labeled training data required:

```
Frame → Segmentation Mask (yolo11m-seg: exact player pixels, no grass/background)
      → 3D HSV Histogram (Hue 8 × Saturation 8 × Value 4 = 256 features)
      → K-Means (k=2, k-means++ init)
      → Temporal Voting (20-frame sliding window)
      → Lock (consistent assignment after 20 frames)
      → GK Override (>70% goalkeeper class → neutral team)
```

The Value channel in the histogram discriminates teams with similar hues
(e.g. dark red vs. light red kits). Enable debug mode to visualize each stage:

```bash
python src/main_seg.py \
    --source_video_path video.mp4 \
    --mode TEAM_CLASSIFICATION \
    --debug
```

---

## Development Workflow

### Training New Models

Training notebooks are in `training/notebooks/`:

| Notebook | Purpose | Base Model |
|---|---|---|
| `train_player_detector.ipynb` | Player/GK/Referee/Ball detection | YOLOv11/12 |
| `train_ball_detector.ipynb` | Ball-only detection (high recall) | YOLOv11-Medium |
| `train_pitch_keypoint_detector.ipynb` | 32-keypoint pitch pose | YOLOv11-Medium Pose |

After training, copy `best.pt` to `models/active/` and update the path constant
in `src/main_seg.py` **and** `web/backend/config.py` (`REQUIRED_MODELS`).

### Running Tests

```bash
# Python suite (interpolation, exports, trackers, web backend)
python -m pytest tests/ --ignore=tests/test_seg.py

# Frontend suite (from web/frontend/)
npx vitest run
```

`tests/test_seg.py` requires the YOLO models and a GPU — run it manually.

### BoT-SORT Tracker Tuning

Edit `configs/futebol_botsort.yaml` to adjust tracker behavior:

```yaml
track_buffer: 60         # Frames to keep lost tracks (↑ = more ID persistence)
track_high_thresh: 0.6   # Initial detection confidence threshold
gmc_method: sparseOptFlow # Camera motion compensation method
match_thresh: 0.8        # IOU matching threshold
```

---

## Technical Documentation

| Document | Content |
|---|---|
| [Technical Report](docs/RELATORIO_TECNICO.md) | Full system architecture, model training configs, performance metrics |
| [Meeting Summary](docs/RESUMO_TECNICO_REUNIAO.md) | Team classification, interpolation, and BoT-SORT improvements |
| [BoT-SORT Migration](docs/BOTSORT_REFACTOR.md) | Norfair/ByteTrack → BoT-SORT refactoring details |
| [Interpolation Refactor](docs/INTERPOLATION_REFACTOR.md) | Dual-pass → single-pass ball interpolation architecture |
| [Web UI Guide](web/README.md) | Backend/frontend setup, processor Python selection, API reference |
| [CLI Examples](docs/run_examples.md) | Tested command-line invocations |
| [Install Guide](docs/install_instructions.md) | Step-by-step environment setup |

---

## License

- **YOLO models & Ultralytics**: [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0)
- **Sports module** (fork of [roboflow/sports](https://github.com/roboflow/sports)): MIT
- **FootAR application code**: AGPL-3.0

---

## Acknowledgements

- [Roboflow](https://roboflow.com/) — Original sports analysis framework and datasets
- [Ultralytics](https://ultralytics.com/) — YOLO model architecture and training pipeline
- [Supervision](https://supervision.roboflow.com/) — Video processing and annotation toolkit
- [VSports](https://vsports.pt/) — Liga Portugal match footage
