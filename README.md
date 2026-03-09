# FootAR — Automatic Football Video Analysis

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![YOLO](https://img.shields.io/badge/Ultralytics-YOLOv11%2F12-purple.svg)](https://docs.ultralytics.com/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-red.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Liga Portugal](https://img.shields.io/badge/Liga%20Portugal-2024%2F25-green.svg)](#)

Real-time computer vision system for analyzing professional football match footage from **Liga Portugal 2024/25**. Detects players, goalkeepers, referees, and the ball; tracks persistent identities across frames; classifies teams by uniform color; and projects tactical positions onto a 2D pitch radar.

**~25 FPS** on NVIDIA RTX 4070 | **6 operational modes** | **Single-pass ball interpolation**

---

## Features

| Capability | Method | Model |
|---|---|---|
| **Player Detection** | YOLOv12-Large (4 classes: ball, GK, player, referee) | `player_y12l_footar_best.pt` |
| **Ball Detection** | YOLOv11-Medium + InferenceSlicer (640×640 tiles) | `ball_y11m_footar_best.pt` |
| **Pitch Keypoints** | YOLOv11-Medium Pose (32 keypoints, FIFA standard) | `pitch_v11m_640_footar_best.pt` |
| **Multi-Object Tracking** | BoT-SORT with GMC (Sparse Optical Flow, 60-frame buffer) | Native YOLO `.track()` |
| **Team Classification** | HSV 2D Histogram → K-Means → Temporal Voting + Soft Lock | Unsupervised (no model) |
| **Ball Interpolation** | Real-time single-pass linear interpolation (30-frame buffer) | Algorithmic |
| **Tactical Radar** | Homography projection (cv2.findHomography → 2D pitch) | Geometric |

### Processing Modes

```
PITCH_DETECTION      →  Visualize detected pitch keypoints with confidence colors
PLAYER_DETECTION     →  Bounding box detection of all players/GK/referees
BALL_DETECTION       →  Ball tracking with real-time interpolation for missed frames
PLAYER_TRACKING      →  Persistent IDs via BoT-SORT with camera motion compensation
TEAM_CLASSIFICATION  →  Color-based team assignment with temporal voting
RADAR                →  Full pipeline: detection + tracking + teams + 2D pitch radar
```

---

## Architecture

```
FootAR/
├── src/                        # Application source code
│   ├── main.py                 # CLI entry point (all 6 modes)
│   └── sports/                 # Core library
│       ├── annotators/         # Pitch drawing, point projection
│       ├── common/             # Ball tracker, interpolator, team classifier, view transformer
│       └── configs/            # Soccer pitch geometry (32 vertices, FIFA dimensions)
│
├── configs/                    # Runtime configuration
│   └── futebol_botsort.yaml    # BoT-SORT tracker parameters
│
├── models/                     # YOLO model weights
│   ├── active/                 # Production models (loaded by main.py)
│   │   ├── player_y12l_footar_best.pt
│   │   ├── ball_y11m_footar_best.pt
│   │   └── pitch_v11m_640_footar_best.pt
│   └── archive/                # Previous experiments (kept for reproducibility)
│
├── training/                   # Model training pipeline
│   ├── notebooks/              # Jupyter notebooks for player/ball/pitch training
│   └── datasets/               # YOLO-format datasets (gitignored)
│
├── tests/                      # Test suite
│   ├── test_interpolation.py   # Ball interpolator unit tests
│   └── check_metrics.py        # Training metrics analyzer
│
├── docs/                       # Technical documentation
├── scripts/                    # Setup & utility scripts
├── videos/                     # Input/output video data (gitignored)
│   ├── input/                  # Source match footage
│   ├── output/                 # Processed results
│   └── raw/                    # Uncut VSports recordings
│
├── requirements.txt            # Python dependencies
├── check_system.py             # Sanity check script
└── project_structure.md        # Architecture audit document
```

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

This script validates that all model files, dependencies, and Python imports resolve correctly.

### 3. Run the Pipeline

```bash
# Full RADAR mode (detection + tracking + teams + pitch projection)
python src/main.py \
    --source_video_path videos/input/lp1/round1/goal_01.mp4 \
    --target_video_path output.mp4 \
    --device cuda \
    --mode RADAR

# Real-time display (no output file)
python src/main.py \
    --source_video_path videos/input/lp1/round1/goal_01.mp4 \
    --device cuda \
    --mode TEAM_CLASSIFICATION

# Process entire directory
python src/main.py \
    --source_video_path videos/input/lp1/round1/ \
    --target_video_path videos/output/lp1/round1/ \
    --device cuda \
    --mode RADAR

# CPU fallback
python src/main.py \
    --source_video_path video.mp4 \
    --target_video_path out.mp4 \
    --device cpu \
    --mode PLAYER_DETECTION
```

### CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--source_video_path` | `str` | *required* | Path to video file or directory |
| `--target_video_path` | `str` | `None` | Output path (omit for real-time display) |
| `--device` | `str` | `cuda` | `cuda` or `cpu` |
| `--mode` | `Mode` | `RADAR` | Processing mode (see above) |
| `--debug` | `flag` | `False` | Save team classification debug images |
| `--debug_output_dir` | `str` | `debug_team_output` | Debug image output directory |

---

## Team Classification Pipeline

The team classifier uses an unsupervised approach — no labeled training data required:

```
Frame → Player Crop (top 50%, center 60%)
      → HSV Green Mask (remove grass pixels)
      → 2D Histogram (Hue × Saturation, 8×8 = 64 features)
      → K-Means (k=2, k-means++ init)
      → Temporal Voting (30-frame sliding window)
      → Soft Lock (consistent assignment after 30 frames)
      → GK Override (>70% goalkeeper class → neutral team)
```

Enable debug mode to visualize each stage:

```bash
python src/main.py \
    --source_video_path video.mp4 \
    --mode TEAM_CLASSIFICATION \
    --debug \
    --debug_output_dir docs/images/pipeline
```

---

## Development Workflow

### Training New Models

Training notebooks are in `training/notebooks/`:

| Notebook | Purpose | Base Model |
|---|---|---|
| `train_player_detector.ipynb` | Player/GK/Referee/Ball detection | YOLOv12-Large |
| `train_ball_detector.ipynb` | Ball-only detection (high recall) | YOLOv11-Medium |
| `train_pitch_keypoint_detector.ipynb` | 32-keypoint pitch pose | YOLOv11-Medium Pose |

Datasets live in `training/datasets/` (gitignored). Download from Roboflow:
- [football-players-and-ball](https://universe.roboflow.com/) — 4 classes
- [football-ball-detection](https://universe.roboflow.com/) — ball only
- [football-field-detection](https://universe.roboflow.com/) — 32 keypoints

After training, copy `best.pt` to `models/active/` and update the path constant in `src/main.py`.

### Running Tests

```bash
# Ball interpolation unit tests
python tests/test_interpolation.py

# Analyze training metrics from CSV logs
python tests/check_metrics.py
```

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
| [CLI Examples](docs/run_examples.md) | Tested command-line invocations |
| [Install Guide](docs/install_instructions.md) | Step-by-step environment setup |

---

## Model Performance

| Model | mAP50 | Recall | Precision | Input Size |
|---|---|---|---|---|
| Player (YOLOv12-L) | 98.9% (player) | 97.1% (GK) | 96.7% (referee) | 1280px |
| Ball (YOLOv11-M) | 74.6% | 68.0% | 87.9% | 1024px |
| Pitch (YOLOv11-M Pose) | 65.8% | — | — | 640px |

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
