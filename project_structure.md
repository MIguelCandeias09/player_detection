# FootAR — Project Structure Audit & Reorganization Plan

> **Date:** 2026-02-13  
> **Author:** Automated Architecture Review  
> **Scope:** Full repository at `FootAR/`

---

## 1. Context Summary

**FootAR** is a computer vision system for automatic analysis of professional football (soccer) match videos from Liga Portugal 2024/25. Built on a modified fork of [roboflow/sports](https://github.com/roboflow/sports), it provides:

- **Object Detection** — Players, goalkeepers, referees, ball (YOLOv11/12)
- **Multi-Object Tracking** — Persistent IDs via BoT-SORT with GMC (Sparse Optical Flow)
- **Team Classification** — Unsupervised clustering (HSV histogram + K-Means + temporal voting)
- **Tactical Mapping** — 2D→3D homography projection to virtual pitch (RADAR mode)
- **Ball Interpolation** — Real-time single-pass gap filling (30-frame buffer)

**Stack:** Python 3.11, PyTorch 2.0, CUDA 12.1, Ultralytics YOLO, Supervision, OpenCV 4.8  
**Performance:** ~25 FPS on NVIDIA RTX 3060, ~8 FPS on CPU

The current repository is a working prototype with significant technical debt: 6 backup copies of the main script, duplicated datasets, scattered model weights, orphaned cache files, and no clear separation between application code, training artifacts, and raw data.

---

## 2. Current State — Full Tree

```
FootAR/
├── .git/
├── .github/
│   └── instructions/
│       └── snyk_rules.instructions.md
├── .gitignore
├── .vscode/
├── PA-IA.05_FootAR_GanttChart.xlsx
├── readme.txt
├── RELATORIO_TECNICO.md                          # 541 lines — full technical report
├── RESUMO_TECNICO_REUNIAO.md                     # 295 lines — meeting summary
│
├── datasets_and_models/                          # ⚠ DUPLICATE — same data exists inside roboflow_sports_footar/
│   ├── football-field-detection-1/
│   │   ├── data.yaml
│   │   ├── README.dataset.txt
│   │   ├── README.roboflow.txt
│   │   ├── test/images/ + labels/
│   │   ├── train/images/ + labels/
│   │   └── valid/images/ + labels/
│   └── football-players-and-ball-1/
│       ├── data.yaml
│       ├── yolo11n.pt                            # ⚠ Pretrained base — not project-specific
│       ├── yolo11s.pt                            # ⚠ Pretrained base — not project-specific
│       ├── first_train/best.pt                   # ⚠ Early training artifact
│       ├── runs/detect/                          # ⚠ 22 subfolders of old training runs
│       │   ├── train/ through train14/
│       │   ├── val/, val2/
│       │   └── predict/ through predict6/
│       ├── test/images/ + labels/
│       ├── train/images/ + labels/
│       └── valid/images/ + labels/
│
├── OriginalVideosUncut/                          # 7.54 GB raw VSports footage
│   ├── estádios selecionados.txt
│   └── lp1/
│       ├── round1/                               # 135 files
│       └── round2/                               # 74 files
│
└── roboflow_sports_footar/                       # ── MAIN APPLICATION ──
    ├── main.py                                   # 1104 lines — CURRENT entry point
    ├── 1-main_BEFORE_MAX_TEAM_ID.py              # ⛔ OBSOLETE backup (551 lines)
    ├── 2-main_MAX_TEAM_ID.py                     # ⛔ OBSOLETE backup (561 lines)
    ├── 3-main_MAX_TEAM_ID_PROGRESSIVE.py         # ⛔ OBSOLETE backup (566 lines)
    ├── main - 2025-02-17.py                      # ⛔ OBSOLETE backup (639 lines)
    ├── main - 2025-02-19.py                      # ⛔ OBSOLETE backup (632 lines)
    ├── main (1).py                               # ⛔ OBSOLETE backup (1110 lines)
    │
    ├── check_metrics.py                          # Utility — reads YOLO CSV results
    ├── test_interpolation.py                     # Unit test — ball interpolator
    │
    ├── README.md                                 # Original Roboflow README
    ├── BOTSORT_REFACTOR.md                       # Refactoring docs — BoT-SORT migration
    ├── INTERPOLATION_REFACTOR.md                 # Refactoring docs — ball interpolation
    ├── requirements.txt                          # ⚠ Incomplete (only ultralytics, gdown)
    ├── install_instructions.txt                  # Manual install steps
    ├── run command examples.txt                  # 168 lines of tested CLI examples
    ├── setup.sh                                  # Downloads models + demo videos
    ├── futebol_botsort.yaml                      # BoT-SORT tracker config
    │
    ├── team_training_crops.png                   # ⚠ Debug visualization — root level
    ├── team_training_3d_lab.png                  # ⚠ Debug visualization — root level
    ├── team_kdistance_graph.png                  # ⚠ Debug visualization — root level
    │
    ├── __pycache__/                              # ⛔ Should be gitignored/deleted
    │   └── main.cpython-311.pyc
    │
    ├── sports/                                   # ── CORE LIBRARY ──
    │   ├── __init__.py
    │   ├── __pycache__/                          # ⛔ Stale bytecode
    │   ├── annotators/
    │   │   ├── __init__.py
    │   │   ├── __pycache__/                      # ⛔ Stale bytecode
    │   │   └── soccer.py                         # draw_pitch(), draw_points_on_pitch()
    │   ├── common/
    │   │   ├── __init__.py
    │   │   ├── __pycache__/                      # ⛔ Stale bytecode (includes myColorThief orphan)
    │   │   ├── ball.py                           # BallAnnotator, BallTracker
    │   │   ├── ball_interpolator.py              # RealTimeBallInterpolator
    │   │   ├── team.py                           # TeamClassifier (522 lines)
    │   │   └── view.py                           # ViewTransformer (homography)
    │   └── configs/
    │       ├── __init__.py
    │       ├── __pycache__/                      # ⛔ Stale bytecode
    │       └── soccer.py                         # SoccerPitchConfiguration
    │
    ├── data/                                     # ── MODEL WEIGHTS (1.08 GB) ──
    │   ├── player_y12l_footar_best.pt            # ✅ ACTIVE — player detection
    │   ├── ball_y11m_footar_best.pt              # ✅ ACTIVE — ball detection
    │   ├── football-pitch-detection-mike_640_v11m.pt  # ✅ ACTIVE — pitch keypoints
    │   ├── pitch_y11x_keypoint_best.pt           # ⚠ EXPERIMENTAL (failed, needs 1280)
    │   ├── football-pitch-detection-mike_1280.pt # ⚠ Unused variant
    │   ├── football-pitch-detection-mike_640.pt  # ⚠ Unused variant
    │   ├── football-pitch-detection.pt           # ⚠ Original Roboflow (superseded)
    │   ├── football-player-detection.pt          # ⚠ Original Roboflow (superseded)
    │   ├── football-player-detection_mike.pt     # ⚠ Early custom model (superseded)
    │   ├── football-ball-detection.pt            # ⚠ Original Roboflow (superseded)
    │   ├── ball_y12m_footar_best.pt              # ⚠ Experimental variant
    │   ├── ball_y12m_8gb_best.pt                 # ⚠ Experimental variant
    │   ├── ball_y12s_best.pt                     # ⚠ Experimental (recall 30%)
    │   ├── ball_y12s_optimized_best.pt           # ⚠ Experimental variant
    │   ├── player_y12l_8gb_best.pt               # ⚠ Experimental variant
    │   └── yolo12s.pt                            # ⚠ Pretrained base (not custom)
    │
    ├── datasets_and_models/                      # ⚠ DUPLICATE of root-level dataset
    │   └── football-players-and-ball-1/
    │       ├── data.yaml
    │       ├── temp_data_for_training.yaml        # ⚠ Debug config (val→train)
    │       ├── test/ train/ valid/
    │
    ├── notebooks/                                # ── TRAINING NOTEBOOKS ──
    │   ├── train_ball_detector.ipynb
    │   ├── train_pitch_keypoint_detector.ipynb
    │   ├── train_player_detector.ipynb
    │   ├── yolo11m.pt, yolo11m-pose.pt           # ⚠ Pretrained bases
    │   ├── yolo11n.pt, yolo11x-pose.pt           # ⚠ Pretrained bases
    │   ├── yolo12l.pt, yolo12m.pt, yolo12s.pt    # ⚠ Pretrained bases
    │   ├── datasets/                             # ⚠ Yet another copy of datasets
    │   │   ├── football-ball-detection-2/
    │   │   └── football-field-detection-12/
    │   └── runs/                                 # Training run outputs
    │       ├── detect/ (12+ training runs)
    │       └── pose/ (3+ training runs)
    │
    ├── imagens_relatorio/                        # Debug visualizations (20 PNGs)
    │
    ├── runs/                                     # ⚠ Another runs directory
    │   ├── .DS_Store                             # ⛔ macOS artifact
    │   └── pose/train/weights/
    │
    └── videos/                                   # ── INPUT/OUTPUT VIDEOS (2.14 GB) ──
        └── lp1/
            ├── round1/                           # 121 source clips
            ├── round2/                           # 106 source clips
            └── out_melhores/                     # 30 best processed outputs
```

---

## 3. Proposed Architecture

```
FootAR/
├── .git/
├── .github/
│   └── instructions/
│       └── snyk_rules.instructions.md
├── .gitignore                                    # Updated: add __pycache__, .DS_Store, *.png debug
├── .vscode/
│
├── README.md                                     # Consolidated project README (merge readme.txt + Roboflow README)
├── LICENSE                                       # AGPL-3.0 (YOLO) + MIT (sports module)
├── project_structure.md                          # This document
│
├── docs/                                         # ── ALL DOCUMENTATION ──
│   ├── RELATORIO_TECNICO.md                      # Technical report
│   ├── RESUMO_TECNICO_REUNIAO.md                 # Meeting summary
│   ├── BOTSORT_REFACTOR.md                       # BoT-SORT migration notes
│   ├── INTERPOLATION_REFACTOR.md                 # Interpolation refactor notes
│   ├── install_instructions.md                   # Setup guide (renamed from .txt)
│   ├── run_examples.md                           # CLI examples (renamed from .txt)
│   ├── stadiums.md                               # Selected stadiums (renamed from .txt)
│   ├── PA-IA.05_FootAR_GanttChart.xlsx           # Project timeline
│   └── images/                                   # Report/debug visualizations
│       ├── team_training_crops.png
│       ├── team_training_3d_lab.png
│       ├── team_kdistance_graph.png
│       └── pipeline/                             # Pipeline stage visualizations
│           ├── 1_crops_frame{50,100,150,200}.png
│           ├── 2_hsv_masks_frame{50,100,150,200}.png
│           ├── 3_histograms_frame{50,100,150,200}.png
│           ├── 4_kmeans_pca_frame{50,100,150,200}.png
│           └── 5_result_frame{50,100,150,200}.png
│
├── src/                                          # ── APPLICATION CODE ──
│   ├── main.py                                   # Entry point (current main.py)
│   └── sports/                                   # Core library
│       ├── __init__.py
│       ├── annotators/
│       │   ├── __init__.py
│       │   └── soccer.py
│       ├── common/
│       │   ├── __init__.py
│       │   ├── ball.py
│       │   ├── ball_interpolator.py
│       │   ├── team.py
│       │   └── view.py
│       └── configs/
│           ├── __init__.py
│           └── soccer.py
│
├── configs/                                      # ── RUNTIME CONFIGS ──
│   └── futebol_botsort.yaml                      # BoT-SORT tracker parameters
│
├── models/                                       # ── TRAINED MODEL WEIGHTS ──
│   ├── active/                                   # Production models (referenced by main.py)
│   │   ├── player_y12l_footar_best.pt
│   │   ├── ball_y11m_footar_best.pt
│   │   └── pitch_v11m_640_footar_best.pt         # Renamed from football-pitch-detection-mike_640_v11m.pt
│   └── archive/                                  # Kept for reproducibility, not loaded at runtime
│       ├── pitch_y11x_keypoint_best.pt
│       ├── football-pitch-detection-mike_1280.pt
│       ├── football-pitch-detection-mike_640.pt
│       ├── ball_y12m_footar_best.pt
│       ├── ball_y12m_8gb_best.pt
│       ├── ball_y12s_optimized_best.pt
│       └── player_y12l_8gb_best.pt
│
├── training/                                     # ── MODEL TRAINING ──
│   ├── notebooks/
│   │   ├── train_ball_detector.ipynb
│   │   ├── train_pitch_keypoint_detector.ipynb
│   │   └── train_player_detector.ipynb
│   └── datasets/                                 # Single source of truth for datasets
│       ├── football-players-and-ball/
│       │   ├── data.yaml
│       │   ├── test/ train/ valid/
│       ├── football-ball-detection/
│       │   ├── data.yaml
│       │   ├── test/ train/ valid/
│       └── football-field-detection/
│           ├── data.yaml
│           ├── test/ train/ valid/
│
├── tests/                                        # ── TEST SUITE ──
│   ├── test_interpolation.py
│   └── check_metrics.py
│
├── scripts/                                      # ── UTILITY SCRIPTS ──
│   ├── setup.sh                                  # Download models + demo videos
│   └── requirements.txt                          # Full dependency list
│
└── videos/                                       # ── VIDEO DATA (gitignored) ──
    ├── input/
    │   └── lp1/
    │       ├── round1/
    │       └── round2/
    ├── output/
    │   └── lp1/
    │       └── out_melhores/
    └── raw/                                      # Uncut VSports footage
        └── lp1/
            ├── round1/
            └── round2/
```

---

## 4. Action Registry — File Migration Table

### 4.1 Application Code

| Old Path | New Path | Action |
|----------|----------|--------|
| `roboflow_sports_footar/main.py` | `src/main.py` | **Move** |
| `roboflow_sports_footar/sports/**` | `src/sports/**` | **Move** (entire module, excluding `__pycache__/`) |

### 4.2 Configuration

| Old Path | New Path | Action |
|----------|----------|--------|
| `roboflow_sports_footar/futebol_botsort.yaml` | `configs/futebol_botsort.yaml` | **Move** |

### 4.3 Documentation

| Old Path | New Path | Action |
|----------|----------|--------|
| `readme.txt` | `README.md` | **Rename + merge** with Roboflow README |
| `RELATORIO_TECNICO.md` | `docs/RELATORIO_TECNICO.md` | **Move** |
| `RESUMO_TECNICO_REUNIAO.md` | `docs/RESUMO_TECNICO_REUNIAO.md` | **Move** |
| `roboflow_sports_footar/BOTSORT_REFACTOR.md` | `docs/BOTSORT_REFACTOR.md` | **Move** |
| `roboflow_sports_footar/INTERPOLATION_REFACTOR.md` | `docs/INTERPOLATION_REFACTOR.md` | **Move** |
| `roboflow_sports_footar/README.md` | *(merge into root README.md)* | **Merge + Delete** |
| `roboflow_sports_footar/install_instructions.txt` | `docs/install_instructions.md` | **Move + Rename** (.txt → .md) |
| `roboflow_sports_footar/run command examples.txt` | `docs/run_examples.md` | **Move + Rename** |
| `OriginalVideosUncut/estádios selecionados.txt` | `docs/stadiums.md` | **Move + Rename** |
| `PA-IA.05_FootAR_GanttChart.xlsx` | `docs/PA-IA.05_FootAR_GanttChart.xlsx` | **Move** |

### 4.4 Visualizations / Report Images

| Old Path | New Path | Action |
|----------|----------|--------|
| `roboflow_sports_footar/team_training_crops.png` | `docs/images/team_training_crops.png` | **Move** |
| `roboflow_sports_footar/team_training_3d_lab.png` | `docs/images/team_training_3d_lab.png` | **Move** |
| `roboflow_sports_footar/team_kdistance_graph.png` | `docs/images/team_kdistance_graph.png` | **Move** |
| `roboflow_sports_footar/imagens_relatorio/*.png` (20 files) | `docs/images/pipeline/*.png` | **Move** |

### 4.5 Model Weights

| Old Path | New Path | Action |
|----------|----------|--------|
| `roboflow_sports_footar/data/player_y12l_footar_best.pt` | `models/active/player_y12l_footar_best.pt` | **Move** |
| `roboflow_sports_footar/data/ball_y11m_footar_best.pt` | `models/active/ball_y11m_footar_best.pt` | **Move** |
| `roboflow_sports_footar/data/football-pitch-detection-mike_640_v11m.pt` | `models/active/pitch_v11m_640_footar_best.pt` | **Move + Rename** |
| `roboflow_sports_footar/data/pitch_y11x_keypoint_best.pt` | `models/archive/pitch_y11x_keypoint_best.pt` | **Move** |
| `roboflow_sports_footar/data/football-pitch-detection-mike_1280.pt` | `models/archive/football-pitch-detection-mike_1280.pt` | **Move** |
| `roboflow_sports_footar/data/football-pitch-detection-mike_640.pt` | `models/archive/football-pitch-detection-mike_640.pt` | **Move** |
| `roboflow_sports_footar/data/ball_y12m_footar_best.pt` | `models/archive/ball_y12m_footar_best.pt` | **Move** |
| `roboflow_sports_footar/data/ball_y12m_8gb_best.pt` | `models/archive/ball_y12m_8gb_best.pt` | **Move** |
| `roboflow_sports_footar/data/ball_y12s_optimized_best.pt` | `models/archive/ball_y12s_optimized_best.pt` | **Move** |
| `roboflow_sports_footar/data/player_y12l_8gb_best.pt` | `models/archive/player_y12l_8gb_best.pt` | **Move** |

### 4.6 Training

| Old Path | New Path | Action |
|----------|----------|--------|
| `roboflow_sports_footar/notebooks/train_ball_detector.ipynb` | `training/notebooks/train_ball_detector.ipynb` | **Move** |
| `roboflow_sports_footar/notebooks/train_pitch_keypoint_detector.ipynb` | `training/notebooks/train_pitch_keypoint_detector.ipynb` | **Move** |
| `roboflow_sports_footar/notebooks/train_player_detector.ipynb` | `training/notebooks/train_player_detector.ipynb` | **Move** |
| `roboflow_sports_footar/notebooks/datasets/football-ball-detection-2/` | `training/datasets/football-ball-detection/` | **Move + Rename** |
| `roboflow_sports_footar/notebooks/datasets/football-field-detection-12/` | `training/datasets/football-field-detection/` | **Move + Rename** |
| `datasets_and_models/football-players-and-ball-1/` | `training/datasets/football-players-and-ball/` | **Move + Rename** (keep one copy) |
| `datasets_and_models/football-field-detection-1/` | *(merged with above)* | **Delete** (duplicate) |

### 4.7 Tests & Scripts

| Old Path | New Path | Action |
|----------|----------|--------|
| `roboflow_sports_footar/test_interpolation.py` | `tests/test_interpolation.py` | **Move** |
| `roboflow_sports_footar/check_metrics.py` | `tests/check_metrics.py` | **Move** |
| `roboflow_sports_footar/setup.sh` | `scripts/setup.sh` | **Move** |
| `roboflow_sports_footar/requirements.txt` | `scripts/requirements.txt` | **Move** (update to include all deps) |

### 4.8 Videos

| Old Path | New Path | Action |
|----------|----------|--------|
| `roboflow_sports_footar/videos/lp1/round1/` | `videos/input/lp1/round1/` | **Move** |
| `roboflow_sports_footar/videos/lp1/round2/` | `videos/input/lp1/round2/` | **Move** |
| `roboflow_sports_footar/videos/lp1/out_melhores/` | `videos/output/lp1/out_melhores/` | **Move** |
| `OriginalVideosUncut/lp1/` | `videos/raw/lp1/` | **Move** |

---

## 5. Deletion List

### 5.1 Obsolete Backup Scripts

| File | Size | Justification |
|------|------|---------------|
| `roboflow_sports_footar/1-main_BEFORE_MAX_TEAM_ID.py` | 551 lines | Superseded by `main.py`. Evolution is in git history. |
| `roboflow_sports_footar/2-main_MAX_TEAM_ID.py` | 561 lines | Superseded by `main.py`. |
| `roboflow_sports_footar/3-main_MAX_TEAM_ID_PROGRESSIVE.py` | 566 lines | Superseded by `main.py`. |
| `roboflow_sports_footar/main - 2025-02-17.py` | 639 lines | Date-stamped backup. Use git for versioning. |
| `roboflow_sports_footar/main - 2025-02-19.py` | 632 lines | Date-stamped backup. Use git for versioning. |
| `roboflow_sports_footar/main (1).py` | 1110 lines | Near-duplicate of current `main.py`. |

### 5.2 Superseded / Failed Model Weights

| File | Size | Justification |
|------|------|---------------|
| `roboflow_sports_footar/data/football-pitch-detection.pt` | 133.7 MB | Original Roboflow model — superseded by custom-trained variant. |
| `roboflow_sports_footar/data/football-player-detection.pt` | 130.5 MB | Original Roboflow model — superseded by `player_y12l_footar_best.pt`. |
| `roboflow_sports_footar/data/football-player-detection_mike.pt` | 18.4 MB | Early custom model — superseded by `player_y12l_footar_best.pt`. |
| `roboflow_sports_footar/data/football-ball-detection.pt` | 130.4 MB | Original Roboflow model (uses slicer, slow) — superseded. |
| `roboflow_sports_footar/data/ball_y12s_best.pt` | 18.0 MB | Failed experiment (30% recall). No production use. |
| `roboflow_sports_footar/data/yolo12s.pt` | 18.1 MB | Generic pretrained base — not project-specific. Download on demand. |
| `datasets_and_models/football-players-and-ball-1/yolo11n.pt` | ~6 MB | Pretrained base — not project-specific. |
| `datasets_and_models/football-players-and-ball-1/yolo11s.pt` | ~22 MB | Pretrained base — not project-specific. |
| `roboflow_sports_footar/notebooks/yolo11m.pt` | ~40 MB | Pretrained base — download on demand during training. |
| `roboflow_sports_footar/notebooks/yolo11m-pose.pt` | ~40 MB | Pretrained base — download on demand during training. |
| `roboflow_sports_footar/notebooks/yolo11n.pt` | ~6 MB | Pretrained base — download on demand during training. |
| `roboflow_sports_footar/notebooks/yolo11x-pose.pt` | ~100 MB | Pretrained base — download on demand during training. |
| `roboflow_sports_footar/notebooks/yolo12l.pt` | ~80 MB | Pretrained base — download on demand during training. |
| `roboflow_sports_footar/notebooks/yolo12m.pt` | ~40 MB | Pretrained base — download on demand during training. |
| `roboflow_sports_footar/notebooks/yolo12s.pt` | ~18 MB | Pretrained base — download on demand during training. |

**Estimated space recovered from model deletions: ~900 MB**

### 5.3 Stale Cache & OS Artifacts

| File | Justification |
|------|---------------|
| `roboflow_sports_footar/__pycache__/` (entire dir) | Compiled bytecode — auto-generated. |
| `roboflow_sports_footar/sports/__pycache__/` | Stale bytecode. |
| `roboflow_sports_footar/sports/annotators/__pycache__/` | Stale bytecode. |
| `roboflow_sports_footar/sports/common/__pycache__/` | Stale bytecode. Contains orphan `myColorThief.cpython-311.pyc` (source deleted). |
| `roboflow_sports_footar/sports/configs/__pycache__/` | Stale bytecode. |
| `roboflow_sports_footar/runs/.DS_Store` | macOS metadata artifact. |
| `roboflow_sports_footar/runs/pose/.DS_Store` | macOS metadata artifact. |
| `roboflow_sports_footar/runs/pose/train/.DS_Store` | macOS metadata artifact. |

### 5.4 Debug / Temporary Files

| File | Justification |
|------|---------------|
| `roboflow_sports_footar/datasets_and_models/football-players-and-ball-1/temp_data_for_training.yaml` | Debug config (validation set pointed to training set). Should not exist in repo. |
| `datasets_and_models/football-players-and-ball-1/first_train/best.pt` | Earliest training artifact — superseded by all subsequent models. |

### 5.5 Duplicate Data

| Path | Justification |
|------|---------------|
| `datasets_and_models/football-field-detection-1/` | Duplicate — same dataset exists at `roboflow_sports_footar/notebooks/datasets/football-field-detection-12/` (newer version). |
| `roboflow_sports_footar/datasets_and_models/` | Duplicate — subset of `datasets_and_models/` at root. |
| `datasets_and_models/football-players-and-ball-1/runs/` (22 subfolders) | Old training runs — results captured in reports. Not needed for production. |

### 5.6 Old Training Runs

| Path | Justification |
|------|---------------|
| `roboflow_sports_footar/runs/` | Contains only one pose training run + `.DS_Store` files. Best weights already saved to `data/`. |
| `roboflow_sports_footar/notebooks/runs/` | 15+ training run directories. Best weights already extracted to `data/`. Keep only if re-training is planned. |
| `datasets_and_models/football-players-and-ball-1/runs/` | 22 old run directories. Best weights already extracted. |

### 5.7 Empty / Zero-byte Files

| File | Justification |
|------|---------------|
| `roboflow_sports_footar/test_norfair.mp4` (if exists, 0 bytes) | Empty output from abandoned Norfair tracker experiment. |

---

## 6. `.gitignore` Updates

The current `.gitignore` blocks all `.md` files (`*.md`), which prevents documentation from being tracked. Proposed replacement:

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
*.egg
venv/
.venv/

# OS
.DS_Store
Thumbs.db

# IDE
.idea/

# Project-specific: large binary data (gitignored, use DVC or Git LFS)
models/**/*.pt
videos/
training/datasets/
training/notebooks/runs/

# Temporary
*.tmp
*.bak
```

**Key change:** Remove `*.md` from gitignore so that documentation is version-controlled.

---

## 7. Additional Recommendations

### 7.1 Immediate Actions (Priority 1)
1. **Delete all 6 backup main scripts** — git history preserves evolution
2. **Delete all `__pycache__/` directories** — add to `.gitignore`
3. **Delete `.DS_Store` files** — add to `.gitignore`
4. **Fix `.gitignore`** — remove `*.md` exclusion, add `__pycache__/`
5. **Update `requirements.txt`** to include all actual dependencies:
   ```
   ultralytics>=8.1.0
   supervision>=0.18.0
   opencv-python>=4.8.0
   numpy>=1.24.0
   umap-learn>=0.5.4
   scikit-learn>=1.3.0
   matplotlib>=3.7.0
   torch>=2.0.0
   gdown>=4.7.0
   ```

### 7.2 Short-Term (Priority 2)
1. **Consolidate datasets** — single `training/datasets/` directory, remove all duplicates
2. **Separate active vs archive models** — only 3 models are loaded at runtime
3. **Move source code to `src/`** — clear separation from data/docs/config
4. **Update model paths in `main.py`** — after moving weights to `models/active/`

### 7.3 Medium-Term (Priority 3)
1. **Add `pyproject.toml`** — modern Python project metadata and dependency management
2. **Add argument for model directory** — make model paths configurable, not hardcoded
3. **Extract hardcoded constants** from `main.py` into `configs/` YAML files
4. **Consider Git LFS** — for the 3 active model weights (~120 MB total)
5. **Add CI** — run `test_interpolation.py` on push

---

## 8. Space Impact Summary

| Category | Current Size | After Cleanup |
|----------|-------------|---------------|
| Obsolete model weights | ~900 MB | 0 MB (deleted) |
| Backup main scripts | ~140 KB | 0 KB (deleted) |
| Duplicate datasets | ~500 MB+ | 0 MB (deduplicated) |
| Old training runs | ~200 MB+ | 0 MB (deleted) |
| `__pycache__` + `.DS_Store` | ~2 MB | 0 MB (deleted) |
| **Total space recovered** | — | **~1.6 GB+** |

---

*This document should be reviewed before executing the migration. Update model paths in `src/main.py` and `scripts/setup.sh` after moving files.*
