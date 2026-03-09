"""
FootAR System Sanity Check
============================
Validates the project structure, dependencies, and importability
of core modules after migration to the new src/ architecture.

Usage:
    python check_system.py
"""

import os
import sys
import importlib
from pathlib import Path

# ============================================================================
# RESOLVE PROJECT ROOT (this script lives at the project root)
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

# Add src/ to Python path so sports.* modules can be imported
sys.path.insert(0, str(SRC_DIR))


# ============================================================================
# ANSI COLORS
# ============================================================================
class C:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def ok(msg: str):
    print(f"  {C.GREEN}[OK]{C.RESET}  {msg}")

def fail(msg: str):
    print(f"  {C.RED}[FAIL]{C.RESET}  {msg}")

def warn(msg: str):
    print(f"  {C.YELLOW}[WARN]{C.RESET}  {msg}")

def header(title: str):
    print(f"\n{C.BOLD}{C.CYAN}{'─'*60}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  {title}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'─'*60}{C.RESET}")


# ============================================================================
# CHECK 1: CRITICAL FILE PATHS
# ============================================================================
def check_paths() -> int:
    header("1. Critical File Paths")
    errors = 0

    critical_files = {
        # Application code
        "src/main.py":                              "Main entry point",
        "src/sports/__init__.py":                   "Sports package init",
        "src/sports/annotators/soccer.py":          "Pitch drawing module",
        "src/sports/common/ball.py":                "Ball tracker & annotator",
        "src/sports/common/ball_interpolator.py":   "Real-time ball interpolation",
        "src/sports/common/team.py":                "Team classifier (HSV + K-Means)",
        "src/sports/common/view.py":                "Homography view transformer",
        "src/sports/configs/soccer.py":             "Pitch configuration (32 keypoints)",

        # Configuration
        "configs/futebol_botsort.yaml":             "BoT-SORT tracker config",

        # Model weights (production)
        "models/active/player_y12l_footar_best.pt": "Player detection model (YOLOv12-L)",
        "models/active/ball_y11m_footar_best.pt":   "Ball detection model (YOLOv11-M)",
        "models/active/pitch_v11m_640_footar_best.pt": "Pitch keypoint model (YOLOv11-M Pose)",
    }

    for rel_path, description in critical_files.items():
        full_path = PROJECT_ROOT / rel_path
        if full_path.exists():
            size = full_path.stat().st_size
            size_str = f"({size / 1024 / 1024:.1f} MB)" if size > 1024 * 1024 else f"({size / 1024:.1f} KB)"
            ok(f"{rel_path} {size_str} — {description}")
        else:
            fail(f"{rel_path} — {description}")
            errors += 1

    # Optional directories
    optional_dirs = {
        "tests":              "Test suite",
        "docs":               "Documentation",
        "scripts":            "Utility scripts",
        "training/notebooks": "Training notebooks",
        "models/archive":     "Archived model weights",
    }

    print()
    for rel_path, description in optional_dirs.items():
        full_path = PROJECT_ROOT / rel_path
        if full_path.exists() and full_path.is_dir():
            count = sum(1 for _ in full_path.rglob("*") if _.is_file())
            ok(f"{rel_path}/ ({count} files) — {description}")
        else:
            warn(f"{rel_path}/ — {description} (not found)")

    return errors


# ============================================================================
# CHECK 2: PYTHON DEPENDENCIES
# ============================================================================
def check_dependencies() -> int:
    header("2. Python Dependencies")
    errors = 0

    required_packages = [
        ("ultralytics",     "YOLO model inference"),
        ("supervision",     "Video processing & annotations"),
        ("cv2",             "OpenCV computer vision"),
        ("numpy",           "Numerical computing"),
        ("sklearn",         "K-Means clustering (team classification)"),
        ("matplotlib",      "Visualization & debug plots"),
        ("tqdm",            "Progress bars"),
    ]

    for module_name, purpose in required_packages:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "unknown")
            ok(f"{module_name} ({version}) — {purpose}")
        except ImportError:
            fail(f"{module_name} — {purpose} (not installed)")
            errors += 1

    # Optional but recommended
    optional_packages = [
        ("umap",            "UMAP dimensionality reduction"),
        ("gdown",           "Google Drive model downloader"),
    ]

    print()
    for module_name, purpose in optional_packages:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "unknown")
            ok(f"{module_name} ({version}) — {purpose} [optional]")
        except ImportError:
            warn(f"{module_name} — {purpose} [optional, not installed]")

    return errors


# ============================================================================
# CHECK 3: PYTORCH & CUDA
# ============================================================================
def check_torch() -> int:
    header("3. PyTorch & CUDA")
    errors = 0

    try:
        import torch
        ok(f"torch {torch.__version__}")

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / (1024**3)
            ok(f"CUDA available — {device_name} ({vram:.1f} GB VRAM)")
            ok(f"CUDA version: {torch.version.cuda}")
            ok(f"cuDNN version: {torch.backends.cudnn.version()}")
        else:
            warn("CUDA not available — will fall back to CPU (slower)")

        # Quick tensor test
        device = "cuda" if torch.cuda.is_available() else "cpu"
        t = torch.zeros(1, device=device)
        ok(f"Tensor creation on '{device}' — OK")
        del t

    except ImportError:
        fail("torch not installed")
        errors += 1
    except Exception as e:
        fail(f"torch test failed: {e}")
        errors += 1

    return errors


# ============================================================================
# CHECK 4: SPORTS MODULE IMPORTS (dry run)
# ============================================================================
def check_imports() -> int:
    header("4. Sports Module Imports (src/)")
    errors = 0

    imports_to_test = [
        ("sports.configs.soccer",       "SoccerPitchConfiguration"),
        ("sports.common.view",          "ViewTransformer"),
        ("sports.common.team",          "TeamClassifier"),
        ("sports.common.ball",          "BallTracker"),
        ("sports.common.ball",          "BallAnnotator"),
        ("sports.common.ball_interpolator", "RealTimeBallInterpolator"),
        ("sports.common.ball_interpolator", "InterpolatedBallAnnotator"),
        ("sports.annotators.soccer",    "draw_pitch"),
        ("sports.annotators.soccer",    "draw_points_on_pitch"),
    ]

    for module_path, class_name in imports_to_test:
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            ok(f"from {module_path} import {class_name}")
        except ImportError as e:
            fail(f"from {module_path} import {class_name} — ImportError: {e}")
            errors += 1
        except AttributeError:
            fail(f"from {module_path} import {class_name} — class not found")
            errors += 1

    return errors


# ============================================================================
# CHECK 5: CLASS INSTANTIATION (dry run)
# ============================================================================
def check_instantiation() -> int:
    header("5. Class Instantiation (Dry Run)")
    errors = 0

    # 5a. SoccerPitchConfiguration
    try:
        from sports.configs.soccer import SoccerPitchConfiguration
        config = SoccerPitchConfiguration()
        assert config.width == 7000, f"width={config.width}"
        assert config.length == 12000, f"length={config.length}"
        assert len(config.vertices) == 32, f"vertices={len(config.vertices)}"
        assert len(config.edges) >= 30, f"edges={len(config.edges)}"
        ok(f"SoccerPitchConfiguration — {len(config.vertices)} vertices, {len(config.edges)} edges, {config.length}×{config.width} cm")
    except Exception as e:
        fail(f"SoccerPitchConfiguration — {e}")
        errors += 1

    # 5b. TeamClassifier
    try:
        from sports.common.team import TeamClassifier
        tc = TeamClassifier()
        assert tc.HISTORY_LENGTH == 30
        assert tc.HIST_BINS == [8, 8]
        ok(f"TeamClassifier — hist_bins={tc.HIST_BINS}, history={tc.HISTORY_LENGTH}")
    except Exception as e:
        fail(f"TeamClassifier — {e}")
        errors += 1

    # 5c. RealTimeBallInterpolator
    try:
        from sports.common.ball_interpolator import RealTimeBallInterpolator
        interp = RealTimeBallInterpolator(buffer_size=30)
        assert interp.buffer_size == 30
        ok(f"RealTimeBallInterpolator — buffer_size={interp.buffer_size}")
    except Exception as e:
        fail(f"RealTimeBallInterpolator — {e}")
        errors += 1

    # 5d. ViewTransformer (requires numpy arrays)
    try:
        import numpy as np
        from sports.common.view import ViewTransformer
        source = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        target = np.array([[0, 0], [200, 0], [200, 200], [0, 200]], dtype=np.float32)
        vt = ViewTransformer(source=source, target=target)
        transformed = vt.transform_points(np.array([[50, 50]], dtype=np.float32))
        assert transformed.shape == (1, 2), f"Expected (1,2), got {transformed.shape}"
        ok(f"ViewTransformer — homography computed, transform_points OK")
    except Exception as e:
        fail(f"ViewTransformer — {e}")
        errors += 1

    # 5e. draw_pitch
    try:
        from sports.annotators.soccer import draw_pitch
        from sports.configs.soccer import SoccerPitchConfiguration
        pitch = draw_pitch(config=SoccerPitchConfiguration())
        assert pitch is not None and pitch.shape[2] == 3
        ok(f"draw_pitch — output shape {pitch.shape}")
    except Exception as e:
        fail(f"draw_pitch — {e}")
        errors += 1

    return errors


# ============================================================================
# CHECK 6: MODEL PATH RESOLUTION
# ============================================================================
def check_model_paths() -> int:
    header("6. Model Path Resolution (as main.py sees them)")
    errors = 0

    # Simulate what main.py does
    parent_dir = str(SRC_DIR)  # src/
    project_root = str(PROJECT_ROOT)  # FootAR/

    model_paths = {
        "PLAYER_DETECTION_MODEL_PATH": os.path.join(project_root, "models", "active", "player_y12l_footar_best.pt"),
        "BALL_DETECTION_MODEL_PATH":   os.path.join(project_root, "models", "active", "ball_y11m_footar_best.pt"),
        "PITCH_DETECTION_MODEL_PATH":  os.path.join(project_root, "models", "active", "pitch_v11m_640_footar_best.pt"),
        "BOTSORT_CONFIG_PATH":         os.path.join(project_root, "configs", "futebol_botsort.yaml"),
    }

    for var_name, path in model_paths.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            ok(f"{var_name} -> {os.path.relpath(path, project_root)} ({size_mb:.1f} MB)")
        else:
            fail(f"{var_name} -> {path} (NOT FOUND)")
            errors += 1

    return errors


# ============================================================================
# MAIN
# ============================================================================
def main():
    print(f"\n{C.BOLD}{'='*60}{C.RESET}")
    print(f"{C.BOLD}  FootAR System Sanity Check{C.RESET}")
    print(f"{C.BOLD}  Project Root: {PROJECT_ROOT}{C.RESET}")
    print(f"{C.BOLD}  Python: {sys.version.split()[0]}{C.RESET}")
    print(f"{C.BOLD}{'='*60}{C.RESET}")

    total_errors = 0
    total_errors += check_paths()
    total_errors += check_dependencies()
    total_errors += check_torch()
    total_errors += check_imports()
    total_errors += check_instantiation()
    total_errors += check_model_paths()

    # Final verdict
    print(f"\n{C.BOLD}{'='*60}{C.RESET}")
    if total_errors == 0:
        print(f"{C.BOLD}{C.GREEN}  SYSTEM READY — All {6} checks passed{C.RESET}")
        print(f"{C.GREEN}  Run: python src/main.py --source_video_path <video> --mode RADAR{C.RESET}")
    else:
        print(f"{C.BOLD}{C.RED}  {total_errors} ERROR(S) DETECTED{C.RESET}")
        print(f"{C.RED}  Fix the issues above before running the pipeline.{C.RESET}")
    print(f"{C.BOLD}{'='*60}{C.RESET}\n")

    return total_errors


if __name__ == "__main__":
    sys.exit(main())
