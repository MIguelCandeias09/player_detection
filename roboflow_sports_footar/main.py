import argparse
from enum import Enum
from typing import Iterator, List
from collections import deque

import os
import time
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
import statistics

# Try to import matplotlib for display
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print('matplotlib not available - real-time display disabled')

print('importing sports files')
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.ball_interpolator import RealTimeBallInterpolator, InterpolatedBallAnnotator
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration


PARENT_DIR = os.path.dirname(os.path.abspath(__file__))

# BoT-SORT Tracker Configuration (with GMC for camera motion compensation)
BOTSORT_CONFIG_PATH = os.path.join(PARENT_DIR, 'futebol_botsort.yaml')
# Use the newly trained YOLOv12m player detector (98.9% player mAP50, 97.1% goalkeeper, 96.7% referee)
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/player_y12l_footar_best.pt')
# PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection_mike.pt')  # old model
# PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/yolo12s.pt')
# PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection.pt')
# PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection-mike_1280.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection-mike_640_v11m.pt')  # Modelo anterior (mais estável)
# PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/pitch_y11x_keypoint_best.pt')  # YOLOv11x-pose 32-keypoint (65.8% mAP50) - PRECISA MELHORIAS
# Use the newly trained YOLOv11m ball detector (74.6% mAP50, 68.0% recall, 87.9% precision @ 1024px)
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/ball_y11m_footar_best.pt')
# BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/ball_y12s_optimized_best.pt')  # previous version
# BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/ball_y12s_best.pt')  # old version (recall 30%)
# BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-ball-detection.pt')   # this model is too slow

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 5
CONFIG = SoccerPitchConfiguration()

# Team colors: Team 0 (Red), Team 1 (Blue), Goalkeeper (Orange), Referee (Yellow)
#          Team 0     Team 1     Goalkeeper Referee
COLORS = ['#FF1744', '#2196F3', '#FF6347', '#FFD700', '#8000FF']
VERTEX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    thickness=2,
    edges=CONFIG.edges,
)
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    base=20,
    height=15,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)

startT = time.perf_counter()
performance_times = {}
def performanceMeter(desc: str = ''):
    global startT, performance_times
    
    elapsed = time.perf_counter() - startT
    if desc:
        performance_times[desc] = elapsed
    startT = time.perf_counter()

class Mode(Enum):
    """
    Enum class representing different modes of operation for Soccer AI video analysis.
    """
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    BALL_DETECTION = 'BALL_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR = 'RADAR'


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.

    Args:
        frame (np.ndarray): The frame from which to extract crops.
        detections (sv.Detections): Detected objects with bounding boxes.

    Returns:
        List[np.ndarray]: List of cropped images.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]

def get_crops_with_tracker_id(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.

    Args:
        frame (np.ndarray): The frame from which to extract crops.
        detections (sv.Detections): Detected objects with bounding boxes.

    Returns:
        List[np.ndarray]: List of cropped images.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy], [tracker_id for tracker_id in detections.tracker_id]


# BoT-SORT tracking is now handled natively by YOLO model.track()
# No need for external tracker initialization or ID mapping functions


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.array,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Resolve the team IDs for detected goalkeepers based on the proximity to team
    centroids.

    Args:
        players (sv.Detections): Detections of all players.
        players_team_id (np.array): Array containing team IDs of detected players.
        goalkeepers (sv.Detections): Detections of goalkeepers.

    Returns:
        np.ndarray: Array containing team IDs for the detected goalkeepers.

    This function calculates the centroids of the two teams based on the positions of
    the players. Then, it assigns each goalkeeper to the nearest team's centroid by
    calculating the distance between each goalkeeper and the centroids of the two teams.
    """
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)

def safe_train_team_classifier(crops: List[np.ndarray], device: str):
    """Train TeamClassifier but fall back to a dummy classifier when training fails or no crops.

    Returns an object with a predict(crops) method that returns a list/array of team ids.
    """
    class DummyClassifier:
        def predict(self, crops_inner):
            # return zeros for each crop
            return np.zeros(len(crops_inner), dtype=int)

    if crops is None or len(crops) == 0:
        return DummyClassifier()

    try:
        tc = TeamClassifier(device=device)
        tc.fit(crops)
        return tc
    except Exception as e:
        print('TeamClassifier training failed, falling back to DummyClassifier:', e)
        return DummyClassifier()


def render_radar(
    detections: sv.Detections,
    balls: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray
) -> np.ndarray:

    # ⚠️ PROTEÇÃO: Verificar se keypoints não está vazio
    if keypoints is None or len(keypoints.xy) == 0 or len(keypoints.xy[0]) == 0:
        # Sem keypoints - retornar radar vazio
        return np.zeros((CONFIG.width, CONFIG.length, 3), dtype=np.uint8)
    
    # Filter keypoints: use only those with HIGH confidence (actually detected by model)
    # This prevents the model from "searching" for invisible keypoints
    if keypoints.confidence is not None and len(keypoints.confidence) > 0:
        # Use confidence threshold: only keypoints with conf > 0.5 are truly visible
        mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1) & (keypoints.confidence[0] > 0.5)
    else:
        # Fallback: just check coordinates are valid (old behavior)
        mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    
    # print('mask', mask)
    
    # ⚠️ PROTEÇÃO: Verificar se há keypoints suficientes após filtro
    if np.sum(mask) < 4:
        # Menos de 4 keypoints válidos - não dá para fazer transformação
        # Retornar radar vazio
        return np.zeros((CONFIG.width, CONFIG.length, 3), dtype=np.uint8)
    
    try:
        transformer = ViewTransformer(
            source=keypoints.xy[0][mask].astype(np.float32),
            target=np.array(CONFIG.vertices)[mask].astype(np.float32)
        )
        xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        xyBalls = balls.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        transformed_xy = transformer.transform_points(points=xy)
        transformed_xyBalls = transformer.transform_points(points=xyBalls)

        transformed_xy_0_100 = [[item[0] / CONFIG.length * 100, item[1] / CONFIG.width * 100] for item in transformed_xy]
        # print('transformed_xy', transformed_xy)

        # print('transformed_xy_0_100', transformed_xy_0_100)
        # print('------')

        radar = draw_pitch(config=CONFIG)
        radar = draw_points_on_pitch(config=CONFIG, xy=transformed_xy[color_lookup == 0], face_color=sv.Color.from_hex(COLORS[0]), radius=20, pitch=radar)
        radar = draw_points_on_pitch(config=CONFIG, xy=transformed_xy[color_lookup == 1], face_color=sv.Color.from_hex(COLORS[1]), radius=20, pitch=radar)
        radar = draw_points_on_pitch(config=CONFIG, xy=transformed_xy[color_lookup == 2], face_color=sv.Color.from_hex(COLORS[2]), radius=20, pitch=radar)
        radar = draw_points_on_pitch(config=CONFIG, xy=transformed_xy[color_lookup == 3], face_color=sv.Color.from_hex(COLORS[3]), radius=20, pitch=radar)
        
        radar = draw_points_on_pitch(config=CONFIG, xy=transformed_xyBalls, face_color=sv.Color.from_hex(COLORS[4]), radius=20, pitch=radar)
        return radar
    except Exception as e:
        print(e)
        return []


def run_pitch_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run pitch detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        annotated_frame = frame.copy()
        
        # Draw keypoints for each detection
        if len(keypoints) > 0:
            for detection_idx in range(len(keypoints)):
                # Get visible keypoints for this detection
                kp_xy = keypoints.xy[detection_idx]  # Shape: (32, 2)
                kp_conf = keypoints.confidence[detection_idx] if keypoints.confidence is not None else None  # Shape: (32,)
                
                # Draw each keypoint with different colors based on confidence
                for kp_idx, (x, y) in enumerate(kp_xy):
                    # Check if keypoint is visible (confidence > 0 or coordinates not 0,0)
                    is_visible = (kp_conf is None or kp_conf[kp_idx] > 0) and (x > 0 or y > 0)
                    
                    if is_visible:
                        # Color based on confidence:
                        # High confidence (>0.5) = GREEN (modelo tem certeza)
                        # Medium confidence (0.2-0.5) = YELLOW (modelo tem dúvidas)
                        # Low confidence (<0.2) = RED (falso positivo provável)
                        if kp_conf is not None:
                            conf = kp_conf[kp_idx]
                            if conf > 0.5:
                                color = (0, 255, 0)  # Verde: confiante
                            elif conf > 0.2:
                                color = (0, 255, 255)  # Amarelo: incerto
                            else:
                                color = (0, 0, 255)  # Vermelho: baixa confiança
                        else:
                            color = (0, 255, 0)  # Verde por padrão
                        
                        # Draw keypoint circle
                        cv2.circle(annotated_frame, (int(x), int(y)), 8, color, -1)
                        # Draw label with confidence
                        if kp_idx < len(CONFIG.labels):
                            label = CONFIG.labels[kp_idx]
                            if kp_conf is not None:
                                label += f" {kp_conf[kp_idx]:.2f}"
                            cv2.putText(annotated_frame, label, (int(x) + 10, int(y) - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        yield annotated_frame


def run_player_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1920, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)

        yield annotated_frame


def run_ball_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run ball detection with real-time interpolation using fixed-size buffer.
    
    This function implements a single-pass video processing approach with delayed output:
    1. Read frame N → Detect ball → Store in buffer
    2. When buffer is full: Interpolate gaps → Output oldest frame (N-30)
    3. After loop: Flush remaining frames from buffer
    
    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames with interpolated ball positions.
    """
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    
    # 🎯 Real-time interpolator with fixed buffer (30 frames = ~1.2s @ 25fps)
    interpolator = RealTimeBallInterpolator(buffer_size=30)
    annotator = InterpolatedBallAnnotator(radius=8, trail_length=15)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=1024, conf=0.30, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=callback,
        slice_wh=(640, 640),
    )

    # 📊 MAIN LOOP: Process frames and fill buffer
    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=0.1)
        balls = detections[detections.class_id == BALL_CLASS_ID]
        
        # 🎯 FILTER 1: Remove detections with very small area (noise/feet)
        if len(balls) > 0:
            areas = (balls.xyxy[:, 2] - balls.xyxy[:, 0]) * (balls.xyxy[:, 3] - balls.xyxy[:, 1])
            min_area = 100  # Minimum ball area in 640x640 slice
            balls = balls[areas >= min_area]
        
        # 🎯 FILTER 2: Keep only HIGHEST confidence detection (only 1 ball exists)
        if len(balls) > 0:
            best_idx = np.argmax(balls.confidence)
            balls = balls[best_idx:best_idx+1]
        
        # ⏱️ Add frame to buffer (with interpolation logic)
        buffered_frame = interpolator.add_frame(frame, balls)
        
        # 📤 OUTPUT: Only yield when buffer is full (delayed output with interpolation)
        if buffered_frame is not None:
            annotated_frame = annotator.annotate(buffered_frame.frame, buffered_frame)
            yield annotated_frame
    
    # 🔚 FLUSH: Process remaining frames in buffer
    for buffered_frame in interpolator.flush_buffer():
        annotated_frame = annotator.annotate(buffered_frame.frame, buffered_frame)
        yield annotated_frame


def run_player_tracking(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player tracking using YOLO's native BoT-SORT tracker with GMC.
    
    BoT-SORT provides superior tracking for football videos through:
    - Global Motion Compensation (GMC): Handles camera panning/zooming
    - High track buffer (60 frames): Maintains IDs through occlusions
    - ReID features: Re-identifies players after long disappearances

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames with tracked players.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    
    for frame in frame_generator:
        # 🎯 YOLO native tracking with BoT-SORT + GMC
        results = player_detection_model.track(
            frame,
            imgsz=1280,
            conf=0.1,  # Low threshold for tracker to catch everything
            persist=True,  # CRITICAL: Maintain IDs across frames
            tracker=BOTSORT_CONFIG_PATH,  # Custom BoT-SORT config with GMC
            verbose=False
        )
        
        # Extract detections with tracker IDs
        detections = sv.Detections.from_ultralytics(results[0])
        
        # Filter out detections without tracker IDs (if any)
        if detections.tracker_id is not None:
            valid_mask = detections.tracker_id != -1  # -1 means no ID assigned
            detections = detections[valid_mask]
        
        labels = [str(int(tid)) for tid in detections.tracker_id] if detections.tracker_id is not None else []

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(annotated_frame, detections, labels=labels)

        yield annotated_frame


def run_team_classification(source_video_path: str, device: str, debug: bool = False, debug_output_dir: str = "debug_team_output") -> Iterator[np.ndarray]:
    """
    Run team classification on a video and yield annotated frames with team colors.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').
        debug (bool): Enable debug mode to save visualization images.
        debug_output_dir (str): Directory to save debug images.

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE)

    # 🎯 Initialize new TeamClassifier (with optional debug mode)
    team_classifier = TeamClassifier(debug=debug, debug_output_dir=debug_output_dir)
    print('✅ TeamClassifier initialized with voting system')
    if debug:
        print(f'🔬 DEBUG MODE: Images will be saved to {debug_output_dir}/')

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    
    for frame in frame_generator:
        # 🎯 YOLO native tracking with BoT-SORT + GMC
        results = player_detection_model.track(
            frame,
            imgsz=1280,
            conf=0.1,
            persist=True,
            tracker=BOTSORT_CONFIG_PATH,
            verbose=False
        )
        
        detections = sv.Detections.from_ultralytics(results[0])
        
        # Filter out detections without tracker IDs
        if detections.tracker_id is not None:
            valid_mask = detections.tracker_id != -1
            detections = detections[valid_mask]

        # 🎯 Assign teams using new system
        players = detections[detections.class_id == PLAYER_CLASS_ID]
        players = team_classifier.assign_team(frame, players)
        
        if hasattr(players, 'team_id') and players.team_id is not None:
            players_team_id = players.team_id
        else:
            players_team_id = np.array([], dtype=int)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])

        # Team1 = 0; Team2 = 1; Goalkeeper = 2; Referee = 3
        # Convert -1 (unclassified) to 4 (neutral color) to avoid negative index error
        safe_players_team_id = np.where(players_team_id == -1, 4, players_team_id)
        color_lookup = np.array(
                safe_players_team_id.tolist() +
                # goalkeepers_team_id.tolist() +
                [2] * len(goalkeepers) +  # goalkeeper color = 2
                [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(annotated_frame, detections, labels, custom_color_lookup=color_lookup)

        yield annotated_frame


def run_radar(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    print('run_radar')
    performanceMeter()

    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    print('video_info', video_info)

    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE)
    performanceMeter('initializing models')

    # 🎯 Initialize new TeamClassifier (no pre-training needed)
    team_classifier = TeamClassifier()  # Uses internal HISTORY_LENGTH=30
    print('✅ TeamClassifier initialized with voting system')
    
    performanceMeter('initializing team classifier')

    ## store all teamIds for each tracker Id
    # frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    # tracker = sv.ByteTrack(minimum_consecutive_frames=3)

    # outsideTrackerIds_Team = {}
    # tempDetections = []
    # tempTeamIds = []
    # for frame in frame_generator:
    #     tempResult = player_detection_model(frame, imgsz=1280, verbose=False)[0]
    #     _tempDetectionsOriginal = sv.Detections.from_ultralytics(tempResult)
    #     _tempDetections = tracker.update_with_detections(_tempDetectionsOriginal)
    #     _tempDetections = _tempDetections[_tempDetections.class_id == PLAYER_CLASS_ID]
    #     tempDetections.append(_tempDetections)
    #     tempCrops = get_crops(frame, _tempDetections)
    #     _tempTeamIds = team_classifier.predict(tempCrops)
    #     tempTeamIds.append(_tempTeamIds)

    # for i in range(len(tempDetections)):
    #     for j in range(len(tempDetections[i].tracker_id)):
    #         if str(tempDetections[i].tracker_id[j]) not in outsideTrackerIds_Team:
    #             outsideTrackerIds_Team[str(tempDetections[i].tracker_id[j])] = []
    #         outsideTrackerIds_Team[str(tempDetections[i].tracker_id[j])].append(int(tempTeamIds[i][j]))
    # performanceMeter('classifying all crops to teamIds')


    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    
    # 🎯 Ball detector with real-time interpolation (usando InferenceSlicer como no modo BALL_DETECTION)
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    ball_interpolator = RealTimeBallInterpolator(buffer_size=30)  # 30 frames = ~1.2s delay @ 25fps
    ball_annotator = InterpolatedBallAnnotator(radius=8, trail_length=15)
    
    # 🎯 InferenceSlicer para detetar bolas pequenas (igual ao modo BALL_DETECTION)
    def ball_slicer_callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=1024, conf=0.30, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)
    
    ball_slicer = sv.InferenceSlicer(
        callback=ball_slicer_callback,
        slice_wh=(640, 640),
    )

    last_keypoints = None
    last_detections = None
    
    frame_counter = 0
    
    # 🎯 Buffer de dados para sincronização (players, keypoints, etc)
    # Como a bola tem delay de 30 frames, precisamos armazenar os outros dados também
    sync_buffer = deque(maxlen=30)

    for frame in frame_generator:
        # print('        new frame')
        frameStartT = time.perf_counter()
        performanceMeter()
        frame_counter += 1

        result = pitch_detection_model(frame, verbose=False)[0]
        
        # Try to extract keypoints, create empty if model doesn't support it
        try:
            # Check if result has keypoints attribute and it's not None
            if hasattr(result, 'keypoints') and result.keypoints is not None and hasattr(result.keypoints, 'xy'):
                keypoints = sv.KeyPoints.from_ultralytics(result)
            else:
                # Create empty keypoints if model doesn't support pose detection
                keypoints = sv.KeyPoints(xy=np.empty((1, 0, 2)))
        except (AttributeError, TypeError, Exception) as e:
            # Fallback: create empty keypoints structure
            keypoints = sv.KeyPoints(xy=np.empty((1, 0, 2)))
        
        # 🎯 YOLO native tracking with BoT-SORT + GMC for players
        # imgsz=1280 para manter consistência com TEAM_CLASSIFICATION (tracker IDs estáveis)
        results = player_detection_model.track(
            frame,
            imgsz=1280,
            conf=0.1,  # Low threshold for tracker
            persist=True,  # Maintain IDs
            tracker=BOTSORT_CONFIG_PATH,
            verbose=False
        )
        
        detections = sv.Detections.from_ultralytics(results[0])
        
        # Filter out detections without tracker IDs
        if detections.tracker_id is not None:
            valid_mask = detections.tracker_id != -1
            detections = detections[valid_mask]

        if last_detections is None:  # first frame
            last_detections = detections
        else:
            if len(detections) == 0:
                detections = last_detections
            else:
                last_detections = detections

        if last_keypoints is None:  # first frame
            last_keypoints = keypoints
        else:
            if len(keypoints.xy) == 0:
                # print('KEYPOINTS LENGHT == 0    !!!!!!!!!!!!!!!!!!!!!!')
                keypoints = last_keypoints
            else:
                if np.count_nonzero(keypoints.xy[0]) < 8:   ## needs at least 4 points of XY pairs
                    # print('KEYPOINTS LESS THAN 8    !!!!!!!!!!!!!!!!!!!!!!')
                    keypoints = last_keypoints
                else:
                    # print('KEYPOINTS MORE THAN 8    !!!!!!!!!!!!!!!!!!!!!!')
                    last_keypoints = keypoints

        # print('keypoints', keypoints)


        # detections = smoother.update_with_detections(detections)
        performanceMeter('getting Player and Pitch detections')
        
        # 🎯 NEW: Assign teams APENAS a Players (class_id == 2)
        players = detections[detections.class_id == PLAYER_CLASS_ID]
        players = team_classifier.assign_team(frame, players)
        
        # Extract team_id from players (deve ser APENAS 0 ou 1, nunca -1)
        if hasattr(players, 'team_id') and players.team_id is not None:
            players_team_id = players.team_id
            # Force: Se algum player ainda tiver -1, converte para 0 (fallback)
            players_team_id = np.where(players_team_id == -1, 0, players_team_id)
        else:
            # Fallback: no teams detected
            players_team_id = np.array([], dtype=int)
        
        # 🔍 DEBUG: Log team assignments
        if frame_counter % 50 == 0:
            print(f"\n📊 Frame {frame_counter} - Team Classification:")
            print(f"   Players: {len(players)}")
            if players.tracker_id is not None and len(players.tracker_id) > 0:
                print(f"   Tracker IDs (first 5): {players.tracker_id[:min(5, len(players.tracker_id))]}")
            if len(players_team_id) > 0:
                print(f"   Team IDs (first 5): {players_team_id[:min(5, len(players_team_id))]}")
                team0_count = np.sum(players_team_id == 0)
                team1_count = np.sum(players_team_id == 1)
                print(f"   🔴 Team 0 (RED): {team0_count} players")
                print(f"   🔵 Team 1 (BLUE): {team1_count} players")

        performanceMeter('assigning player teams with voting')

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        # goalkeepers_team_id = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)
        referees = detections[detections.class_id == REFEREE_CLASS_ID]
        
        # 🎯 Ball detection usando InferenceSlicer (igual ao modo BALL_DETECTION)
        # Divide a imagem em slices 640x640 para detetar bolas pequenas
        balls = ball_slicer(frame).with_nms(threshold=0.1)
        balls = balls[balls.class_id == BALL_CLASS_ID] if len(balls) > 0 else balls
        
        # 🎯 FILTER 1: Remove detections with very small area (noise/feet)
        if len(balls) > 0:
            areas = (balls.xyxy[:, 2] - balls.xyxy[:, 0]) * (balls.xyxy[:, 3] - balls.xyxy[:, 1])
            min_area = 100  # Bola mínima: ~10x10 pixels (igual ao modo BALL_DETECTION)
            balls = balls[areas >= min_area]
        
        # 🎯 FILTER 2: Keep only the HIGHEST confidence detection (only 1 ball exists)
        if len(balls) > 0:
            best_idx = np.argmax(balls.confidence)
            balls = balls[best_idx:best_idx+1]  # Keep only the best one
        performanceMeter('ball detection with slicer')

        # ⏱️ BUFFER STRATEGY: Store frame + all data, process with delay
        detections_merged = sv.Detections.merge([players, goalkeepers, referees])
        
        # Merge team_id arrays: players (0/1) + goalkeepers (-1) + referees (-1)
        merged_team_ids = np.concatenate([
            players_team_id,
            np.full(len(goalkeepers), -1, dtype=int),
            np.full(len(referees), -1, dtype=int)
        ])
        
        # Color lookup based on team_id:
        # Team 0 → 0 (Red)
        # Team 1 → 1 (Blue)
        # GK (team_id=-1, class_id=1) → 2 (Orange)
        # Referee (team_id=-1, class_id=3) → 3 (Yellow)
        color_lookup = []
        for i in range(len(detections_merged)):
            team_id = merged_team_ids[i]
            if team_id != -1:
                # Has valid team (0 or 1) → Use team color
                color_lookup.append(team_id)
            else:
                # Neutral (-1) → Determine by class_id
                class_id = detections_merged.class_id[i]
                if class_id == 1:  # Goalkeeper
                    color_lookup.append(2)
                elif class_id == 3:  # Referee
                    color_lookup.append(3)
                else:
                    color_lookup.append(0)  # Fallback
        
        color_lookup = np.array(color_lookup)
        
        # 🏷️ NEW LABEL LOGIC: Hierarchy based on team_id
        # 1. If team_id != -1 → Show team name (ignore class_id completely)
        # 2. If team_id == -1 → Check class_id (GK or REF)
        labels = []
        if detections_merged.tracker_id is not None:
            for i, tid in enumerate(detections_merged.tracker_id):
                if tid is not None:
                    team_id = merged_team_ids[i]
                    
                    # PRIORITY 1: Has team assignment (0 or 1)
                    if team_id != -1:
                        team_name = "T0" if team_id == 0 else "T1"
                        labels.append(f"{tid} ({team_name})")
                    
                    # PRIORITY 2: No team (-1) → Use class_id
                    else:
                        class_id = detections_merged.class_id[i]
                        if class_id == 1:  # Goalkeeper
                            labels.append(f"{tid} (GK)")
                        elif class_id == 3:  # Referee
                            labels.append(f"{tid} (REF)")
                        else:
                            labels.append(f"{tid}")
                else:
                    labels.append("?")
        else:
            labels = ["?"] * len(detections_merged)
        
        # Store current frame data in sync buffer
        sync_buffer.append({
            'frame': frame.copy(),
            'detections': detections_merged,
            'color_lookup': color_lookup,
            'labels': labels,
            'keypoints': keypoints,
            'frame_counter': frame_counter
        })
        
        # 🎯 Add ball detection to interpolator (triggers interpolation)
        buffered_ball = ball_interpolator.add_frame(frame, balls)
        
        # 📤 OUTPUT: Only when buffer is full (delayed render with interpolated ball)
        if buffered_ball is not None and len(sync_buffer) == 30:
            # Get oldest frame data (synchronized with interpolated ball)
            oldest_data = sync_buffer[0]  # Don't pop yet, deque handles it automatically
            
            # Render frame with interpolated ball
            annotated_frame = oldest_data['frame'].copy()
            annotated_frame = ELLIPSE_ANNOTATOR.annotate(
                annotated_frame, 
                oldest_data['detections'], 
                custom_color_lookup=oldest_data['color_lookup']
            )
            annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
                annotated_frame, 
                oldest_data['detections'], 
                oldest_data['labels'], 
                custom_color_lookup=oldest_data['color_lookup']
            )
            
            # Annotate with interpolated ball
            annotated_frame = ball_annotator.annotate(annotated_frame, buffered_ball)
            
            # 📊 Display ball confidence in top white area
            if buffered_ball.detection is not None and buffered_ball.confidence > 0:
                ball_conf = buffered_ball.confidence
                conf_text = f"Ball Confidence: {ball_conf:.2%}"
                
                # Get frame dimensions
                h, w = annotated_frame.shape[:2]
                
                # Text properties
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                
                # Get text size for background
                (text_w, text_h), baseline = cv2.getTextSize(conf_text, font, font_scale, thickness)
                
                # Position: top-right corner with padding
                padding = 15
                text_x = w - text_w - padding
                text_y = padding + text_h
                
                # Draw semi-transparent background
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, 
                            (text_x - 10, text_y - text_h - 5), 
                            (text_x + text_w + 10, text_y + baseline + 5), 
                            (255, 255, 255), 
                            -1)
                annotated_frame = cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0)
                
                # Draw text (color based on confidence)
                color = (0, 255, 0) if ball_conf > 0.5 else (0, 165, 255)  # Green if >50%, Orange otherwise
                cv2.putText(annotated_frame, conf_text, 
                          (text_x, text_y), 
                          font, font_scale, color, thickness, cv2.LINE_AA)
            
            performanceMeter('annotating frame with Elipse, and Ball')

            # Render radar with interpolated ball
            startT = time.perf_counter()
            RADAR_SCALE = 4
            h, w, _ = annotated_frame.shape
            
            # Convert buffered ball to sv.Detections for radar rendering
            ball_for_radar = ball_interpolator.get_detection_as_sv_detections(buffered_ball)
            radar = render_radar(
                oldest_data['detections'], 
                ball_for_radar, 
                oldest_data['keypoints'], 
                oldest_data['color_lookup']
            )
            
            if len(radar) > 0:
                radar = sv.resize_image(radar, (w // RADAR_SCALE, h // RADAR_SCALE))
                radar_h, radar_w, _ = radar.shape
                rect = sv.Rect(
                    x=w // 2 - radar_w // 2,
                    y=h - radar_h,
                    width=radar_w,
                    height=radar_h
                )
                annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)

            performanceMeter('rendering Radar in frame')
            
            # Print detailed timing every 30 frames
            total_frame_time = time.perf_counter() - frameStartT
            if oldest_data['frame_counter'] % 30 == 0:
                print(f'\n=== Frame {oldest_data["frame_counter"]} Performance (delayed output) ===')
                print(f'Total frame time: {total_frame_time*1000:.1f}ms ({1/total_frame_time:.1f} FPS)')
                for desc, t in performance_times.items():
                    print(f'  {desc}: {t*1000:.1f}ms')
            
            yield annotated_frame
    
    # 🔚 FLUSH: Process remaining frames in buffer after loop ends
    print(f"\n🔚 Flushing {len(ball_interpolator.buffer)} remaining frames from buffer...")
    remaining_balls = ball_interpolator.flush_buffer()
    
    for i, buffered_ball in enumerate(remaining_balls):
        if i < len(sync_buffer):
            oldest_data = sync_buffer[i]
            
            annotated_frame = oldest_data['frame'].copy()
            annotated_frame = ELLIPSE_ANNOTATOR.annotate(
                annotated_frame, 
                oldest_data['detections'], 
                custom_color_lookup=oldest_data['color_lookup']
            )
            annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
                annotated_frame, 
                oldest_data['detections'], 
                oldest_data['labels'], 
                custom_color_lookup=oldest_data['color_lookup']
            )
            annotated_frame = ball_annotator.annotate(annotated_frame, buffered_ball)
            
            # Render radar
            RADAR_SCALE = 4
            h, w, _ = annotated_frame.shape
            ball_for_radar = ball_interpolator.get_detection_as_sv_detections(buffered_ball)
            radar = render_radar(
                oldest_data['detections'], 
                ball_for_radar, 
                oldest_data['keypoints'], 
                oldest_data['color_lookup']
            )
            
            if len(radar) > 0:
                radar = sv.resize_image(radar, (w // RADAR_SCALE, h // RADAR_SCALE))
                radar_h, radar_w, _ = radar.shape
                rect = sv.Rect(
                    x=w // 2 - radar_w // 2,
                    y=h - radar_h,
                    width=radar_w,
                    height=radar_h
                )
                annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)
            
            yield annotated_frame


def main(source_video_path: str, target_video_path: str, device: str, mode: Mode, debug: bool = False, debug_output_dir: str = "debug_team_output") -> None:
    print('main')

    if mode == Mode.PITCH_DETECTION:
        frame_generator = run_pitch_detection(source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_DETECTION:
        frame_generator = run_player_detection(source_video_path=source_video_path, device=device)
    elif mode == Mode.BALL_DETECTION:
        frame_generator = run_ball_detection(source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_TRACKING:
        frame_generator = run_player_tracking(source_video_path=source_video_path, device=device)
    elif mode == Mode.TEAM_CLASSIFICATION:
        frame_generator = run_team_classification(source_video_path=source_video_path, device=device, debug=debug, debug_output_dir=debug_output_dir)
    elif mode == Mode.RADAR:
        frame_generator = run_radar(source_video_path=source_video_path, device=device)
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    video_info = sv.VideoInfo.from_video_path(source_video_path)

    if target_video_path is not None:
        print(f'Processing and saving video: {source_video_path} -> {target_video_path}')
        frame_count = 0
        with sv.VideoSink(target_video_path, video_info) as sink:
            for frame in frame_generator:
                sink.write_frame(frame)
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f'Processed {frame_count}/{video_info.total_frames} frames', end='\r')
                
                # Show frame in window
                try:
                    cv2.imshow("Player Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print('\nStopped by user (pressed q)')
                        break
                except:
                    pass  # Ignore if display not available
        
        try:
            cv2.destroyAllWindows()
        except:
            pass
        print(f'\nDone! Output saved to: {target_video_path}')
    else:
        print(f'Processing video in real-time: {source_video_path}')
        print('Processing... (close the window or Ctrl+C to stop)')
        
        if MATPLOTLIB_AVAILABLE:
            # Use matplotlib for display
            fig, ax = plt.subplots(figsize=(12, 8))
            plt.ion()  # Interactive mode
            frame_count = 0
            
            # Calculate target frame time based on video FPS
            target_frame_time = 1.0 / video_info.fps
            print(f'Target FPS: {video_info.fps:.2f} (frame time: {target_frame_time*1000:.1f}ms)')
            
            # FPS measurement
            fps_buffer = deque(maxlen=30)  # Track last 30 frames for accurate FPS
            frame_start_time = time.time()
            
            for frame in frame_generator:
                frame_count += 1
                
                # Convert BGR to RGB for matplotlib
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                ax.clear()
                
                # Calculate actual FPS
                current_time = time.time()
                frame_time = current_time - frame_start_time
                fps_buffer.append(frame_time)
                actual_fps = 1.0 / np.mean(fps_buffer) if len(fps_buffer) > 0 else 0
                
                ax.imshow(frame_rgb)
                ax.set_title(f'Player Detection - Frame {frame_count}/{video_info.total_frames} | Target: {video_info.fps:.1f} FPS | Actual: {actual_fps:.1f} FPS')
                ax.axis('off')
                
                # Control playback speed - only slow down if we're processing faster than target
                # If processing is slower, display immediately
                if frame_time < target_frame_time:
                    sleep_time = target_frame_time - frame_time
                    plt.pause(sleep_time)
                else:
                    plt.pause(0.00001)  # Minimal pause to update display
                
                frame_start_time = time.time()
                
                if not plt.fignum_exists(fig.number):
                    print(f'\nWindow closed by user after {frame_count} frames')
                    break
            
            plt.close()
            print(f'Processed {frame_count}/{video_info.total_frames} frames total')
            if len(fps_buffer) > 0:
                print(f'Average FPS: {1.0 / np.mean(fps_buffer):.2f}')
        else:
            # Fallback: just process without display and print progress
            frame_count = 0
            for frame in frame_generator:
                frame_count += 1
                if frame_count % 10 == 0:
                    print(f'Processed {frame_count}/{video_info.total_frames} frames', end='\r')
            print(f'\nProcessed {frame_count}/{video_info.total_frames} frames total (no display available)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Football Player Detection with BoT-SORT Tracking (GMC enabled)')
    parser.add_argument('--source_video_path', type=str, required=True, help='Path to input video or directory')
    parser.add_argument('--target_video_path', type=str, required=False, help='Path to save output video (optional)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to run inference on')
    parser.add_argument('--mode', type=Mode, default=Mode.RADAR, help='Processing mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (saves visualization images)')
    parser.add_argument('--debug_output_dir', type=str, default='debug_team_output', help='Directory for debug images')
    args = parser.parse_args()

    print('🎯 Football Analysis System - BoT-SORT with GMC')
    print('args', args)

    if os.path.isdir(args.source_video_path):
        # if args.target_video_path is None:
        #     args.target_video_path = os.path.join(args.source_video_path, 'out')

        if args.target_video_path is not None:
            if not os.path.exists(args.target_video_path):
                os.mkdir(args.target_video_path)

        # if not os.path.isdir(args.target_video_path):
        #     print('target is not dir')
        #     exit()

        print('source is dir', args.source_video_path)
        files = [f for f in os.listdir(args.source_video_path) if os.path.isfile(os.path.join(args.source_video_path, f))]
        print('\n'.join(files))
        for file in files:
            sourcePath = os.path.join(args.source_video_path, file)
            targetPath = None
            if args.target_video_path is not None:
                targetPath = os.path.join(args.target_video_path, file)
            print('\nProcessing file', sourcePath, ' --> ', targetPath)
            try:
                main(
                    source_video_path=sourcePath,
                    target_video_path=targetPath,
                    device=args.device,
                    mode=args.mode,
                    debug=args.debug,
                    debug_output_dir=args.debug_output_dir
                )
            except Exception as err:
                print('Error', err)
    else:
        # if args.target_video_path is None:
        #     print('target is not defined')
        #     exit()

        main(
            source_video_path=args.source_video_path,
            target_video_path=args.target_video_path,
            device=args.device,
            mode=args.mode,
            debug=args.debug,
            debug_output_dir=args.debug_output_dir
        )
        
