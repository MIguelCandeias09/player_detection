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
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration


# Norfair imports (optional)
try:
    from norfair import Detection as NorfairDetection, Tracker as NorfairTracker
    from norfair import distances as norfair_distances
except Exception:
    NorfairDetection = None
    NorfairTracker = None
    norfair_distances = None
    print('Norfair not available - please pip install norfair to enable improved tracking')


PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Use the existing trained models for players and pitch
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection_mike.pt')
# PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/yolo12s.pt')
# PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection.pt')
# PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection-mike_1280.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection-mike_640_v11m.pt')  # Modelo anterior (mais estável)
# PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/pitch_y11x_keypoint_best.pt')  # YOLOv11x-pose 32-keypoint (65.8% mAP50) - PRECISA MELHORIAS
# Use the newly trained YOLOv12 ball detector weights (optimized version)
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/ball_y12s_optimized_best.pt')
# BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/ball_y12s_best.pt')  # previous version (recall 30%)
# BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-ball-detection.pt')   # this model is too slow

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 5
CONFIG = SoccerPitchConfiguration()

# COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700', '#8000FF']
#          Team 1     Team 2     Goalkeeper Referee
COLORS = ['#c7c7c7', '#1055e8', '#FF6347', '#FFD700', '#8000FF']
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


def init_norfair_tracker(distance_threshold: float = 80.0) -> object:
    """Initialize a Norfair tracker with a reasonable pixel distance threshold.

    Returns a Tracker instance or None if Norfair is not installed.
    """
    if NorfairTracker is None:
        return None
    
    # Use scalar euclidean distance (simple and works)
    def euclidean_distance(detection, tracked_object):
        """Calculate euclidean distance between a detection and tracked object."""
        return np.linalg.norm(detection.points - tracked_object.estimate)
    
    return NorfairTracker(
        distance_function=euclidean_distance, 
        distance_threshold=distance_threshold
    )


def norfair_update_and_get_ids(tracker: object, detections: sv.Detections) -> List[object]:
    """Update Norfair tracker with detections and return a list of tracker ids aligned with detections order.

    We use the bottom-center (anchor) of each detection as the single point to track.
    If Norfair is not available or there are no detections, returns a list of None with matching length.
    """
    if tracker is None or NorfairDetection is None:
        # return placeholder None ids
        return [None] * len(detections)

    centers = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    if centers is None or len(centers) == 0:
        return [None] * len(detections)

    # Create Norfair detections with data payload containing original index
    norfair_dets = [
        NorfairDetection(points=cent.reshape(1, 2), data={'index': i}) 
        for i, cent in enumerate(centers)
    ]
    
    # Update tracker
    tracked_objects = tracker.update(detections=norfair_dets)

    # Initialize IDs array with None
    ids = [None] * len(detections)
    
    # Map tracked objects back to detection indices using data payload
    for to in tracked_objects:
        if to.last_detection is not None and to.last_detection.data is not None:
            idx = to.last_detection.data.get('index')
            if idx is not None and idx < len(ids):
                ids[idx] = to.id

    return ids

    return ids


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

    # Filter keypoints: use only those with HIGH confidence (actually detected by model)
    # This prevents the model from "searching" for invisible keypoints
    if keypoints.confidence is not None and len(keypoints.confidence) > 0:
        # Use confidence threshold: only keypoints with conf > 0.5 are truly visible
        mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1) & (keypoints.confidence[0] > 0.5)
    else:
        # Fallback: just check coordinates are valid (old behavior)
        mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    
    # print('mask', mask)
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
    Run ball detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=callback,
        # overlap_filter_strategy=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=0.1)
        balls = detections[detections.class_id == BALL_CLASS_ID]
        detections = ball_tracker.update(balls)
        annotated_frame = frame.copy()
        annotated_frame = ball_annotator.annotate(annotated_frame, detections)

        yield annotated_frame


def run_player_tracking(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player tracking on a video and yield annotated frames with tracked players.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    if TRACKER_CHOICE == 'norfair':
        tracker = init_norfair_tracker()
    else:
        tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        # Update tracker: ByteTrack has update_with_detections; Norfair uses helper
        if TRACKER_CHOICE == 'norfair' and NorfairTracker is not None:
            # get ids aligned with detections
            ids = norfair_update_and_get_ids(tracker, detections)
            # attach tracker ids to detections object (fill with None where missing)
            detections.tracker_id = ids
        else:
            detections = tracker.update_with_detections(detections)

        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(annotated_frame, detections, labels=labels)

        yield annotated_frame


def run_team_classification(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run team classification on a video and yield annotated frames with team colors.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE)

    crops = []
    # i = 0
    # for frame in tqdm(frame_generator, desc='collecting crops'):
    print('collecting crops')
    for frame in frame_generator:
        # print('collecting', i)
        # i+= 1
        # if i > 1:
        #     print('breaking')
        #     break
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    print(f'Collected {len(crops)} crops for team classification')
    team_classifier = safe_train_team_classifier(crops, device=device)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    if TRACKER_CHOICE == 'norfair':
        tracker = init_norfair_tracker()
    else:
        tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        if TRACKER_CHOICE == 'norfair' and NorfairTracker is not None:
            ids = norfair_update_and_get_ids(tracker, detections)
            detections.tracker_id = ids
        else:
            detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)
        # normalize predictions: replace None with 0 and ensure integers
        if players_team_id is None:
            players_team_id = np.array([], dtype=int)
        else:
            players_team_id = np.array([0 if p is None else int(p) for p in players_team_id], dtype=int)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])

        # Team1 = 0; Team2 = 1; Goalkeeper = 2; Referee = 3
        color_lookup = np.array(
                players_team_id.tolist() +
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

    teamColorTrainingCrops = []
    print('collecting crops to train')
    # for frame in tqdm(frame_generator, desc='collecting crops'):
    MAX_TRAIN_DETECTIONS = 200  # Aumentado de 100 para 200 (mais amostras = melhor separação)
    for frame in frame_generator:
        if len(teamColorTrainingCrops) > MAX_TRAIN_DETECTIONS:
            print('reached', MAX_TRAIN_DETECTIONS, 'crops: stopping')
            break
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.class_id == PLAYER_CLASS_ID]
        detections = detections[detections.confidence > 0.75]  # Aumentado para melhor qualidade
        teamColorTrainingCrops+= get_crops(frame, detections)

    performanceMeter('collecting teamColorTrainingCrops from all frames')
    
    # Use safe training with fallback to dummy classifier if not enough crops
    print(f'Collected {len(teamColorTrainingCrops)} crops for team classification')
    team_classifier = safe_train_team_classifier(teamColorTrainingCrops, device=device)
    
    performanceMeter('training team recognition model')

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
    if TRACKER_CHOICE == 'norfair':
        tracker = init_norfair_tracker()
    else:
        tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    # smoother = sv.DetectionsSmoother(length=3)



    # ball detector
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    ball_tracker = BallTracker(buffer_size=50)  # Aumentado para mais histórico
    ball_annotator = BallAnnotator(radius=8, buffer_size=15)  # Aumentado para melhor visualização
    # end of ball detector

    insideTrackerIds_Team = {}

    last_keypoints = None
    last_detections = None
    
    frame_counter = 0

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
        
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        originalDetections = sv.Detections.from_ultralytics(result)
        
        # Update tracker based on choice
        if TRACKER_CHOICE == 'norfair' and NorfairTracker is not None and tracker is not None:
            ids = norfair_update_and_get_ids(tracker, originalDetections)
            detections = originalDetections
            detections.tracker_id = np.array(ids) if ids else np.array([])
        else:
            detections = tracker.update_with_detections(originalDetections)

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
        
        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)
        # normalize predictions: replace None with 0 and ensure integers
        if players_team_id is None:
            players_team_id = np.array([], dtype=int)
        else:
            players_team_id = np.array([0 if p is None else int(p) for p in players_team_id], dtype=int)

        # FIX players team ID - use temporal smoothing with tracker history
        if players.tracker_id is not None and len(players.tracker_id) > 0:
            for i in range(min(len(players.tracker_id), len(players_team_id))):
                if players.tracker_id[i] is not None:
                    tracker_id_str = str(players.tracker_id[i])
                    if tracker_id_str not in insideTrackerIds_Team:
                        insideTrackerIds_Team[tracker_id_str] = []
                    insideTrackerIds_Team[tracker_id_str].append(int(players_team_id[i]))

        # use teamId from the last 75 frames (3 sec) classifications of insideTrackerIds_Team
        smoothed_team_ids = []
        if players.tracker_id is not None and len(players.tracker_id) > 0:
            for tracker_id in players.tracker_id:
                if tracker_id is not None:
                    tracker_id_str = str(tracker_id)
                    if tracker_id_str in insideTrackerIds_Team and len(insideTrackerIds_Team[tracker_id_str]) > 0:
                        # Use mode (most common) instead of mean for more stable classification
                        recent_ids = insideTrackerIds_Team[tracker_id_str][-75:]
                        smoothed_id = max(set(recent_ids), key=recent_ids.count)
                        smoothed_team_ids.append(smoothed_id)
                    else:
                        smoothed_team_ids.append(0)
                else:
                    smoothed_team_ids.append(0)
        else:
            # No tracker IDs, use raw predictions
            smoothed_team_ids = players_team_id.tolist() if len(players_team_id) > 0 else []

        players_team_id = np.array(smoothed_team_ids, dtype=int) if len(smoothed_team_ids) > 0 else np.array([], dtype=int)
        # END OF FIX players team ID

        # print('players_team_id', players_team_id)

        performanceMeter('predicting player Teams')

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        # goalkeepers_team_id = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)
        referees = detections[detections.class_id == REFEREE_CLASS_ID]
        # use the dedicated ball detection model per frame (single-class ball model)
        # Model trained with imgsz=1280, using very low conf for better recall (especially for ball in air)
        # Using agnostic_nms to avoid suppressing multiple ball detections
        ball_result = ball_detection_model(
            frame, 
            imgsz=1280, 
            conf=0.05,          # Muito baixo para pegar bola no ar
            iou=0.3,            # IoU baixo para não suprimir detecções próximas
            agnostic_nms=True,  # NMS independente de classe
            max_det=10,         # Permite múltiplas detecções
            verbose=False
        )[0]
        balls = sv.Detections.from_ultralytics(ball_result)
        performanceMeter('filtering detections')

        ballAnnotations = ball_tracker.update(balls)
        performanceMeter('update Ball tracker')

        # print('players', players)
        # print('balls', balls)

        detections = sv.Detections.merge([players, goalkeepers, referees])
        performanceMeter('merging detections')

        # Team1 = 0; Team2 = 1; Goalkeeper = 2; Referee = 3
        color_lookup = np.array(
            players_team_id.tolist() +
            # goalkeepers_team_id.tolist() +
            [2] * len(goalkeepers) +  # goalkeeper color = 2
            [REFEREE_CLASS_ID] * len(referees)
            # [4] * len(balls)
        )
        # Create labels, handling None tracker_id gracefully
        labels = []
        if detections.tracker_id is not None:
            labels = [str(tid) if tid is not None else "?" for tid in detections.tracker_id]
        else:
            labels = ["?"] * len(detections)

        # print('detections', detections)
        performanceMeter('creating colors and labels from detections')

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(annotated_frame, detections, labels, custom_color_lookup=color_lookup)
        annotated_frame = ball_annotator.annotate(annotated_frame, ballAnnotations)
        performanceMeter('annotating frame with Elipse, and Ball')

        startT = time.perf_counter()
        RADAR_SCALE = 4
        h, w, _ = frame.shape
        radar = render_radar(detections, balls, keypoints, color_lookup)
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
        if frame_counter % 30 == 0:
            print(f'\n=== Frame {frame_counter} Performance ===')
            print(f'Total frame time: {total_frame_time*1000:.1f}ms ({1/total_frame_time:.1f} FPS)')
            for desc, t in performance_times.items():
                print(f'  {desc}: {t*1000:.1f}ms')
        
        yield annotated_frame


def main(source_video_path: str, target_video_path: str, device: str, mode: Mode) -> None:
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
        frame_generator = run_team_classification(source_video_path=source_video_path, device=device)
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
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source_video_path', type=str, required=True)
    parser.add_argument('--target_video_path', type=str, required=False)
    parser.add_argument('--device', type=str, default='cuda')   # cpu || cuda
    parser.add_argument('--mode', type=Mode, default=Mode.RADAR)
    parser.add_argument('--tracker', type=str, default='bytetrack', choices=['bytetrack', 'norfair'], help='Tracker backend to use')
    args = parser.parse_args()

    print('args', args)

    # global selection of tracker backend
    TRACKER_CHOICE = args.tracker

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
                    mode=args.mode
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
            mode=args.mode
        )
        
