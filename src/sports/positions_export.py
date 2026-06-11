"""Exporta posicoes por frame (jogadores + bola) em coordenadas reais do campo.

Helpers puros (sem deps pesadas alem de numpy/supervision), no estilo de
stats_export.py, para alimentar o visualizador 3D do frontend.
"""
import json
import os
from collections import defaultdict, deque
from typing import Optional

import numpy as np
import supervision as sv

from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

_CONFIG = SoccerPitchConfiguration()

# Escala: CONFIG usa cm do campo (12000×7000) → metros reais (105×68)
_M_PER_CM_X = 105.0 / _CONFIG.length
_M_PER_CM_Y = 68.0 / _CONFIG.width

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

# Margem de tolerância (m) fora do campo; além disto a homografia é considerada má
_OUT_OF_PITCH_MARGIN_M = 3.0


class PositionsRecorder:
    """Acumula posicoes por frame em metros, suavizadas por tracker_id."""

    def __init__(self, fps: float, smooth_window: int = 5):
        self.fps = float(fps)
        self._pos_history: dict = defaultdict(lambda: deque(maxlen=smooth_window))
        # frame_index → {"players": [...], "ball": {...} | None}
        self._frames: dict = {}

    def update(
        self,
        frame_index: int,
        detections: sv.Detections,
        keypoints: sv.KeyPoints,
        team_ids: np.ndarray,
    ) -> None:
        """Regista as posicoes dos jogadores de um frame (se houver homografia)."""
        if detections.tracker_id is None or len(detections) == 0:
            return

        transformer = _build_transformer(keypoints)
        if transformer is None:
            return

        xy_pixels = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        try:
            xy_pitch_cm = transformer.transform_points(points=xy_pixels)
        except Exception:
            return

        players = []
        for i, tracker_id in enumerate(detections.tracker_id):
            if tracker_id is None or int(tracker_id) == -1:
                continue
            tid = int(tracker_id)

            # Descartar projeções fora do campo (homografia instável) antes de
            # entrarem na janela de suavização
            raw_x_m = float(xy_pitch_cm[i][0]) * _M_PER_CM_X
            raw_y_m = float(xy_pitch_cm[i][1]) * _M_PER_CM_Y
            if not _within_pitch(raw_x_m, raw_y_m):
                continue

            self._pos_history[tid].append(xy_pitch_cm[i])
            smoothed_cm = np.mean(self._pos_history[tid], axis=0)

            x_m = float(smoothed_cm[0]) * _M_PER_CM_X
            y_m = float(smoothed_cm[1]) * _M_PER_CM_Y

            team = int(team_ids[i]) if i < len(team_ids) else -1
            class_id = int(detections.class_id[i]) if detections.class_id is not None else -1

            players.append({
                "id": tid,
                "t": team,
                "c": class_id,
                "x": round(x_m, 2),
                "y": round(y_m, 2),
            })

        if players:
            entry = self._frames.setdefault(int(frame_index), {"players": [], "ball": None})
            entry["players"] = players

    def update_ball(
        self,
        frame_index: int,
        ball_xy_pixels: Optional[np.ndarray],
        keypoints: sv.KeyPoints,
    ) -> None:
        """Regista a posicao (interpolada) da bola para um frame ja registado."""
        if ball_xy_pixels is None:
            return

        transformer = _build_transformer(keypoints)
        if transformer is None:
            return

        ball_xy = np.asarray(ball_xy_pixels, dtype=np.float32).reshape(1, 2)
        try:
            ball_cm = transformer.transform_points(points=ball_xy)[0]
        except Exception:
            return

        ball_x_m = float(ball_cm[0]) * _M_PER_CM_X
        ball_y_m = float(ball_cm[1]) * _M_PER_CM_Y
        if not _within_pitch(ball_x_m, ball_y_m):
            return

        entry = self._frames.setdefault(int(frame_index), {"players": [], "ball": None})
        entry["ball"] = {
            "x": round(ball_x_m, 2),
            "y": round(ball_y_m, 2),
        }

    def to_payload(self) -> dict:
        frames = []
        for frame_index in sorted(self._frames.keys()):
            entry = self._frames[frame_index]
            if not entry["players"] and entry["ball"] is None:
                continue
            frames.append({
                "i": frame_index,
                "players": entry["players"],
                "ball": entry["ball"],
            })
        return {
            "version": 1,
            "fps": self.fps,
            "pitch": {"length_m": PITCH_LENGTH_M, "width_m": PITCH_WIDTH_M},
            "frame_count": frames[-1]["i"] if frames else 0,
            "frames": frames,
        }


def write_positions(stats_output_dir, recorder: PositionsRecorder):
    """Grava positions.json. Devolve o caminho do JSON, ou None se sem dir/dados."""
    if not stats_output_dir:
        return None

    payload = recorder.to_payload()
    if not payload["frames"]:
        return None

    os.makedirs(stats_output_dir, exist_ok=True)
    path = os.path.join(stats_output_dir, "positions.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, separators=(",", ":"))
    return path


def _within_pitch(x_m: float, y_m: float) -> bool:
    return (
        -_OUT_OF_PITCH_MARGIN_M <= x_m <= PITCH_LENGTH_M + _OUT_OF_PITCH_MARGIN_M
        and -_OUT_OF_PITCH_MARGIN_M <= y_m <= PITCH_WIDTH_M + _OUT_OF_PITCH_MARGIN_M
    )


def _build_transformer(keypoints: sv.KeyPoints) -> Optional[ViewTransformer]:
    """Mesma logica do DistanceTracker/render_radar para obter o ViewTransformer."""
    if keypoints is None or len(keypoints.xy) == 0 or len(keypoints.xy[0]) == 0:
        return None

    if keypoints.confidence is not None and len(keypoints.confidence) > 0:
        mask = (
            (keypoints.xy[0][:, 0] > 1)
            & (keypoints.xy[0][:, 1] > 1)
            & (keypoints.confidence[0] > 0.5)
        )
    else:
        mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)

    if np.sum(mask) < 4:
        return None

    try:
        return ViewTransformer(
            source=keypoints.xy[0][mask].astype(np.float32),
            target=np.array(_CONFIG.vertices)[mask].astype(np.float32),
        )
    except Exception:
        return None
