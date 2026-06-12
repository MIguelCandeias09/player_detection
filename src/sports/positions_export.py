"""Exporta posicoes por frame (jogadores + bola) em coordenadas reais do campo.

Helpers puros (sem deps pesadas alem de numpy/supervision), no estilo de
stats_export.py, para alimentar o visualizador 3D do frontend.

A captura (update/update_ball) guarda apenas dados crus em pixels; a projecao
para o campo acontece em to_payload(). Isso permite interpolar os keypoints
entre detecoes do pitch (que so correm a cada N frames) e eliminar o "serrote"
de posicoes que deslizam com o pan da camara e saltam na detecao seguinte.
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
_TARGET_VERTICES = np.array(_CONFIG.vertices, dtype=np.float32)

# Escala: CONFIG usa cm do campo (12000×7000) → metros reais (105×68)
_M_PER_CM_X = 105.0 / _CONFIG.length
_M_PER_CM_Y = 68.0 / _CONFIG.width

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

# Margem de tolerância (m) fora do campo; além disto a homografia é considerada má
_OUT_OF_PITCH_MARGIN_M = 3.0

# Distância máxima (m) para um id novo herdar um id morto (re-identificação)
_STITCH_MAX_DIST_M = 4.0


class PositionsRecorder:
    """Acumula dados crus por frame; projeta e suaviza em to_payload()."""

    def __init__(
        self,
        fps: float,
        smooth_window: int = 5,
        max_gap_seconds: float = 1.0,
        stitch_gap_seconds: float = 2.0,
    ):
        self.fps = float(fps)
        self._smooth_window = max(1, int(smooth_window))
        # Gap (frames) acima do qual o historico de um tracker fica obsoleto e
        # acima do qual nao se interpola keypoints entre detecoes do pitch
        self._max_gap_frames = max(1, int(round(self.fps * max_gap_seconds)))
        # Gap maximo (frames) para um id novo herdar um id morto
        self._stitch_gap_frames = max(1, int(round(self.fps * stitch_gap_seconds)))
        # frame_index → {"kp": (xy, conf), "players": [(tid, px, py, t, c)], "ball": (px, py)}
        self._raw: dict = {}

    def update(
        self,
        frame_index: int,
        detections: sv.Detections,
        keypoints: sv.KeyPoints,
        team_ids: np.ndarray,
    ) -> None:
        """Captura as posicoes em pixels dos jogadores de um frame."""
        if detections.tracker_id is None or len(detections) == 0:
            return

        kp = _keypoint_arrays(keypoints)
        if kp is None:
            return

        xy_pixels = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        players = []
        for i, tracker_id in enumerate(detections.tracker_id):
            if tracker_id is None or int(tracker_id) == -1:
                continue
            team = int(team_ids[i]) if i < len(team_ids) else -1
            class_id = int(detections.class_id[i]) if detections.class_id is not None else -1
            players.append((
                int(tracker_id),
                float(xy_pixels[i][0]),
                float(xy_pixels[i][1]),
                team,
                class_id,
            ))

        if players:
            entry = self._frame_entry(frame_index, kp)
            entry["players"] = players

    def update_ball(
        self,
        frame_index: int,
        ball_xy_pixels: Optional[np.ndarray],
        keypoints: sv.KeyPoints,
    ) -> None:
        """Captura a posicao em pixels (interpolada) da bola para um frame."""
        if ball_xy_pixels is None:
            return

        kp = _keypoint_arrays(keypoints)
        if kp is None:
            return

        ball = np.asarray(ball_xy_pixels, dtype=np.float32).ravel()
        if ball.size < 2:
            return

        entry = self._frame_entry(frame_index, kp)
        entry["ball"] = (float(ball[0]), float(ball[1]))

    def to_payload(self) -> dict:
        sorted_frames = sorted(self._raw.keys())
        transformers = self._frame_transformers(sorted_frames)

        # Passo 1: projetar tudo (sem suavizar) e recolher o span de cada tracker
        projected = []  # (frame_index, [(tid, cm, team, class_id)], ball | None)
        spans: dict = {}
        for frame_index in sorted_frames:
            transformer = transformers.get(frame_index)
            if transformer is None:
                continue
            entry = self._raw[frame_index]

            accepted = []
            if entry["players"]:
                pixels = np.array([(p[1], p[2]) for p in entry["players"]], dtype=np.float32)
                pitch_cm = _transform_or_none(transformer, pixels)
                if pitch_cm is not None:
                    for (tid, _px, _py, team, class_id), cm in zip(entry["players"], pitch_cm):
                        # Descartar projeções fora do campo (homografia instável)
                        # antes de entrarem na janela de suavização
                        raw_x_m = float(cm[0]) * _M_PER_CM_X
                        raw_y_m = float(cm[1]) * _M_PER_CM_Y
                        if not _within_pitch(raw_x_m, raw_y_m):
                            continue
                        accepted.append((tid, cm, team, class_id))
                        span = spans.get(tid)
                        if span is None:
                            spans[tid] = {
                                "first": frame_index, "first_cm": cm,
                                "last": frame_index, "last_cm": cm,
                            }
                        else:
                            span["last"] = frame_index
                            span["last_cm"] = cm

            ball = None
            if entry["ball"] is not None:
                ball_px = np.asarray(entry["ball"], dtype=np.float32).reshape(1, 2)
                ball_cm = _transform_or_none(transformer, ball_px)
                if ball_cm is not None:
                    ball_x_m = float(ball_cm[0][0]) * _M_PER_CM_X
                    ball_y_m = float(ball_cm[0][1]) * _M_PER_CM_Y
                    if _within_pitch(ball_x_m, ball_y_m):
                        ball = {"x": round(ball_x_m, 2), "y": round(ball_y_m, 2)}

            if accepted or ball is not None:
                projected.append((frame_index, accepted, ball))

        # Ids novos que continuam tracks mortos herdam o id antigo
        remap = _stitch_ids(spans, self._stitch_gap_frames, _STITCH_MAX_DIST_M)

        # Passo 2: suavizar e emitir com os ids re-mapeados
        pos_history: dict = defaultdict(lambda: deque(maxlen=self._smooth_window))
        last_seen_frame: dict = {}
        team_votes: dict = defaultdict(lambda: defaultdict(int))
        class_votes: dict = defaultdict(lambda: defaultdict(int))

        frames = []
        for frame_index, accepted, ball in projected:
            players = []
            for tid, cm, team, class_id in accepted:
                rid = remap.get(tid, tid)

                # Historico obsoleto apos um gap longo: senao o jogador
                # "desliza" desde a posicao antiga ao reaparecer
                last_seen = last_seen_frame.get(rid)
                if last_seen is not None and frame_index - last_seen > self._max_gap_frames:
                    pos_history[rid].clear()
                last_seen_frame[rid] = frame_index

                pos_history[rid].append(cm)
                smoothed_cm = np.mean(pos_history[rid], axis=0)
                team_votes[rid][team] += 1
                class_votes[rid][class_id] += 1
                players.append({
                    "id": rid,
                    "x": round(float(smoothed_cm[0]) * _M_PER_CM_X, 2),
                    "y": round(float(smoothed_cm[1]) * _M_PER_CM_Y, 2),
                })

            if players or ball is not None:
                frames.append({"i": frame_index, "players": players, "ball": ball})

        roster = [
            {"id": tid, "t": _majority(team_votes[tid]), "c": _majority(class_votes[tid])}
            for tid in sorted(team_votes.keys())
        ]

        return {
            "version": 1,
            "fps": self.fps,
            "pitch": {"length_m": PITCH_LENGTH_M, "width_m": PITCH_WIDTH_M},
            "frame_count": frames[-1]["i"] if frames else 0,
            "roster": roster,
            "frames": frames,
        }

    def _frame_entry(self, frame_index: int, kp) -> dict:
        entry = self._raw.setdefault(int(frame_index), {"kp": None, "players": None, "ball": None})
        if entry["kp"] is None:
            entry["kp"] = kp
        return entry

    def _frame_transformers(self, sorted_frames: list) -> dict:
        """ViewTransformer efetivo por frame.

        Detecao fresca → os proprios keypoints; entre detecoes proximas →
        interpolacao linear dos keypoints visiveis em ambas; caso contrario →
        ultimos keypoints validos (hold, comportamento antigo).
        """
        if not sorted_frames:
            return {}

        # Detecoes frescas = frames cujos valores diferem do frame capturado
        # anterior (frames "hold" partilham o mesmo array reutilizado)
        fresh = []
        prev_xy = None
        for frame_index in sorted_frames:
            xy, conf = self._raw[frame_index]["kp"]
            if prev_xy is None or (xy is not prev_xy and not np.array_equal(xy, prev_xy)):
                fresh.append((frame_index, xy, conf))
            prev_xy = xy

        exact_cache: dict = {}

        def exact_transformer(frame_index, xy, conf):
            if frame_index not in exact_cache:
                mask = _visible_mask(xy, conf)
                if mask is None or int(np.sum(mask)) < 4:
                    exact_cache[frame_index] = None
                else:
                    exact_cache[frame_index] = _make_transformer(xy[mask], _TARGET_VERTICES[mask])
            return exact_cache[frame_index]

        result: dict = {}
        fresh_idx = 0
        for frame_index in sorted_frames:
            while fresh_idx + 1 < len(fresh) and fresh[fresh_idx + 1][0] <= frame_index:
                fresh_idx += 1
            frame_a, xy_a, conf_a = fresh[fresh_idx]

            if frame_index == frame_a:
                result[frame_index] = exact_transformer(frame_a, xy_a, conf_a)
                continue

            nxt = fresh[fresh_idx + 1] if fresh_idx + 1 < len(fresh) else None
            if nxt is not None and nxt[0] - frame_a <= self._max_gap_frames:
                frame_b, xy_b, conf_b = nxt
                mask_a = _visible_mask(xy_a, conf_a)
                mask_b = _visible_mask(xy_b, conf_b)
                if mask_a is not None and mask_b is not None:
                    common = mask_a & mask_b
                    if int(np.sum(common)) >= 4:
                        t = (frame_index - frame_a) / float(frame_b - frame_a)
                        blended = xy_a[common] + (xy_b[common] - xy_a[common]) * t
                        transformer = _make_transformer(blended, _TARGET_VERTICES[common])
                        if transformer is not None:
                            result[frame_index] = transformer
                            continue

            result[frame_index] = exact_transformer(frame_a, xy_a, conf_a)
        return result


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


def _stitch_ids(spans: dict, max_gap_frames: int, max_dist_m: float) -> dict:
    """Mapa tid→id herdado: um track novo que nasce pouco depois e perto do fim
    de um track morto e o mesmo jogador re-identificado pelo tracker."""
    remap: dict = {}
    open_tracks: dict = {}  # id raiz → (ultimo frame, ultima posicao cm)
    for tid in sorted(spans.keys(), key=lambda t: (spans[t]["first"], t)):
        span = spans[tid]
        best = None
        for root, (last_frame, last_cm) in open_tracks.items():
            gap = span["first"] - last_frame
            if gap <= 0 or gap > max_gap_frames:
                continue
            dx_m = float(span["first_cm"][0] - last_cm[0]) * _M_PER_CM_X
            dy_m = float(span["first_cm"][1] - last_cm[1]) * _M_PER_CM_Y
            dist_m = (dx_m * dx_m + dy_m * dy_m) ** 0.5
            if dist_m > max_dist_m:
                continue
            if best is None or (dist_m, gap) < (best[0], best[1]):
                best = (dist_m, gap, root)

        root = tid
        if best is not None:
            root = best[2]
            remap[tid] = root

        current = open_tracks.get(root)
        if current is None or span["last"] > current[0]:
            open_tracks[root] = (span["last"], span["last_cm"])
    return remap


def _majority(votes: dict) -> int:
    """Valor mais votado; em empate ganha o visto primeiro (determinista)."""
    return max(votes.items(), key=lambda item: item[1])[0]


def _within_pitch(x_m: float, y_m: float) -> bool:
    return (
        -_OUT_OF_PITCH_MARGIN_M <= x_m <= PITCH_LENGTH_M + _OUT_OF_PITCH_MARGIN_M
        and -_OUT_OF_PITCH_MARGIN_M <= y_m <= PITCH_WIDTH_M + _OUT_OF_PITCH_MARGIN_M
    )


def _keypoint_arrays(keypoints: sv.KeyPoints):
    """Devolve (xy, confidence) do primeiro conjunto de keypoints, ou None."""
    if keypoints is None or len(keypoints.xy) == 0 or len(keypoints.xy[0]) == 0:
        return None
    conf = None
    if keypoints.confidence is not None and len(keypoints.confidence) > 0:
        conf = keypoints.confidence[0]
    return keypoints.xy[0], conf


def _visible_mask(xy: np.ndarray, conf) -> Optional[np.ndarray]:
    """Mesma logica do DistanceTracker/render_radar para keypoints visiveis."""
    if xy.shape[0] != _TARGET_VERTICES.shape[0]:
        return None
    mask = (xy[:, 0] > 1) & (xy[:, 1] > 1)
    if conf is not None and len(conf) == xy.shape[0]:
        mask = mask & (conf > 0.5)
    return mask


def _make_transformer(source: np.ndarray, target: np.ndarray) -> Optional[ViewTransformer]:
    if len(source) < 4:
        return None
    try:
        return ViewTransformer(
            source=np.asarray(source, dtype=np.float32),
            target=np.asarray(target, dtype=np.float32),
        )
    except Exception:
        return None


def _transform_or_none(transformer: ViewTransformer, points: np.ndarray):
    try:
        return transformer.transform_points(points=points)
    except Exception:
        return None
