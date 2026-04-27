from collections import defaultdict, deque
from typing import Optional

import numpy as np
import supervision as sv

from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

_CONFIG = SoccerPitchConfiguration()

# Escala: CONFIG usa cm do campo (12000×7000) → metros reais (105×68)
_M_PER_CM_X = 105.0 / _CONFIG.length   # 105 / 12000 ≈ 0.00875 m/cm
_M_PER_CM_Y = 68.0 / _CONFIG.width     # 68  /  7000 ≈ 0.00971 m/cm


class DistanceTracker:
    """Acumula distâncias percorridas por tracker_id usando coordenadas do radar."""

    def __init__(self, fps: float, smooth_window: int = 5, max_jump_m: float = 0.5):
        self.fps = fps
        self.smooth_window = smooth_window
        self.max_jump_m = max_jump_m

        self._pos_history: dict = defaultdict(lambda: deque(maxlen=smooth_window))
        self._last_smoothed: dict = {}
        self._total_distance_m: dict = defaultdict(float)
        self._max_speed_kmh: dict = defaultdict(float)

    # ------------------------------------------------------------------
    # Métodos chamados externamente para processar frames e imprimir relatório
    # ------------------------------------------------------------------

    def update(
        self,
        frame_index: int,
        detections: sv.Detections,
        keypoints: sv.KeyPoints,
    ) -> None:
        """Processa um frame: transforma coords, suaviza, acumula distâncias."""
        if detections.tracker_id is None or len(detections) == 0:
            return

        transformer = self._build_transformer(keypoints)
        if transformer is None:
            return

        xy_pixels = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        try:
            xy_pitch_cm = transformer.transform_points(points=xy_pixels)
        except Exception:
            return

        for i, tracker_id in enumerate(detections.tracker_id):
            if tracker_id is None or int(tracker_id) == -1:
                continue
            tid = int(tracker_id)
            pos_cm = xy_pitch_cm[i]

            self._pos_history[tid].append(pos_cm)
            smoothed = np.mean(self._pos_history[tid], axis=0)

            if tid in self._last_smoothed:
                delta_cm = smoothed - self._last_smoothed[tid]
                # Converter delta de cm para metros com escala real
                delta_m = np.array([delta_cm[0] * _M_PER_CM_X,
                                    delta_cm[1] * _M_PER_CM_Y])
                dist_m = float(np.linalg.norm(delta_m))

                if dist_m <= self.max_jump_m:
                    self._total_distance_m[tid] += dist_m
                    speed_kmh = dist_m * self.fps * 3.6
                    if speed_kmh > self._max_speed_kmh[tid]:
                        self._max_speed_kmh[tid] = speed_kmh

            self._last_smoothed[tid] = smoothed

    def print_report(self) -> None:
        """Imprime relatório de distâncias e velocidade máxima por tracker_id."""
        if not self._total_distance_m:
            print("\n[DistanceTracker] Sem dados recolhidos.")
            return

        print("\n" + "=" * 50)
        print("         DISTANCE TRACKER — RELATÓRIO")
        print("=" * 50)
        print(f"  {'ID':>4}  {'Distância (km)':>14}  {'Vel. Máx. (km/h)':>16}")
        print("-" * 50)
        for tid in sorted(self._total_distance_m.keys()):
            dist_km = self._total_distance_m[tid] / 1000.0
            speed = self._max_speed_kmh.get(tid, 0.0)
            print(f"  {tid:>4}  {dist_km:>14.3f}  {speed:>15.1f}")
        print("=" * 50)

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _build_transformer(self, keypoints: sv.KeyPoints) -> Optional[ViewTransformer]:
        """Replica a lógica de render_radar para obter o ViewTransformer."""
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
