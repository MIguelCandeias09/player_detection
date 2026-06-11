from collections import defaultdict, deque
from typing import Optional

import numpy as np
import supervision as sv

from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

_CONFIG = SoccerPitchConfiguration()

# CONFIG uses virtual pitch cm (12000x7000); convert to real meters.
_M_PER_CM_X = 105.0 / _CONFIG.length
_M_PER_CM_Y = 68.0 / _CONFIG.width


class PossessionTracker:
    """Accumulates team possession with closest-player logic in real coordinates."""

    POSSESSION_RADIUS_M = 1.5
    SMOOTH_WINDOW = 5

    def __init__(
        self,
        fps: float,
        possession_radius_m: float = POSSESSION_RADIUS_M,
        smooth_window: int = SMOOTH_WINDOW,
    ):
        self.fps = fps
        self.possession_radius_m = possession_radius_m
        self.smooth_window = smooth_window

        self._frames_team = {0: 0, 1: 0}
        self._frames_player: dict = defaultdict(int)
        self._player_team: dict = {}
        self._loose_frames = 0
        self._unassigned_frames = 0
        self._total_frames = 0
        self._skipped_frames = 0
        self._candidate_buffer = deque(maxlen=smooth_window)
        self._current_team: Optional[int] = None
        self._seen_frames: set[int] = set()

    def update(
        self,
        frame_index: int,
        ball_xy_pixels: Optional[np.ndarray],
        players_detections: sv.Detections,
        players_team_ids: np.ndarray,
        keypoints: sv.KeyPoints,
    ) -> None:
        """Process one synchronized frame and assign possession or loose ball."""
        if frame_index in self._seen_frames:
            return
        self._seen_frames.add(frame_index)

        if ball_xy_pixels is None:
            return

        transformer = self._build_transformer(keypoints)
        if transformer is None:
            return

        try:
            ball_xy = np.asarray(ball_xy_pixels, dtype=np.float32).reshape(1, 2)
            ball_pitch_cm = transformer.transform_points(points=ball_xy)
        except Exception:
            return

        if (
            players_detections is None
            or players_team_ids is None
            or len(players_detections) == 0
        ):
            self._record_frame(candidate_team=None, candidate_player_id=None)
            return

        team_ids = np.asarray(players_team_ids, dtype=int)
        if len(team_ids) != len(players_detections):
            self._skipped_frames += 1
            return

        eligible_mask = np.isin(team_ids, [0, 1])
        if not np.any(eligible_mask):
            self._record_frame(candidate_team=None, candidate_player_id=None)
            return

        try:
            xy_pixels = players_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            xy_pitch_cm = transformer.transform_points(points=xy_pixels)
        except Exception:
            return

        eligible_indices = np.flatnonzero(eligible_mask)
        eligible_xy_cm = xy_pitch_cm[eligible_indices]
        deltas_cm = eligible_xy_cm - ball_pitch_cm[0]
        deltas_m = np.column_stack((
            deltas_cm[:, 0] * _M_PER_CM_X,
            deltas_cm[:, 1] * _M_PER_CM_Y,
        ))
        distances_m = np.linalg.norm(deltas_m, axis=1)

        nearest_local_idx = int(np.argmin(distances_m))
        nearest_distance_m = float(distances_m[nearest_local_idx])
        if nearest_distance_m > self.possession_radius_m:
            self._record_frame(candidate_team=None, candidate_player_id=None)
            return

        nearest_idx = int(eligible_indices[nearest_local_idx])
        candidate_team = int(team_ids[nearest_idx])
        candidate_player_id = self._tracker_id_at(players_detections, nearest_idx)

        self._record_frame(
            candidate_team=candidate_team,
            candidate_player_id=candidate_player_id,
        )

    def print_report(self) -> None:
        """Print team possession and top-player report."""
        if self._total_frames == 0:
            print("\n[PossessionTracker] Sem dados recolhidos.")
            if self._skipped_frames:
                print(f"[PossessionTracker] Frames ignorados por dados inconsistentes: {self._skipped_frames}")
            return

        print("\n" + "=" * 50)
        print("        POSSESSION TRACKER - RELATORIO")
        print("=" * 50)
        print(f"Frames analisados: {self._total_frames}")
        if self._skipped_frames:
            print(f"Frames ignorados: {self._skipped_frames}")
        print(
            f"Bola solta: {self._loose_frames} frames "
            f"({self._pct(self._loose_frames):.1f}%)"
        )
        if self._unassigned_frames:
            print(
                f"Sem posse atribuida: {self._unassigned_frames} frames "
                f"({self._pct(self._unassigned_frames):.1f}%)"
            )
        print(
            f"Posse Team 0: {self._frames_team[0]} frames "
            f"({self._possession_pct(self._frames_team[0]):.1f}%)"
        )
        print(
            f"Posse Team 1: {self._frames_team[1]} frames "
            f"({self._possession_pct(self._frames_team[1]):.1f}%)"
        )

        if self._frames_player:
            print("-" * 50)
            print("Top 5 jogadores por tempo com bola")
            ranked_players = sorted(
                self._frames_player.items(),
                key=lambda item: (-item[1], item[0]),
            )[:5]
            for tracker_id, frames in ranked_players:
                team_id = self._player_team.get(tracker_id, "?")
                seconds = frames / self.fps if self.fps else 0.0
                print(f"ID {tracker_id} (T{team_id}): {seconds:.1f}s")
        print("=" * 50)

    def to_dict(self) -> dict:
        """Serializa o relatorio de posse para JSON (frontend web)."""
        top_players = []
        if self._frames_player:
            ranked = sorted(
                self._frames_player.items(),
                key=lambda item: (-item[1], item[0]),
            )[:5]
            for tracker_id, frames in ranked:
                seconds = frames / self.fps if self.fps else 0.0
                top_players.append({
                    "tracker_id": tracker_id,
                    "team": self._player_team.get(tracker_id),
                    "seconds": round(seconds, 1),
                })
        return {
            "frames_analyzed": self._total_frames,
            "loose_pct": round(self._pct(self._loose_frames), 1),
            "team": {
                "0": {"pct": round(self._possession_pct(self._frames_team[0]), 1)},
                "1": {"pct": round(self._possession_pct(self._frames_team[1]), 1)},
            },
            "top_players": top_players,
        }

    def player_seconds(self) -> dict:
        """Tempo (s) com bola por tracker_id, para juntar a tabela de jogadores."""
        if not self.fps:
            return {tracker_id: 0.0 for tracker_id in self._frames_player}
        return {
            tracker_id: frames / self.fps
            for tracker_id, frames in self._frames_player.items()
        }

    def _record_frame(
        self,
        candidate_team: Optional[int],
        candidate_player_id: Optional[int],
    ) -> None:
        self._candidate_buffer.append(candidate_team)
        smoothed_team = self._smoothed_team()

        self._total_frames += 1

        if candidate_team is None:
            self._loose_frames += 1

        if smoothed_team is not None:
            self._current_team = smoothed_team
        elif self._current_team is None and candidate_team is not None:
            self._current_team = candidate_team

        if self._current_team is None:
            self._unassigned_frames += 1
            return

        self._frames_team[self._current_team] += 1

        if candidate_team == self._current_team and candidate_player_id is not None:
            self._frames_player[candidate_player_id] += 1
            self._player_team[candidate_player_id] = self._current_team

    def _smoothed_team(self) -> Optional[int]:
        if not self._candidate_buffer:
            return None

        threshold = len(self._candidate_buffer) // 2 + 1
        count_team_0 = sum(candidate == 0 for candidate in self._candidate_buffer)
        count_team_1 = sum(candidate == 1 for candidate in self._candidate_buffer)

        if count_team_0 >= threshold:
            return 0
        if count_team_1 >= threshold:
            return 1
        return None

    def _pct(self, frames: int) -> float:
        return 100.0 * frames / self._total_frames if self._total_frames else 0.0

    def _possession_pct(self, frames: int) -> float:
        assigned_frames = self._frames_team[0] + self._frames_team[1]
        return 100.0 * frames / assigned_frames if assigned_frames else 0.0

    def _tracker_id_at(self, detections: sv.Detections, index: int) -> Optional[int]:
        if detections.tracker_id is None:
            return None

        tracker_id = detections.tracker_id[index]
        if tracker_id is None or int(tracker_id) == -1:
            return None
        return int(tracker_id)

    def _build_transformer(self, keypoints: sv.KeyPoints) -> Optional[ViewTransformer]:
        """Replicate render_radar logic to build the ViewTransformer."""
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
