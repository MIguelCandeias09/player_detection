from collections import defaultdict
from typing import Optional

import cv2
import numpy as np
import supervision as sv

from sports.annotators.soccer import draw_pitch
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

_CONFIG = SoccerPitchConfiguration()


class HeatmapTracker:
    """Acumula posicoes dos jogadores em coordenadas do campo e gera heatmaps.

    Espelha a logica do DistanceTracker: usa keypoints para construir um
    ViewTransformer e converter posicoes de pixel -> cm do campo. Acumula
    presencas em grids 2D: por equipa (0 e 1), por jogador (tracker_id),
    da bola e global (todos os jogadores).

    API de selecao (para o frontend escolher que heatmap mostrar):
        render_global(), render_team(0|1), render_player(tracker_id),
        render_ball() -> imagem BGR do campo com o heatmap.
        list_players() -> ids disponiveis ordenados por presenca.
    """

    # Resolucao do grid (colunas x linhas) sobre o campo (length x width).
    GRID_COLS = 120
    GRID_ROWS = 68
    # Desvio do blur gaussiano aplicado ao grid antes de colorir (em celulas).
    SMOOTH_SIGMA = 2.5
    # Transparencia do colormap por cima do campo desenhado.
    OVERLAY_ALPHA = 0.6
    # Largura maxima da janela final (px); o composite e reduzido para caber.
    MAX_WINDOW_WIDTH = 1500

    def __init__(
        self,
        grid_cols: int = GRID_COLS,
        grid_rows: int = GRID_ROWS,
        smooth_sigma: float = SMOOTH_SIGMA,
    ):
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.smooth_sigma = smooth_sigma

        # grids[team], grid global, por jogador e bola,
        # indexados [linha (y/width), coluna (x/length)]
        self._grids: dict = {
            0: np.zeros((grid_rows, grid_cols), dtype=np.float64),
            1: np.zeros((grid_rows, grid_cols), dtype=np.float64),
        }
        self._grid_all = np.zeros((grid_rows, grid_cols), dtype=np.float64)
        self._grids_player: dict = defaultdict(
            lambda: np.zeros((grid_rows, grid_cols), dtype=np.float64)
        )
        self._grid_ball = np.zeros((grid_rows, grid_cols), dtype=np.float64)

        self._samples: dict = defaultdict(int)
        self._samples_player: dict = defaultdict(int)
        self._samples_ball = 0
        self._player_team: dict = {}
        # A bola chega com delay (sync buffer) e tambem no flush final;
        # evitar contar o mesmo frame duas vezes.
        self._ball_seen_frames: set = set()

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def update(
        self,
        frame_index: int,
        detections: sv.Detections,
        keypoints: sv.KeyPoints,
        team_ids: np.ndarray,
    ) -> None:
        """Processa um frame: transforma coords e acumula presencas no grid."""
        if detections is None or len(detections) == 0 or team_ids is None:
            return

        team_ids = np.asarray(team_ids, dtype=int)
        if len(team_ids) != len(detections):
            return

        transformer = self._build_transformer(keypoints)
        if transformer is None:
            return

        xy_pixels = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        try:
            xy_pitch_cm = transformer.transform_points(points=xy_pixels)
        except Exception:
            return

        for i in range(len(detections)):
            team = int(team_ids[i])
            if team not in (0, 1):
                continue  # ignora guarda-redes (2/3), arbitros e bola

            col, row = self._cell_index(xy_pitch_cm[i])
            if col is None:
                continue

            self._grids[team][row, col] += 1.0
            self._grid_all[row, col] += 1.0
            self._samples[team] += 1

            tracker_id = self._tracker_id_at(detections, i)
            if tracker_id is not None:
                self._grids_player[tracker_id][row, col] += 1.0
                self._samples_player[tracker_id] += 1
                self._player_team[tracker_id] = team

    def update_ball(
        self,
        frame_index: int,
        ball_xy_pixels,
        keypoints: sv.KeyPoints,
    ) -> None:
        """Acumula a posicao da bola (em pixels) no grid da bola."""
        if frame_index in self._ball_seen_frames:
            return
        self._ball_seen_frames.add(frame_index)

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

        col, row = self._cell_index(ball_pitch_cm[0])
        if col is None:
            return

        self._grid_ball[row, col] += 1.0
        self._samples_ball += 1

    # ------------------------------------------------------------------
    # Selecao de heatmaps (interface para o frontend)
    # ------------------------------------------------------------------

    def render_global(self) -> np.ndarray:
        """Heatmap de todos os jogadores combinados."""
        return self.render(self._grid_all)

    def render_team(self, team_id: int) -> np.ndarray:
        """Heatmap de uma equipa (0 ou 1)."""
        if team_id not in self._grids:
            raise ValueError(f"team_id invalido: {team_id} (esperado 0 ou 1)")
        return self.render(self._grids[team_id])

    def render_player(self, tracker_id: int) -> np.ndarray:
        """Heatmap de um jogador individual (tracker_id)."""
        tracker_id = int(tracker_id)
        if tracker_id not in self._grids_player:
            return self.render(np.zeros((self.grid_rows, self.grid_cols)))
        return self.render(self._grids_player[tracker_id])

    def render_ball(self) -> np.ndarray:
        """Heatmap da bola."""
        return self.render(self._grid_ball)

    def list_players(self) -> list:
        """Jogadores com dados, ordenados por presenca (mais frames primeiro).

        Devolve lista de dicts: {"tracker_id", "team", "samples"} — pronto a
        serializar para o frontend popular o seletor de heatmaps.
        """
        ranked = sorted(
            self._samples_player.items(),
            key=lambda item: (-item[1], item[0]),
        )
        return [
            {
                "tracker_id": tracker_id,
                "team": self._player_team.get(tracker_id),
                "samples": samples,
            }
            for tracker_id, samples in ranked
        ]

    def print_report(self) -> None:
        """Imprime resumo de amostras recolhidas por equipa."""
        total = self._samples[0] + self._samples[1]
        if total == 0:
            print("\n[HeatmapTracker] Sem dados recolhidos.")
            return

        print("\n" + "=" * 50)
        print("         HEATMAP TRACKER - RELATORIO")
        print("=" * 50)
        print(f"  Amostras Team 0: {self._samples[0]}")
        print(f"  Amostras Team 1: {self._samples[1]}")
        print(f"  Amostras totais: {total}")
        print(f"  Amostras bola:   {self._samples_ball}")
        print(f"  Jogadores com heatmap: {len(self._samples_player)}")
        if self._samples_player:
            print("-" * 50)
            print("  Heatmaps individuais disponiveis (top 10):")
            for entry in self.list_players()[:10]:
                print(
                    f"    ID {entry['tracker_id']} (T{entry['team']}): "
                    f"{entry['samples']} amostras"
                )
        print("=" * 50)

    def show(self, window_name: str = "Heatmaps") -> None:
        """Mostra os heatmaps (Team 0, Team 1 / Todos, Bola) numa janela."""
        if self._samples[0] + self._samples[1] + self._samples_ball == 0:
            print("[HeatmapTracker] Sem dados para mostrar heatmap.")
            return

        panel_t0 = self._labeled_panel(self.render_team(0), "Team 0")
        panel_t1 = self._labeled_panel(self.render_team(1), "Team 1")
        panel_all = self._labeled_panel(self.render_global(), "Todos")
        panel_ball = self._labeled_panel(self.render_ball(), "Bola")

        # Grelha 2x2: equipas em cima, Todos e Bola em baixo.
        top_row = np.hstack([panel_t0, panel_t1])
        bottom_row = np.hstack([panel_all, panel_ball])

        composite = np.vstack([top_row, bottom_row])
        composite = self._fit_to_window(composite)

        try:
            cv2.imshow(window_name, composite)
            print("\n[HeatmapTracker] A mostrar heatmaps. Carrega numa tecla para fechar.")
            cv2.waitKey(0)
            cv2.destroyWindow(window_name)
        except Exception as exc:
            print(f"[HeatmapTracker] Nao foi possivel mostrar a janela: {exc}")

    def render(self, grid: np.ndarray) -> np.ndarray:
        """Desenha o heatmap de um grid por cima do campo."""
        pitch = draw_pitch(config=_CONFIG)
        h, w = pitch.shape[:2]

        if grid.max() <= 0:
            return pitch

        # Suavizar o grid (kernel gaussiano em celulas).
        smoothed = cv2.GaussianBlur(grid, ksize=(0, 0), sigmaX=self.smooth_sigma)

        # Redimensionar para o tamanho do campo desenhado.
        resized = cv2.resize(smoothed, (w, h), interpolation=cv2.INTER_LINEAR)

        # Normalizar [0..255] e aplicar colormap.
        norm = resized / resized.max()
        heat_u8 = (norm * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)

        # Mascara: so misturar onde ha presenca (mantem campo limpo nas zonas vazias).
        mask = (norm > 0.02).astype(np.float32)[..., None]
        blended = pitch.astype(np.float32) * (1 - mask * self.OVERLAY_ALPHA) + \
            colored.astype(np.float32) * (mask * self.OVERLAY_ALPHA)
        return blended.astype(np.uint8)

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _cell_index(self, pos_cm: np.ndarray):
        """Converte uma posicao em cm do campo para (coluna, linha) do grid."""
        x_cm, y_cm = float(pos_cm[0]), float(pos_cm[1])
        if not (0 <= x_cm <= _CONFIG.length and 0 <= y_cm <= _CONFIG.width):
            return None, None

        col = int(x_cm / _CONFIG.length * self.grid_cols)
        row = int(y_cm / _CONFIG.width * self.grid_rows)
        col = min(max(col, 0), self.grid_cols - 1)
        row = min(max(row, 0), self.grid_rows - 1)
        return col, row

    def _tracker_id_at(self, detections: sv.Detections, index: int) -> Optional[int]:
        if detections.tracker_id is None:
            return None

        tracker_id = detections.tracker_id[index]
        if tracker_id is None or int(tracker_id) == -1:
            return None
        return int(tracker_id)

    def _fit_to_window(self, image: np.ndarray) -> np.ndarray:
        """Reduz a imagem para a largura maxima da janela, mantendo o aspeto."""
        h, w = image.shape[:2]
        if w <= self.MAX_WINDOW_WIDTH:
            return image
        scale = self.MAX_WINDOW_WIDTH / w
        new_size = (self.MAX_WINDOW_WIDTH, int(h * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    def _labeled_panel(self, image: np.ndarray, label: str) -> np.ndarray:
        """Coloca um titulo no canto superior esquerdo do painel."""
        out = image.copy()
        cv2.putText(out, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return out

    def _build_transformer(self, keypoints: sv.KeyPoints) -> Optional[ViewTransformer]:
        """Replica a logica de render_radar para obter o ViewTransformer."""
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
