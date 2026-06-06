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
    presencas num grid 2D por equipa (0 e 1) e num grid global (todos).
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

        # grids[team] e grids global, indexados [linha (y/width), coluna (x/length)]
        self._grids: dict = {
            0: np.zeros((grid_rows, grid_cols), dtype=np.float64),
            1: np.zeros((grid_rows, grid_cols), dtype=np.float64),
        }
        self._grid_all = np.zeros((grid_rows, grid_cols), dtype=np.float64)
        self._samples: dict = defaultdict(int)

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
        print("=" * 50)

    def show(self, window_name: str = "Heatmaps") -> None:
        """Gera os heatmaps (Team 0, Team 1, Global) e mostra-os numa janela."""
        if self._samples[0] + self._samples[1] == 0:
            print("[HeatmapTracker] Sem dados para mostrar heatmap.")
            return

        panel_t0 = self._labeled_panel(self.render(self._grids[0]), "Team 0")
        panel_t1 = self._labeled_panel(self.render(self._grids[1]), "Team 1")
        panel_all = self._labeled_panel(self.render(self._grid_all), "Todos")

        # Linha de cima: Team 0 e Team 1 lado a lado.
        top_row = np.hstack([panel_t0, panel_t1])
        top_w = top_row.shape[1]

        # Linha de baixo: "Todos" centrado na largura da linha de cima.
        all_h, all_w = panel_all.shape[:2]
        bottom_row = np.zeros((all_h, top_w, 3), dtype=np.uint8)
        x_off = (top_w - all_w) // 2
        bottom_row[:, x_off:x_off + all_w] = panel_all

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
