from sklearn.cluster import KMeans
import numpy as np
import cv2
from collections import deque, Counter

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from sklearn.decomposition import PCA
    _MATPLOTLIB_OK = True
except ImportError:
    _MATPLOTLIB_OK = False


class TeamClassifierSeg:
    """
    Classificador de equipas usando MASCARAS DE SEGMENTACAO + Histogramas 3D (HSV) + K-Means + Votação Temporal.

    Usa a máscara de segmentação do yolo11m-seg para isolar exatamente os pixels
    do jogador — sem relva, sem fundo. Extrai histogramas 3D (Hue + Saturation + Value)
    para discriminar equipas mesmo com cores semelhantes (ex: vermelho escuro vs claro).

    Args:
        debug (bool): Se True, imprime info no terminal e abre janelas de debug.
    """
    def __init__(self, debug=False):
        self.team_kmeans = None
        self.previous_centers = None
        self.player_team_history = {}
        self.player_class_history = {}
        self.locked_player_teams = {}
        self.HISTORY_LENGTH = 20
        self.CLASS_HISTORY_LENGTH = 60
        self.LOCK_THRESHOLD = 20
        self.GK_CONSISTENCY_THRESHOLD = 0.7

        # Histograma 3D: H(8) × S(8) × V(4) = 256 features
        self.HIST_BINS = [8, 8, 4]
        self.HIST_RANGES = [0, 180, 0, 256, 0, 256]
        self.HIST_CHANNELS = [0, 1, 2]

        self.debug = debug
        self.debug_frame_count = 0
        self.DEBUG_SHOW_EVERY = 30  # atualiza janelas a cada N frames

    # ------------------------------------------------------------------
    # Debug visual helpers
    # ------------------------------------------------------------------

    def _show_crop_grid(self, frame, bboxes, masks_full_res, track_ids, team_ids):
        """Mostra janela com grid dos crops mascarados de cada jogador, com bordas por equipa."""
        CROP_W, CROP_H = 80, 120
        BORDER = 4
        TEAM_COLORS_BGR = {0: (0, 0, 220), 1: (220, 50, 0), -1: (120, 120, 120)}

        crops = []
        for idx, (bbox, tid, team_id) in enumerate(zip(bboxes, track_ids, team_ids)):
            y1, y2 = int(bbox[1]), int(bbox[3])
            x1, x2 = int(bbox[0]), int(bbox[2])
            if y1 >= y2 or x1 >= x2:
                continue

            crop = frame[y1:y2, x1:x2].copy()

            if masks_full_res is not None and idx < len(masks_full_res) and masks_full_res[idx] is not None:
                mask_crop = masks_full_res[idx][y1:y2, x1:x2]
                crop[~mask_crop] = 0

            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (CROP_W, CROP_H))

            color = TEAM_COLORS_BGR.get(int(team_id), (120, 120, 120))
            crop = cv2.copyMakeBorder(crop, BORDER, BORDER, BORDER, BORDER,
                                      cv2.BORDER_CONSTANT, value=color)
            label = f'#{tid} T{team_id}'
            cv2.putText(crop, label, (3, 13), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, (255, 255, 255), 1, cv2.LINE_AA)
            crops.append(crop)

        if not crops:
            return

        per_row = min(8, len(crops))
        rows = []
        for i in range(0, len(crops), per_row):
            row = crops[i:i + per_row]
            while len(row) < per_row:
                row.append(np.zeros_like(crops[0]))
            rows.append(np.hstack(row))
        grid = np.vstack(rows)
        cv2.imshow('Debug: Player Crops', grid)
        cv2.waitKey(1)

    def _show_kmeans_scatter(self, features, kmeans_labels):
        """Mostra janela com scatter 2D (PCA) dos clusters do K-Means."""
        if not _MATPLOTLIB_OK or len(features) < 3:
            return
        try:
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(features)
            labels = np.array(kmeans_labels)

            fig, ax = plt.subplots(figsize=(6, 4))
            palette = ['#FF1744', '#2196F3']

            for cluster_id in [0, 1]:
                mask = labels == cluster_id
                if mask.any():
                    ax.scatter(reduced[mask, 0], reduced[mask, 1],
                               c=palette[cluster_id], label=f'Team {cluster_id}',
                               alpha=0.75, s=70, edgecolors='white', linewidths=0.4)

            if self.previous_centers is not None:
                try:
                    centers_2d = pca.transform(self.previous_centers)
                    for i, (cx, cy) in enumerate(centers_2d):
                        ax.scatter(cx, cy, c=palette[i], marker='X', s=220,
                                   edgecolors='black', linewidths=1.5, zorder=5)
                except Exception:
                    pass

            ax.set_title(f'K-Means (PCA 2D) — frame {self.debug_frame_count}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.25)
            plt.tight_layout()

            # Renderizar para numpy array e mostrar com cv2
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            img = np.asarray(canvas.buffer_rgba())
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            plt.close(fig)

            cv2.imshow('Debug: K-Means Clusters', img_bgr)
            cv2.waitKey(1)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def get_player_feature(self, frame, bbox, mask=None, mask_is_full_res=False):
        """
        Extrai histograma 3D (H+S+V) usando máscara de segmentação.

        Se a máscara estiver disponível, isola exatamente os pixels do jogador.
        Sem máscara, faz fallback ao método HSV com remoção de relva.

        Args:
            frame: Frame BGR completo
            bbox: Bounding box [x1, y1, x2, y2]
            mask: Máscara 2D (H, W) ou None.
            mask_is_full_res: Se True, salta o cv2.resize (otimização).

        Returns:
            np.ndarray: Vetor de 256 features (8×8×4) ou None se falhar
        """
        y1, y2 = int(bbox[1]), int(bbox[3])
        x1, x2 = int(bbox[0]), int(bbox[2])
        if y1 >= y2 or x1 >= x2:
            return None

        if mask is not None:
            if mask_is_full_res:
                mask_resized = mask
            else:
                h_frame, w_frame = frame.shape[:2]
                mask_resized = cv2.resize(mask.astype(np.float32), (w_frame, h_frame)) > 0.5

            player_crop_raw = frame[y1:y2, x1:x2]
            mask_crop = mask_resized[y1:y2, x1:x2]
            player_crop = player_crop_raw.copy()
            player_crop[~mask_crop] = 0

            if player_crop.size == 0:
                return None

            h, w = player_crop.shape[:2]
            if h >= 60:
                y_start = int(h * 0.10)
                y_end = int(h * 0.55)
            else:
                y_start = 0
                y_end = int(h * 0.55)
            player_crop = player_crop[y_start:y_end, :]
            mask_crop = mask_crop[y_start:y_end, :]

            mask_uint8 = mask_crop.astype(np.uint8) * 255

            if cv2.countNonZero(mask_uint8) < 10:
                return None

            hsv = cv2.cvtColor(player_crop, cv2.COLOR_BGR2HSV)

            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])
            grass_mask = cv2.inRange(hsv, lower_green, upper_green)
            non_grass_mask = cv2.bitwise_not(grass_mask)
            final_mask = cv2.bitwise_and(mask_uint8, non_grass_mask)

            if cv2.countNonZero(final_mask) < 10:
                final_mask = mask_uint8

            hist = cv2.calcHist([hsv], self.HIST_CHANNELS, final_mask,
                                self.HIST_BINS, self.HIST_RANGES)

        else:
            image = frame[y1:y2, x1:x2]
            h, w, _ = image.shape
            image = image[int(h * 0.10):int(h * 0.55), int(w * 0.2):int(w * 0.8)]

            if image.size == 0:
                return None

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_green = np.array([30, 40, 40])
            upper_green = np.array([90, 255, 255])
            grass_mask = cv2.inRange(hsv, lower_green, upper_green)
            non_grass_mask = cv2.bitwise_not(grass_mask)

            if cv2.countNonZero(non_grass_mask) < 10:
                return None

            hist = cv2.calcHist([hsv], self.HIST_CHANNELS, non_grass_mask,
                                self.HIST_BINS, self.HIST_RANGES)

        hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist.flatten()

    # ------------------------------------------------------------------
    # Main classification
    # ------------------------------------------------------------------

    def assign_team(self, frame, player_detections, masks=None):
        """
        Atribui equipas com base em histogramas extraídos das máscaras de segmentação.

        Args:
            frame: Frame BGR completo
            player_detections: sv.Detections com as deteções de jogadores
            masks: np.ndarray [N, H_mask, W_mask] ou None

        Returns:
            sv.Detections com 'team_id' (-1=Neutro, 0=Team A, 1=Team B)
        """
        self.debug_frame_count += 1

        if len(player_detections) == 0:
            player_detections.team_id = np.array([], dtype=int)
            return player_detections

        PLAYER_CLASS_ID = 2
        GOALKEEPER_CLASS_ID = 1

        # FASE 0: Histórico de classes (para override GK)
        if player_detections.tracker_id is not None and player_detections.class_id is not None:
            for i in range(len(player_detections)):
                track_id = int(player_detections.tracker_id[i])
                raw_class_id = int(player_detections.class_id[i])
                if track_id not in self.player_class_history:
                    self.player_class_history[track_id] = deque(maxlen=self.CLASS_HISTORY_LENGTH)
                self.player_class_history[track_id].append(raw_class_id)

        # Pré-processar máscaras em batch
        masks_full_res = None
        if masks is not None and len(masks) > 0:
            h_frame, w_frame = frame.shape[:2]
            m0 = masks[0]
            if m0.shape[0] == h_frame and m0.shape[1] == w_frame:
                masks_full_res = masks.astype(bool)
            else:
                masks_full_res = np.empty((len(masks), h_frame, w_frame), dtype=bool)
                for k in range(len(masks)):
                    masks_full_res[k] = cv2.resize(
                        masks[k].astype(np.float32), (w_frame, h_frame)
                    ) > 0.5

        # FASE 1: Extrair features de Players + recolher dados para debug
        player_features = []
        player_indices = []
        debug_bboxes = []
        debug_track_ids = []

        should_show = self.debug and (self.debug_frame_count % self.DEBUG_SHOW_EVERY == 0)

        for i in range(len(player_detections)):
            class_id = int(player_detections.class_id[i]) if player_detections.class_id is not None else -1

            if class_id == PLAYER_CLASS_ID:
                bbox = player_detections.xyxy[i]
                mask_i = masks_full_res[i] if masks_full_res is not None else None
                feature = self.get_player_feature(frame, bbox, mask=mask_i, mask_is_full_res=True)

                if feature is not None:
                    player_features.append(feature)
                    player_indices.append(i)
                    if should_show:
                        debug_bboxes.append(bbox)
                        tid = int(player_detections.tracker_id[i]) if player_detections.tracker_id is not None else -1
                        debug_track_ids.append(tid)

        # FASE 2: Treinar K-Means
        kmeans_labels_debug = []
        if len(player_features) > 1:
            init = self.previous_centers if self.previous_centers is not None else "k-means++"
            n_init = 1 if self.previous_centers is not None else 10
            current_kmeans = KMeans(n_clusters=2, init=init, n_init=n_init, max_iter=100)
            current_kmeans.fit(player_features)
            kmeans_labels_debug = current_kmeans.labels_.tolist()

            swap_votes = []
            for idx_in_list, i in enumerate(player_indices):
                if player_detections.tracker_id is None or i >= len(player_detections.tracker_id):
                    continue
                track_id = int(player_detections.tracker_id[i])
                if track_id in self.locked_player_teams:
                    locked_team = self.locked_player_teams[track_id]
                    predicted_team = current_kmeans.predict([player_features[idx_in_list]])[0]
                    swap_votes.append(predicted_team != locked_team)

            if len(swap_votes) >= 2:
                self.swap_labels = sum(swap_votes) > len(swap_votes) / 2
            elif self.previous_centers is not None:
                centers = current_kmeans.cluster_centers_
                dist_00 = np.linalg.norm(centers[0] - self.previous_centers[0])
                dist_01 = np.linalg.norm(centers[0] - self.previous_centers[1])
                dist_10 = np.linalg.norm(centers[1] - self.previous_centers[0])
                dist_11 = np.linalg.norm(centers[1] - self.previous_centers[1])
                self.swap_labels = (dist_00 + dist_11) > (dist_01 + dist_10)
            else:
                centers = current_kmeans.cluster_centers_
                self.swap_labels = np.sum(centers[0]) > np.sum(centers[1])

            centers = current_kmeans.cluster_centers_
            if self.swap_labels:
                self.previous_centers = np.array([centers[1], centers[0]])
            else:
                self.previous_centers = centers.copy()

            self.team_kmeans = current_kmeans

        team_ids = np.full(len(player_detections), -1, dtype=int)

        if not self.team_kmeans:
            player_detections.team_id = team_ids
            return player_detections

        # FASE 3: Classificar jogadores + Soft Lock + Votação Temporal
        CORRECTION_THRESHOLD = 0.85

        for idx_in_list, i in enumerate(player_indices):
            if player_detections.tracker_id is None or i >= len(player_detections.tracker_id):
                team_id = self.team_kmeans.predict([player_features[idx_in_list]])[0]
                if getattr(self, 'swap_labels', False):
                    team_id = 1 - team_id
                team_ids[i] = int(team_id)
                continue

            track_id = int(player_detections.tracker_id[i])
            current_team_id = self.team_kmeans.predict([player_features[idx_in_list]])[0]

            if getattr(self, 'swap_labels', False):
                current_team_id = 1 - current_team_id

            if track_id not in self.player_team_history:
                self.player_team_history[track_id] = deque(maxlen=self.HISTORY_LENGTH)
            self.player_team_history[track_id].append(current_team_id)

            if track_id in self.locked_player_teams:
                locked_team = self.locked_player_teams[track_id]
                opposite_team = 1 - locked_team

                if len(self.player_team_history[track_id]) >= self.LOCK_THRESHOLD:
                    opposite_count = list(self.player_team_history[track_id]).count(opposite_team)
                    opposite_ratio = opposite_count / len(self.player_team_history[track_id])

                    if opposite_ratio >= CORRECTION_THRESHOLD:
                        self.locked_player_teams[track_id] = opposite_team
                        self.player_team_history[track_id].clear()
                        team_ids[i] = opposite_team
                    else:
                        team_ids[i] = locked_team
                else:
                    team_ids[i] = locked_team
                continue

            if len(self.player_team_history[track_id]) >= self.LOCK_THRESHOLD:
                locked_team = Counter(self.player_team_history[track_id]).most_common(1)[0][0]
                self.locked_player_teams[track_id] = locked_team
                self.player_team_history[track_id].clear()
                final_team_id = locked_team
            else:
                final_team_id = Counter(self.player_team_history[track_id]).most_common(1)[0][0]

            team_ids[i] = int(final_team_id)

        # FASE 4: Override GK por consistência de classe
        if player_detections.tracker_id is not None:
            for i in range(len(player_detections)):
                track_id = int(player_detections.tracker_id[i])
                if track_id in self.player_class_history and len(self.player_class_history[track_id]) >= 30:
                    history = list(self.player_class_history[track_id])
                    gk_ratio = history.count(GOALKEEPER_CLASS_ID) / len(history)
                    if gk_ratio >= self.GK_CONSISTENCY_THRESHOLD and team_ids[i] in [0, 1]:
                        team_ids[i] = -1

        # DEBUG: terminal
        if self.debug:
            mask_count = np.sum([m is not None for m in (masks if masks is not None else [])])
            print(f"[TeamClassifierSeg] frame={self.debug_frame_count} players={len(player_indices)} "
                  f"masks={mask_count} locked={len(self.locked_player_teams)}")

        # DEBUG: janelas visuais (a cada DEBUG_SHOW_EVERY frames)
        if should_show and len(player_features) > 1:
            debug_team_ids = [team_ids[i] for i in player_indices]
            self._show_crop_grid(frame, debug_bboxes, masks_full_res, debug_track_ids, debug_team_ids)
            self._show_kmeans_scatter(player_features, kmeans_labels_debug)

        player_detections.team_id = team_ids
        return player_detections
