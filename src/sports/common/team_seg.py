from sklearn.cluster import KMeans
import numpy as np
import cv2
from collections import deque, Counter


class TeamClassifierSeg:
    """
    Classificador de equipas usando MASCARAS DE SEGMENTACAO + Histogramas + K-Means + Votação Temporal.

    Em vez de remover a relva por HSV (aproximação), usa a máscara de segmentação
    do yolo11m-seg para isolar exatamente os pixels do jogador — sem relva, sem fundo.

    Args:
        debug (bool): Se True, imprime informação de debug no terminal.
    """
    def __init__(self, debug=False):
        self.team_kmeans = None
        self.previous_centers = None
        self.player_team_history = {}
        self.player_class_history = {}
        self.locked_player_teams = {}
        self.HISTORY_LENGTH = 30
        self.CLASS_HISTORY_LENGTH = 60
        self.LOCK_THRESHOLD = 30
        self.GK_CONSISTENCY_THRESHOLD = 0.7

        self.HIST_BINS = [8, 8]
        self.HIST_RANGES = [0, 180, 0, 256]

        self.debug = debug

    def get_player_feature(self, frame, bbox, mask=None, mask_is_full_res=False):
        """
        Extrai histograma 2D (H+S) usando máscara de segmentação.

        Se a máscara estiver disponível, isola exatamente os pixels do jogador.
        Sem máscara, faz fallback ao método HSV (compatibilidade).

        Args:
            frame: Frame BGR completo
            bbox: Bounding box [x1, y1, x2, y2]
            mask: Máscara 2D (H, W) ou None. Se mask_is_full_res=True, assume-se
                  que já está no shape do frame e é booleana (evita resize por jogador).
            mask_is_full_res: Se True, salta o cv2.resize (otimização).

        Returns:
            np.ndarray: Vetor de 64 features ou None se falhar
        """
        y1, y2 = int(bbox[1]), int(bbox[3])
        x1, x2 = int(bbox[0]), int(bbox[2])
        if y1 >= y2 or x1 >= x2:
            return None

        if mask is not None:
            if mask_is_full_res:
                mask_resized = mask  # já booleana, já no shape do frame
            else:
                # Redimensionar máscara para dimensões do frame
                h_frame, w_frame = frame.shape[:2]
                mask_resized = cv2.resize(mask.astype(np.float32), (w_frame, h_frame)) > 0.5

            # Crop direto (sem copiar o frame inteiro)
            player_crop_raw = frame[y1:y2, x1:x2]
            mask_crop = mask_resized[y1:y2, x1:x2]
            player_crop = player_crop_raw.copy()
            player_crop[~mask_crop] = 0

            if player_crop.size == 0:
                return None

            # Foco na metade superior (zona da camisola)
            h, w = player_crop.shape[:2]
            player_crop = player_crop[0:int(h * 0.5), :]
            mask_crop = mask_crop[0:int(h * 0.5), :]

            mask_uint8 = mask_crop.astype(np.uint8) * 255

            if cv2.countNonZero(mask_uint8) < 10:
                return None

            hsv = cv2.cvtColor(player_crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], mask_uint8, self.HIST_BINS, self.HIST_RANGES)

        else:
            # Fallback: crop + remoção de relva por HSV (método original)
            image = frame[y1:y2, x1:x2]
            h, w, _ = image.shape
            image = image[0:int(h * 0.5), int(w * 0.2):int(w * 0.8)]

            if image.size == 0:
                return None

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_green = np.array([30, 40, 40])
            upper_green = np.array([90, 255, 255])
            grass_mask = cv2.inRange(hsv, lower_green, upper_green)
            non_grass_mask = cv2.bitwise_not(grass_mask)

            if cv2.countNonZero(non_grass_mask) < 10:
                return None

            hist = cv2.calcHist([hsv], [0, 1], non_grass_mask, self.HIST_BINS, self.HIST_RANGES)

        hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist.flatten()

    def assign_team(self, frame, player_detections, masks=None):
        """
        Atribui equipas com base em histogramas extraídos das máscaras de segmentação.

        Args:
            frame: Frame BGR completo
            player_detections: sv.Detections com as deteções de jogadores
            masks: np.ndarray [N, H_mask, W_mask] ou None — máscaras do yolo11m-seg,
                   em que N corresponde à mesma ordem de player_detections

        Returns:
            sv.Detections com 'team_id' (-1=Neutro, 0=Team A, 1=Team B)
        """
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

        # Pré-processar máscaras em batch (resize uma vez por frame, não por jogador)
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

        # FASE 1: Extrair features apenas de Players
        player_features = []
        player_indices = []

        for i in range(len(player_detections)):
            class_id = int(player_detections.class_id[i]) if player_detections.class_id is not None else -1

            if class_id == PLAYER_CLASS_ID:
                bbox = player_detections.xyxy[i]
                mask_i = masks_full_res[i] if masks_full_res is not None else None
                feature = self.get_player_feature(frame, bbox, mask=mask_i, mask_is_full_res=True)

                if feature is not None:
                    player_features.append(feature)
                    player_indices.append(i)

        # FASE 2: Treinar K-Means com histogramas de jogadores
        if len(player_features) > 1:
            init = self.previous_centers if self.previous_centers is not None else "k-means++"
            n_init = 1 if self.previous_centers is not None else 3
            current_kmeans = KMeans(n_clusters=2, init=init, n_init=n_init, max_iter=100)
            current_kmeans.fit(player_features)

            swap_votes = []
            for idx_in_list, i in enumerate(player_indices):
                if player_detections.tracker_id is None or i >= len(player_detections.tracker_id):
                    continue
                track_id = int(player_detections.tracker_id[i])
                if track_id in self.locked_player_teams:
                    locked_team = self.locked_player_teams[track_id]
                    feature = player_features[idx_in_list]
                    predicted_team = current_kmeans.predict([feature])[0]
                    swap_votes.append(predicted_team != locked_team)

            if len(swap_votes) >= 2:
                swap_count = sum(swap_votes)
                self.swap_labels = swap_count > len(swap_votes) / 2
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
        CORRECTION_THRESHOLD = 0.9

        for idx_in_list, i in enumerate(player_indices):
            if player_detections.tracker_id is None or i >= len(player_detections.tracker_id):
                feature = player_features[idx_in_list]
                team_id = self.team_kmeans.predict([feature])[0]
                if getattr(self, 'swap_labels', False):
                    team_id = 1 - team_id
                team_ids[i] = int(team_id)
                continue

            track_id = int(player_detections.tracker_id[i])
            feature = player_features[idx_in_list]
            current_team_id = self.team_kmeans.predict([feature])[0]

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
                    gk_count = history.count(GOALKEEPER_CLASS_ID)
                    gk_ratio = gk_count / len(history)
                    if gk_ratio >= self.GK_CONSISTENCY_THRESHOLD and team_ids[i] in [0, 1]:
                        team_ids[i] = -1

        if self.debug:
            mask_count = np.sum([m is not None for m in (masks if masks is not None else [])])
            print(f"[TeamClassifierSeg] players={len(player_indices)} masks={mask_count} locked={len(self.locked_player_teams)}")

        player_detections.team_id = team_ids
        return player_detections
