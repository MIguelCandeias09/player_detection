from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import cv2
import os
from collections import deque, Counter
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt

class TeamClassifier:
    """
    Classificador de equipas usando HISTOGRAMAS DE COR + K-Means Global + Votação Temporal.
    
    Vantagens do Histograma vs Cor Média:
    - Captura distribuição de cores (riscas, padrões, gradientes)
    - Robusto a variações de iluminação (usa apenas H+S, ignora V)
    - K-Means agrupa por "assinatura de cor" completa, não apenas "cor única"
    
    Args:
        debug (bool): Se True, guarda imagens de debug do processo de classificação
        debug_output_dir (str): Diretório para guardar imagens de debug
    """
    def __init__(self, debug=False, debug_output_dir="debug_team_output"):
        self.team_kmeans = None
        self.previous_centers = None   # Centros do K-Means do frame anterior (estabilização)
        self.player_team_history = {}  # Histórico {track_id: deque([0,1,0...])}
        self.player_class_history = {} # Histórico {track_id: deque([2,1,2...])} - Override GK por Consistência
        self.locked_player_teams = {}  # Bloqueio {track_id: 0 ou 1} - Equipa DEFINITIVA
        self.HISTORY_LENGTH = 30       # Memória de votação (30 frames ~1.2s)
        self.CLASS_HISTORY_LENGTH = 60 # Histórico estendido para deteção de GK consistente
        self.LOCK_THRESHOLD = 30       # Frames necessários para bloquear equipa
        self.GK_CONSISTENCY_THRESHOLD = 0.7  # 70% de frames como GK → Override
        
        # Configuração do Histograma 2D (H+S, ignora V para robustez à luz)
        self.HIST_BINS = [8, 8]        # [Hue bins, Saturation bins] → 64 features
        self.HIST_RANGES = [0, 180, 0, 256]  # OpenCV: Hue=0-180, Sat=0-256
        
        #  DEBUG: Configuração
        self.debug = debug
        self.debug_output_dir = debug_output_dir
        self.debug_frame_counter = 0
        self.debug_saved_frames = set()  # Para não guardar frames repetidos
        
        if self.debug:
            os.makedirs(self.debug_output_dir, exist_ok=True)
            print(f" DEBUG MODE ENABLED - Output dir: {self.debug_output_dir}")

    def _save_debug_images(self, frame, player_detections, player_features, player_indices, team_ids):
        """
        Guarda imagens de debug do processo de classificação.
        Só guarda nos frames 50, 100, 150 para não encher o disco.
        """
        # Só guarda em frames específicos (50, 100, 150, 200, ...)
        if self.debug_frame_counter not in [50, 100, 150, 200, 300] or self.debug_frame_counter in self.debug_saved_frames:
            return
        
        self.debug_saved_frames.add(self.debug_frame_counter)
        frame_id = self.debug_frame_counter
        
        print(f"\n🔬 DEBUG: A guardar imagens do frame {frame_id}...")
        
        try:
            # =====================================================================
            # 1. CROPS DAS CAMISOLAS
            # =====================================================================
            num_players = min(8, len(player_indices))
            if num_players > 0:
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                fig.suptitle(f'Frame {frame_id}: Extração ROI - Camisola (Top 50%, Center 60%)', fontsize=14, fontweight='bold')
                
                crops_data = []
                for idx in range(8):
                    ax = axes[idx // 4, idx % 4]
                    
                    if idx < num_players:
                        i = player_indices[idx]
                        bbox = player_detections.xyxy[i]
                        y1, y2 = int(bbox[1]), int(bbox[3])
                        x1, x2 = int(bbox[0]), int(bbox[2])
                        
                        full_crop = frame[y1:y2, x1:x2]
                        if full_crop.size > 0:
                            h, w = full_crop.shape[:2]
                            shirt_crop = full_crop[0:int(h*0.5), int(w*0.2):int(w*0.8)]
                            
                            display = full_crop.copy()
                            cv2.rectangle(display, (int(w*0.2), 0), (int(w*0.8), int(h*0.5)), (0, 255, 255), 2)
                            
                            ax.imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
                            tid = int(player_detections.tracker_id[i]) if player_detections.tracker_id is not None else idx
                            team = team_ids[i] if i < len(team_ids) else -1
                            ax.set_title(f'ID:{tid} | Equipa:{team}', fontsize=10)
                            
                            if shirt_crop.size > 0:
                                crops_data.append((shirt_crop, tid, team))
                    ax.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.debug_output_dir, f'1_crops_frame{frame_id}.png'), dpi=150)
                plt.close()
                print(f"   ✅ 1_crops_frame{frame_id}.png")
                
                # =====================================================================
                # 2. MÁSCARAS HSV
                # =====================================================================
                if crops_data:
                    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                    fig.suptitle(f'Frame {frame_id}: Máscara HSV - Remoção de Relva (Verde)', fontsize=14, fontweight='bold')
                    
                    hsv_data = []
                    for idx in range(8):
                        ax = axes[idx // 4, idx % 4]
                        
                        if idx < len(crops_data):
                            shirt, tid, team = crops_data[idx]
                            hsv = cv2.cvtColor(shirt, cv2.COLOR_BGR2HSV)
                            
                            lower_green = np.array([30, 40, 40])
                            upper_green = np.array([90, 255, 255])
                            grass_mask = cv2.inRange(hsv, lower_green, upper_green)
                            non_grass = cv2.bitwise_not(grass_mask)
                            
                            result = cv2.bitwise_and(shirt, shirt, mask=non_grass)
                            hsv_data.append((hsv, non_grass, tid, team))
                            
                            combined = np.hstack([shirt, result])
                            ax.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
                            pct = cv2.countNonZero(non_grass) / non_grass.size * 100
                            ax.set_title(f'ID:{tid} | {pct:.0f}% útil', fontsize=10)
                        ax.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.debug_output_dir, f'2_hsv_masks_frame{frame_id}.png'), dpi=150)
                    plt.close()
                    print(f"   ✅ 2_hsv_masks_frame{frame_id}.png")
                    
                    # =====================================================================
                    # 3. HISTOGRAMAS 2D
                    # =====================================================================
                    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                    fig.suptitle(f'Frame {frame_id}: Histogramas 2D (Hue × Saturation) - 64 Features', fontsize=14, fontweight='bold')
                    
                    for idx in range(8):
                        ax = axes[idx // 4, idx % 4]
                        
                        if idx < len(hsv_data):
                            hsv, mask, tid, team = hsv_data[idx]
                            hist = cv2.calcHist([hsv], [0, 1], mask, [8, 8], [0, 180, 0, 256])
                            hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                            
                            im = ax.imshow(hist.T, origin='lower', aspect='auto', 
                                          extent=[0, 180, 0, 256], cmap='hot')
                            ax.set_xlabel('Hue')
                            ax.set_ylabel('Saturation')
                            color = 'red' if team == 0 else ('blue' if team == 1 else 'gray')
                            ax.set_title(f'ID:{tid} | Eq:{team}', fontsize=10, color=color)
                            plt.colorbar(im, ax=ax, fraction=0.046)
                        else:
                            ax.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.debug_output_dir, f'3_histograms_frame{frame_id}.png'), dpi=150)
                    plt.close()
                    print(f"   ✅ 3_histograms_frame{frame_id}.png")
                
                # =====================================================================
                # 4. K-MEANS CLUSTERING (PCA Visualization)
                # =====================================================================
                if len(player_features) >= 4 and self.team_kmeans is not None:
                    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                    fig.suptitle(f'Frame {frame_id}: Clustering K-Means (k=2) - Projeção PCA', fontsize=14, fontweight='bold')
                    
                    features = np.array(player_features)
                    pca = PCA(n_components=2)
                    features_2d = pca.fit_transform(features)
                    centers_2d = pca.transform(self.team_kmeans.cluster_centers_)
                    
                    colors_map = {0: '#FF1744', 1: '#2196F3', -1: '#888888'}
                    
                    for idx, (x, y) in enumerate(features_2d):
                        if idx < len(player_indices):
                            i = player_indices[idx]
                            team = team_ids[i] if i < len(team_ids) else -1
                            c = colors_map.get(team, '#888888')
                            ax.scatter(x, y, c=c, s=200, edgecolors='black', linewidths=2, zorder=3)
                            tid = int(player_detections.tracker_id[i]) if player_detections.tracker_id is not None else idx
                            ax.annotate(f'{tid}', (x, y), textcoords="offset points", 
                                       xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')
                    
                    ax.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                              c=['#FF1744', '#2196F3'], s=400, marker='X', 
                              edgecolors='black', linewidths=3, zorder=4, label='Centróides')
                    
                    ax.set_xlabel('Componente Principal 1', fontsize=12)
                    ax.set_ylabel('Componente Principal 2', fontsize=12)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Add legend
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor='#FF1744', label='Equipa 0 (Vermelho)'),
                        Patch(facecolor='#2196F3', label='Equipa 1 (Azul)'),
                    ]
                    ax.legend(handles=legend_elements, loc='upper right')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.debug_output_dir, f'4_kmeans_pca_frame{frame_id}.png'), dpi=150)
                    plt.close()
                    print(f"   ✅ 4_kmeans_pca_frame{frame_id}.png")
                
                # =====================================================================
                # 5. FRAME FINAL ANOTADO
                # =====================================================================
                fig_final = frame.copy()
                colors_bgr = {0: (68, 23, 255), 1: (243, 150, 33), -1: (128, 128, 128)}  # BGR
                
                for idx, i in enumerate(player_indices):
                    if i >= len(player_detections.xyxy):
                        continue
                    bbox = player_detections.xyxy[i]
                    x1, y1, x2, y2 = map(int, bbox)
                    team = team_ids[i] if i < len(team_ids) else -1
                    color = colors_bgr.get(team, (128, 128, 128))
                    
                    cv2.rectangle(fig_final, (x1, y1), (x2, y2), color, 3)
                    tid = int(player_detections.tracker_id[i]) if player_detections.tracker_id is not None else idx
                    label = f"ID:{tid} T:{team}"
                    cv2.putText(fig_final, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                cv2.imwrite(os.path.join(self.debug_output_dir, f'5_result_frame{frame_id}.png'), fig_final)
                print(f"   ✅ 5_result_frame{frame_id}.png")
                
            print(f"🔬 DEBUG: Imagens guardadas em {self.debug_output_dir}/")
            
        except Exception as e:
            print(f"⚠️ DEBUG Error: {e}")

    def get_player_feature(self, frame, bbox):
        """
        Extrai HISTOGRAMA 2D de cor da camisola (H+S) em vez de cor média.
        
        Pipeline:
        1. Crop da camisola (top 50%, center 60%)
        2. Máscara HSV para remover relva (verde)
        3. Calcula histograma 2D nos canais Hue + Saturation
        4. Normaliza e flatten para vetor de 64 features
        
        Returns:
            np.ndarray: Histograma normalizado (64 features) ou None se falhar
        """
        # Proteção contra bbox inválida
        y1, y2 = int(bbox[1]), int(bbox[3])
        x1, x2 = int(bbox[0]), int(bbox[2])
        if y1 >= y2 or x1 >= x2:
            return None

        image = frame[y1:y2, x1:x2]
        
        # CROP: Focar na camisola (metade superior, centro 60%)
        h, w, _ = image.shape
        image = image[0:int(h*0.5), int(w*0.2):int(w*0.8)]
        
        if image.size == 0:
            return None

        # FILTRO DE RELVA: Máscara HSV para remover verde
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([90, 255, 255])
        grass_mask = cv2.inRange(hsv, lower_green, upper_green)
        non_grass_mask = cv2.bitwise_not(grass_mask)
        
        # Verificar se há pixels válidos (camisola visível)
        if cv2.countNonZero(non_grass_mask) < 10:
            return None  # Só havia relva, sem camisola

        # HISTOGRAMA 2D: Hue + Saturation (ignora Value para robustez à luz)
        hist = cv2.calcHist(
            [hsv],                  # Imagem HSV
            [0, 1],                 # Canais: H=0, S=1 (ignora V=2)
            non_grass_mask,         # Máscara (apenas pixels não-verdes)
            self.HIST_BINS,         # [8, 8] = 64 bins
            self.HIST_RANGES        # H:[0,180], S:[0,256]
        )
        
        # NORMALIZAR: 0 a 1 (para que magnitudes não dominem o K-Means)
        hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        
        # FLATTEN: Converter matriz 8x8 → vetor 1D de 64 features
        feature_vector = hist.flatten()
        
        return feature_vector

    def assign_team(self, frame, player_detections):
        """
        🛑 FILTRAGEM RADICAL: Apenas Players (class_id=2) contam para classificação.
        
        Pipeline:
        - FASE 0: Estabiliza class_id via votação temporal (corrige oscilações YOLO)
        - FASE 1: Extrai histogramas APENAS de Players estabilizados
        - FASE 2: Treina K-Means com histogramas limpos
        - FASE 3: Atribui equipas com Soft Lock + Autocorreção
        
        Args:
            frame: Video frame (BGR)
            player_detections: sv.Detections com TODAS as deteções
            
        Returns:
            sv.Detections com 'team_id' (-1=Neutro, 0=Team A, 1=Team B)
        """
        if len(player_detections) == 0:
            player_detections.team_id = np.array([], dtype=int)
            return player_detections
        
        PLAYER_CLASS_ID = 2  # Player
        GOALKEEPER_CLASS_ID = 1  # Goalkeeper
        
        # ============================================================================
        # FASE 0: 📊 HISTÓRICO DE CLASSES (Para Override GK Consistente)
        # ============================================================================
        # Guarda histórico LONGO (60 frames) para detetar GK com cor semelhante a equipa.
        # NÃO sobrescreve class_id aqui - apenas acumula dados para decisão posterior.
        
        if player_detections.tracker_id is not None and player_detections.class_id is not None:
            for i in range(len(player_detections)):
                track_id = int(player_detections.tracker_id[i])
                raw_class_id = int(player_detections.class_id[i])
                
                # Adiciona classe atual ao histórico (deque com 60 frames)
                if track_id not in self.player_class_history:
                    self.player_class_history[track_id] = deque(maxlen=self.CLASS_HISTORY_LENGTH)
                
                self.player_class_history[track_id].append(raw_class_id)
        
        # ============================================================================
        # FASE 1: 🔍 Recolher features (histogramas) APENAS de Players ESTABILIZADOS
        # ============================================================================
        player_features = []
        player_indices = []  # Índices ESTRITAMENTE de jogadores
        
        for i in range(len(player_detections)):
            class_id = int(player_detections.class_id[i]) if player_detections.class_id is not None else -1
            
            # 🛑 BLOQUEIO TOTAL: Se não é jogador, não existe para o K-Means
            if class_id == PLAYER_CLASS_ID:
                bbox = player_detections.xyxy[i]
                feature = self.get_player_feature(frame, bbox)
                
                if feature is not None:
                    player_features.append(feature)
                    player_indices.append(i)

        # ============================================================================
        # FASE 2: 🧠 Treinar K-Means APENAS com histogramas de jogadores (zero ruído)
        # ============================================================================
        if len(player_features) > 1:
            current_kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, max_iter=300)
            current_kmeans.fit(player_features)
            
            # 🎯 NOVA ESTRATÉGIA: Usar jogadores BLOQUEADOS para determinar swap
            # Se já temos jogadores com equipa fixa, usamos eles como referência
            # Isto é MUITO mais estável que comparar centros de K-Means
            
            swap_votes = []  # Lista de votos: True = precisa swap, False = não precisa
            
            for idx_in_list, i in enumerate(player_indices):
                if player_detections.tracker_id is None or i >= len(player_detections.tracker_id):
                    continue
                    
                track_id = int(player_detections.tracker_id[i])
                
                # Se este jogador já está BLOQUEADO, usa-o como referência
                if track_id in self.locked_player_teams:
                    locked_team = self.locked_player_teams[track_id]
                    feature = player_features[idx_in_list]
                    predicted_team = current_kmeans.predict([feature])[0]
                    
                    # Se a predição difere do bloqueio, vota por swap
                    swap_votes.append(predicted_team != locked_team)
            
            # 🎯 DECISÃO: Se maioria dos bloqueados indica swap, então swap
            if len(swap_votes) >= 2:  # Precisa de pelo menos 2 jogadores bloqueados
                swap_count = sum(swap_votes)
                self.swap_labels = swap_count > len(swap_votes) / 2
            elif self.previous_centers is not None:
                # Fallback: Usa comparação de centros (método antigo) se poucos bloqueados
                centers = current_kmeans.cluster_centers_
                dist_00 = np.linalg.norm(centers[0] - self.previous_centers[0])
                dist_01 = np.linalg.norm(centers[0] - self.previous_centers[1])
                dist_10 = np.linalg.norm(centers[1] - self.previous_centers[0])
                dist_11 = np.linalg.norm(centers[1] - self.previous_centers[1])
                
                if (dist_00 + dist_11) > (dist_01 + dist_10):
                    self.swap_labels = True
                else:
                    self.swap_labels = False
            else:
                # Primeira iteração: Usa critério de escuridão (fallback)
                centers = current_kmeans.cluster_centers_
                if np.sum(centers[0]) > np.sum(centers[1]):
                    self.swap_labels = True
                else:
                    self.swap_labels = False
            
            # Guarda centros para próximo frame (fallback)
            centers = current_kmeans.cluster_centers_
            if self.swap_labels:
                self.previous_centers = np.array([centers[1], centers[0]])
            else:
                self.previous_centers = centers.copy()
            
            self.team_kmeans = current_kmeans

        # Inicializa TODOS com -1 (neutro)
        team_ids = np.full(len(player_detections), -1, dtype=int)

        # Se não conseguiu treinar (< 2 jogadores), retorna tudo como neutro
        if not self.team_kmeans:
            player_detections.team_id = team_ids
            return player_detections

        # ============================================================================
        # FASE 3: ⚽ Classificar APENAS os jogadores + Soft Lock (Autocorreção) + Votação Temporal
        # ============================================================================
        CORRECTION_THRESHOLD = 0.9  # 90% de certeza para corrigir bloqueio (ex: 27/30 frames)
        
        for idx_in_list, i in enumerate(player_indices):
            if player_detections.tracker_id is None or i >= len(player_detections.tracker_id):
                # Sem tracker_id → Usa K-Means direto
                feature = player_features[idx_in_list]
                team_id = self.team_kmeans.predict([feature])[0]
                if getattr(self, 'swap_labels', False):
                    team_id = 1 - team_id
                team_ids[i] = int(team_id)
                continue
            
            track_id = int(player_detections.tracker_id[i])
            
            # 🧠 K-MEANS: Calcula equipa ATUAL (mesmo se bloqueado - para autocorreção)
            feature = player_features[idx_in_list]
            current_team_id = self.team_kmeans.predict([feature])[0]
            
            # Aplica swap se necessário
            if getattr(self, 'swap_labels', False):
                current_team_id = 1 - current_team_id

            # 📊 VOTAÇÃO TEMPORAL: SEMPRE adiciona ao histórico (bloqueado ou não)
            if track_id not in self.player_team_history:
                self.player_team_history[track_id] = deque(maxlen=self.HISTORY_LENGTH)
            
            self.player_team_history[track_id].append(current_team_id)
            
            # 🔒 VERIFICAÇÃO DE BLOQUEIO: Se já está bloqueado, verifica se precisa correção
            if track_id in self.locked_player_teams:
                locked_team = self.locked_player_teams[track_id]
                opposite_team = 1 - locked_team
                
                # Verifica se histórico tem evidência forte da equipa OPOSTA
                if len(self.player_team_history[track_id]) >= self.LOCK_THRESHOLD:
                    opposite_count = list(self.player_team_history[track_id]).count(opposite_team)
                    opposite_ratio = opposite_count / len(self.player_team_history[track_id])
                    
                    # 🔄 AUTOCORREÇÃO: Se >90% dos frames indicam equipa oposta, corrige bloqueio
                    if opposite_ratio >= CORRECTION_THRESHOLD:
                        self.locked_player_teams[track_id] = opposite_team
                        self.player_team_history[track_id].clear()
                        team_ids[i] = opposite_team
                    else:
                        # Mantém bloqueio atual
                        team_ids[i] = locked_team
                else:
                    # Histórico ainda curto, mantém bloqueio
                    team_ids[i] = locked_team
                continue
            
            # 🔐 LÓGICA DE BLOQUEIO INICIAL: Se não está bloqueado e atingiu threshold
            if len(self.player_team_history[track_id]) >= self.LOCK_THRESHOLD:
                # Calcula MODA (equipa mais frequente)
                locked_team = Counter(self.player_team_history[track_id]).most_common(1)[0][0]
                self.locked_player_teams[track_id] = locked_team
                # Limpa histórico (inicia fase de autocorreção)
                self.player_team_history[track_id].clear()
                final_team_id = locked_team
            else:
                # Ainda em fase de votação inicial → Usa MODA atual
                final_team_id = Counter(self.player_team_history[track_id]).most_common(1)[0][0]

            team_ids[i] = int(final_team_id)
        
        # ============================================================================
        # FASE 4: 🥅 OVERRIDE GK POR CONSISTÊNCIA DE CLASSE
        # ============================================================================
        # Edge Case: GK com cor similar a uma equipa é classificado erradamente.
        # Solução: Se YOLO diz "GK" consistentemente (>70% em 60 frames), força team_id=-1
        
        if player_detections.tracker_id is not None:
            for i in range(len(player_detections)):
                track_id = int(player_detections.tracker_id[i])
                
                # Verifica se tem histórico de classes acumulado
                if track_id in self.player_class_history and len(self.player_class_history[track_id]) >= 30:
                    history = list(self.player_class_history[track_id])
                    
                    # Conta quantas vezes YOLO disse "Goalkeeper (1)"
                    gk_count = history.count(GOALKEEPER_CLASS_ID)
                    gk_ratio = gk_count / len(history)
                    
                    # Se >70% das classificações YOLO são GK (1) E cor diz que é jogador (0/1)
                    if gk_ratio >= self.GK_CONSISTENCY_THRESHOLD and team_ids[i] in [0, 1]:
                        # OVERRIDE: Força classificação como GK (team_id=-1)
                        team_ids[i] = -1

        # 🔬 DEBUG: Guardar imagens de debug se ativado
        if self.debug:
            self.debug_frame_counter += 1
            self._save_debug_images(frame, player_detections, player_features, player_indices, team_ids)

        # Retorna: Players com 0/1, GK/Referee com -1
        player_detections.team_id = team_ids
        return player_detections