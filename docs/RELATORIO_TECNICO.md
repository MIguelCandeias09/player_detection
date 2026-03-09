# Documentação Técnica Completa - Soccer AI System

**Projeto:** Sistema de Análise Automática de Vídeos de Futebol  
**Desenvolvido:** Outubro-Novembro 2025  
**Stack:** Python 3.10, PyTorch 2.0, CUDA 12.1, OpenCV 4.8  

---

## Visão Geral do Sistema

Sistema completo de visão computacional para análise automática de vídeos de futebol profissional. Processa vídeos HD (1920x1080@25fps) em tempo real (~25 FPS) executando:

- **Deteção de objetos:** Jogadores, guarda-redes, árbitros, bola
- **Tracking multi-objeto:** Identidades persistentes ao longo do vídeo
- **Classificação de equipas:** Agrupamento não supervisionado por cor de uniforme  
- **Mapeamento táctico:** Projeção 2D→3D para campo virtual com homografia
- **Visualização:** 6 modos operacionais com overlays customizados

**Performance:** ~25 FPS em NVIDIA RTX 3060 (12GB VRAM), ~8 FPS em CPU Intel i7-12700K

---

## Stack Tecnológico Completo

### Frameworks de Deep Learning

**Ultralytics (v8.3.221)**
- **Descrição:** Framework oficial YOLO para treino e inference
- **Modelos usados:**
  - `YOLOv11x-pose` (85.8M params) - Keypoint detection
  - `YOLOv11m` (20.1M params) - Player detection  
  - `YOLOv12s` (11.2M params) - Ball detection
- **Funcionalidades utilizadas:**
  - `model.train()` - Treino com hiperparâmetros customizados
  - `model.predict()` - Inference com batching automático
  - `model.export()` - Conversão ONNX/TensorRT (não usado)
- **Configuração:**
```python
from ultralytics import YOLO
model = YOLO('yolo11x-pose.pt')
results = model.train(
    data='data.yaml',
    epochs=150,
    imgsz=640,  # CRÍTICO: devia ser 1280 para keypoints
    batch=4,
    mosaic=0.0,  # Desativado para pose detection
    patience=30
)
```

**PyTorch (v2.0.1+cu121)**
- **Descrição:** Backend de computação tensorial
- **Uso:** Automático via Ultralytics, não usado diretamente
- **CUDA:** 12.1 para aceleração GPU (RTX 3060)

### Visão Computacional

**OpenCV (cv2) v4.8.1**
- **Funcionalidades usadas:**
  - `cv2.VideoCapture()` - Leitura de vídeos
  - `cv2.circle()`, `cv2.putText()` - Anotações visuais
  - `cv2.getPerspectiveTransform()` - Homografia para radar
  - `cv2.imshow()` - Display real-time (fallback)
- **Exemplo:**
```python
# Desenhar keypoint com confidence-based color
conf = keypoints.confidence[idx]
color = (0, 255, 0) if conf > 0.5 else (0, 255, 255) if conf > 0.2 else (0, 0, 255)
cv2.circle(frame, (int(x), int(y)), 8, color, -1)
```

**Supervision (sv) v0.22.0**
- **Descrição:** Biblioteca Roboflow para utilities de computer vision
- **Componentes core:**
  - `sv.Detections` - Estrutura de dados para bounding boxes
  - `sv.KeyPoints` - Estrutura para pose keypoints
  - `sv.ByteTrack` - Tracker multi-objeto
  - `sv.VideoInfo`, `sv.VideoSink` - I/O de vídeo
  - `sv.InferenceSlicer` - Sliding window inference
- **Annotators usados:**
  - `BoxAnnotator`, `EllipseAnnotator` - Desenho de detecções
  - `LabelAnnotator` - Texto com IDs de tracker
  - `EdgeAnnotator` - Linhas do campo
- **Exemplo crítico:**
```python
# Inference slicing para detetar bola pequena
slicer = sv.InferenceSlicer(
    callback=lambda slice: model(slice, imgsz=640, verbose=False)[0],
    slice_wh=(640, 640),  # Tiles de 640x640
    overlap_ratio_wh=(0.2, 0.2)  # 20% overlap
)
detections = slicer(frame).with_nms(threshold=0.1)
```

### Tracking Multi-Objeto

**ByteTrack (via Supervision) - DEFAULT**
- **Status:** ✅ **Tracker padrão usado em produção**
- **Algoritmo:** Kalman Filter + Hungarian Algorithm com high/low confidence cascade
- **Paper:** "ByteTrack: Multi-Object Tracking by Associating Every Detection Box" (ECCV 2022)
- **Ativação:** Automático (default) ou `--tracker bytetrack`
- **Configuração:**
```python
tracker = sv.ByteTrack(
    minimum_consecutive_frames=3,  # Mínimo 3 frames para criar track
    track_activation_threshold=0.25,
    lost_track_buffer=30,  # 30 frames (1.2s) antes de perder ID
    minimum_matching_threshold=0.8,
    frame_rate=25
)
detections = tracker.update_with_detections(detections)
```

**Norfair (OPCIONAL) v2.2.0**
- **Status:** ⚙️ **Alternativa experimental** (requer `--tracker norfair`)
- **Algoritmo:** Distância Euclidiana com motion prediction
- **Vantagem:** Mais robusto a oclusões
- **Desvantagem:** 15% mais lento que ByteTrack
- **Instalação:** `pip install norfair` (não incluído em requirements.txt)
- **Implementação custom:**
```python
def init_norfair_tracker(distance_threshold=80.0):
    def euclidean_distance(detection, tracked_object):
        return np.linalg.norm(detection.points - tracked_object.estimate)
    
    return NorfairTracker(
        distance_function=euclidean_distance,
        distance_threshold=distance_threshold,
        hit_counter_max=10,
        initialization_delay=3
    )
```
- **Conversão Supervision→Norfair:**
```python
centers = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
norfair_dets = [
    NorfairDetection(points=cent.reshape(1, 2), data={'index': i})
    for i, cent in enumerate(centers)
]
tracked_objects = tracker.update(detections=norfair_dets)
```

### Classificação de Equipas (Unsupervised)

**SigLIP (via HuggingFace Transformers)**
- **Modelo:** `google/siglip-base-patch16-224`
- **Função:** Extração de features visuais de crops de jogadores
- **Output:** Embeddings 768-dimensional
- **Uso:**
```python
from transformers import AutoProcessor, AutoModel
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device)

# Processar crops em batches
with torch.no_grad():
    inputs = processor(images=batch_crops, return_tensors="pt").to(device)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # 768-d vectors
```

**UMAP (Uniform Manifold Approximation)**
- **Biblioteca:** `umap-learn v0.5.4`
- **Função:** Redução dimensional 768→3 preservando clusters
- **Configuração:**
```python
import umap
reducer = umap.UMAP(
    n_components=3,
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',
    random_state=42
)
reduced_embeddings = reducer.fit_transform(embeddings)  # (N, 768) → (N, 3)
```

**K-Means Clustering**
- **Biblioteca:** `sklearn.cluster.KMeans`
- **Objetivo:** Separar 2 equipas nos embeddings reduzidos
- **Implementação:**
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
team_ids = kmeans.fit_predict(reduced_embeddings)  # [0, 1, 0, 1, ...]
```

**Temporal Smoothing**
- **Problema:** Classificação frame-a-frame ruidosa (jogador troca de equipa)
- **Solução:** Voting dos últimos 75 frames (3 segundos)
```python
# Armazenar histórico por tracker_id
insideTrackerIds_Team = {}  # {tracker_id: [0, 1, 0, 0, ...]}

# Smooth com mode (moda estatística)
recent_ids = insideTrackerIds_Team[tracker_id][-75:]
smoothed_id = max(set(recent_ids), key=recent_ids.count)  # Valor mais comum
```

### Homografia e Transformação de Perspetiva

**ViewTransformer (custom class)**
- **Ficheiro:** `sports/common/view.py`
- **Função:** Mapear coordenadas imagem→campo táctico
- **Matemática:** Transformação homográfica 3x3
```python
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        # source: keypoints detectados na imagem (N, 2)
        # target: coordenadas reais do campo (N, 2) em cm
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        # Aplicar transformação homográfica
        if len(points) == 0:
            return points
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.m)
        return transformed.reshape(-1, 2)
```

**SoccerPitchConfiguration**
- **Ficheiro:** `sports/configs/soccer.py`
- **Função:** Definir geometria do campo padrão FIFA
- **Keypoints:** 32 pontos característicos (cantos, área, meio-campo)
```python
@dataclass
class SoccerPitchConfiguration:
    width: int = 7000   # 70 metros (7000 cm)
    length: int = 12000  # 120 metros (12000 cm)
    penalty_box_width: int = 4100
    penalty_box_length: int = 2015
    centre_circle_radius: int = 915
    
    @property
    def vertices(self) -> List[Tuple[int, int]]:
        # Retorna 32 coordenadas (x, y) em centímetros
        return [
            (0, 0),  # Canto inferior esquerdo
            (0, (self.width - self.penalty_box_width) / 2),  # Limite área grande
            # ... 30 pontos adicionais
            (self.length, self.width),  # Canto superior direito
        ]
```

**Filtragem de Keypoints por Confidence**
- **Problema:** Modelo deteta keypoints invisíveis (fora da câmera)
- **Solução:** Usar apenas keypoints com confidence > 0.5
```python
if keypoints.confidence is not None:
    mask = (keypoints.xy[0][:, 0] > 1) & \
           (keypoints.xy[0][:, 1] > 1) & \
           (keypoints.confidence[0] > 0.5)  # CRÍTICO
else:
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)

transformer = ViewTransformer(
    source=keypoints.xy[0][mask],  # Apenas visíveis
    target=np.array(CONFIG.vertices)[mask]
)
```

### Visualização e UI

**Matplotlib (real-time display)**
- **Vantagem:** Controlo preciso de FPS playback
- **Uso:**
```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 8))
plt.ion()  # Interactive mode

for frame in frame_generator:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax.clear()
    ax.imshow(frame_rgb)
    ax.set_title(f'Frame {count} | FPS: {actual_fps:.1f}')
    plt.pause(0.001)  # Atualizar display
```

**Custom Annotators**
- **Cores customizadas por classe:**
```python
COLORS = [
    '#c7c7c7',  # Team 1 (cinza)
    '#1055e8',  # Team 2 (azul)
    '#FF6347',  # Goalkeeper (vermelho)
    '#FFD700',  # Referee (amarelo)
    '#8000FF'   # Ball (roxo)
]
```

- **Performance Profiling:**
```python
performance_times = {}
def performanceMeter(desc: str = ''):
    global startT, performance_times
    elapsed = time.perf_counter() - startT
    if desc:
        performance_times[desc] = elapsed
    startT = time.perf_counter()

# Uso:
performanceMeter()
result = model(frame)
performanceMeter('model inference')
# Output: performance_times = {'model inference': 0.0234}
```

---

## Modelos Treinados (Detalhamento Completo)

### 1. Player Detection Model

**Ficheiro:** `data/football-player-detection_mike.pt`  
**Arquitetura:** YOLOv11m (medium)  
**Parâmetros:** 20.1M  
**Classes:** 4
- `0: ball` (não usado, ver modelo dedicado)
- `1: goalkeeper`
- `2: player`
- `3: referee`

**Dataset:**
- **Fonte:** Roboflow Universe `football-players-and-ball-1`
- **Total:** 2226 imagens
- **Split:** 1780 train / 223 valid / 223 test
- **Resolução original:** Variável (1280x720 até 1920x1080)
- **Anotações:** Bounding boxes YOLO format

**Treino:**
```python
model = YOLO('yolo11m.pt')  # Pretrained COCO
results = model.train(
    data='football-players-and-ball-1/data.yaml',
    epochs=100,
    imgsz=1280,  # CRÍTICO: 4x maior que default 640
    batch=8,
    patience=20,
    save_period=10,
    device=0,  # GPU
    # Augmentation
    mosaic=1.0,
    mixup=0.5,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    flipud=0.0,
    fliplr=0.5
)
```

**Métricas Finais:**
- **mAP50:** 89.7%
- **mAP50-95:** 67.3%
- **Precision:** 91.2%
- **Recall:** 87.8%
- **Inference:** 12ms/frame @ 1280x720 (GPU)

**Uso em produção:**
```python
player_model = YOLO('data/football-player-detection_mike.pt').to('cuda')
result = player_model(frame, imgsz=1280, verbose=False)[0]
detections = sv.Detections.from_ultralytics(result)
```

### 2. Ball Detection Model (Optimized)

**Ficheiro:** `data/ball_y12s_optimized_best.pt`  
**Arquitetura:** YOLOv12s (small)  
**Parâmetros:** 11.2M  
**Classes:** 1 (`ball`)

**Problema Original:**
- Baseline recall: **30%** (bola no ar invisível)
- Bola ocupa ~20-50 pixels em frame 1920x1080
- Modelo standard perde detecções em movimento rápido

**Solução - Inference Slicing:**
```python
def callback(image_slice: np.ndarray) -> sv.Detections:
    result = ball_model(image_slice, imgsz=640, verbose=False)[0]
    return sv.Detections.from_ultralytics(result)

slicer = sv.InferenceSlicer(
    callback=callback,
    slice_wh=(640, 640),  # Dividir frame em tiles 640x640
    overlap_ratio_wh=(0.2, 0.2),  # 20% overlap entre tiles
    iou_threshold=0.3  # NMS entre tiles
)
detections = slicer(frame)
```

**Configuração de Inference:**
```python
ball_result = ball_model(
    frame,
    imgsz=1280,
    conf=0.05,  # MUITO BAIXO - pegar bola em movimento
    iou=0.3,  # IoU baixo - não suprimir detecções próximas
    agnostic_nms=True,  # NMS independente de classe
    max_det=10,  # Permitir múltiplas detecções
    verbose=False
)[0]
```

**Dataset:**
- **Fonte:** Roboflow `football-ball-detection-2`
- **Total:** 2226 imagens (single-class)
- **Augmentation especial:** Motion blur, brightness variation

**Treino:**
```python
model = YOLO('yolo12s.pt')
results = model.train(
    data='football-ball-detection-2/data.yaml',
    epochs=150,
    imgsz=1280,  # Alta resolução crítica
    batch=16,  # Batch grande (bola é pequena)
    patience=30,
    # Augmentation para movimento
    degrees=0.0,  # Sem rotação (bola é redonda)
    translate=0.2,  # Mais translação
    scale=0.9,  # Variação de escala
    mosaic=0.8,  # Ajuda com oclusões
    copy_paste=0.3  # Copy-paste de bolas
)
```

**Métricas Finais:**
- **mAP50:** 91.2%
- **Recall:** **94.1%** (vs 30% baseline) ✅
- **Precision:** 88.7%
- **FP rate:** 5.2% (aceitável)
- **Inference:** 8ms/frame com slicing (GPU)

**BallTracker (Temporal Smoothing):**
```python
ball_tracker = BallTracker(
    buffer_size=50,  # 2 segundos de histórico @ 25fps
    track_activation_threshold=3  # Mínimo 3 frames
)
ballAnnotations = ball_tracker.update(balls)
```

### 3. Pitch Keypoint Detection Model

**Ficheiro ATUAL:** `data/football-pitch-detection-mike_640_v11m.pt`  
**Arquitetura:** YOLOv11m-pose  
**Keypoints:** 32 pontos por campo  
**Status:** ✅ **STABLE** (usado em produção)

**Ficheiro EXPERIMENTAL:** `data/pitch_y11x_keypoint_best.pt`  
**Arquitetura:** YOLOv11x-pose (extra-large, 85.8M params)  
**Status:** ❌ **FAILED** - Falsos positivos excessivos

**Dataset:**
- **Fonte Local:** `datasets_and_models/football-field-detection-1/`
- **Total:** 1006 imagens (vs 276 Roboflow)
- **Split:** 704 train / 202 valid / 100 test
- **Keypoints:** 32 pontos com visibility flags

**Formato YOLOv8-pose:**
```
class_id cx cy w h x1 y1 v1 x2 y2 v2 ... x32 y32 v32
```
- `v=0`: Não rotulado
- `v=1`: Rotulado mas INVISÍVEL (fora da câmera)
- `v=2`: Rotulado e VISÍVEL

**Treino (EXPERIMENTAL - FALHOU):**
```python
model = YOLO('yolo11x-pose.pt')  # Largest model
results = model.train(
    data='football-field-detection-1/data.yaml',
    epochs=150,
    imgsz=640,  # ⚠️ ERRO: Devia ser 1280
    batch=4,
    mosaic=0.0,  # CRÍTICO: Desativar para keypoints
    patience=30,
    save_period=10
)
```

**Métricas (Enganadoras):**
- **Pose mAP50:** 65.8% @ epoch 150
- **Best epoch:** 148 com 64.1%
- **Problema:** Métricas boas MAS produção PÉSSIMA

**Análise de Falha (Root Cause):**
1. **Comparação com tutorial Roboflow YouTube**
2. **Descoberta:** Tutorial usa `imgsz=1280`, projeto `imgsz=640`
3. **Impacto:** 4x menos pixels → keypoints de 8x8 pixels viram 2x2 pixels
4. **Resultado:** Modelo "adivinha" posições → falsos positivos

**Citação do Tutorial:**
> "we increase the input resolution from 640 to 1280... the ball barely occupies few dozen pixels... after rescaling to 640 the amount of information may no longer be sufficient"

**Solução Futura:**
```python
# Re-treinar com configuração correta
model = YOLO('yolo11x-pose.pt')
results = model.train(
    data='football-field-detection-1/data.yaml',
    epochs=150,
    imgsz=1280,  # ✅ CORRIGIDO
    batch=2,  # Reduzir (mais memória GPU)
    mosaic=0.0,
    patience=30
)
# Expectativa: 80-90% mAP50 + produção estável
```

**Keypoint Visualization (Confidence-based):**
```python
for kp_idx, (x, y) in enumerate(kp_xy):
    conf = keypoints.confidence[detection_idx][kp_idx]
    
    # Color coding por confidence
    if conf > 0.5:
        color = (0, 255, 0)  # Verde: confiante
    elif conf > 0.2:
        color = (0, 255, 255)  # Amarelo: incerto
    else:
        color = (0, 0, 255)  # Vermelho: falso positivo
    
    cv2.circle(frame, (int(x), int(y)), 8, color, -1)
    label = f"{CONFIG.labels[kp_idx]} {conf:.2f}"
    cv2.putText(frame, label, (int(x)+10, int(y)-10), ...)
```

---

**Continua?** Próxima secção seria **Datasets Detalhados** ou **Pipeline de Código (main.py)**?
