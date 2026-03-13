from ultralytics import YOLO

modelo_base = YOLO('yolo11s-seg.pt')

# classes=[0, 32] -> 0 é 'person' (pessoas) e 32 é 'sports ball' (bola)
# save=True -> Para ele guardar um vídeo .mp4 com o resultado final na pasta 'runs/predict'
resultados = modelo_base(r"C:\_FOOTAR\PD_FOOTAR\videos\input\lp1\round1\J2-Benfica-CasaPia_1-0.mp4", show=True, classes=[0, 32], save=True)