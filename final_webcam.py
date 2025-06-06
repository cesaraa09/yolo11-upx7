import cv2
import numpy as np
import torch
import json
import time
from ultralytics import YOLO
from datetime import datetime

VIDEO_PATH = 1
MODEL_PATH = 'yolo11s.pt'
COORDS_PATH = 'areas.json'
RESULTS_JSON = 'resultados2.json'
FRAME_SKIP = 10
VEICULOS = [2]
STATIONARY_THRESHOLD = 3
CONFIDENCE_THRESHOLD = 0.5

# Inicia JSON limpo
with open(RESULTS_JSON, 'w') as f:
    json.dump([], f)

def salvar_incremental_json(novo_dado, caminho):
    try:
        with open(caminho, 'r+') as f:
            dados = json.load(f)
            dados.append(novo_dado)
            f.seek(0)
            json.dump(dados, f, indent=2)
            f.truncate()
    except Exception as e:
        print(f"[ERRO] Falha ao salvar no JSON: {e}")

def carregar_areas(video_nome, caminho_json):
    with open(caminho_json, 'r') as f:
        dados = json.load(f)
    if video_nome not in dados:
        print(f"[AVISO] Nenhuma área encontrada para o vídeo: {video_nome}")
        return {}
    return {k: np.array(v, dtype=np.int32) for k, v in dados[video_nome].items()}

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH).to(device)

cap = cv2.VideoCapture(VIDEO_PATH)
areas = carregar_areas("webcam", COORDS_PATH)

frame_count = 0
fps = 30
frame_duration = FRAME_SKIP / fps
historico_centros = {}

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    results = model.track(frame, persist=True, classes=VEICULOS, verbose=False, conf=CONFIDENCE_THRESHOLD)[0]
    dados_areas = {nome: {"veiculos": 0, "tipos": {}, "parados": 0} for nome in areas}

    if results.boxes is not None:
        ids = results.boxes.id.cpu().numpy() if results.boxes.id is not None else [None]*len(results.boxes.cls)
        for box, cls, obj_id in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy(), ids):
            cls = int(cls)
            if cls not in VEICULOS:
                continue

            x1, y1, x2, y2 = box
            ponto = ((x1 + x2) / 2, (y1 + y2) / 2)

            for nome_area, poly in areas.items():
                if cv2.pointPolygonTest(poly, ponto, False) >= 0:
                    nome_cls = model.model.names[cls] if hasattr(model.model, 'names') else str(cls)
                    dados_areas[nome_area]["veiculos"] += 1
                    dados_areas[nome_area]["tipos"].setdefault(nome_cls, 0)
                    dados_areas[nome_area]["tipos"][nome_cls] += 1

                    if obj_id is not None:
                        historico_centros.setdefault(obj_id, []).append(ponto)
                        if len(historico_centros[obj_id]) > STATIONARY_THRESHOLD:
                            recentes = historico_centros[obj_id][-STATIONARY_THRESHOLD:]
                            dists = [np.linalg.norm(np.array(recentes[-1]) - np.array(p)) for p in recentes[:-1]]
                            if all(d < 2.0 for d in dists):
                                dados_areas[nome_area]["parados"] += 1
                    break

    timestamp = datetime.now().strftime("%H:%M:%S")
    salvar_incremental_json({
        "timestamp": timestamp,
        "areas": dados_areas
    }, RESULTS_JSON)

    annotated = results.plot() if results.boxes is not None else frame.copy()
    for nome_area, poly in areas.items():
        cv2.polylines(annotated, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(annotated, nome_area, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.putText(annotated, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('YOLOv11 + Tracking', annotated)

    elapsed = time.time() - start_time
    time.sleep(max(0, frame_duration - elapsed))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Resultados salvos incrementalmente em: {RESULTS_JSON}")
