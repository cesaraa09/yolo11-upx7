import cv2
import numpy as np
import torch
import json
import os
import time
from ultralytics import YOLO
from datetime import datetime

# configs basicas mutaveis 
VIDEO_PATH = 'video2.webm'  # nome do video é o mesmo nome das coordenadas no json
MODEL_PATH = 'yolo11s.pt'  # mudar para yolo11n quando necessario (pc mais fraco e etc)
COORDS_PATH = 'areas.json'  
RESULTS_JSON = 'resultados.json'  
FRAME_SKIP = 8 
VEICULOS = [2]  # carros 2, motos 3, onibus 5 e caminhões 7
STATIONARY_THRESHOLD = 3  # depende do video, mudar caso os "parados" estejam errados
#IOU_THRESHOLD = 0.5
#CONFIDENCE_THRESHOLD = 0.6

def carregar_areas(video_nome, caminho_json):
    with open(caminho_json, 'r') as f:
        dados = json.load(f)
    if video_nome not in dados:
        print(f"[AVISO] Nenhuma área encontrada para o vídeo: {video_nome}")
        return {}
    return {k: np.array(v, dtype=np.int32) for k, v in dados[video_nome].items()}


def salvar_json(dados, caminho):
    with open(caminho, 'w') as f:
        json.dump(dados, f, indent=2)

model = YOLO(MODEL_PATH).to("cuda")
cap = cv2.VideoCapture(VIDEO_PATH)
areas = carregar_areas(VIDEO_PATH, COORDS_PATH)

frame_count = 0
resultados = []
fps = cap.get(cv2.CAP_PROP_FPS)
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

    results = model.track(frame, persist=True, classes=VEICULOS, verbose=False)[0]

    dados_areas = {nome: {"veiculos": 0, "tipos": {}, "parados": 0} for nome in areas}

    if results.boxes is not None:
        ids = results.boxes.id.cpu().numpy() if results.boxes.id is not None else [None]*len(results.boxes.cls)

        for box, cls, obj_id in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy(), ids):
            cls = int(cls)
            if cls not in VEICULOS:
                continue

            x1, y1, x2, y2 = box
            x_centro = (x1 + x2) / 2
            y_centro = (y1 + y2) / 2
            ponto = (x_centro, y_centro)

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

    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    resultados.append({
        "timestamp": round(timestamp, 2),
        "areas": dados_areas
    })

    annotated = results.plot() if results.boxes is not None else frame.copy()
    for nome_area, poly in areas.items():
        cv2.polylines(annotated, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(annotated, nome_area, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.putText(annotated, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('YOLOv11 + Tracking', annotated)

    elapsed = time.time() - start_time
    tempo_espera = max(0, frame_duration - elapsed)
    time.sleep(tempo_espera)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
salvar_json(resultados, RESULTS_JSON)
print(f"Resultados salvos em: {RESULTS_JSON}")