import cv2
import numpy as np
import torch
import json
import os
import time
from ultralytics import YOLO
from datetime import datetime

# configs basicas mutaveis 
VIDEO_PATH = 0  # usar índice da webcam, 0 geralmente é a webcam padrão (DroidCam, etc)
MODEL_PATH = 'yolo11s.pt'  # mudar para yolo11n quando necessario (pc mais fraco e etc)
COORDS_PATH = 'areas.json'  
RESULTS_JSON = 'resultados2.json'  # novo arquivo de saída
FRAME_SKIP = 8 
VEICULOS = [2]  # carros 2, motos 3, onibus 5 e caminhões 7
STATIONARY_THRESHOLD = 3  # depende do video, mudar caso os "parados" estejam errados
#IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.3

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
areas = carregar_areas("webcam", COORDS_PATH)  # nome do vídeo é substituído por "webcam"

frame_count = 0
resultados = []
fps = 30  # valor fixo, pois webcam nem sempre retorna corretamente via cap.get
frame_duration = FRAME_SKIP / fps

historico_centros = {}

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    ##frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) ## coloca caso o vídeo não esteja na orientação certa
    ##frame = cv2.flip(frame, 1)  ## descomente se a imagem estiver espelhada
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

    timestamp = time.time()  # substitui get(CAP_PROP_POS_MSEC), pois é em tempo real
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
