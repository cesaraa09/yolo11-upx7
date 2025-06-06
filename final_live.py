import cv2
import numpy as np
import torch
import json
import time
from ultralytics import YOLO
from datetime import datetime  # Import necessário para timestamp legível

VIDEO_PATH = "https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/1749261647/ei/70hDaJWsGeW59cYP56jxsQI/ip/2804:7f1:6807:c67:b156:5486:73a6:71bd/id/ByED80IKdIU.3/itag/96/source/yt_live_broadcast/requiressl/yes/ratebypass/yes/live/1/sgoap/gir%3Dyes%3Bitag%3D140/sgovp/gir%3Dyes%3Bitag%3D137/rqh/1/hls_chunk_host/rr4---sn-8p8v-bg0ll.googlevideo.com/xpc/EgVo2aDSNQ%3D%3D/playlist_duration/30/manifest_duration/30/bui/AY1jyLOmd-FMSxO781YzCBf6rRBfENP2ClVmvHLvWz-UQixl0G4A8K5aU2aMBew7wSj9BFSYK18vjfl8/spc/l3OVKbQk62fp8LvP_2QXFCcPcIHPueMtZ7wDT_DVf5HfhB4PegqOwTLsjsoFpwpPbenFJ9g/vprv/1/playlist_type/DVR/initcwndbps/2803750/met/1749240048,/mh/jK/mm/44/mn/sn-8p8v-bg0ll/ms/lva/mv/m/mvi/4/pl/48/rms/lva,lva/dover/11/pacing/0/keepalive/yes/fexp/51355912/mt/1749239575/sparams/expire,ei,ip,id,itag,source,requiressl,ratebypass,live,sgoap,sgovp,rqh,xpc,playlist_duration,manifest_duration,bui,spc,vprv,playlist_type/sig/AJfQdSswRgIhAIGzNcivciVxKPX7p-SnEls-LG3I8wtSzAecyNv5evOMAiEA_t8--6tGoPEvQVvVHdhPXDEzBwFyqhmYFfH2VN5V9OU%3D/lsparams/hls_chunk_host,initcwndbps,met,mh,mm,mn,ms,mv,mvi,pl,rms/lsig/APaTxxMwRAIgPM_NcdRXz-xSnFonk0Slj7aczUiF25NENVIXciX5onsCIBRIfpbpDfoziN0_AOYKJrsqj4gekY_apuWbcmTaArKV/playlist/index.m3u8"
MODEL_PATH = 'yolo11s.pt'
COORDS_PATH = 'areas.json'
RESULTS_JSON = 'resultados3.json'  # JSON que o Streamlit lê
FRAME_SKIP = 10
VEICULOS = [2]
STATIONARY_THRESHOLD = 3
CONFIDENCE_THRESHOLD = 0.5

# Inicializa JSON limpo no início do programa (apenas uma vez)
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
areas = carregar_areas("live", COORDS_PATH)

frame_count = 0
fps = cap.get(cv2.CAP_PROP_FPS) or 30
frame_duration = FRAME_SKIP / fps

historico_centros = {}
ultimo_area_por_obj = {}

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
    transicoes = []

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
                        # Verifica veículo parado
                        if len(historico_centros[obj_id]) > STATIONARY_THRESHOLD:
                            recentes = historico_centros[obj_id][-STATIONARY_THRESHOLD:]
                            dists = [np.linalg.norm(np.array(recentes[-1]) - np.array(p)) for p in recentes[:-1]]
                            if all(d < 2.0 for d in dists):
                                dados_areas[nome_area]["parados"] += 1

                        # Verifica transição entre áreas
                        if obj_id in ultimo_area_por_obj and ultimo_area_por_obj[obj_id] != nome_area:
                            transicoes.append({
                                "obj_id": int(obj_id),
                                "de": ultimo_area_por_obj[obj_id],
                                "para": nome_area,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # timestamp legível
                            })
                        ultimo_area_por_obj[obj_id] = nome_area
                    break

    salvar_incremental_json({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # timestamp legível aqui também
        "areas": dados_areas,
        "transicoes": transicoes
    }, RESULTS_JSON)

    # Desenho opcional (pode comentar para acelerar)
    annotated = results.plot() if results.boxes is not None else frame.copy()
    for nome_area, poly in areas.items():
        cv2.polylines(annotated, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(annotated, nome_area, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imshow('YOLOv11 + Tracking', annotated)

    elapsed = time.time() - start_time
    time.sleep(max(0, frame_duration - elapsed))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Resultados salvos incrementalmente em: {RESULTS_JSON}")
