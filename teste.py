import cv2
import torch
import time
import numpy as np
from ultralytics import YOLO

# Função letterbox (preserva proporção com padding)
def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    shape = image.shape[:2]  # height, width
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, r, dw, dh

# Verifica GPU
assert torch.cuda.is_available(), "CUDA não está disponível"
model = YOLO('yolo11n.pt').to('cuda')

vehicle_classes = [2, 3, 5, 7]

# Abre vídeo
video_path = 'video1.mp4'
cap = cv2.VideoCapture(video_path)

# Pega FPS real do vídeo
fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / fps)  # Delay em milissegundos

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    start_time = time.time()
    original = frame.copy()

    # Preprocessamento (preserva proporção)
    resized, r, dw, dh = letterbox(frame, new_shape=(640, 640))
    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Prepara imagem para inferência
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to('cuda')

    # Inferência
    results = model(img_tensor, verbose=False)[0]

    # Desenha as caixas no frame original
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls in vehicle_classes:
            x1, y1, x2, y2 = box.xyxy[0]
            # Reverte escala e padding
            x1 = int((x1 - dw) / r)
            y1 = int((y1 - dh) / r)
            x2 = int((x2 - dw) / r)
            y2 = int((y2 - dh) / r)
            conf = box.conf[0].item()
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(original, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(original, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    # Mostra FPS da inferência (não afeta o tempo real do vídeo)
    elapsed = time.time() - start_time
    fps_disp = 1.0 / elapsed
    cv2.putText(original, f"FPS: {fps_disp:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("YOLOv11 - Veículos", original)

    # Espera o tempo certo entre quadros
    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
