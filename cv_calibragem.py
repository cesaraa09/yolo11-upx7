# -*- coding: cp1252 -*-
import cv2
import numpy as np
drawing = False
pts = []

# input do mouse retorna as coordenadas no video
def draw_polygon(event, x, y, flags, param):
    global drawing, pts

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pts = [(x, y)]

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        pts.append((x, y))

        cv2.polylines(frame, [np.array(pts)], isClosed=True, color=(0, 255, 0), thickness=2)

        print("Coordenadas do poligono:")
        for point in pts:
            print(point)

        
        pts = []

cap = cv2.VideoCapture('video2.webm')  

cv2.namedWindow('Video')
cv2.setMouseCallback('Video', draw_polygon)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow('Video', frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()