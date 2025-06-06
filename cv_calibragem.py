import cv2
import numpy as np

frame = None
pts = []
temp_frame = None
area_index = 1  # Você pode usar para nomear: area_rua1, rua2, etc.

def draw_polygon(event, x, y, flags, param):
    global pts, frame, temp_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))

    elif event == cv2.EVENT_MOUSEMOVE:
        if len(pts) > 0:
            temp_frame = frame.copy()
            cv2.polylines(temp_frame, [np.array(pts + [(x, y)])], isClosed=False, color=(0, 255, 255), thickness=1)

    elif event == cv2.EVENT_RBUTTONDOWN and len(pts) >= 3:
        cv2.polylines(frame, [np.array(pts)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.imshow('Definir Áreas', frame)

        print("\nCoordenadas do polígono:")
        for p in pts:
            print(p)
        print("\n--- Copie e cole no seu JSON:")
        print(f'"area_rua{area_index}": {pts},')
        pts.clear()

# === Fonte do vídeo (link .m3u8 de stream ao vivo) ===
VIDEO_SOURCE = "https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/1749261647/ei/70hDaJWsGeW59cYP56jxsQI/ip/2804:7f1:6807:c67:b156:5486:73a6:71bd/id/ByED80IKdIU.3/itag/96/source/yt_live_broadcast/requiressl/yes/ratebypass/yes/live/1/sgoap/gir%3Dyes%3Bitag%3D140/sgovp/gir%3Dyes%3Bitag%3D137/rqh/1/hls_chunk_host/rr4---sn-8p8v-bg0ll.googlevideo.com/xpc/EgVo2aDSNQ%3D%3D/playlist_duration/30/manifest_duration/30/bui/AY1jyLOmd-FMSxO781YzCBf6rRBfENP2ClVmvHLvWz-UQixl0G4A8K5aU2aMBew7wSj9BFSYK18vjfl8/spc/l3OVKbQk62fp8LvP_2QXFCcPcIHPueMtZ7wDT_DVf5HfhB4PegqOwTLsjsoFpwpPbenFJ9g/vprv/1/playlist_type/DVR/initcwndbps/2803750/met/1749240048,/mh/jK/mm/44/mn/sn-8p8v-bg0ll/ms/lva/mv/m/mvi/4/pl/48/rms/lva,lva/dover/11/pacing/0/keepalive/yes/fexp/51355912/mt/1749239575/sparams/expire,ei,ip,id,itag,source,requiressl,ratebypass,live,sgoap,sgovp,rqh,xpc,playlist_duration,manifest_duration,bui,spc,vprv,playlist_type/sig/AJfQdSswRgIhAIGzNcivciVxKPX7p-SnEls-LG3I8wtSzAecyNv5evOMAiEA_t8--6tGoPEvQVvVHdhPXDEzBwFyqhmYFfH2VN5V9OU%3D/lsparams/hls_chunk_host,initcwndbps,met,mh,mm,mn,ms,mv,mvi,pl,rms/lsig/APaTxxMwRAIgPM_NcdRXz-xSnFonk0Slj7aczUiF25NENVIXciX5onsCIBRIfpbpDfoziN0_AOYKJrsqj4gekY_apuWbcmTaArKV/playlist/index.m3u8"

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("Erro ao abrir o vídeo ou stream.")
    exit()

cv2.namedWindow('Definir Áreas')
cv2.setMouseCallback('Definir Áreas', draw_polygon)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Frame não capturado.")
        break

    temp_frame = frame.copy()
    if len(pts) > 0:
        cv2.polylines(temp_frame, [np.array(pts)], isClosed=False, color=(0, 255, 255), thickness=1)

    cv2.imshow('Definir Áreas', temp_frame)
    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
