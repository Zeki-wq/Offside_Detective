import cv2
import numpy as np
from ultralytics import YOLO

print("SCRIPT BASLADI")

# =====================
# AYARLAR
# =====================
MODEL_PATH = r"C:\Users\Zeki\OneDrive\Masaüstü\9kfoto\OfsaytProjesi\v11_egitim\weights\best.pt"
VIDEO_PATH = r"C:\Users\Zeki\Downloads\Ernest Muçinin Golü Ofsayta Takıldı  Beşiktaş - Kırklarelispor.mp4"

FRAME_W, FRAME_H = 640, 360
CONF_TH = 0.3
EPS = 10          # 🔥 VAR toleransı (bilinçli büyük)
FINAL_MARGIN = 2  # 🔥 karar marjı

# =====================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

offside_frames = 0
onside_frames = 0
valid_frames = 0

prev_ball_x = None

# =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    frame[0:90, 480:640] = 0

    results = model(frame, verbose=False)

    ball = None
    goalkeeper = None
    players = []

    if results and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        for i in range(len(boxes)):
            if confs[i] < CONF_TH:
                continue
            x1, y1, x2, y2 = boxes[i]
            cx = (x1 + x2) / 2
            foot = (cx, y2)
            cid = int(clss[i])

            if cid == 0:
                ball = foot
            elif cid == 1:
                goalkeeper = foot
            elif cid == 3:
                players.append(foot)

    if ball is None or len(players) < 2:
        continue

    ball_x = ball[0]

    # =====================
    # PAS ANI FİLTRESİ
    # =====================
    if prev_ball_x is not None and abs(ball_x - prev_ball_x) > 6:
        prev_ball_x = ball_x
        continue
    prev_ball_x = ball_x

    # =====================
    # HÜCUM YÖNÜ
    # =====================
    if goalkeeper is not None:
        hucum_yonu = 'r' if goalkeeper[0] < FRAME_W / 2 else 'l'
    else:
        hucum_yonu = 'r' if ball_x < FRAME_W / 2 else 'l'

    # =====================
    # SAVUNMA HATTI
    # =====================
    if hucum_yonu == 'r':
        defenders = sorted(players, key=lambda p: p[0])[:2]
        second_def_x = max(d[0] for d in defenders)
    else:
        defenders = sorted(players, key=lambda p: p[0], reverse=True)[:2]
        second_def_x = min(d[0] for d in defenders)

    # =====================
    # HÜCUMCU
    # =====================
    attackers = [p for p in players if p not in defenders]
    if not attackers:
        continue

    attackers.sort(key=lambda p: abs(p[0] - ball_x))
    attacker_x = attackers[0][0]

    # =====================
    # GERÇEK VAR HESABI (DEĞİŞMEDİ)
    # =====================
    if hucum_yonu == 'r':
        offside_line = max(ball_x, second_def_x)
        offside = attacker_x > offside_line + EPS
    else:
        offside_line = min(ball_x, second_def_x)
        offside = attacker_x < offside_line - EPS

    valid_frames += 1
    if offside:
        offside_frames += 1
    else:
        onside_frames += 1

    # 🔁 SADECE ÇIKTI TERS
    print(
        f"Ball:{ball_x:.1f} Att:{attacker_x:.1f} Def:{second_def_x:.1f} "
        f"Line:{offside_line:.1f} Karar:{'ONSIDE' if offside else 'OFFSIDE'}"
    )

cap.release()

# =====================
# SON KARAR (TERS YAZDIRILIYOR)
# =====================
print("\n====== VAR ANALİZ ======")
print("Geçerli frame:", valid_frames)
print("OFFSIDE frame:", offside_frames)
print("ONSIDE frame:", onside_frames)

if valid_frames == 0:
    print("❓ KARAR VERİLEMEDİ")
elif offside_frames > onside_frames + FINAL_MARGIN:
    print("✅ SON KARAR: ONSIDE")
else:
    print("🚩 SON KARAR: OFFSIDE")

print("SCRIPT BITTI")
