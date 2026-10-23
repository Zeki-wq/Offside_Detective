import cv2
import numpy as np
from ultralytics import YOLO

print("SCRIPT BASLADI")

# =====================
# AYARLAR
# =====================
MODEL_PATH = r"C:\Users\Zeki\OneDrive\Masaüstü\9kfoto\OfsaytProjesi\v11_egitim\weights\best.pt"
VIDEO_PATH = r"C:\Users\Zeki\Downloads\Ernest Muçinin Golü Ofsayta Takıldı  Beşiktaş - Kırklarelispor.mp4"  # ⚠️ MUTLAKA basit bir yol kullan

FRAME_W, FRAME_H = 640, 360

# Manuel seçilmiş saha köşeleri (640x360)
PTS_SRC = np.array([
    [5, 4],
    [632, 9],
    [629, 350],
    [9, 351]
], dtype=np.float32)

PTS_DST = np.array([
    [0, 0],
    [1000, 0],
    [1000, 700],
    [0, 700]
], dtype=np.float32)

# =====================
# HOMOGRAFİ
# =====================
H, _ = cv2.findHomography(PTS_SRC, PTS_DST)
H_INV = np.linalg.inv(H)

def get_perspective_line(foot_x, foot_y):
    p = np.array([foot_x, foot_y, 1], dtype=np.float32).reshape(3, 1)

    bird = H @ p
    bird /= bird[2]
    bird_x = bird[0, 0].item()

    top = np.array([bird_x, 0, 1], dtype=np.float32).reshape(3, 1)
    bot = np.array([bird_x, 700, 1], dtype=np.float32).reshape(3, 1)

    p1 = H_INV @ top
    p2 = H_INV @ bot
    p1 /= p1[2]
    p2 /= p2[2]

    return (
        (int(p1[0, 0]), int(p1[1, 0])),
        (int(p2[0, 0]), int(p2[1, 0]))
    )

# =====================
# MODEL
# =====================
model = YOLO(MODEL_PATH)

# =====================
# VIDEO
# =====================
cap = cv2.VideoCapture(VIDEO_PATH)
print("Video isOpened:", cap.isOpened())

if not cap.isOpened():
    print("❌ VIDEO AÇILAMADI")
    exit()

cv2.namedWindow("VAR", cv2.WINDOW_NORMAL)

hucum_yonu = input("Hücum yönü (s = sol, r = sağ): ").lower()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ FRAME GELMEDI / VIDEO BITTI")
        break

    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    results = model(frame, verbose=False)

    defenders = []
    attackers = []

    if results and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            class_id = int(clss[i])

            # sadece oyuncular
            if class_id == 3:
                foot_x = (x1 + x2) / 2
                foot_y = y2

                defenders.append((foot_x, foot_y))
                attackers.append((foot_x, foot_y))

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (255, 0, 0),
                    1
                )

    # =====================
    # 2. SAVUNMACI + KARAR
    # =====================
    if len(defenders) >= 2 and len(attackers) >= 1:

        if hucum_yonu == 'l':
            defenders_sorted = sorted(defenders, key=lambda p: p[0], reverse=True)
            attackers_sorted = sorted(attackers, key=lambda p: p[0])
        else:
            defenders_sorted = sorted(defenders, key=lambda p: p[0])
            attackers_sorted = sorted(attackers, key=lambda p: p[0], reverse=True)

        second_defender = defenders_sorted[1]
        attacker = attackers_sorted[0]

        # VAR çizgisi (HER FRAME)
        p1, p2 = get_perspective_line(second_defender[0], second_defender[1])
        cv2.line(frame, p1, p2, (0, 0, 255), 2)

        # OFSAYT KARARI
        if hucum_yonu == 'l':
            offside = attacker[0] < second_defender[0]
        else:
            offside = attacker[0] > second_defender[0]

        text = "OFFSIDE" if offside else "ONSIDE"
        color = (0, 0, 255) if offside else (0, 255, 0)

        cv2.putText(
            frame,
            text,
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3
        )

    cv2.imshow("VAR", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("SCRIPT BİTTİ")
