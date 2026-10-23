import cv2
import numpy as np

img = cv2.imread(
    "C:\\Users\\Zeki\\Downloads\\turkiye-hirvatistan-macinda-ofsayta-takildi-iste-o-pozisyon-1680030243253.jpeg"
)
orig = img.copy()

ref_line = None
ball = None
points = []

def mouse(event, x, y, flags, param):
    global ref_line, ball, points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

        # 1️⃣ Kale çizgisi (2 nokta)
        if ref_line is None and len(points) == 2:
            ref_line = (points[0], points[1])
            points.clear()

        # 2️⃣ Top noktası
        elif ref_line is not None and ball is None:
            ball = (x, y)
            points.clear()

def draw_offside_line(img, ref_line, point):
    (x1, y1), (x2, y2) = ref_line
    px, py = point

    # Kale çizgisi yönü
    dx = x2 - x1
    dy = y2 - y1

    # 🔥 KRİTİK NOKTA: DİK VEKTÖR
    nx = -dy
    ny = dx

    length = np.hypot(nx, ny)
    nx /= length
    ny /= length

    # Uzatılmış ofsayt çizgisi
    p1 = (int(px - 3000 * nx), int(py - 3000 * ny))
    p2 = (int(px + 3000 * nx), int(py + 3000 * ny))

    cv2.line(img, p1, p2, (255, 255, 255), 3)

cv2.namedWindow("OFFSIDE")
cv2.setMouseCallback("OFFSIDE", mouse)

while True:
    img = orig.copy()

    if ref_line:
        cv2.line(img, ref_line[0], ref_line[1], (0, 255, 0), 2)

    if ball:
        cv2.circle(img, ball, 6, (0, 0, 255), -1)

    if ref_line and ball:
        draw_offside_line(img, ref_line, ball)

    cv2.imshow("OFFSIDE", img)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    if key == ord('r'):  # reset
        ref_line = None
        ball = None
        points.clear()

cv2.destroyAllWindows()
