import cv2
import numpy as np
import os
import sys
from math import atan2, cos, sin, pi, hypot

# ================= CLI INPUT =================
ANGLE_FRAMES = int(sys.argv[1])
VIDEO_PATH = sys.argv[2]
# ============================================

# ================= PARAMETERS ==================
MIN_BLOB_AREA_RATIO = 0.0008
LINE_POS_RATIO = 0.55
MAX_CENTROID_DIST = 60
MAX_MISSED_FRAMES = 6
MAX_ANGLE_DEG = 30
# ==============================================


def signed_distance(px, py, p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((px - x1)*(y2 - y1) - (py - y1)*(x2 - x1)) / hypot(y2 - y1, x2 - x1)


def estimate_motion_angle(cap, bg, kernel, min_blob_area):
    vectors = []

    for _ in range(ANGLE_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break

        fg = bg.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, 1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, 2)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        num_labels, _, stats, cents = cv2.connectedComponentsWithStats(fg, 8)

        best = None
        best_area = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > best_area and area > min_blob_area:
                best_area = area
                best = cents[i]

        if best is not None:
            vectors.append(best)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if len(vectors) < 2:
        return pi / 2

    dx = vectors[-1][0] - vectors[0][0]
    dy = vectors[-1][1] - vectors[0][1]

    if dy < 0:
        dx, dy = -dx, -dy

    angle = atan2(dy, dx)
    max_a = MAX_ANGLE_DEG * pi / 180
    angle = np.clip(angle, pi/2 - max_a, pi/2 + max_a)

    return angle


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        sys.exit(1)

    ret, frame0 = cap.read()
    if not ret:
        sys.exit(1)

    H, W = frame0.shape[:2]

    bg = cv2.createBackgroundSubtractorMOG2(500, 32, False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    min_blob_area = int(MIN_BLOB_AREA_RATIO * W * H)

    motion_angle = estimate_motion_angle(cap, bg, kernel, min_blob_area)

    cap.release()
    cap = cv2.VideoCapture(VIDEO_PATH)
    bg = cv2.createBackgroundSubtractorMOG2(500, 32, False)

    line_angle = motion_angle + pi / 2
    cx, cy = W // 2, int(LINE_POS_RATIO * H)
    L = max(W, H)

    dx = int(cos(line_angle) * L)
    dy = int(sin(line_angle) * L)
    fixed_line = ((cx - dx, cy - dy), (cx + dx, cy + dy))

    objects = []
    total_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg = bg.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, 1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, 2)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        num_labels, _, stats, cents = cv2.connectedComponentsWithStats(fg, 8)

        detections = [
            cents[i] for i in range(1, num_labels)
            if stats[i, cv2.CC_STAT_AREA] >= min_blob_area
        ]

        for obj in objects:
            obj["matched"] = False

        new_objs = []

        for cx_, cy_ in detections:
            best, best_d = None, MAX_CENTROID_DIST
            for obj in objects:
                if obj["matched"]:
                    continue
                d = hypot(cx_ - obj["centroid"][0], cy_ - obj["centroid"][1])
                if d < best_d:
                    best, best_d = obj, d

            if best:
                best["centroid"] = (cx_, cy_)
                best["matched"] = True
                best["missed"] = 0
            else:
                new_objs.append({
                    "centroid": (cx_, cy_),
                    "matched": True,
                    "missed": 0,
                    "counted": False
                })

        objects = [
            o for o in objects
            if (o["matched"] or (o.update({"missed": o["missed"] + 1}) or o["missed"] <= MAX_MISSED_FRAMES))
        ] + new_objs

        for obj in objects:
            if obj["counted"]:
                continue

            d = signed_distance(
                obj["centroid"][0],
                obj["centroid"][1],
                fixed_line[0],
                fixed_line[1]
            )

            if "prev_d" in obj and obj["prev_d"] > 0 and d <= 0:
                total_count += 1
                obj["counted"] = True

            obj["prev_d"] = d

    cap.release()
    print(total_count)


if __name__ == "__main__":
    main()
