import cv2
import numpy as np
import os
import sys
from collections import deque

# ================= USER INPUT =================
VIDEO_PATH = "vehant_hackathon_video_1.avi"
# ==============================================

# ================= OUTPUT ======================
OUT_DIR = "main_attempt_solution"
OUT_VIDEO_NAME = "trial.mp4"
# ==============================================

# ================= PARAMETERS ==================
MIN_BLOB_AREA_RATIO = 0.0008
LINE_POS_RATIO = 0.5           # middle line
MAX_CENTROID_DIST = 60
MAX_MISSED_FRAMES = 6

LINE_OCCUPANCY_THRESHOLD = 0.40
MAX_NEW_VEHICLES = 3
WINDOW_FRAMES = 4
# ==============================================


def main():
    if not os.path.exists(VIDEO_PATH):
        print("Video not found")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, OUT_VIDEO_NAME)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        sys.exit(1)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (W, H)
    )

    min_blob_area = int(MIN_BLOB_AREA_RATIO * W * H)
    line_y = int(LINE_POS_RATIO * H)

    bg = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=32,
        detectShadows=False
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    objects = []
    total_count = 0
    frame_idx = 0
    recent_increments = deque()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # ---------- FOREGROUND ----------
        fg = bg.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, 1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, 2)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        # ---------- CONNECTED COMPONENTS ----------
        num_labels, _, stats, cents = cv2.connectedComponentsWithStats(fg, 8)

        detections = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_blob_area:
                continue

            x, y, w, h = stats[i, :4]
            cx, cy = cents[i]
            detections.append((cx, cy, x, y, w, h, area))

        # ---------- TRACK UPDATE ----------
        for obj in objects:
            obj["matched"] = False

        new_objs = []

        for cx, cy, x, y, w, h, area in detections:
            best, best_d = None, MAX_CENTROID_DIST
            for obj in objects:
                if obj["matched"]:
                    continue
                d = np.hypot(cx - obj["centroid"][0], cy - obj["centroid"][1])
                if d < best_d:
                    best_d, best = d, obj

            if best:
                best["centroid"] = (cx, cy)
                best["bbox"] = (x, y, w, h)
                best["area"] = area
                best["missed"] = 0
                best["matched"] = True
            else:
                new_objs.append({
                    "centroid": (cx, cy),
                    "bbox": (x, y, w, h),
                    "area": area,
                    "missed": 0,
                    "counted": False,
                    "matched": True
                })

        survivors = []
        for obj in objects:
            if not obj["matched"]:
                obj["missed"] += 1
            if obj["missed"] <= MAX_MISSED_FRAMES:
                survivors.append(obj)

        objects = survivors + new_objs

        # ---------- LINE OCCUPANCY ----------
        crossing = []
        intervals = []

        for obj in objects:
            if obj["counted"]:
                continue
            x, y, w, h = obj["bbox"]
            if y <= line_y <= y + h:
                crossing.append(obj)
                intervals.append((x, x + w))

        # merge x-intervals
        intervals.sort()
        covered = 0
        ps, pe = None, None
        for s, e in intervals:
            if ps is None:
                ps, pe = s, e
            elif s <= pe:
                pe = max(pe, e)
            else:
                covered += pe - ps
                ps, pe = s, e
        if ps is not None:
            covered += pe - ps

        line_occupancy = covered / W

        # clean rate limiter
        while recent_increments and frame_idx - recent_increments[0] > WINDOW_FRAMES:
            recent_increments.popleft()

        if line_occupancy > LINE_OCCUPANCY_THRESHOLD:
            if crossing:
                obj = max(crossing, key=lambda o: o["area"])
                total_count += 1
                obj["counted"] = True
                recent_increments.append(frame_idx)
        else:
            allowed = MAX_NEW_VEHICLES - len(recent_increments)
            if allowed > 0:
                crossing.sort(key=lambda o: o["area"], reverse=True)
                for obj in crossing[:allowed]:
                    total_count += 1
                    obj["counted"] = True
                    recent_increments.append(frame_idx)

        # ---------- VISUALIZATION ----------
        vis = frame.copy()
        cv2.line(vis, (0, line_y), (W, line_y), (0, 255, 255), 2)
        cv2.putText(
            vis,
            f"Vehicle Count: {total_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )

        writer.write(vis)

        # 🔴 THESE WINDOWS WERE MISSING BEFORE — NOW FIXED
        cv2.imshow("Motion Blobs (Detector)", fg)
        cv2.imshow("Original + Count", vis)

        print(
            f"\rFrame {frame_idx:05d} | LineOcc: {line_occupancy:.2f} | Count: {total_count}",
            end=""
        )

        if cv2.waitKey(1) & 0xFF == 27:
            break

    print("\nSaved to:", out_path)
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
