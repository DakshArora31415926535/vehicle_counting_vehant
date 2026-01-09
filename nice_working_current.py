import cv2
import numpy as np
import os
import sys

# ================= USER INPUT =================
VIDEO_PATH = "vehant_hackathon_video_12.mp4"   # <<< CHANGE THIS
# ==============================================

# ================= OUTPUT ======================
OUT_DIR = "main_attempt_solution"
OUT_VIDEO_NAME = "trial.mp4"
# ==============================================

# ================= PARAMETERS ==================
MIN_BLOB_AREA_RATIO = 0.0008

LINE_POS_RATIO = 0.62          # <<< moved up
MAX_CENTROID_DIST = 70
MAX_MISSED_FRAMES = 6
MIN_LIFE_FRAMES = 6

MIN_ASPECT_RATIO = 0.6         # <<< shape filter
MIN_SOLIDITY = 0.35            # <<< shape filter
# ==============================================


def main():
    if not os.path.exists(VIDEO_PATH):
        print("Video not found")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, OUT_VIDEO_NAME)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Could not open video")
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

    frame_area = W * H
    min_blob_area = int(MIN_BLOB_AREA_RATIO * frame_area)
    line_y = int(LINE_POS_RATIO * H)

    bg = cv2.createBackgroundSubtractorMOG2(
        history=700,
        varThreshold=40,
        detectShadows=False
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    objects = []
    total_count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        fg = bg.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, 1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, 2)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            fg, connectivity=8
        )

        detections = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_blob_area:
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            aspect_ratio = w / max(h, 1)
            solidity = area / max(w * h, 1)

            # -------- SHAPE FILTER --------
            if aspect_ratio < MIN_ASPECT_RATIO:
                continue
            if solidity < MIN_SOLIDITY:
                continue

            cx, cy = centroids[i]
            detections.append((cx, cy, x, y, w, h))

        # ---------- TRACK UPDATE ----------
        for obj in objects:
            obj["matched"] = False

        new_objects = []

        for cx, cy, x, y, w, h in detections:
            best = None
            best_dist = MAX_CENTROID_DIST

            for obj in objects:
                if obj["matched"]:
                    continue
                px, py = obj["centroid"]
                d = np.hypot(cx - px, cy - py)
                if d < best_dist:
                    best_dist = d
                    best = obj

            if best is not None:
                best["centroid"] = (cx, cy)
                best["bbox"] = (x, y, w, h)
                best["missed"] = 0
                best["life"] += 1
                best["matched"] = True
            else:
                new_objects.append({
                    "centroid": (cx, cy),
                    "bbox": (x, y, w, h),
                    "missed": 0,
                    "life": 1,
                    "counted": False,
                    "matched": True
                })

        survivors = []
        for obj in objects:
            if not obj["matched"]:
                obj["missed"] += 1
            if obj["missed"] <= MAX_MISSED_FRAMES:
                survivors.append(obj)

        objects = survivors + new_objects

        # ---------- COUNT ----------
        for obj in objects:
            if obj["counted"]:
                continue
            if obj["life"] < MIN_LIFE_FRAMES:
                continue

            x, y, w, h = obj["bbox"]
            if y <= line_y <= y + h:
                total_count += 1
                obj["counted"] = True

        # ---------- VISUAL ----------
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

        cv2.imshow("Motion Blobs (White)", fg)
        cv2.imshow("Original + Count", vis)

        print(
            f"\rFrame {frame_idx:05d} | Active: {len(objects)} | Count: {total_count}",
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
