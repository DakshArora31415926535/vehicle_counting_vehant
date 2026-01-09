import cv2
import numpy as np
import os
import sys

# ================= USER INPUT =================
VIDEO_PATH = "vehant_hackathon_video_10.mp4"   # <<< CHANGE THIS
# ==============================================

# ================= OUTPUT ======================
OUT_DIR = "main_attempt_solution"
OUT_VIDEO_NAME = "output_conventional_DATASET_10.mp4"
# ==============================================


# ================= PARAMETERS ==================
MIN_BLOB_AREA_RATIO = 0.0008
MAX_BLOB_AREA_RATIO = 0.08

LINE_POS_RATIO = 0.82
MAX_CENTROID_DIST = 50
MAX_MISSED_FRAMES = 6
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
    max_blob_area = int(MAX_BLOB_AREA_RATIO * frame_area)
    count_line_y = int(LINE_POS_RATIO * H)

    bg = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=32,
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

        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            fg, connectivity=8
        )

        detections = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_blob_area or area > max_blob_area:
                continue
            cx, cy = centroids[i]
            detections.append((int(cx), int(cy)))

        # -------- UPDATE OBJECTS --------
        for obj in objects:
            obj["matched"] = False

        new_objects = []

        for cx, cy in detections:
            best = None
            best_dist = MAX_CENTROID_DIST

            for obj in objects:
                if obj["matched"]:
                    continue
                px, py = obj["centroid"]
                d = np.hypot(cx - px, cy - py)
                if d < best_dist and cy < py:
                    best_dist = d
                    best = obj

            if best is not None:
                best["centroid"] = (cx, cy)
                best["missed"] = 0
                best["matched"] = True

                if not best["counted"] and best["prev_y"] > count_line_y >= cy:
                    total_count += 1
                    best["counted"] = True

                best["prev_y"] = cy
            else:
                new_objects.append({
                    "centroid": (cx, cy),
                    "prev_y": cy,
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

        objects = survivors + new_objects

        # -------- VISUALIZATION --------
        vis = frame.copy()

        cv2.line(vis, (0, count_line_y), (W, count_line_y), (0, 255, 255), 2)

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
            f"\rFrame {frame_idx:05d} | "
            f"Active: {len(objects)} | "
            f"Count: {total_count}",
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
