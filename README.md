# Vehicle Counting Using Classical Computer Vision

## Overview

This project estimates the **total number of vehicles** appearing in a traffic video captured by a **static camera**, using **only classical computer vision techniques**.

No deep learning models, object detectors, or learning-based approaches are used.

The solution is designed to be **robust, deterministic, and fully compliant** with the constraints of the Vehant Technologies Vehicle Count Hackathon, with emphasis on engineering robustness, explainability, and reproducibility.

---

# Problem Statement

## Input

- A traffic video file (`video_path`)
- Video characteristics:
  - Static camera (no camera motion)
  - Vehicles move predominantly away from the camera

## Output

A single integer representing the total number of vehicles appearing in the video.

No annotation files or ground-truth labels are required.

---

# Key Observations

The solution relies on several properties of the provided videos:

- Static camera throughout the recording
- Vehicles move in a dominant direction
- Background remains largely unchanged
- Vehicles appear as coherent moving regions

These characteristics make the task well suited for **motion-based classical computer vision** rather than appearance-based object detection.

---

# Methodology

The complete pipeline consists of the following stages:

1. Background subtraction
2. Morphological filtering
3. Connected component extraction
4. Motion direction estimation
5. Automatic counting line placement
6. Centroid-based tracking
7. Line-crossing based vehicle counting

Each stage is deterministic and independently explainable.

---

# Background Subtraction

A **MOG2 background subtractor** is used to separate moving objects from the static background.

Since the camera is fixed, background modeling remains stable over time.

The foreground mask is refined using morphological operations:

- **Opening** removes small noise
- **Closing** fills holes inside vehicle regions

Connected components are then extracted.

Very small components (determined using a frame-relative area threshold) are discarded to suppress noise and non-vehicle motion.

---

# Motion Direction Estimation

Instead of assuming a fixed horizontal counting line, the algorithm estimates the dominant traffic direction directly from the video.

During an initialization phase:

- The centroid of the largest moving region is tracked over several frames.
- Its displacement provides an estimate of the dominant motion vector.

The counting line is then placed **perpendicular** to this direction.

This makes the solution robust to:

- Camera tilt
- Slight road curvature
- Different camera orientations
- Perspective effects

without requiring any hardcoded assumptions.

---
<img width="905" height="493" alt="image" src="https://github.com/user-attachments/assets/1eeb1d82-ebe9-4964-bd50-c0a9a64d66b4" />

# Tracking and Counting

Moving regions are tracked using centroid proximity between consecutive frames.

Each track stores a short history of missed detections to improve robustness against:

- Temporary occlusions
- Imperfect foreground segmentation
- Short interruptions in motion detection

A vehicle is counted exactly once when its centroid crosses the counting line.

Crossing is detected using a **signed distance test**, which identifies a change in the centroid's side relative to the counting line.

This avoids:

- Double counting
- Bounding-box overlap heuristics
- Object re-identification

---

# Dual Independent Execution Strategy

Background subtraction can occasionally be sensitive to:

- Initialization effects
- Sparse traffic
- Early-frame noise

To improve robustness, the entire pipeline is executed **twice**.

### Run 1

Motion estimation uses the first **100 frames**.

### Run 2

Motion estimation uses the first **500 frames**.

Each run is executed in a completely separate Python process, ensuring:

- Independent background models
- No shared OpenCV state
- No memory contamination

This improves reliability while remaining fully deterministic.

---

# Final Decision Strategy

Let:

- **c100** = vehicle count from the 100-frame run
- **c500** = vehicle count from the 500-frame run

The relative difference is computed as:

```text
|c100 − c500| / max(c100, c500)
```

Decision rule:

- If the difference is **greater than 50%**, choose the **higher count** (to reduce severe under-counting).
- Otherwise, choose the **lower count** as a conservative estimate to reduce over-counting caused by blob fragmentation or segmentation noise.

This strategy is:

- Deterministic
- Video-agnostic
- Based on observed pipeline behavior rather than video-specific heuristics

---

# Repository Structure

```
.
├── main.py
├── worker.py
├── requirements.txt
└── README.md
```

### `main.py`

Entry point of the solution.

Contains the required:

- `Solution` class
- `forward(video_path)` method

### `worker.py`

Implements the complete classical computer vision pipeline.

Each execution prints only the final vehicle count.

### `requirements.txt`

Lists all required external dependencies.

### `README.md`

Project documentation describing the methodology, assumptions, implementation, and robustness strategy.

---

# Evaluation Procedure

The evaluator executes the solution as follows:

```python
from main import Solution

solution = Solution()

count = solution.forward(video_path)
```

The returned value is a single integer representing the estimated number of vehicles.

No command-line arguments or manual intervention are required.

---

# Dependencies

The project depends only on:

- OpenCV (`opencv-python`)
- NumPy

All remaining imports are from Python's standard library.

---

# Assumptions

- Static camera
- Predominantly one-directional vehicle motion
- Vehicles appear as separable moving regions

---

# Limitations

Like any background-subtraction-based approach, the solution may experience reduced accuracy under certain conditions:

- Heavy occlusions
- Extremely slow-moving vehicles
- Sudden lighting changes

These limitations are inherent to classical motion-based computer vision techniques.

---

# Hackathon Compliance

This solution fully complies with the challenge rules.

- ✅ Classical computer vision only
- ✅ No deep learning
- ✅ No learning-based models
- ✅ No hardcoded video-specific logic
- ✅ Deterministic and reproducible
- ✅ Automatic execution
- ✅ No manual intervention
- ✅ Designed for robustness across videos

---

# Conclusion

This project focuses on robust engineering and interpretable classical computer vision rather than model complexity.

By combining:

- Motion-based segmentation
- Automatic motion-direction estimation
- Adaptive counting line placement
- Centroid tracking
- Dual independent executions

the solution provides a deterministic and explainable vehicle counting pipeline that remains fully compliant with the constraints of the Vehant Technologies Vehicle Count Hackathon.
