import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import argparse

from dataclasses import dataclass
from typing import List


# ============================
# KEYPOINT CONFIG
# ============================
KEYPOINT_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle",
]

SKELETON_EDGES = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,11),(6,12),(11,12),
    (5,7),(7,9),(6,8),(8,10),
    (11,13),(13,15),(12,14),(14,16),
]


# ============================
# DATA CLASSES
# ============================
@dataclass
class KeyPoint:
    name: str
    index: int
    x: float
    y: float
    confidence: float

    @classmethod
    def from_movenet_tensor(cls, index, raw_y, raw_x, score):
        return cls(
            name=KEYPOINT_NAMES[index],
            index=index,
            x=float(np.clip(raw_x, 0, 1)),
            y=float(np.clip(raw_y, 0, 1)),
            confidence=float(np.clip(score, 0, 1)),
        )

    def to_pixel(self, w, h):
        return int(self.x * w), int(self.y * h)


@dataclass
class Pose:
    keypoints: List[KeyPoint]

    def visible(self, min_conf=0.3):
        return [kp for kp in self.keypoints if kp.confidence >= min_conf]


# ============================
# MODEL
# ============================
class MoveNetOutput:
    URL = "https://tfhub.dev/google/movenet/singlepose/lightning/4"

    def __init__(self):
        print("Loading MoveNet...")
        module = hub.load(self.URL)
        self.model = module.signatures["serving_default"]
        print("Loaded ✓")

    def process(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (192,192))
        tensor = tf.cast(tf.expand_dims(img,0), tf.int32)

        outputs = self.model(input=tensor)
        data = outputs["output_0"].numpy()[0,0]

        kps = []
        for i in range(17):
            y,x,c = data[i]
            kps.append(KeyPoint.from_movenet_tensor(i,y,x,c))

        return Pose(kps)


# ============================
# SMOOTHING
# ============================
def smooth_keypoints(prev_pose, curr_pose, alpha=0.4, conf_threshold=0.3):
    if prev_pose is None:
        return curr_pose

    smoothed = []

    for p, c in zip(prev_pose.keypoints, curr_pose.keypoints):
        if c.confidence < conf_threshold:
            smoothed.append(p)
            continue

        x = alpha * c.x + (1 - alpha) * p.x
        y = alpha * c.y + (1 - alpha) * p.y

        smoothed.append(KeyPoint(c.name, c.index, x, y, c.confidence))

    return Pose(smoothed)


def clamp_motion(prev_pose, curr_pose, max_delta=0.05):
    if prev_pose is None:
        return curr_pose

    clamped = []

    for p, c in zip(prev_pose.keypoints, curr_pose.keypoints):
        dx = np.clip(c.x - p.x, -max_delta, max_delta)
        dy = np.clip(c.y - p.y, -max_delta, max_delta)

        clamped.append(KeyPoint(c.name, c.index, p.x+dx, p.y+dy, c.confidence))

    return Pose(clamped)


# ============================
# DRAW
# ============================
def draw_pose(frame, pose, min_conf=0.3):
    h,w = frame.shape[:2]

    pts = {}
    for kp in pose.keypoints:
        pts[kp.index] = kp.to_pixel(w,h) if kp.confidence >= min_conf else None

    for i,j in SKELETON_EDGES:
        if pts[i] and pts[j]:
            cv2.line(frame, pts[i], pts[j], (0,255,255), 2)

    for kp in pose.keypoints:
        pt = pts[kp.index]
        if pt:
            cv2.circle(frame, pt, 5, (0,255,0), -1)

    return frame


# ============================
# PIPELINE
# ============================
class PosePipeline:
    def __init__(self, alpha=0.4, max_delta=0.05):
        self.model = MoveNetOutput()
        self.prev_pose = None
        self.alpha = alpha
        self.max_delta = max_delta

    def run(self, frame):
        pose = self.model.process(frame)
        pose = clamp_motion(self.prev_pose, pose, self.max_delta)
        pose = smooth_keypoints(self.prev_pose, pose, self.alpha)

        self.prev_pose = pose
        return draw_pose(frame.copy(), pose), pose


# ============================
# VIDEO PROCESSING
# ============================
def run_and_save(input_path, output_path="output.mp4", alpha=0.4, max_delta=0.05):

    pipeline = PosePipeline(alpha=alpha, max_delta=max_delta)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w,h)
    )

    print(f"Processing {input_path} → {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, _ = pipeline.run(frame)
        out.write(annotated)

        cv2.imshow("Pose", annotated)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"✅ Saved: {output_path}")


# ============================
# CLI ENTRY POINT
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoveNet Pose Estimation with Smoothing")

    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", default="output.mp4", help="Output video path")
    parser.add_argument("--alpha", type=float, default=0.4, help="Smoothing factor (0-1)")
    parser.add_argument("--max_delta", type=float, default=0.05, help="Max joint movement per frame")

    args = parser.parse_args()

    run_and_save(
        input_path=args.input,
        output_path=args.output,
        alpha=args.alpha,
        max_delta=args.max_delta
    )