import swatahVision as sv
import numpy as np
import cv2


class BlazeFaceInference:
    def __init__(self, model_path: str, conf_thresh: float = 0.5):
        self.conf_thresh = conf_thresh

        self.model = sv.Model(
            model=model_path,
            engine=sv.Engine.ONNX,
            hardware=sv.Hardware.CPU
        )

        self.box_annotator = sv.BoxAnnotator(color=sv.Color.YELLOW)

    def preprocess(self, frame):
        H, W = frame.shape[:2]

        img128 = cv2.resize(frame, (128, 128))
        img_np = img128.astype(np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))[None, ...]

        return img_np, (H, W)

    def inference(self, frame):
        img_np, shape = self.preprocess(frame)

        outs = self.model.runtime_engine.session.run(
            None,
            {"image": img_np}
        )

        return outs, shape

    def postprocess(self, outs, shape):
        H, W = shape

        boxes = outs[0][0]
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, 16)

        scores = outs[1][0] if len(outs) > 1 else np.ones(len(boxes))
        scores = np.atleast_1d(scores)

        xyxy, confs = [], []

        for det, score in zip(boxes, scores):
            if score < self.conf_thresh:
                continue

            top_y, top_x, bot_y, bot_x = det[:4]

            x1 = int(top_x * W)
            y1 = int(top_y * H)
            x2 = int(bot_x * W)
            y2 = int(bot_y * H)

            if x2 - x1 < 5 or y2 - y1 < 5:
                continue

            xyxy.append([x1, y1, x2, y2])
            confs.append(score)

        if len(xyxy) == 0:
            return sv.Detections.empty(), boxes, scores

        detections = sv.Detections(
            xyxy=np.array(xyxy),
            confidence=np.array(confs),
            class_id=np.zeros(len(xyxy), dtype=int)
        )

        return detections, boxes, scores

    def draw_landmarks_and_labels(self, frame, boxes, scores, shape):
        H, W = shape

        for det, score in zip(boxes, scores):
            if score < self.conf_thresh:
                continue

            (
                top_y, top_x, bot_y, bot_x,
                ley_x, ley_y, rey_x, rey_y,
                nose_x, nose_y, mou_x, mou_y,
                lea_x, lea_y, rea_x, rea_y
            ) = det

            x1 = int(top_x * W)
            y1 = int(top_y * H)

            # 🔴 Confidence label
            cv2.putText(
                frame,
                f"{score:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            # 🔵 Landmarks
            landmarks = [
                (ley_x, ley_y),
                (rey_x, rey_y),
                (nose_x, nose_y),
                (mou_x, mou_y),
                (lea_x, lea_y),
                (rea_x, rea_y)
            ]

            for nx, ny in landmarks:
                cx = int(nx * W)
                cy = int(ny * H)
                cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)

        return frame

    def process_frame(self, frame):
        outs, shape = self.inference(frame)
        detections, boxes, scores = self.postprocess(outs, shape)

        frame = self.box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )

        frame = self.draw_landmarks_and_labels(frame, boxes, scores, shape)

        return frame

    def run_webcam(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            output = self.process_frame(frame)

            cv2.imshow("BlazeFace Webcam", output)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break

        cap.release()
        cv2.destroyAllWindows()

    def run_video(self, video_path):
        cap = cv2.VideoCapture(video_path)

        #FPS of video
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps) if fps > 0 else 30  # fallback

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            output = self.process_frame(frame)

            cv2.imshow("BlazeFace Video", output)

            if cv2.waitKey(delay) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


# ---------------------------------------------
# Run
# ---------------------------------------------
if __name__ == "__main__":
    model = BlazeFaceInference(
        model_path="blazeface.onnx",
        conf_thresh=0.3
    )

    # 🔹 Choose one:
    model.run_webcam()
    #model.run_video("your video path")