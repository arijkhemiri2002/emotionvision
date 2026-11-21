"""Inference engine: face detection (YuNet) -> preprocessing -> ONNX Runtime -> Grad-CAM.

Kept free of torch so the API Docker image stays small (numpy + opencv + onnxruntime).
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from src.explain.cam import cam_for_class, overlay

FER_SIZE = 48          # resolution of the training images
FACE_MARGIN = 0.15     # extra context around the detected box (FER+ crops include forehead/chin)


class EmotionEngine:
    def __init__(self, onnx_path: str, config_path: str, detector_path: str,
                 uncertainty_threshold: float = 0.40, min_face_score: float = 0.7):
        cfg = json.loads(Path(config_path).read_text())
        self.model_name: str = cfg["model"]
        self.classes: list[str] = cfg["classes"]
        self.input_size: int = cfg["input_size"]
        self.mean = np.array(cfg["mean"], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(cfg["std"], dtype=np.float32).reshape(3, 1, 1)
        self.onnx_path = onnx_path
        self.uncertainty_threshold = uncertainty_threshold

        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.detector = cv2.FaceDetectorYN.create(detector_path, "", (320, 320), min_face_score, 0.3, 50)

    # ------------------------------------------------------------------ faces
    def detect_faces(self, img_bgr: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        h, w = img_bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img_bgr)
        boxes = []
        if faces is not None:
            for f in faces:
                x, y, bw, bh, score = int(f[0]), int(f[1]), int(f[2]), int(f[3]), float(f[-1])
                boxes.append(self._square_box(x, y, bw, bh, w, h) + (score,))
        return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)

    @staticmethod
    def _square_box(x, y, bw, bh, w, h) -> tuple[int, int, int, int]:
        """Square box around the face + margin, clipped to the image."""
        side = int(max(bw, bh) * (1 + 2 * FACE_MARGIN))
        cx, cy = x + bw // 2, y + bh // 2
        x0, y0 = max(cx - side // 2, 0), max(cy - side // 2, 0)
        x1, y1 = min(x0 + side, w), min(y0 + side, h)
        return x0, y0, max(x1 - x0, 1), max(y1 - y0, 1)

    # ------------------------------------------------------------ preprocess
    def preprocess(self, face_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Same pipeline as training: grayscale, 48x48 (FER domain), resize, 3 channels, normalize."""
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray48 = cv2.resize(gray, (FER_SIZE, FER_SIZE), interpolation=cv2.INTER_AREA)
        up = cv2.resize(gray48, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        rgb = np.repeat(up[None, :, :], 3, axis=0).astype(np.float32) / 255.0
        x = ((rgb - self.mean) / self.std)[None].astype(np.float32)
        return x, cv2.cvtColor(up, cv2.COLOR_GRAY2RGB)

    # --------------------------------------------------------------- predict
    def predict(self, image_bytes: bytes, explain: bool = False, max_faces: int = 5) -> dict:
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
        h, w = img.shape[:2]
        t0 = time.perf_counter()

        boxes = self.detect_faces(img)[:max_faces]
        face_detected = bool(boxes)
        if not face_detected:                       # fall back to the whole image
            side = min(h, w)
            boxes = [((w - side) // 2, (h - side) // 2, side, side, 0.0)]

        crops = [img[y:y + bh, x:x + bw] for x, y, bw, bh, _ in boxes]
        prepped = [self.preprocess(c) for c in crops]
        batch = np.concatenate([p[0] for p in prepped], axis=0)
        logits, cams = self.session.run(["logits", "cams"], {"input": batch})
        probs = _softmax(logits)

        faces = []
        for i, (x, y, bw, bh, score) in enumerate(boxes):
            k = int(probs[i].argmax())
            face = {
                "box": {"x": x, "y": y, "width": bw, "height": bh},
                "detection_score": round(score, 4),
                "label": self.classes[k],
                "confidence": round(float(probs[i, k]), 4),
                "uncertain": bool(probs[i, k] < self.uncertainty_threshold),
                "probabilities": {c: round(float(p), 4) for c, p in zip(self.classes, probs[i], strict=True)},
                "heatmap_png_base64": None,
            }
            if explain:
                face_rgb = prepped[i][1]
                heat = cam_for_class(cams[i], k, face_rgb.shape[:2], mode="contrastive")
                face["heatmap_png_base64"] = _png_b64(overlay(face_rgb, heat))
            faces.append(face)

        return {
            "faces": faces,
            "n_faces": len(faces) if face_detected else 0,
            "face_detected": face_detected,
            "image_width": w,
            "image_height": h,
            "inference_ms": round((time.perf_counter() - t0) * 1000, 2),
        }


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _png_b64(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode() if ok else ""
