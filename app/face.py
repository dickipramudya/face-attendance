"""Face detection (YuNet) and recognition (SFace), both from the OpenCV model zoo.

Both models are Apache-2.0, so this can go into commercial work. The popular
InsightFace weights are licensed for non-commercial research only, which is
why they are not used here.
"""
import threading

import cv2
import numpy as np

from . import config

# Detection runs on a downscaled copy; recognition always uses full resolution.
# 480 px wide is about 55 ms on the box (640 px: about 90 ms) and still finds a
# face 2.5 m from the 720p camera. Registration uses 640 for steadier landmarks.
LIVE_DET_WIDTH = 480
REG_DET_WIDTH = 640


class FaceEngine:
    """One per camera thread: OpenCV DNN nets are not safe to share across threads."""

    def __init__(self, det_width=LIVE_DET_WIDTH):
        m = config.MODELS
        self.det_width = det_width
        # score threshold 0.85, NMS 0.3, at most 50 faces per frame
        self.det = cv2.FaceDetectorYN.create(
            str(m / "face_detection_yunet_2023mar.onnx"), "", (det_width, det_width * 9 // 16), 0.85, 0.3, 50)
        self.rec = cv2.FaceRecognizerSF.create(str(m / "face_recognition_sface_2021dec.onnx"), "")
        self._size = (det_width, det_width * 9 // 16)

    def detect(self, frame):
        """Faces in full-resolution coordinates: Nx15 (box x,y,w,h, 5 landmarks, score)."""
        h, w = frame.shape[:2]
        scale = self.det_width / w
        size = (self.det_width, int(round(h * scale)))
        if size != self._size:
            self.det.setInputSize(size)
            self._size = size
        _, faces = self.det.detect(cv2.resize(frame, size))
        if faces is None:
            return np.zeros((0, 15), np.float32)
        faces = faces.copy()
        faces[:, :14] /= scale
        return faces

    def embed(self, frame, face) -> np.ndarray:
        """Unit-length 128-d embedding of one detected face."""
        crop = self.rec.alignCrop(frame, face)
        f = self.rec.feature(crop).flatten().astype(np.float32)
        return f / (np.linalg.norm(f) + 1e-9)

    def crop(self, frame, face, pad=0.35):
        """A loose square crop around a face, for thumbnails and snapshots."""
        x, y, w, h = face[:4]
        s = max(w, h) * (1 + pad)
        cx, cy = x + w / 2, y + h / 2
        x0, y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
        x1, y1 = int(min(frame.shape[1], cx + s / 2)), int(min(frame.shape[0], cy + s / 2))
        return frame[y0:y1, x0:x1].copy()


class Gallery:
    """Every registered embedding in one matrix, so matching is a single dot product."""

    def __init__(self, db):
        self.db = db
        self.lock = threading.Lock()
        self.ids = np.zeros((0,), np.int64)
        self.mat = np.zeros((0, 128), np.float32)
        self.names = {}
        self.reload()

    def reload(self):
        rows = self.db.all_embeddings()
        names = {p["id"]: p["name"] for p in self.db.people()}
        with self.lock:
            self.ids = np.array([r[0] for r in rows], np.int64)
            self.mat = np.stack([r[1] for r in rows]) if rows else np.zeros((0, 128), np.float32)
            self.names = names

    def match(self, emb):
        """(person_id, name, score) of the best match, or (None, None, best_score)."""
        threshold = config.load()["match_threshold"]
        with self.lock:
            if len(self.ids) == 0:
                return None, None, 0.0
            sims = self.mat @ emb
            best = int(np.argmax(sims))
            score = float(sims[best])
            if score < threshold:
                return None, None, score
            pid = int(self.ids[best])
            return pid, self.names.get(pid, "?"), score
