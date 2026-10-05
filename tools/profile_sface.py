import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2, numpy as np
from app.face import FaceEngine
from app import config
cap = cv2.VideoCapture("/dev/video1", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
for _ in range(10): cap.read()
frame = cap.read()[1]
eng = FaceEngine()
face = eng.detect(frame)[0]
aligned = eng.rec.alignCrop(frame, face)
ref = eng.embed(frame, face)
print([e for e in dir(cv2.dnn) if "ENGINE" in e])
path = str(config.MODELS / "face_recognition_sface_2021dec.onnx")
for name, kw in (("new", {}), ("classic", {"engine": cv2.dnn.ENGINE_CLASSIC})):
    net = cv2.dnn.readNetFromONNX(path, **kw)
    def run():
        net.setInput(cv2.dnn.blobFromImage(aligned, 1, (112, 112), (0, 0, 0), True, False))
        v = net.forward().flatten(); return v / np.linalg.norm(v)
    run(); t0 = time.perf_counter()
    for _ in range(10): v = run()
    print(name, f"{(time.perf_counter() - t0) * 100:.1f} ms", "cos vs FaceRecognizerSF", float(v @ ref))
t0 = time.perf_counter()
for _ in range(10): eng.rec.alignCrop(frame, face)
print("alignCrop", f"{(time.perf_counter() - t0) * 100:.1f} ms")
