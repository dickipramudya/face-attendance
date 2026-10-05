"""Quick check: open both cameras, time capture + face detection + recognition."""
import time, cv2, numpy as np, sys
from pathlib import Path
M = str(Path(__file__).resolve().parent.parent / "models") + "/"
det = cv2.FaceDetectorYN.create(M + "face_detection_yunet_2023mar.onnx", "", (640, 360), 0.8, 0.3, 5000)
rec = cv2.FaceRecognizerSF.create(M + "face_recognition_sface_2021dec.onnx", "")
cv2.setNumThreads(4)

for dev, (w, h) in (("/dev/video1", (1280, 720)), ("/dev/video3", (1280, 720))):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 30)
    ok, frame = cap.read()
    if not ok:
        print(dev, "no frame"); continue
    print(dev, "frame", frame.shape)
    t0 = time.time(); n = 0
    while time.time() - t0 < 3:
        ok, frame = cap.read(); n += 1
    print(f"  capture+decode: {n/3:.1f} fps")
    small = cv2.resize(frame, (640, 360))
    t = time.time()
    for _ in range(10):
        _, faces = det.detect(small)
    print(f"  YuNet 640x360: {(time.time()-t)*100:.0f} ms/frame, faces={0 if faces is None else len(faces)}")
    if faces is not None:
        f = faces[0].copy(); f[:14] *= frame.shape[1] / 640
        t = time.time()
        for _ in range(10):
            emb = rec.feature(rec.alignCrop(frame, f))
        print(f"  SFace per face: {(time.time()-t)*100:.0f} ms, emb {emb.shape}")
    cv2.imwrite(f"/tmp/snap_{dev[-1]}.jpg", frame)
    cap.release()
