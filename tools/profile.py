"""Times each per-frame stage on the box, with the service stopped."""
import sys, time
sys.path.insert(0, "/opt/absensi")
import cv2
from app.face import FaceEngine
from app import render

cap = cv2.VideoCapture(sys.argv[1] if len(sys.argv) > 1 else "/dev/video1", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
for _ in range(10): cap.read()
print("cv2 threads", cv2.getNumThreads())
eng = FaceEngine()

def t(label, fn, n=10):
    fn(); t0 = time.perf_counter()
    for _ in range(n): r = fn()
    print(f"{label:28s} {(time.perf_counter() - t0) / n * 1000:7.1f} ms"); return r

t("grab", cap.grab)
frame = t("read (grab+decode)", lambda: cap.read()[1])
faces = t("detect 640x360", lambda: eng.detect(frame))
print("faces", len(faces))
if len(faces):
    t("embed (align+SFace)", lambda: eng.embed(frame, faces[0]))
    f2 = frame.copy()
    t("face_box", lambda: render.face_box(f2, *faces[0][:4], "Dicki Pramudya", "92% match", render.GREEN))
t("imencode 1280x720 q75", lambda: cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75]))
t("frame.copy", frame.copy)
for n in (1, 2, 4):
    cv2.setNumThreads(n)
    t(f"detect threads={n}", lambda: eng.detect(frame))
cv2.setNumThreads(4)
for size in ((480, 270), (320, 180)):
    eng.det.setInputSize(size)
    def d():
        _, f = eng.det.detect(cv2.resize(frame, size)); return f
    r = t(f"detect {size[0]}x{size[1]}", d)
    print("   faces", 0 if r is None else len(r), "width", None if r is None else round(float(r[0][2]) * 1280 / size[0]))
