"""Face attendance server: cameras, recognition, monitor output and web UI.

    venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8090
"""
import asyncio
import base64
import csv
import io
import logging
import secrets
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .attendance import Attendance, today
from .camera import CameraWorker
from .db import Database
from .display import FramebufferDisplay
from .face import REG_DET_WIDTH, FaceEngine, Gallery

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("main")

WEB = Path(__file__).parent / "web"
DATA = config.DATA
FACES = DATA / "faces"

app = FastAPI(title="Face Attendance")
state = {}


# ── startup ─────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    s = config.load()
    FACES.mkdir(parents=True, exist_ok=True)
    db = Database(DATA / "attendance.db")
    gallery = Gallery(db)
    att = Attendance(db)

    def on_identified(worker, track, frame, face):
        ev = att.record(track.person_id, track.name, worker.cfg, track.score, worker.engine.crop(frame, face))
        if ev is None:
            return None
        return ("Checked in " if ev["kind"] == "in" else "Checked out ") + ev["time"]

    workers = [CameraWorker(c, gallery, on_identified) for c in s["cameras"] if c.get("enabled", True)]
    for w in workers:
        w.start()
    display = FramebufferDisplay(workers, att)
    display.start()
    state.update(db=db, gallery=gallery, att=att, workers=workers, display=display,
                 reg_engine=None, reg_lock=threading.Lock(), pending={})
    log.info("started with %d camera(s), %d registered face(s)", len(workers), len(gallery.ids))


@app.on_event("shutdown")
def shutdown():
    for w in state.get("workers", []):
        w.stop()
    if state.get("display"):
        state["display"].stop()


def worker(cam_id):
    for w in state["workers"]:
        if w.id == cam_id:
            return w
    raise HTTPException(404, "unknown camera")


def reg_engine():
    if state["reg_engine"] is None:
        state["reg_engine"] = FaceEngine(REG_DET_WIDTH)
    return state["reg_engine"]


# ── pages ───────────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/", response_class=HTMLResponse)
def page_index():
    return (WEB / "index.html").read_text()


@app.get("/{page}", response_class=HTMLResponse)
def page(page: str):
    f = WEB / f"{page}.html"
    if page not in {"register", "people", "report", "settings"} or not f.exists():
        raise HTTPException(404)
    return f.read_text()


@app.get("/data/{path:path}")
def data_file(path: str):
    """Face photos and snapshots. Only files under faces/ and snaps/ are served."""
    p = (DATA / path).resolve()
    if not (str(p).startswith(str(FACES.resolve())) or str(p).startswith(str((DATA / "snaps").resolve()))):
        raise HTTPException(404)
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


# ── live ────────────────────────────────────────────────────────────────

@app.get("/stream/{cam_id}")
async def stream(cam_id: str):
    w = worker(cam_id)

    async def gen():
        last = None
        while True:
            jpg = w.jpeg
            if jpg is not None and jpg is not last:
                last = jpg
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/monitor.jpg")
def monitor_preview(w: int = 1920, h: int = 1080):
    """The picture the HDMI monitor shows, so the layout can be checked without one."""
    img = state["display"].compose(max(640, min(w, 1920)), max(360, min(h, 1080)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buf.tobytes(), media_type="image/jpeg")


@app.get("/api/state")
def api_state():
    s = config.load()
    return {
        "org_name": s["org_name"], "mode": s["mode"], "today": today(),
        "cameras": [{"id": w.id, "name": w.cfg["name"], "role": w.cfg.get("role"), "status": w.status,
                     "fps": round(w.fps, 1), "faces": len(w.tracks)} for w in state["workers"]],
        "recent": list(state["att"].recent),
        "inside": [r for r in state["att"].day_report(today()) if r["inside"]] if s["mode"] == "gate" else [],
        "registered": len(state["db"].people()),
    }


# ── registration ────────────────────────────────────────────────────────

def _face_from(frame, uploaded=False):
    """(face_row, None) for exactly one usable face, else (None, error)."""
    eng = reg_engine()
    faces = eng.detect(frame)
    if len(faces) == 0:
        return None, ("No face found in this photo. Use a clear, front-facing photo." if uploaded
                      else "No face found. Look straight at the camera, in good light.")
    faces = faces[np.argsort(-faces[:, 2] * faces[:, 3])]
    if len(faces) > 1 and faces[1][2] * faces[1][3] > 0.4 * faces[0][2] * faces[0][3]:
        return None, ("This photo has more than one face. Use a photo of this person alone." if uploaded
                      else "More than one face in view. Only the person registering should be in frame.")
    # SFace works on a 112 px aligned crop; much smaller faces give weak matches.
    if faces[0][2] < (64 if uploaded else 80):
        return None, ("The face in this photo is too small. Use a closer photo (head and shoulders)." if uploaded
                      else "Face too small. Come closer to the camera.")
    return faces[0], None


def _decode_upload(data: bytes):
    """BGR image upright as the phone shot it (EXIF rotation applied), or None."""
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:
        return None
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def _pending_add(frame, face):
    eng = reg_engine()
    emb = eng.embed(frame, face)
    crop = eng.crop(frame, face)
    token = secrets.token_hex(8)
    state["pending"][token] = (emb, crop, time.time())
    # drop stale captures
    for k in [k for k, v in state["pending"].items() if time.time() - v[2] > 900]:
        state["pending"].pop(k, None)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {"token": token, "image": "data:image/jpeg;base64," + base64.b64encode(buf).decode()}


@app.post("/api/capture/{cam_id}")
def api_capture(cam_id: str):
    frame = worker(cam_id).latest_frame()
    if frame is None:
        raise HTTPException(409, "Camera has no picture yet")
    with state["reg_lock"]:
        face, err = _face_from(frame)
        if err:
            return JSONResponse({"error": err}, 422)
        return _pending_add(frame, face)


@app.post("/api/upload-photo")
async def api_upload(photo: UploadFile = File(...)):
    frame = _decode_upload(await photo.read())
    if frame is None:
        log.info("upload %s rejected: not a readable image", photo.filename)
        return JSONResponse({"error": "Cannot read this file. Use a JPG or PNG photo (iPhone HEIC is not supported)."}, 422)
    if max(frame.shape[:2]) > 1920:
        f = 1920 / max(frame.shape[:2])
        frame = cv2.resize(frame, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    with state["reg_lock"]:
        face, err = _face_from(frame, uploaded=True)
        # Some apps strip the EXIF rotation tag, leaving a portrait shot sideways.
        for rot in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180):
            if face is not None:
                break
            turned = cv2.rotate(frame, rot)
            face, _ = _face_from(turned, uploaded=True)
            if face is not None:
                frame, err = turned, None
        if err:
            log.info("upload %s (%dx%d) rejected: %s", photo.filename, frame.shape[1], frame.shape[0], err)
            return JSONResponse({"error": err}, 422)
        return _pending_add(frame, face)


@app.post("/api/people")
def api_add_person(name: str = Form(...), emp_no: str = Form(""), dept: str = Form(""), phone: str = Form(""),
                   consent: str = Form(""), tokens: str = Form(...)):
    if consent != "yes":
        raise HTTPException(400, "Consent is required to store a face")
    toks = [t for t in tokens.split(",") if t in state["pending"]]
    if not toks:
        raise HTTPException(400, "Add at least one face photo (the photos expire after 15 minutes)")
    db = state["db"]
    try:
        pid = db.add_person(emp_no.strip(), name.strip(), dept.strip(), phone.strip())
    except Exception:
        raise HTTPException(409, "That ID number is already registered")
    folder = FACES / str(pid)
    folder.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(toks):
        emb, crop, _ = state["pending"].pop(t)
        rel = f"faces/{pid}/{i + 1}.jpg"
        cv2.imwrite(str(DATA / rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        db.add_face(pid, emb, rel)
    state["gallery"].reload()
    return {"id": pid}


@app.get("/api/people")
def api_people():
    return state["db"].people()


@app.delete("/api/people/{pid}")
def api_delete_person(pid: int):
    """Removes the person, their face data and photos, and their attendance history."""
    for e in state["db"].q("SELECT snapshot FROM events WHERE person_id = ? AND snapshot != ''", (pid,)):
        (DATA / e["snapshot"]).unlink(missing_ok=True)
    state["db"].delete_person(pid)
    shutil.rmtree(FACES / str(pid), ignore_errors=True)
    state["gallery"].reload()
    state["att"].forget(pid)
    return {"ok": True}


# ── reports ─────────────────────────────────────────────────────────────

@app.get("/api/report")
def api_report(day: str = ""):
    return {"day": day or today(), "rows": state["att"].day_report(day or today())}


@app.get("/api/report.csv")
def api_report_csv(day_from: str, day_to: str):
    rows = state["db"].events_between(day_from, day_to)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["date", "time", "emp_no", "name", "dept", "type", "camera", "match"])
    for r in rows:
        t = time.strftime("%H:%M:%S", time.localtime(r["ts"]))
        w.writerow([r["day"], t, r["emp_no"] or "", r["name"], r["dept"], "check-in" if r["kind"] == "in" else "check-out",
                    r["camera_id"], f"{r['score']:.2f}"])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="attendance_{day_from}_{day_to}.csv"'})


# ── settings ────────────────────────────────────────────────────────────

@app.get("/api/settings")
def api_settings():
    return config.load()


@app.post("/api/settings")
async def api_save_settings(body: dict):
    allowed = {k: v for k, v in body.items() if k in
               {"org_name", "mode", "work_start", "work_end", "late_grace_min", "match_threshold",
                "cooldown_sec", "display_enabled", "cameras"}}
    s = config.save(allowed)
    # Camera names and roles apply live; device changes need a restart.
    for w in state["workers"]:
        for c in s["cameras"]:
            if c["id"] == w.id:
                w.cfg.update({k: c[k] for k in ("name", "role") if k in c})
    return s
