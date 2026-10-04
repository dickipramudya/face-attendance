"""One worker per camera: capture, detect every face, recognise, track, annotate.

Several people can be in view at once. Each face gets a track (matched frame
to frame by box overlap), and a track's name comes from a vote over its last
few recognitions, so one blurry frame cannot flip a label. Recognition is
re-run on a confident track only now and then, which keeps the CPU free for
crowded frames.
"""
import logging
import threading
import time
from collections import Counter, deque

import cv2

from . import render
from .face import FaceEngine

log = logging.getLogger("camera")

PROCESS_FPS = 6          # detection/recognition passes per second
DECODE_FPS = 8           # camera frames decoded per second (slightly above PROCESS_FPS)
MAX_RECOGNITIONS_PER_PASS = 2
IDLE_DETECT_SEC = 1.0    # detector interval while the scene is empty and still
MOTION_LEVEL = 2.0       # mean grey-level change (0-255) that counts as movement
RECHECK_SEC = 2.0        # re-recognise a confident track this often
TRACK_TTL = 1.0          # forget a track not seen for this long
VOTES_NEEDED = 2         # agreeing recognitions before a name is shown


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class Track:
    _next = 1

    def __init__(self, box):
        self.id = Track._next
        Track._next += 1
        self.box = box
        self.seen = time.time()
        self.votes = deque(maxlen=5)      # person_id or 0 for "unknown"
        self.scores = {}
        self.last_rec = 0.0
        self.person_id = None
        self.name = None
        self.score = 0.0
        self.reported = False

    def vote(self, pid, name, score):
        self.votes.append(pid or 0)
        if pid:
            self.scores[pid] = (name, score)
        top, count = Counter(self.votes).most_common(1)[0]
        if count >= VOTES_NEEDED:
            if top:
                self.person_id, (self.name, self.score) = top, self.scores[top]
            else:
                self.person_id, self.name, self.score = None, None, score

    @property
    def confident(self):
        return len(self.votes) >= VOTES_NEEDED


class CameraWorker:
    def __init__(self, cam_cfg, gallery, on_identified):
        self.cfg = cam_cfg
        self.id = cam_cfg["id"]
        self.gallery = gallery
        self.on_identified = on_identified   # (camera_worker, track, frame, face_row) -> optional sub-label
        self.engine = None
        self.cap = None
        self.frame = None                    # latest raw frame
        self.frame_lock = threading.Lock()
        self.annotated = None                # latest frame with boxes and names
        self.jpeg = None
        self.tracks = []
        self.sublabels = {}                  # track id -> (text, until)
        self.status = "starting"
        self.fps = 0.0
        self.running = False
        self._prev_small = None              # tiny grey copy of the last frame, for motion
        self._last_detect = 0.0

    # ── lifecycle ───────────────────────────────────────────────────────
    def start(self):
        self.running = True
        threading.Thread(target=self._capture_loop, name=f"cap-{self.id}", daemon=True).start()
        threading.Thread(target=self._process_loop, name=f"proc-{self.id}", daemon=True).start()

    def stop(self):
        self.running = False

    def _open(self):
        cap = cv2.VideoCapture(self.cfg["device"], cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.get("width", 1280))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.get("height", 720))
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap if cap.isOpened() else None

    def _capture_loop(self):
        """Keeps only the newest frame, so processing never works on stale video.

        The camera is drained at its full rate with grab(), which does not
        decode; a JPEG is decoded only as often as the processing loop can use
        one. Decoding every 720p frame from two cameras would eat most of the CPU.
        """
        next_decode = 0.0
        while self.running:
            if self.cap is None:
                self.cap = self._open()
                if self.cap is None:
                    self.status = "camera not found"
                    time.sleep(3)
                    continue
                self.status = "live"
            if not self.cap.grab():
                self.status = "no signal, retrying"
                self.cap.release()
                self.cap = None
                time.sleep(1)
                continue
            now = time.time()
            if now < next_decode:
                continue
            ok, frame = self.cap.retrieve()
            if not ok:
                continue
            next_decode = now + 1.0 / DECODE_FPS
            with self.frame_lock:
                self.frame = frame

    def _process_loop(self):
        self.engine = FaceEngine()
        period = 1.0 / PROCESS_FPS
        last = time.time()
        while self.running:
            t0 = time.time()
            with self.frame_lock:
                frame = None if self.frame is None else self.frame.copy()
            if frame is None:
                time.sleep(0.1)
                continue
            try:
                self._process(frame)
            except Exception:
                log.exception("processing failed on %s", self.id)
            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(1e-3, now - last))
            last = now
            time.sleep(max(0.0, period - (now - t0)))

    # ── per frame ───────────────────────────────────────────────────────
    def _process(self, frame):
        now = time.time()

        # Nobody in view and nothing moving: look for faces only once a second,
        # so an idle terminal stays cool instead of running the detector flat out.
        small = cv2.cvtColor(cv2.resize(frame, (80, 45), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        moving = self._prev_small is None or cv2.absdiff(small, self._prev_small).mean() > MOTION_LEVEL
        self._prev_small = small
        if not self.tracks and not moving and now - self._last_detect < IDLE_DETECT_SEC:
            self._annotate(frame, now)
            return
        self._last_detect = now

        faces = self.engine.detect(frame)

        # Match detections to existing tracks by overlap, greedily.
        unmatched = list(range(len(faces)))
        assigned = {}
        for tr in sorted(self.tracks, key=lambda t: -t.seen):
            best, best_iou = None, 0.3
            for i in unmatched:
                o = iou(tr.box, tuple(faces[i][:4]))
                if o > best_iou:
                    best, best_iou = i, o
            if best is not None:
                tr.box, tr.seen = tuple(faces[best][:4]), now
                assigned[tr.id] = best
                unmatched.remove(best)
        for i in unmatched:
            tr = Track(tuple(faces[i][:4]))
            self.tracks.append(tr)
            assigned[tr.id] = i
        self.tracks = [t for t in self.tracks if now - t.seen < TRACK_TTL]

        # Recognise new or unsure tracks first, confident ones occasionally. One
        # recognition costs about 110 ms here, so a crowded frame is named a few
        # faces per pass rather than stalling the video.
        due = [t for t in self.tracks if t.id in assigned
               and faces[assigned[t.id]][2] >= 40        # too few pixels to recognise reliably
               and (not t.confident or now - t.last_rec > RECHECK_SEC)]
        due.sort(key=lambda t: (t.confident, len(t.votes), t.last_rec))
        for tr in due[:MAX_RECOGNITIONS_PER_PASS]:
            pid, name, score = self.gallery.match(self.engine.embed(frame, faces[assigned[tr.id]]))
            tr.vote(pid, name, score)
            tr.last_rec = now

        for tr in self.tracks:
            i = assigned.get(tr.id)
            if i is None:
                continue
            if tr.person_id and not tr.reported:
                tr.reported = True
                sub = self.on_identified(self, tr, frame, faces[i])
                if sub:
                    self.sublabels[tr.id] = (sub, now + 6)

        self._annotate(frame, now)

    def _annotate(self, frame, now):
        for tr in self.tracks:
            x, y, w, h = tr.box
            if tr.person_id:
                sub = self.sublabels.get(tr.id)
                sub = sub[0] if sub and sub[1] > now else f"{tr.score * 100:.0f}% match"
                render.face_box(frame, x, y, w, h, tr.name, sub, render.GREEN)
            elif tr.confident:
                render.face_box(frame, x, y, w, h, "Unknown", "not registered", render.AMBER)
            else:
                render.face_box(frame, x, y, w, h, "Checking…", None, render.BLUE)
        self.annotated = frame
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            self.jpeg = buf.tobytes()

    def latest_frame(self):
        """A copy of the newest raw (unannotated) frame, or None."""
        with self.frame_lock:
            return None if self.frame is None else self.frame.copy()
