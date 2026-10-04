"""Turns "person X seen on camera Y" into check-in / check-out records.

Two modes, matching how the industry does it:
  gate  - each camera has a direction. Entrance camera = check-in, exit camera
          = check-out. This is the turnstile model, and it also tells you who is
          inside right now.
  kiosk - cameras are punch points. The first scan of the day is the check-in,
          and every later scan moves the check-out forward ("first in, last
          out"), which is what typical attendance terminals and payroll use.
"""
import threading
import time
from collections import deque
from datetime import datetime, timedelta

import cv2

from . import config

DATA = config.DATA


def today(ts=None):
    return datetime.fromtimestamp(ts or time.time()).strftime("%Y-%m-%d")


def hhmm(ts):
    return datetime.fromtimestamp(ts).strftime("%H:%M")


def greeting(ts):
    h = datetime.fromtimestamp(ts).hour
    return "Good morning" if h < 11 else "Good afternoon" if h < 15 else "Good evening" if h < 19 else "Good night"


def minutes_late(first_in_ts, settings):
    start = datetime.strptime(settings["work_start"], "%H:%M").time()
    t = datetime.fromtimestamp(first_in_ts)
    due = datetime.combine(t.date(), start) + timedelta(minutes=settings["late_grace_min"])
    late = (t - due).total_seconds() / 60
    return int(late) if late > 0 else 0


class Attendance:
    def __init__(self, db):
        self.db = db
        self.lock = threading.Lock()
        self.last_seen = {}                 # (person_id, camera_id) -> ts
        self.recent = deque(maxlen=12)      # newest first, for the screens

    def record(self, person_id, name, camera, score, face_img):
        """Logs a sighting if outside the cooldown. Returns the event dict or None."""
        s = config.load()
        now = time.time()
        key = (person_id, camera["id"])
        with self.lock:
            if now - self.last_seen.get(key, 0) < s["cooldown_sec"]:
                return None
            self.last_seen[key] = now

            day = today(now)
            if s["mode"] == "gate":
                kind = "out" if camera.get("role") == "out" else "in"
            else:
                had = self.db.q("SELECT 1 FROM events WHERE person_id = ? AND day = ? LIMIT 1", (person_id, day))
                kind = "out" if had else "in"

            snap = ""
            if face_img is not None and face_img.size:
                rel = f"snaps/{day}/{person_id}_{int(now)}.jpg"
                path = DATA / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), face_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                snap = rel
            eid = self.db.add_event(person_id, camera["id"], kind, now, day, score, snap)

            first_in = self.db.q("SELECT MIN(ts) AS t FROM events WHERE person_id = ? AND day = ? AND kind = 'in'",
                                 (person_id, day))[0]["t"]
            late = minutes_late(first_in, s) if kind == "in" and first_in == now else 0
            ev = {
                "id": eid, "person_id": person_id, "name": name, "camera": camera["name"],
                "camera_id": camera["id"], "kind": kind, "ts": now, "time": hhmm(now),
                "score": round(score, 3), "snapshot": snap, "late_min": late,
                "greeting": greeting(now) if kind == "in" else "Goodbye",
            }
            self.recent.appendleft(ev)
            return ev

    def forget(self, person_id):
        """Drops a deleted person from the on-screen event list."""
        with self.lock:
            kept = [e for e in self.recent if e["person_id"] != person_id]
            self.recent.clear()
            self.recent.extend(kept)
            for key in [k for k in self.last_seen if k[0] == person_id]:
                del self.last_seen[key]

    # ── reports ─────────────────────────────────────────────────────────
    def day_report(self, day):
        """One row per registered person: first check-in, last check-out, status."""
        s = config.load()
        events = self.db.q("SELECT * FROM events WHERE day = ? ORDER BY ts", (day,))
        by_person = {}
        for e in events:
            by_person.setdefault(e["person_id"], []).append(e)
        rows = []
        for p in self.db.people():
            evs = by_person.get(p["id"], [])
            ins = [e["ts"] for e in evs if e["kind"] == "in"]
            outs = [e["ts"] for e in evs if e["kind"] == "out"]
            first_in = min(ins) if ins else None
            last_out = max(outs) if outs else None
            if s["mode"] == "kiosk" and not outs and len(evs) > 1:
                last_out = evs[-1]["ts"]
            inside = bool(evs) and evs[-1]["kind"] == "in"
            if not evs:
                status = "Absent"
            elif first_in is None:
                status = "Check-out only"
            else:
                late = minutes_late(first_in, s)
                status = f"Late {late} min" if late else "On time"
            rows.append({
                "person_id": p["id"], "emp_no": p["emp_no"] or "", "name": p["name"], "dept": p["dept"],
                "photo": p["photo"], "first_in": hhmm(first_in) if first_in else "",
                "last_out": hhmm(last_out) if last_out else "", "status": status,
                "inside": inside, "events": len(evs),
            })
        return rows
