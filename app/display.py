"""Draws the attendance screen straight to the HDMI framebuffer (/dev/fb0).

No desktop or browser is needed on the box, which keeps RAM free for
recognition. Layout: header (company, clock, mode), the camera views with
names over every face, then a strip of the latest check-ins/outs. A big
greeting banner shows for a few seconds after each new event.
"""
import fcntl
import logging
import mmap
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import config, render

log = logging.getLogger("display")

FB = "/dev/fb0"
KDSETMODE, KD_TEXT, KD_GRAPHICS = 0x4B3A, 0x00, 0x01
BANNER_SEC = 5
HDMI_STATUS = Path("/sys/class/drm/card0-HDMI-A-1/status")


def hdmi_connected():
    """No monitor, no drawing: composing the screen costs CPU the cameras need.
    Boxes without this sysfs node are assumed to have a monitor."""
    try:
        return HDMI_STATUS.read_text().strip() != "disconnected"
    except OSError:
        return True


class FramebufferDisplay:
    def __init__(self, workers, attendance):
        self.workers = workers
        self.att = attendance
        self.running = False
        self.thumbs = {}

    def start(self):
        if not os.path.exists(FB):
            log.warning("no framebuffer, monitor output disabled")
            return
        self.running = True
        threading.Thread(target=self._loop, name="display", daemon=True).start()

    def stop(self):
        self.running = False

    def _geometry(self):
        sysfs = Path("/sys/class/graphics/fb0")
        w, h = map(int, (sysfs / "virtual_size").read_text().split(","))
        bpp = int((sysfs / "bits_per_pixel").read_text())
        stride = int((sysfs / "stride").read_text())
        return w, h, bpp, stride

    def _loop(self):
        tty = None
        try:
            # Stop the text console drawing over the picture.
            tty = os.open("/dev/tty1", os.O_RDWR)
            fcntl.ioctl(tty, KDSETMODE, KD_GRAPHICS)
        except OSError:
            tty = None
        try:
            w, h, bpp, stride = self._geometry()
            log.info("framebuffer %dx%d %dbpp", w, h, bpp)
            with open(FB, "r+b") as f:
                fb = mmap.mmap(f.fileno(), stride * h)
                view = np.frombuffer(fb, dtype=np.uint8).reshape(h, stride)
                while self.running:
                    t0 = time.time()
                    if not config.load().get("display_enabled", True) or not hdmi_connected():
                        time.sleep(2)
                        continue
                    canvas = self.compose(w, h)
                    if bpp == 32:
                        view[:, : w * 4] = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA).reshape(h, w * 4)
                    elif bpp == 16:
                        view[:, : w * 2] = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGR565).reshape(h, w * 2)
                    time.sleep(max(0.0, 0.12 - (time.time() - t0)))
        except Exception:
            log.exception("display stopped")
        finally:
            if tty is not None:
                try:
                    fcntl.ioctl(tty, KDSETMODE, KD_TEXT)
                    os.close(tty)
                except OSError:
                    pass

    # ── layout ──────────────────────────────────────────────────────────
    def compose(self, W, H):
        s = config.load()
        c = np.full((H, W, 3), render.BG, np.uint8)
        u = max(1, W // 160)                     # layout unit
        header_h = u * 9
        footer_h = int(H * 0.24)

        # Header
        cv2.rectangle(c, (0, 0), (W, header_h), render.PANEL, -1)
        put = render.put_text
        put(c, s["org_name"].upper(), u * 3, u * 2, u * 4, render.WHITE, bold=True)
        mode = "ENTRANCE / EXIT" if s["mode"] == "gate" else "ATTENDANCE KIOSK"
        put(c, f"Face Attendance · {mode}", u * 3, u * 6 + u // 2, int(u * 2.2), render.MUTED)
        now = datetime.now()
        clock = now.strftime("%H:%M:%S")
        cw, _ = render.text_size(clock, u * 5, bold=True)
        put(c, clock, W - cw - u * 3, u * 1, u * 5, render.WHITE, bold=True)
        date = now.strftime("%A, %d %B %Y")
        dw, _ = render.text_size(date, int(u * 2.2))
        put(c, date, W - dw - u * 3, u * 6 + u // 2, int(u * 2.2), render.MUTED)

        # Camera tiles
        live = [wk for wk in self.workers if wk.cfg.get("enabled", True)]
        area_y, area_h = header_h + u * 2, H - header_h - footer_h - u * 4
        n = max(1, len(live))
        gap = u * 2
        tile_w = min((W - gap * (n + 1)) // n, int(area_h * 16 / 9))
        tile_h = int(tile_w * 9 / 16)
        x = (W - (tile_w * n + gap * (n - 1))) // 2
        y = area_y + (area_h - tile_h) // 2
        for wk in live:
            frame = wk.annotated
            if frame is not None:
                c[y:y + tile_h, x:x + tile_w] = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            else:
                cv2.rectangle(c, (x, y), (x + tile_w, y + tile_h), render.PANEL, -1)
                put(c, wk.status, x + u * 3, y + tile_h // 2, u * 3, render.MUTED)
            role = {"in": "CHECK-IN", "out": "CHECK-OUT"}.get(wk.cfg.get("role"), "SCAN") if s["mode"] == "gate" else "SCAN"
            color = render.GREEN if role == "CHECK-IN" else render.BLUE if role == "CHECK-OUT" else render.AMBER
            put(c, f" {role} · {wk.cfg['name']} ", x + u, y + u, int(u * 2.4), render.DARK, color, bold=True, pad=u // 2 + 2)
            x += tile_w + gap

        # Footer: latest events
        fy = H - footer_h
        cv2.rectangle(c, (0, fy), (W, H), render.PANEL, -1)
        put(c, "LATEST", u * 3, fy + u * 2, int(u * 2.2), render.MUTED, bold=True)
        recent = list(self.att.recent)
        card_w = (W - u * 6) // 4
        thumb = footer_h - u * 9
        for i, ev in enumerate(recent[:4]):
            cx = u * 3 + i * card_w
            cy = fy + u * 6
            img = self._thumb(ev, thumb)
            if img is not None:
                c[cy:cy + thumb, cx:cx + thumb] = img
            tx = cx + thumb + u * 2
            put(c, ev["name"], tx, cy, int(u * 2.8), render.WHITE, bold=True)
            label = "IN " if ev["kind"] == "in" else "OUT "
            col = render.GREEN if ev["kind"] == "in" else render.BLUE
            put(c, label + ev["time"], tx, cy + u * 4, int(u * 2.4), col, bold=True)
            if ev.get("late_min"):
                put(c, f"Late {ev['late_min']} min", tx, cy + u * 7 + u // 2, int(u * 2), render.AMBER)

        # Greeting banner for the newest event
        if recent and time.time() - recent[0]["ts"] < BANNER_SEC:
            ev = recent[0]
            text = f"{ev['greeting']}, {ev['name']}!"
            sub = ("Checked in " if ev["kind"] == "in" else "Checked out ") + ev["time"]
            if ev.get("late_min"):
                sub += f"  ·  late {ev['late_min']} min"
            bh = u * 14
            by = area_y + area_h - bh
            overlay = c.copy()
            cv2.rectangle(overlay, (0, by), (W, by + bh), (25, 90, 25) if ev["kind"] == "in" else (110, 70, 20), -1)
            cv2.addWeighted(overlay, 0.85, c, 0.15, 0, c)
            tw, _ = render.text_size(text, u * 5, bold=True)
            put(c, text, (W - tw) // 2, by + u * 2, u * 5, render.WHITE, bold=True)
            sw, _ = render.text_size(sub, u * 3)
            put(c, sub, (W - sw) // 2, by + u * 8 + u // 2, u * 3, render.WHITE)
        return c

    def _thumb(self, ev, size):
        key = (ev["snapshot"], size)
        if key not in self.thumbs:
            img = cv2.imread(str(config.DATA / ev["snapshot"])) if ev["snapshot"] else None
            self.thumbs[key] = None if img is None else cv2.resize(img, (size, size))
            if len(self.thumbs) > 64:
                self.thumbs.pop(next(iter(self.thumbs)))
        return self.thumbs[key]
