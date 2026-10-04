"""Drawing helpers: anti-aliased text via Pillow (cached), pasted into OpenCV frames."""
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# BGR
GREEN = (120, 220, 61)
AMBER = (72, 182, 255)
RED = (79, 77, 255)
BLUE = (255, 170, 80)
WHITE = (240, 240, 240)
MUTED = (168, 151, 138)
BG = (20, 15, 11)
PANEL = (33, 24, 18)
DARK = (20, 20, 20)


@lru_cache(maxsize=64)
def _font(size, bold):
    try:
        return ImageFont.truetype(FONT_BOLD if bold else FONT, size)
    except OSError:
        return ImageFont.load_default()


@lru_cache(maxsize=1024)
def text_img(text, size, color=WHITE, bg=None, bold=False, pad=0):
    """Rendered text as (BGR image, alpha mask or None), cached by content.

    With a background colour the image is an opaque tag; without one, the mask
    lets the text blend over video.
    """
    font = _font(size, bold)
    l, t, r, b = font.getbbox(text)
    w, h = max(1, r - l + 2 * pad), max(1, b - t + 2 * pad)
    fill = tuple(reversed(bg)) if bg else (0, 0, 0)
    im = Image.new("RGB", (w, h), fill)
    ImageDraw.Draw(im).text((pad - l, pad - t), text, font=font, fill=tuple(reversed(color)))
    arr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    if bg is not None:
        return arr, None
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).text((pad - l, pad - t), text, font=font, fill=255)
    return arr, np.asarray(m)


def put_text(dst, text, x, y, size, color=WHITE, bg=None, bold=False, pad=0):
    """Draws text with its top-left at (x, y), clipped to dst. Returns (w, h)."""
    img, mask = text_img(text, size, color, bg, bold, pad)
    h, w = img.shape[:2]
    H, W = dst.shape[:2]
    x, y = int(x), int(y)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return w, h
    src = img[y0 - y:y1 - y, x0 - x:x1 - x]
    if mask is None:
        dst[y0:y1, x0:x1] = src
    else:
        m = (mask[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32) / 255.0)[..., None]
        roi = dst[y0:y1, x0:x1].astype(np.float32)
        dst[y0:y1, x0:x1] = (roi * (1 - m) + src.astype(np.float32) * m).astype(np.uint8)
    return w, h


def text_size(text, size, bold=False, pad=0):
    img, _ = text_img(text, size, WHITE, DARK, bold, pad)
    return img.shape[1], img.shape[0]


def face_box(frame, x, y, w, h, label, sub, color):
    """Box with corner accents around a face, name tag above, detail line below."""
    x, y, w, h = int(x), int(y), int(w), int(h)
    t = max(2, frame.shape[1] // 500)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, t, cv2.LINE_AA)
    c = max(10, w // 5)
    for px, py, dx, dy in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        cv2.line(frame, (px, py), (px + dx * c, py), color, t * 2, cv2.LINE_AA)
        cv2.line(frame, (px, py), (px, py + dy * c), color, t * 2, cv2.LINE_AA)

    size = max(18, frame.shape[1] // 28)
    _, th = text_size(label, size, bold=True, pad=8)
    tag_y = y - th - 4 if y - th - 4 >= 0 else y + h + 4
    put_text(frame, label, x, tag_y, size, DARK, color, bold=True, pad=8)
    if sub:
        below = y + h + 4 if tag_y < y else tag_y + th + 2
        put_text(frame, sub, x, below, max(14, size * 2 // 3), WHITE, (45, 45, 45), pad=6)
