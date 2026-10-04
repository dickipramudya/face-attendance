"""Settings, stored as JSON next to the database so they survive updates."""
import json
import threading
from pathlib import Path

BASE = Path("/opt/absensi")
DATA = BASE / "data"
MODELS = BASE / "models"
SETTINGS_FILE = DATA / "settings.json"

DEFAULTS = {
    "org_name": "Demo Company",
    # "gate": one camera per direction (entrance = check-in, exit = check-out).
    # "kiosk": every camera is a check-in point; first scan of the day is the
    #          check-in, the last one is the check-out (how most attendance
    #          terminals work).
    "mode": "gate",
    "cameras": [
        {
            "id": "cam1", "name": "Entrance", "role": "in", "enabled": True,
            "device": "/dev/v4l/by-id/usb-icSpring_WebCamera_20240603201409-video-index0",
            "width": 1280, "height": 720,
        },
        {
            "id": "cam2", "name": "Exit", "role": "out", "enabled": True,
            "device": "/dev/v4l/by-id/usb-Generic_HD_video_20210901000000-video-index0",
            "width": 1280, "height": 720,
        },
    ],
    "work_start": "08:00",
    "work_end": "17:00",
    "late_grace_min": 5,
    # Cosine similarity for SFace. OpenCV's reference value is 0.363; a bit
    # higher trades a few "not recognised" for fewer wrong names.
    "match_threshold": 0.42,
    # Same person, same camera: ignore repeat sightings within this window.
    "cooldown_sec": 60,
    "display_enabled": True,
}

_lock = threading.Lock()
_settings = None


def load() -> dict:
    global _settings
    with _lock:
        if _settings is None:
            DATA.mkdir(parents=True, exist_ok=True)
            s = json.loads(json.dumps(DEFAULTS))
            if SETTINGS_FILE.exists():
                try:
                    s.update(json.loads(SETTINGS_FILE.read_text()))
                except Exception:
                    pass
            _settings = s
        return _settings


def save(new: dict) -> dict:
    global _settings
    with _lock:
        s = json.loads(json.dumps(_settings or DEFAULTS))
        for k, v in new.items():
            if k in DEFAULTS:
                s[k] = v
        SETTINGS_FILE.write_text(json.dumps(s, indent=2))
        _settings = s
        return s
