"""Uploads a camera frame as a normal JPEG, as a phone-style EXIF-rotated JPEG, and as junk."""
import io, json, sys, urllib.request
from PIL import Image

base = "http://localhost:8090"

def frame_from_stream(cam):
    r = urllib.request.urlopen(f"{base}/stream/{cam}", timeout=10)
    buf = b""
    while True:
        buf += r.read(4096)
        a = buf.find(b"\xff\xd8"); b = buf.find(b"\xff\xd9", a + 2)
        if a >= 0 and b > a:
            return buf[a:b + 2]

def upload(name, data, ctype):
    boundary = "xBOUNDARYx"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"photo\"; filename=\"{name}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{base}/api/upload-photo", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        j = json.load(urllib.request.urlopen(req))
        return "OK token " + j["token"]
    except urllib.error.HTTPError as e:
        return f"{e.code} {json.load(e).get('error')}"

jpg = frame_from_stream(sys.argv[1] if len(sys.argv) > 1 else "cam1")
img = Image.open(io.BytesIO(jpg))
print("normal     ", upload("normal.jpg", jpg, "image/jpeg"))
# Phone portrait shot: pixels stored sideways, EXIF Orientation=6 says "rotate 90° CW to view".
side = img.transpose(Image.Transpose.ROTATE_90)
exif = Image.Exif(); exif[0x0112] = 6
b = io.BytesIO(); side.save(b, "JPEG", exif=exif.tobytes())
print("exif-rot   ", upload("phone.jpg", b.getvalue(), "image/jpeg"))
b = io.BytesIO(); side.save(b, "JPEG")
print("sideways   ", upload("sideways-no-exif.jpg", b.getvalue(), "image/jpeg"))
b = io.BytesIO(); img.save(b, "PNG")
print("png        ", upload("photo.png", b.getvalue(), "image/png"))
print("junk       ", upload("notes.txt", b"hello", "text/plain"))
