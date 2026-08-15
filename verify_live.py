import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib import request

root = Path(r"C:\Users\KIRISHIPATHI\OneDrive\Desktop\NEXA_HEALTH_AI")

proc = subprocess.Popen(
    [sys.executable, "app.py"],
    cwd=str(root),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

try:
    deadline = time.time() + 40
    ready = False
    while time.time() < deadline:
        try:
            with request.urlopen("http://127.0.0.1:5000/", timeout=2) as resp:
                print("SERVER_READY", resp.status)
                ready = True
                break
        except Exception:
            time.sleep(1)

    if not ready:
        print("SERVER_TIMEOUT")
        if proc.stdout is not None:
            out = proc.stdout.read()
            print(out[-2000:])
        raise SystemExit(1)

    img_path = root / "data" / "processed" / "test" / "1" / "img_0_1226.jpg"
    if not img_path.exists():
        raise FileNotFoundError(f"Missing image: {img_path}")

    boundary = "----WebKitFormBoundary" + uuid.uuid4().hex
    body = []
    body.append(f"--{boundary}\r\n".encode("utf-8"))
    body.append(b'Content-Disposition: form-data; name="image"; filename="img_0_1226.jpg"\r\n')
    body.append(b"Content-Type: image/jpeg\r\n\r\n")
    body.append(img_path.read_bytes())
    body.append(f"\r\n--{boundary}\r\n".encode("utf-8"))
    body.append(b'Content-Disposition: form-data; name="symptoms"\r\n\r\n')
    body.append(b"pain severe pelvic pressure infertility\r\n")
    body.append(f"--{boundary}--\r\n".encode("utf-8"))
    content = b"".join(body)

    req = request.Request(
        "http://127.0.0.1:5000/predict",
        data=content,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )

    with request.urlopen(req, timeout=80) as resp:
        html = resp.read().decode("utf-8", errors="replace")
        print("STATUS", resp.status)
        print("HAS_RESULT", "Prediction" in html or "Insight is delivered." in html)
        print("HAS_CONFIDENCE", "Confidence" in html or "confidence" in html.lower())
        print("HAS_GRADCAM_NOTICE", "Grad-CAM" in html or "Overlay unavailable" in html)
        print("HTML_SNIPPET_START")
        print(html[:1500])
        print("HTML_SNIPPET_END")
finally:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
