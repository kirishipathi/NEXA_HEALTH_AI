import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib import request

root = Path(r"C:\Users\KIRISHIPATHI\OneDrive\Desktop\NEXA_HEALTH_AI")


def run_case(image_name: str, symptoms: str):
    img_path = root / "data" / "processed" / "test" / image_name.split("/")[0] / image_name.split("/")[1]
    boundary = "----WebKitFormBoundary" + uuid.uuid4().hex
    body = []
    body.append(f"--{boundary}\r\n".encode("utf-8"))
    body.append(b'Content-Disposition: form-data; name="image"; filename="img.jpg"\r\n')
    body.append(b"Content-Type: image/jpeg\r\n\r\n")
    body.append(img_path.read_bytes())
    body.append(f"\r\n--{boundary}\r\n".encode("utf-8"))
    body.append(b'Content-Disposition: form-data; name="symptoms"\r\n\r\n')
    body.append(f"{symptoms}\r\n".encode("utf-8"))
    body.append(f"--{boundary}--\r\n".encode("utf-8"))
    content = b"".join(body)
    req = request.Request(
        "http://127.0.0.1:5000/predict",
        data=content,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with request.urlopen(req, timeout=60) as resp:
        html = resp.read().decode("utf-8", errors="replace")
        print(f"CASE {image_name} STATUS {resp.status}")
        print(f"HAS_RESULT {('Prediction' in html or 'Insight is delivered.' in html)}")
        print(f"HAS_CONFIDENCE {('Confidence' in html or 'confidence' in html.lower())}")
        snippet = html[html.find("Prediction") if "Prediction" in html else html.find("Insight is delivered.") : html.find("Evaluate another case")]
        print(snippet[:300].replace("\n", " "))
        print()


proc = subprocess.Popen([sys.executable, "app.py"], cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
try:
    deadline = time.time() + 40
    ready = False
    while time.time() < deadline:
        try:
            with request.urlopen("http://127.0.0.1:5000/", timeout=2) as r:
                print("SERVER_READY", r.status)
                ready = True
                break
        except Exception:
            time.sleep(1)
    if not ready:
        print("SERVER_TIMEOUT")
        raise SystemExit(1)

    run_case("1/img_0_1226.jpg", "pain severe pelvic pressure infertility")
    run_case("0/img_0_1001.jpg", "no pain fatigue mild")
finally:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
