from pathlib import Path

root = Path(r"c:\Users\KIRISHIPATHI\OneDrive\Desktop\NEXA_HEALTH_AI")
import sys
sys.path.insert(0, str(root))

from app import app
from model.predictor import generate_gradcam_for_upload

img_paths = [
    root / "data" / "processed" / "test" / "0" / "img_0_100.jpg",
    root / "data" / "processed" / "test" / "1" / "img_0_1226.jpg",
]

print("VERIFY_START")
print("IMAGE_EXISTS_0", img_paths[0].exists(), img_paths[0])
print("IMAGE_EXISTS_1", img_paths[1].exists(), img_paths[1])

with app.test_client() as client:
    with img_paths[1].open("rb") as f:
        resp = client.post(
            "/predict",
            data={"image": (f, img_paths[1].name), "symptoms": "pain severe pelvic pressure infertility"},
            content_type="multipart/form-data",
        )
    print("STATUS_CODE", resp.status_code)
    html = resp.get_data(as_text=True)
    print("HAS_ENDOMETRIOSIS", "Endometriosis" in html or "Normal" in html)
    print("HAS_GRADCAM", "gradcam" in html.lower())
    print("HAS_CONFIDENCE", "Confidence" in html or "confidence" in html.lower())
    print("HTML_SNIPPET", html[:500])

for idx, img in enumerate(img_paths, 1):
    rel = generate_gradcam_for_upload(str(img))
    if rel is None:
        print(f"GRADCAM_{idx}_PATH", None)
        print(f"GRADCAM_{idx}_EXISTS", False)
        continue
    out = root / rel
    print(f"GRADCAM_{idx}_PATH", out)
    print(f"GRADCAM_{idx}_EXISTS", out.exists())
    if out.exists():
        print(f"GRADCAM_{idx}_SIZE", out.stat().st_size)

print("VERIFY_END")
