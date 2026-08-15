import mimetypes
import os
import uuid
from pathlib import Path
from urllib import request, error

root = Path(r"c:\Users\KIRISHIPATHI\OneDrive\Desktop\NEXA_HEALTH_AI")
img_path = root / 'data' / 'processed' / 'test' / '1' / 'img_0_1226.jpg'

boundary = '----WebKitFormBoundary' + uuid.uuid4().hex
body = []
body.append(f'--{boundary}\r\n'.encode('utf-8'))
body.append(b'Content-Disposition: form-data; name="image"; filename="img_0_1226.jpg"\r\n')
body.append(b'Content-Type: image/jpeg\r\n\r\n')
body.append(img_path.read_bytes())
body.append(f'\r\n--{boundary}\r\n'.encode('utf-8'))
body.append(b'Content-Disposition: form-data; name="symptoms"\r\n\r\n')
body.append(b'pain severe pelvic pressure infertility\r\n')
body.append(f'--{boundary}--\r\n'.encode('utf-8'))
content = b''.join(body)

req = request.Request(
    'http://127.0.0.1:5000/predict',
    data=content,
    method='POST',
    headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
)

try:
    with request.urlopen(req, timeout=60) as resp:
        html = resp.read().decode('utf-8', errors='replace')
        print('STATUS', resp.status)
        print('HAS_RESULT', 'Prediction' in html or 'Insight is delivered.' in html)
        print('HAS_CONFIDENCE', 'Confidence' in html or 'confidence' in html.lower())
        print('HAS_OVERLAY_MESSAGE', 'Overlay unavailable' in html or 'Grad-CAM' in html)
        print('HTML_SNIPPET', html[:1000])
except error.HTTPError as e:
    print('HTTP_ERROR', e.code)
    print(e.read().decode('utf-8', errors='replace')[:1000])
    raise
