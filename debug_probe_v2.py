import os
import numpy as np
import cv2
import tensorflow as tf

root = r'C:\Users\KIRISHIPATHI\OneDrive\Desktop\NEXA_HEALTH_AI'
model = tf.keras.models.load_model(os.path.join(root, 'model', 'efficientnet_b0_model.keras'))

for folder in ['0', '1']:
    folder_path = os.path.join(root, 'data', 'processed', 'test', folder)
    files = sorted(os.listdir(folder_path))[:5]
    vals = []
    for f in files:
        full = os.path.join(folder_path, f)
        img = cv2.imread(full, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        batch = np.expand_dims(img.astype(np.float32), axis=0)
        prob = float(model.predict(batch, verbose=0)[0, 0])
        vals.append(prob)
        print(f'{folder}/{f}: {prob:.6f}')
    print(f'folder {folder} avg={float(np.mean(vals)):.6f} min={float(np.min(vals)):.6f} max={float(np.max(vals)):.6f}')

all_pos = []
folder_path = os.path.join(root, 'data', 'processed', 'test', '1')
for f in sorted(os.listdir(folder_path)):
    full = os.path.join(folder_path, f)
    img = cv2.imread(full, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    batch = np.expand_dims(img.astype(np.float32), axis=0)
    all_pos.append(float(model.predict(batch, verbose=0)[0, 0]))

arr = np.asarray(all_pos)
print(f'positive_total={len(arr)} mean={float(arr.mean()):.6f} min={float(arr.min()):.6f} max={float(arr.max()):.6f}')
print(f'positive_above_half={int(np.sum(arr >= 0.5))}')
