import os
import cv2
import numpy as np
import tensorflow as tf

root = r'C:\Users\KIRISHIPATHI\OneDrive\Desktop\NEXA_HEALTH_AI'
model = tf.keras.models.load_model(os.path.join(root, 'model', 'efficientnet_b0_model.keras'))

for folder in ['0', '1']:
    folder_path = os.path.join(root, 'data', 'processed', 'test', folder)
    files = sorted(os.listdir(folder_path))[:10]
    print(f'FOLDER {folder} sample_count={len(files)}')
    vals = []
    for f in files:
        p = os.path.join(folder_path, f)
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        batch = np.expand_dims(img.astype(np.float32), axis=0)
        prob = float(model.predict(batch, verbose=0)[0, 0])
        vals.append(prob)
        print(f'  {f}: prob={prob:.6f}')
    print(f'  avg={float(np.mean(vals)):.6f} min={float(np.min(vals)):.6f} max={float(np.max(vals)):.6f}')

positive_folder = os.path.join(root, 'data', 'processed', 'test', '1')
all_positive = sorted(os.listdir(positive_folder))
probs = []
for f in all_positive:
    p = os.path.join(positive_folder, f)
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    batch = np.expand_dims(img.astype(np.float32), axis=0)
    prob = float(model.predict(batch, verbose=0)[0, 0])
    probs.append(prob)
print('ALL_POSITIVE_COUNT', len(probs))
print('ALL_POSITIVE_MEAN', float(np.mean(probs)))
print('ALL_POSITIVE_MIN', float(np.min(probs)))
print('ALL_POSITIVE_MAX', float(np.max(probs)))
print('ALL_POSITIVE_ABOVE_HALF', int(np.sum(np.asarray(probs) >= 0.5)))
