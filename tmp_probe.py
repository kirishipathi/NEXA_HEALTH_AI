import os
import joblib
import pandas as pd

root = r'C:\Users\KIRISHIPATHI\OneDrive\Desktop\NEXA_HEALTH_AI'
os.chdir(root)
model = joblib.load(r'model\symptom_logistic_regression.joblib')
df = pd.read_csv(r'data\symptom_features\simulated_symptom_dataset.csv')
feat = ['pain_level','cycle_irregularity','pain_during_intercourse','pelvic_pressure','heavy_bleeding','infertility_history','fatigue']
print('CLASSES', model.classes_.tolist())
print('POSITIVE_ROW_CHECK')
for _, row in df[df['label']==1].head(5).iterrows():
    x = row[feat].astype(float).values.reshape(1,-1)
    p = model.predict_proba(x)[0]
    print('sample_id=', row['sample_id'], 'features=', row[feat].tolist(), 'proba=', p.tolist(), 'pred=', int(model.predict(x)[0]))
print('NEGATIVE_ROW_CHECK')
for _, row in df[df['label']==0].head(5).iterrows():
    x = row[feat].astype(float).values.reshape(1,-1)
    p = model.predict_proba(x)[0]
    print('sample_id=', row['sample_id'], 'features=', row[feat].tolist(), 'proba=', p.tolist(), 'pred=', int(model.predict(x)[0]))
