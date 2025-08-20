import os, joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

def make_dirs():
    os.makedirs('models', exist_ok=True)
def get_scaler():
    return StandardScaler()
def get_label_encoder():
    return LabelEncoder()
def save_artifacts(scaler, le, path='models'):
    os.makedirs(path, exist_ok=True)
    joblib.dump(scaler, f'{path}/scaler.pkl')
    joblib.dump(le, f'{path}/label_encoder.pkl')
def load_artifacts(path='models'):
    scaler = joblib.load(f'{path}/scaler.pkl')
    le = joblib.load(f'{path}/label_encoder.pkl')
    return scaler, le
