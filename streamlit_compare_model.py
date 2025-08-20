# streamlit_compare_models.py
# Compare classic ML baselines (tree-based) vs LSTM on HAR windows
# - Baselines use hand-crafted features per window
# - LSTM is loaded from models/lstm_har.pt if available (with scaler/label_encoder)

import os, math, time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="HAR: Baselines ", layout="wide")
st.title("⚖️ Classic ML Baselines  — HAR on ax/ay/az")

# --------------------------
# Data helpers
# --------------------------
def synth_har(seconds=240, fs=50, seed=42):
    """Generate a tiny synthetic HAR stream: sitting -> walking -> running."""
    rng = np.random.default_rng(seed)
    acts = ["sitting","walking","running"]
    rows=[]; t_total = seconds*fs; block=t_total//len(acts)
    for i,act in enumerate(acts):
        start=i*block; end=(i+1)*block if i<len(acts)-1 else t_total
        for t in range(start,end):
            tt=t/fs
            if act=="sitting":
                ax=rng.normal(0.02,0.03); ay=rng.normal(0.02,0.03); az=rng.normal(1.0,0.05)
            elif act=="walking":
                f=2.0
                ax=0.4*math.sin(2*math.pi*f*tt)+rng.normal(0,0.05)
                ay=0.4*math.cos(2*math.pi*f*tt)+rng.normal(0,0.05)
                az=1.0+0.2*math.sin(2*math.pi*f*tt+0.5)+rng.normal(0,0.05)
            else:
                f=3.5
                ax=0.7*math.sin(2*math.pi*f*tt)+rng.normal(0,0.08)
                ay=0.7*math.cos(2*math.pi*f*tt)+rng.normal(0,0.08)
                az=1.0+0.35*math.sin(2*math.pi*f*tt+0.8)+rng.normal(0,0.08)
            rows.append((tt,ax,ay,az,act))
    return pd.DataFrame(rows, columns=["timestamp","ax","ay","az","label"])

def window_df(df, window_size=50, step=5, feature_cols=("ax","ay","az"), label_col="label"):
    X,y=[],[]
    labels=df[label_col].values
    feats=df[list(feature_cols)].values
    n=len(df); i=0
    while i+window_size<=n:
        seg=feats[i:i+window_size]
        # majority label in the window
        lab=pd.Series(labels[i:i+window_size]).mode().iloc[0]
        X.append(seg); y.append(lab); i+=step
    return np.array(X), np.array(y)

# --------------------------
# Feature extraction for baselines
# --------------------------
def zero_cross_rate(x):
    # x: (T,) one axis
    return np.mean(np.abs(np.diff(np.signbit(x))))

def spectral_feats(x, fs):
    # Return dominant frequency and spectral centroid
    # x: (T,)
    X = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0/fs)
    if X.sum() == 0:
        return 0.0, 0.0
    dom_idx = int(np.argmax(X[1:])) + 1 if len(X) > 1 else 0  # skip DC
    dom_freq = freqs[dom_idx]
    centroid = float((freqs * X).sum() / (X.sum() + 1e-8))
    return dom_freq, centroid

def extract_features(X, fs=50):
    """
    X: (N, T, F=3) windows
    Returns: (N, D) feature matrix
    Per-axis features: mean, std, min, max, median, rms, energy, zcr, dom_freq, spec_centroid
    Cross-axis: corr(ax,ay), corr(ax,az), corr(ay,az)
    Magnitude: |vec| mean/std
    """
    N, T, F = X.shape
    feats = []
    for i in range(N):
        seg = X[i]  # (T,3)
        cols = []
        mag = np.linalg.norm(seg, axis=1)

        for a in range(3):
            x = seg[:, a]
            mean = float(np.mean(x)); std = float(np.std(x))
            minv = float(np.min(x)); maxv = float(np.max(x)); med = float(np.median(x))
            rms = float(np.sqrt(np.mean(x**2))); energy = float(np.sum(x**2)/T)
            zcr = float(zero_cross_rate(x))
            dom_f, cent = spectral_feats(x, fs)
            cols += [mean,std,minv,maxv,med,rms,energy,zcr,dom_f,cent]

        # correlations
        def safe_corr(a,b):
            if np.std(a)==0 or np.std(b)==0: return 0.0
            return float(np.corrcoef(a,b)[0,1])
        c_xy = safe_corr(seg[:,0], seg[:,1])
        c_xz = safe_corr(seg[:,0], seg[:,2])
        c_yz = safe_corr(seg[:,1], seg[:,2])

        # magnitude stats
        mag_mean = float(np.mean(mag)); mag_std = float(np.std(mag))
        cols += [c_xy, c_xz, c_yz, mag_mean, mag_std]
        feats.append(cols)

    feats = np.array(feats)
    return feats

# --------------------------
# Optional LSTM inference
# --------------------------
def try_load_lstm(num_classes_hint=3):
    """Load trained LSTM + artifacts if present; else return None to skip."""
    model_path = "models/lstm_har.pt"
    scaler_path = "models/scaler.pkl"
    le_path = "models/label_encoder.pkl"
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(le_path)):
        return None
    try:
        import torch, torch.nn as nn
        import joblib
    except Exception:
        return None

    class LSTMClassifier(nn.Module):
        def __init__(self, input_size=3, hidden_size=64, num_layers=1, num_classes=3, dropout=0.0):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                                batch_first=True, dropout=dropout if num_layers>1 else 0.0)
            self.fc = nn.Linear(hidden_size, num_classes)
        def forward(self, x):
            out,_ = self.lstm(x); last=out[:,-1,:]; return self.fc(last)

    le = None
    try:
        import joblib
        scaler = joblib.load(scaler_path)
        le = joblib.load(le_path)
    except Exception:
        return None

    model = LSTMClassifier(num_classes=len(le.classes_))
    import torch
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return dict(model=model, scaler=scaler, le=le)


# --------------------------
# Sidebar controls
# --------------------------
with st.sidebar:
    st.header("Data")
    src = st.radio("Choose data", ["Use synthetic (offline)", "Upload CSV"])
    window = st.slider("Window size", 50, 200, 100, step=10)
    step = st.slider("Hop (overlap)", 10, 150, 50, step=10)
    fs = st.select_slider("Assumed sampling rate (Hz)", [25, 50, 100], value=50)

    st.header("Baseline Model")
    model_name = st.selectbox("Classic model", ["RandomForest", "GradientBoosting"])
    if model_name == "RandomForest":
        n_estimators = st.slider("n_estimators", 10, 400, 10, 50)
        max_depth = st.slider("max_depth (None for auto)", 2, 30, 2, 2)
        if max_depth == 2:  # allow None when dragging low
            max_depth = None
    else:
        n_estimators = st.slider("n_estimators", 10, 400, 10, 50)
        max_depth = st.slider("max_depth (None for auto)", 2, 30, 2, 2)

    test_size = st.select_slider("Test size", [0.1,0.2,0.25,0.3,0.4], value=0.2)
    random_state = st.number_input("Random seed", 0, 9999, 42)


# Load data
if src == "Upload CSV":
    up = st.file_uploader("CSV with columns: timestamp, ax, ay, az, label", type=["csv"])
    if up is None:
        st.stop()
    df = pd.read_csv(up)
    print(len(df))
    st.success("Using your uploaded CSV")
else:
    df = synth_har(seconds=240, fs=fs)

# Make windows
X, y = window_df(df, window_size=window, step=step)
if len(X)==0:
    st.error("No windows produced. Try a smaller window or different hop.")
    st.stop()

# Encode labels once
le_all = LabelEncoder()
y_enc = le_all.fit_transform(y)
classes = list(le_all.classes_)

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc)

# Extract features for baselines
with st.spinner("Extracting features for classic model..."):
    F_train = extract_features(X_train, fs=fs)
    F_test  = extract_features(X_test, fs=fs)
    scaler_feat = StandardScaler().fit(F_train)
    F_train_s = scaler_feat.transform(F_train)
    F_test_s  = scaler_feat.transform(F_test)

# Fit classic model
if model_name == "RandomForest":
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)
else:
    clf = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

with st.spinner(f"Training {model_name}..."):
    clf.fit(F_train_s, y_train)

y_pred_base = clf.predict(F_test_s)
acc_base = accuracy_score(y_test, y_pred_base)


# --------------------------
# Show results
# --------------------------
colA, colB = st.columns([1,1])
with colA:
    st.subheader(f"Classic Model: {model_name}")
    st.write(f"**Accuracy:** {acc_base:.3f}")
    rep = classification_report(y_test, y_pred_base, target_names=classes, output_dict=True)
    st.dataframe(pd.DataFrame(rep).transpose())
    cm = confusion_matrix(y_test, y_pred_base, labels=list(range(len(classes))))
    fig_cm = go.Figure(data=go.Heatmap(z=cm, x=classes, y=classes, zmin=0))
    fig_cm.update_layout(title="Confusion Matrix (Classic)", xaxis_title="Predicted", yaxis_title="True", height=360, margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig_cm, use_container_width=True)

# Peek at features (first few rows)
with st.expander("Peek at engineered features (first 5 rows)"):
    cols = []
    for axis in ["ax","ay","az"]:
        cols += [f"{axis}_mean", f"{axis}_std", f"{axis}_min", f"{axis}_max",
                 f"{axis}_median", f"{axis}_rms", f"{axis}_energy", f"{axis}_zcr",
                 f"{axis}_domfreq", f"{axis}_speccent"]
    cols += ["corr_xy","corr_xz","corr_yz","mag_mean","mag_std"]
    df_feat = pd.DataFrame(F_train[:5], columns=cols)
    st.dataframe(df_feat)
