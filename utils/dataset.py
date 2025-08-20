import pandas as pd, numpy as np, os
from sklearn.model_selection import train_test_split
from .common import get_scaler, get_label_encoder, save_artifacts

def load_synthetic(csv_path='synthetic_har.csv'):
    return pd.read_csv(csv_path)

def window_df(df, window_size=100, step=50, feature_cols=('ax','ay','az'), label_col='label'):
    X,y=[],[]; labels=df[label_col].values; feats=df[list(feature_cols)].values
    n=len(df); i=0
    while i+window_size<=n:
        seg=feats[i:i+window_size]
        lab=pd.Series(labels[i:i+window_size]).mode().iloc[0]
        X.append(seg); y.append(lab); i+=step
    return np.array(X), np.array(y)

def train_val_split(X,y,test_size=0.2,random_state=42):
    return train_test_split(X,y,test_size=test_size,random_state=random_state,stratify=y)

def prepare_data(df, window_size=100, step=50):
    X,y = window_df(df, window_size=window_size, step=step)
    le=get_label_encoder(); y_enc=le.fit_transform(y)
    # scale per-feature
    X2 = X.reshape(-1, X.shape[-1])
    sc=get_scaler(); X2=sc.fit_transform(X2); Xs=X2.reshape(X.shape)
    save_artifacts(sc, le)
    X_train, X_val, y_train, y_val = train_val_split(Xs, y_enc)
    return (X_train,y_train), (X_val,y_val), le
