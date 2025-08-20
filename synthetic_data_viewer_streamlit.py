# streamlit_view_synth.py
# Visualize wearable accelerometer CSVs with time-series, class coverage, and spectrum.
# Expected columns: timestamp, ax, ay, az, [label]  (label optional)

import io, math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="HAR Synthetic Data Viewer", layout="wide")
st.title("👀 HAR Synthetic Data Viewer (ax / ay / az)")

# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.header("Load data")
    f = st.file_uploader("Upload CSV", type=["csv"])
    example = st.checkbox("Load tiny built-in example", value=False)

    st.header("Display")
    show_axes = st.multiselect("Axes to plot", ["ax","ay","az","|magnitude|"], default=["ax","ay","az"])
    smooth = st.slider("Smoothing window (samples)", 1, 51, 1, 2)
    down = st.slider("Downsample (every Nth point)", 1, 10, 1, 1)
    fs_assumed = st.select_slider("Sampling rate (Hz, if timestamp missing)", [25, 50, 100], value=50)

    st.header("Time range")
    auto_range = st.checkbox("Auto full range", value=True)
    t_start = st.number_input("Start (seconds)", value=0.0, step=0.5, disabled=auto_range)
    t_end   = st.number_input("End (seconds)", value=10.0, step=0.5, disabled=auto_range)

# ---------------- Load or synthesize ----------------
def synth_demo(seconds=12, fs=50, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    acts = ["sitting","walking","running"]
    block = seconds // len(acts)
    for i, act in enumerate(acts):
        for n in range(block*fs):
            t = (i*block*fs + n)/fs
            if act == "sitting":
                ax = rng.normal(0.02,0.03); ay = rng.normal(0.02,0.03); az = rng.normal(1.0,0.05)
            elif act == "walking":
                f=2.0
                ax=0.4*np.sin(2*np.pi*f*t)+rng.normal(0,0.05)
                ay=0.4*np.cos(2*np.pi*f*t)+rng.normal(0,0.05)
                az=1.0+0.2*np.sin(2*np.pi*f*t+0.5)+rng.normal(0,0.05)
            else:
                f=3.5
                ax=0.7*np.sin(2*np.pi*f*t)+rng.normal(0,0.08)
                ay=0.7*np.cos(2*np.pi*f*t)+rng.normal(0,0.08)
                az=1.0+0.35*np.sin(2*np.pi*f*t+0.8)+rng.normal(0,0.08)
            rows.append((t, ax, ay, az, act))
    return pd.DataFrame(rows, columns=["timestamp","ax","ay","az","label"])

if example:
    df = synth_demo(seconds=12, fs=fs_assumed)
elif f is not None:
    df = pd.read_csv(f)
else:
    st.info("Upload your CSV (timestamp, ax, ay, az[, label]) or tick 'Load tiny built-in example'.")
    st.stop()

# ---------------- Normalize columns ----------------
cols = {c.lower(): c for c in df.columns}
def get_col(name):
    return cols.get(name, None)

tcol = get_col("timestamp")
axc, ayc, azc = get_col("ax"), get_col("ay"), get_col("az")
lcol = get_col("label")

# If no timestamp, make one using sample index / fs_assumed
if tcol is None:
    df["timestamp"] = np.arange(len(df)) / float(fs_assumed)
    tcol = "timestamp"

# Compute magnitude if needed
if "|magnitude|" in show_axes and axc and ayc and azc:
    df["|magnitude|"] = np.sqrt(df[axc]**2 + df[ayc]**2 + df[azc]**2)

# Optional smoothing (simple moving average)
if smooth and smooth > 1:
    for c in ["ax","ay","az","|magnitude|"]:
        cc = get_col(c) if c != "|magnitude|" else "|magnitude|"
        if cc in df.columns:
            df[cc] = df[cc].rolling(window=smooth, min_periods=1, center=True).mean()

# Downsample for speed if requested
if down > 1:
    df = df.iloc[::down, :].reset_index(drop=True)

# Subset time range
tmin, tmax = float(df[tcol].min()), float(df[tcol].max())
if auto_range:
    t0, t1 = tmin, tmax
else:
    t0 = max(tmin, t_start)
    t1 = min(tmax, t_end)
mask = (df[tcol] >= t0) & (df[tcol] <= t1)
view = df.loc[mask].copy()

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4 = st.tabs(["Time series", "Class coverage", "Spectrum (FFT)", "Stats"])

with tab1:
    st.subheader("Time-series line plot")
    if not show_axes:
        st.warning("Select at least one series to plot.")
    else:
        ycols = [c for c in show_axes if c in view.columns]
        fig = px.line(view, x=tcol, y=ycols, title=f"Signals from {t0:.2f}s to {t1:.2f}s")
        fig.update_xaxes(title="Time (s)")
        fig.update_yaxes(title="Acceleration (approx)")
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Tip: use the sidebar to smooth or downsample, and to set the time range.")

with tab2:
    st.subheader("Class coverage (if labels present)")
    if lcol and lcol in df.columns:
        counts = df[lcol].value_counts().reset_index()
        counts.columns = ["label","count"]
        st.plotly_chart(px.bar(counts, x="label", y="count"), use_container_width=True)
        st.write(counts)
    else:
        st.info("No 'label' column found, so class coverage is skipped.")

with tab3:
    st.subheader("Simple spectrum (FFT)")
    st.caption("Choose one series. If your timestamp is uneven, the FFT is approximate—works best with regular sampling.")
    series_opts = [c for c in ["ax","ay","az","|magnitude|"] if c in view.columns]
    series = st.selectbox("Series", series_opts, index=series_opts.index("|magnitude|") if "|magnitude|" in series_opts else 0)
    # Try to infer sampling rate from timestamps
    if len(view) >= 3:
        dt = np.diff(view[tcol].to_numpy())
        median_dt = np.median(dt[~np.isnan(dt)])
        fs_infer = 1.0/median_dt if median_dt and median_dt>0 else fs_assumed
    else:
        fs_infer = fs_assumed
    st.write(f"Using sampling rate ≈ **{fs_infer:.2f} Hz**")
    x = view[series].to_numpy()
    # Remove mean to focus on oscillation peaks
    x = x - np.mean(x)
    X = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0/fs_infer)
    spec_fig = go.Figure()
    spec_fig.add_trace(go.Scatter(x=freqs, y=X, mode="lines"))
    spec_fig.update_xaxes(title="Frequency (Hz)")
    spec_fig.update_yaxes(title="Amplitude (arb. units)")
    spec_fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=360)
    st.plotly_chart(spec_fig, use_container_width=True)
    if len(X) > 1:
        peak_idx = int(np.argmax(X[1:])) + 1
        st.write(f"Dominant frequency ≈ **{freqs[peak_idx]:.2f} Hz** (skip DC)")

with tab4:
    st.subheader("Quick stats")
    cols_show = [c for c in ["ax","ay","az","|magnitude|"] if c in view.columns]
    stats = {}
    for c in cols_show:
        v = view[c].to_numpy()
        stats[c] = dict(
            mean=float(np.mean(v)),
            std=float(np.std(v)),
            min=float(np.min(v)),
            max=float(np.max(v)),
            rms=float(np.sqrt(np.mean(v**2))),
        )
    if stats:
        st.dataframe(pd.DataFrame(stats).T)
    else:
        st.info("Pick at least one series to show stats.")
