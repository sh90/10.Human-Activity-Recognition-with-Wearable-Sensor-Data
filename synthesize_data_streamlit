import streamlit as st, pandas as pd, numpy as np, math, plotly.express as px

st.set_page_config(page_title='Synthetic HAR Data Generator', layout='wide')
st.title('Generate Synthetic Accelerometer Data')

with st.sidebar:
    fs = st.slider('Sampling rate (Hz)', 10, 200, 50, 5)
    seconds = st.slider('Total seconds', 10, 600, 120, 10)
    seq = st.text_input('Activity sequence (comma-separated)', 'sitting,walking,running')
    jstd = st.slider('Noise (jitter std)', 0.0, 0.2, 0.05, 0.01)
    seed = st.number_input('Random seed', 0, 10_000, 123)
    filename = st.text_input('Output CSV path', 'data/synthetic_custom.csv')

def gen_activity(t, fs, act, jitter_std=0.05):
    if act=='sitting':
        ax=np.random.normal(0.02,0.03); ay=np.random.normal(0.02,0.03); az=np.random.normal(1.0,0.05)
    elif act=='walking':
        f=2.0
        ax=0.4*math.sin(2*math.pi*f*(t/fs))+np.random.normal(0, jitter_std)
        ay=0.4*math.cos(2*math.pi*f*(t/fs))+np.random.normal(0, jitter_std)
        az=1.0+0.2*math.sin(2*math.pi*f*(t/fs)+0.5)+np.random.normal(0, jitter_std)
    else: # running
        f=3.5
        ax=0.7*math.sin(2*math.pi*f*(t/fs))+np.random.normal(0, jitter_std*1.6)
        ay=0.7*math.cos(2*math.pi*f*(t/fs))+np.random.normal(0, jitter_std*1.6)
        az=1.0+0.35*math.sin(2*math.pi*f*(t/fs)+0.8)+np.random.normal(0, jitter_std*1.6)
    return ax, ay, az

def synthesize(seconds=120, fs=50, sequence='sitting,walking,running', jitter_std=0.05, seed=123):
    np.random.seed(seed)
    acts=[a.strip() for a in sequence.split(',') if a.strip()]
    rows=[]; block=seconds//len(acts)
    t=0
    for act in acts:
        n_samp = block*fs
        for i in range(n_samp):
            ax,ay,az = gen_activity(t, fs, act, jitter_std=jitter_std)
            rows.append((t/fs, ax, ay, az, act))
            t += 1
    return pd.DataFrame(rows, columns=['timestamp','ax','ay','az','label'])

if st.button('Generate'):
    df = synthesize(seconds, fs, seq, jstd, seed)
    st.success(f'Generated {len(df)} rows')
    st.dataframe(df.head())
    fig = px.line(df.head(min(1500, len(df))), x='timestamp', y=['ax','ay','az'], title='Preview')
    st.plotly_chart(fig, use_container_width=True)
    df.to_csv(filename, index=False)
    st.info(f'Saved → {filename}')
