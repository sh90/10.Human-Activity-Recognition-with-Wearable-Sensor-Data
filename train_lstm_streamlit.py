import os, time, numpy as np, pandas as pd, plotly.express as px, streamlit as st, torch, torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
from utils.dataset import load_synthetic, prepare_data, window_df
from utils.augment import apply_augs
from utils.common import make_dirs, load_artifacts

st.set_page_config(page_title='HAR with LSTM', layout='wide')
st.title('🏃 Human Activity Recognition (Walking/Running/Sitting)')

with st.sidebar:
    st.header('Data & Model Settings')
    window = st.slider('Window size', 50, 200, 100, step=10)
    step = st.slider('Step (hop)', 10, 150, 50, step=10)
    use_aug = st.checkbox('Use augmentation', True)
    jitter = st.checkbox('Jitter', True); scaling = st.checkbox('Scaling', True)
    permute = st.checkbox('Permute segments', False); timewarp = st.checkbox('Time warp', False)
    epochs = st.slider('Epochs', 1, 10, 3, 1)
    hidden = st.slider('Hidden size', 16, 128, 64, 16)
    layers = st.slider('LSTM layers', 1, 2, 1, 1)
    lr = st.select_slider('Learning rate', [1e-2,5e-3,1e-3,5e-4,1e-4], 1e-3)
    batch = st.select_slider('Batch size', [16,32,64,128], 64)
    up = st.file_uploader('Upload CSV (timestamp, ax, ay, az, label)', type=['csv'])

tab1, tab2, tab3 = st.tabs(['Data preview', 'Train & Evaluate', 'Real-time demo'])

with tab1:
    st.subheader('Signals preview')
    df = pd.read_csv(up) if up else load_synthetic('scripts/synthetic_har.csv')
    st.write(df.head())
    fig = px.line(df.head(1000), x='timestamp', y=['ax','ay','az'], title='~First 20s')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader('Train an LSTM')
    make_dirs()
    (X_train,y_train), (X_val,y_val), le = prepare_data(df, window_size=window, step=step)
    if use_aug: X_train = apply_augs(X_train, use_jitter=jitter, use_scaling=scaling, use_permute=permute, use_timewarp=timewarp)

    class LSTMClassifier(nn.Module):
        def __init__(self, input_size=3, hidden_size=64, num_layers=1, num_classes=3, dropout=0.0):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
            self.fc = nn.Linear(hidden_size, num_classes)
        def forward(self, x):
            out,_ = self.lstm(x); last=out[:,-1,:]; return self.fc(last)

    num_classes=len(np.unique(y_train))
    model=LSTMClassifier(hidden_size=hidden, num_layers=layers, num_classes=num_classes)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device)

    tr=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_train,dtype=torch.float32), torch.tensor(y_train)), batch_size=batch, shuffle=True)
    va=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_val,dtype=torch.float32), torch.tensor(y_val)), batch_size=batch, shuffle=False)

    crit=nn.CrossEntropyLoss(); opt=torch.optim.Adam(model.parameters(), lr=float(lr))
    prog = st.progress(0, text='Training...')
    for e in range(1, epochs+1):
        model.train()
        for xb,yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss=crit(model(xb), yb); loss.backward(); opt.step()
        prog.progress(int(e/epochs*100), text=f'Epoch {e}/{epochs}')

    model.eval(); preds=[]; gts=[]
    with torch.no_grad():
        for xb,yb in va:
            xb=xb.to(device); pr=model(xb).argmax(1).cpu().numpy().tolist()
            preds+=pr; gts+=yb.numpy().tolist()

    st.success('Done!')
    labels=list(le.classes_)
    cm=confusion_matrix(gts, preds, labels=list(range(len(labels))))
    st.write('**Classes:**', labels)
    st.dataframe(pd.DataFrame(cm, index=labels, columns=labels))
    rep=classification_report(gts, preds, target_names=labels, output_dict=True)
    st.dataframe(pd.DataFrame(rep).transpose())
    torch.save(model.state_dict(),'models/lstm_har.pt')
    st.info('Saved → models/lstm_har.pt')

with tab3:
    st.subheader('Simulated real-time classification')
    if not os.path.exists('models/lstm_har.pt'):
        st.warning('Please train a model first.')
    else:
        class LSTMClassifier(nn.Module):
            def __init__(self, input_size=3, hidden_size=64, num_layers=1, num_classes=3, dropout=0.0):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
                self.fc = nn.Linear(hidden_size, num_classes)
            def forward(self, x):
                out,_ = self.lstm(x); last=out[:,-1,:]; return self.fc(last)
        _, le = load_artifacts()
        model=LSTMClassifier(num_classes=len(le.classes_))
        model.load_state_dict(torch.load('models/lstm_har.pt', map_location='cpu')); model.eval()

        X,_ = window_df(df, window_size=window, step=step)
        k = st.slider('Windows to stream', 1, min(50, len(X)), 10)
        ph = st.empty()
        if st.button('Start'):
            with torch.no_grad():
                for i in range(k):
                    x=torch.tensor(X[i:i+1], dtype=torch.float32)
                    lab=le.inverse_transform([model(x).argmax(1).item()])[0]
                    with ph.container():
                        st.metric('Predicted', lab)
                        fig=px.line(pd.DataFrame(X[i].squeeze(), columns=['ax','ay','az']), title=f'Window {i+1}')
                        st.plotly_chart(fig, use_container_width=True)
                    time.sleep(0.4)
