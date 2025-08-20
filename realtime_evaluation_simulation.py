import time, torch
from utils.common import load_artifacts
from scripts.train_lstm import LSTMClassifier
from utils.dataset import load_synthetic, prepare_data

def main():
    _, le = load_artifacts()
    model=LSTMClassifier(num_classes=len(le.classes_))
    model.load_state_dict(torch.load('models/lstm_har.pt', map_location='cpu')); model.eval()
    df=load_synthetic('synthetic_har.csv')
    (_, _), (X_val, y_val), _ = prepare_data(df, window_size=100, step=50)
    print('Streaming 10 windows...')
    with torch.no_grad():
        for i in range(10):
            x=torch.tensor(X_val[i:i+1], dtype=torch.float32)
            pred=model(x).argmax(1).item()
            lab=le.inverse_transform([pred])[0]
            print(f'Window {i+1:02d}: {lab}')
            time.sleep(0.5)

if __name__=='__main__':
    main()
