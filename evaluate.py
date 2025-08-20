
import torch
from sklearn.metrics import classification_report, confusion_matrix
from utils.dataset import load_synthetic, prepare_data
from scripts.train_lstm import LSTMClassifier

def main():
    df = load_synthetic('synthetic_har.csv')
    """
    Turns the raw time-series into overlapping windows (length 100 samples, moving forward 50 samples each time).

    Scales the features and label-encodes the string labels ("sitting", "walking", "running") into integers via le.

    Returns only the validation split here: 
    X_val (windows) and y_val (their numeric labels), plus the label encoder le (so you know which class index corresponds to which string).
    """
    (_, _), (X_val, y_val), le = prepare_data(df, window_size=100, step=50)
    model=LSTMClassifier(num_classes=len(le.classes_))
    model.load_state_dict(torch.load('models/lstm_har.pt', map_location='cpu')); model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_val, dtype=torch.float32))
        preds = logits.argmax(1).numpy()
    print('Classes:', list(le.classes_))
    print(classification_report(y_val, preds, target_names=list(le.classes_)))
    print('Confusion Matrix:\n', confusion_matrix(y_val, preds))
if __name__=='__main__':
    main()
