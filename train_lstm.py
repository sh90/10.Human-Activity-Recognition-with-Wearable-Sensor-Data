import os, argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from utils.dataset import load_synthetic, prepare_data
from utils.augment import apply_augs
from utils.common import make_dirs

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=1, num_classes=3, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out,_ = self.lstm(x); last=out[:,-1,:]; return self.fc(last)

def get_loaders(X_train,y_train,X_val,y_val,batch_size=64):
    tr=TensorDataset(torch.tensor(X_train,dtype=torch.float32), torch.tensor(y_train,dtype=torch.long))
    va=TensorDataset(torch.tensor(X_val,dtype=torch.float32), torch.tensor(y_val,dtype=torch.long))
    return DataLoader(tr,batch_size=batch_size,shuffle=True), DataLoader(va,batch_size=batch_size,shuffle=False)

def train(args):
    make_dirs()
    df = load_synthetic(args.synthetic_path)
    (X_train,y_train),(X_val,y_val),le = prepare_data(df, window_size=args.window, step=args.step)
    if args.use_augment:
        X_train = apply_augs(X_train, use_jitter=args.jitter, use_scaling=args.scaling, use_permute=args.permute, use_timewarp=args.timewarp)
    num_classes=len(np.unique(y_train))
    model=LSTMClassifier(hidden_size=args.hidden, num_layers=args.layers, num_classes=num_classes, dropout=args.dropout)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device)
    tr,va = get_loaders(X_train,y_train,X_val,y_val,batch_size=args.batch)
    crit=nn.CrossEntropyLoss(); opt=torch.optim.Adam(model.parameters(), lr=args.lr)
    best=0.0
    for e in range(1,args.epochs+1):
        model.train(); tot=0.0
        for xb,yb in tqdm(tr, desc=f'Epoch {e}/{args.epochs}'):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); logits=model(xb); loss=crit(logits,yb); loss.backward(); opt.step()
            tot += loss.item()*xb.size(0)
        tr_loss = tot/len(tr.dataset)
        model.eval(); cor=0; tot=0
        with torch.no_grad():
            for xb,yb in va:
                xb,yb=xb.to(device), yb.to(device)
                pr=model(xb).argmax(1); cor+=(pr==yb).sum().item(); tot+=yb.size(0)
        acc=cor/tot; print(f'Epoch {e}: loss={tr_loss:.4f} val_acc={acc:.4f}')
        if acc>best:
            best=acc; torch.save(model.state_dict(),'models/lstm_har.pt'); print('Saved to models/lstm_har.pt')
    print(f'Best val_acc={best:.4f}')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--window', type=int, default=50)
    p.add_argument('--step', type=int, default=5)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--layers', type=int, default=1)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--use_augment', type=lambda s: s.lower()=='true', default=True)
    p.add_argument('--jitter', type=lambda s: s.lower()=='true', default=True)
    p.add_argument('--scaling', type=lambda s: s.lower()=='true', default=True)
    p.add_argument('--permute', type=lambda s: s.lower()=='true', default=False)
    p.add_argument('--timewarp', type=lambda s: s.lower()=='true', default=False)
    p.add_argument('--synthetic_path', type=str, default='synthetic_har.csv')
    train(p.parse_args())
