import os
import glob
import random
import math
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
DATA_DIR = r"D:\ProductX\Ravdees dataset\ravdess"
SAMPLE_RATE = 16000
N_MFCC = 40
MAX_LEN = 200
BATCH_SIZE = 32
NUM_EPOCHS = 30
LR = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHT_DECAY = 1e-5
CLASS_WEIGHT = 1.0
REG_WEIGHT = 1.0
MODEL_SAVE_PATH = "confidence_bilstm.pth"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
EMOTION_MAP = {
    1: 'neutral',
    2: 'calm',
    3: 'happy',
    4: 'sad',
    5: 'angry',
    6: 'fearful',
    7: 'disgust',
    8: 'surprised'
}

CONFIDENCE_MAP = {
    'neutral': 0.85,
    'calm': 0.90,
    'happy': 0.95,
    'sad': 0.40,
    'angry': 0.50,
    'fearful': 0.30,
    'disgust': 0.35,
    'surprised': 0.60
}
def load_audio(path, sr=SAMPLE_RATE):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y
def extract_mfcc(y, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=1024, hop_length=256)
    return mfcc.T
def pad_or_truncate(mfcc, max_len=MAX_LEN):
    if mfcc.shape[0] >= max_len:
        return mfcc[:max_len, :]
    else:
        pad_len = max_len - mfcc.shape[0]
        pad = np.zeros((pad_len, mfcc.shape[1]), dtype=np.float32)
        return np.vstack([mfcc, pad])
class RavdessDataset(Dataset):
    def __init__(self, wav_files, cache_dir="./mfcc_cache"):
        self.files = wav_files
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.labels = [self._get_emotion_from_filename(p) for p in self.files]
        self.le = LabelEncoder().fit(list(EMOTION_MAP.values()))
        self.y_class = [self.le.transform([e])[0] for e in self.labels]
        self.y_reg = [CONFIDENCE_MAP[e] for e in self.labels]
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        path = self.files[idx]
        cache_file = os.path.join(self.cache_dir, os.path.basename(path) + ".npy")
        if os.path.exists(cache_file):
            mfcc = np.load(cache_file)
        else:
            y = load_audio(path)
            mfcc = extract_mfcc(y)
            mfcc = pad_or_truncate(mfcc)
            mfcc = (mfcc - np.mean(mfcc, axis=0, keepdims=True)) / (np.std(mfcc, axis=0, keepdims=True) + 1e-9)
            np.save(cache_file, mfcc.astype(np.float32))
        x = torch.tensor(mfcc, dtype=torch.float32)
        y_c = torch.tensor(self.y_class[idx], dtype=torch.long)
        y_r = torch.tensor(self.y_reg[idx], dtype=torch.float32)
        return x, y_c, y_r
    @staticmethod
    def _get_emotion_from_filename(path):
        parts = os.path.basename(path).replace('.wav', '').split('-')
        try:
            emo_id = int(parts[2])
            return EMOTION_MAP.get(emo_id, 'neutral')
        except:
            return 'neutral'
def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch])
    yc = torch.stack([b[1] for b in batch])
    yr = torch.stack([b[2] for b in batch])
    return xs, yc, yr
class ConfidenceBiLSTM(nn.Module):
    def __init__(self, input_size=N_MFCC, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                 num_classes=8, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        fc_dim = hidden_size * 2
        self.classifier = nn.Sequential(
            nn.Linear(fc_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, num_classes)
        )
        self.regressor = nn.Sequential(
            nn.Linear(fc_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        logits = self.classifier(last)
        conf = self.regressor(last).squeeze(1)
        return logits, conf
def get_file_list(data_dir):
    return sorted(glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True))
def train_one_epoch(model, dl, optim, ce_loss, mse_loss):
    model.train()
    total_loss, all_preds, all_labels, all_conf_preds, all_conf_true = 0, [], [], [], []
    for x, y_c, y_r in tqdm(dl, desc="Train"):
        x, y_c, y_r = x.to(DEVICE), y_c.to(DEVICE), y_r.to(DEVICE)
        optim.zero_grad()
        logits, conf = model(x)
        loss = CLASS_WEIGHT * ce_loss(logits, y_c) + REG_WEIGHT * mse_loss(conf, y_r)
        loss.backward()
        optim.step()
        total_loss += loss.item() * x.size(0)
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(y_c.cpu().numpy())
        all_conf_preds.extend(conf.detach().cpu().numpy())
        all_conf_true.extend(y_r.cpu().numpy())
    avg_loss = total_loss / len(dl.dataset)
    acc = accuracy_score(all_labels, all_preds)
    rmse = math.sqrt(mean_squared_error(all_conf_true, all_conf_preds))
    return avg_loss, acc, rmse
@torch.no_grad()
def eval_model(model, dl, ce_loss, mse_loss):
    model.eval()
    total_loss, all_preds, all_labels, all_conf_preds, all_conf_true = 0, [], [], [], []
    for x, y_c, y_r in tqdm(dl, desc="Eval"):
        x, y_c, y_r = x.to(DEVICE), y_c.to(DEVICE), y_r.to(DEVICE)
        logits, conf = model(x)
        loss = CLASS_WEIGHT * ce_loss(logits, y_c) + REG_WEIGHT * mse_loss(conf, y_r)
        total_loss += loss.item() * x.size(0)
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(y_c.cpu().numpy())
        all_conf_preds.extend(conf.cpu().numpy())
        all_conf_true.extend(y_r.cpu().numpy())
    avg_loss = total_loss / len(dl.dataset)
    acc = accuracy_score(all_labels, all_preds)
    rmse = math.sqrt(mean_squared_error(all_conf_true, all_conf_preds))
    return avg_loss, acc, rmse
def main():
    files = get_file_list(DATA_DIR)
    if not files:
        raise SystemExit(f"No .wav files found in {DATA_DIR}")
    random.shuffle(files)
    ds = RavdessDataset(files)
    n_train = int(0.8 * len(ds))
    train_ds, val_ds = random_split(ds, [n_train, len(ds) - n_train])
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    model = ConfidenceBiLSTM().to(DEVICE)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_score = float("inf")
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        tr_loss, tr_acc, tr_rmse = train_one_epoch(model, train_dl, optim, ce_loss, mse_loss)
        val_loss, val_acc, val_rmse = eval_model(model, val_dl, ce_loss, mse_loss)
        print(f"Train: loss={tr_loss:.4f} acc={tr_acc:.4f} rmse={tr_rmse:.4f}")
        print(f"Val:   loss={val_loss:.4f} acc={val_acc:.4f} rmse={val_rmse:.4f}")
        if val_loss < best_score:
            best_score = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'label_encoder': ds.le.classes_,
                'conf_map': CONFIDENCE_MAP
            }, MODEL_SAVE_PATH)
            print("Model saved.")
if __name__ == "__main__":
    main()
