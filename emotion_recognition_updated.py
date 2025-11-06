import numpy as np
import pandas as pd
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
from imblearn.metrics import specificity_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve

# Gujarati Rasya Emotion Dataset Setup
EMOTIONS = {
    1: 'Shant ras',
    2: 'Hasya ras',
    3: 'Bhayanak ras',
    4: 'Karuna ras',
    5: 'Rudra ras'
}
DATA_PATH = 'personal_dataset'
SAMPLE_RATE = 22050

data = []
for dirname, _, filenames in os.walk(DATA_PATH):
    for filename in filenames:
        if not filename.lower().endswith(".wav"):
            continue
        file_path = os.path.join(dirname, filename)
        identifiers = filename.split('.')[0].split('-')
        if len(identifiers) != 4:
            continue
        try:
            emotion = int(identifiers[0])
            gender = 'male' if identifiers[1] == '00' else 'female'
            drama_id = int(identifiers[2])
            dataset_id = int(identifiers[3])
        except:
            continue
        data.append({
            "Emotion": emotion,
            "Gender": gender,
            "Drama ID": drama_id,
            "Dataset ID": dataset_id,
            "Path": file_path
        })

data = pd.DataFrame(data)
print("Number of files:", len(data))
print(data.head())

# Plot Emotion Distribution
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(x=range(1, 6), height=data['Emotion'].value_counts().sort_index())
ax.set_xticks(ticks=range(1, 6))
ax.set_xticklabels([EMOTIONS[i] for i in range(1, 6)], rotation=45, fontsize=10)
ax.set_xlabel('Emotions (Rasya)')
ax.set_ylabel('Number of Samples')
plt.tight_layout()
plt.show()

def pre_emphasis(signal, alpha=0.97):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

signals = []
for i, file_path in enumerate(data.Path):
    audio, _ = librosa.load(file_path, duration=3, offset=0.5, sr=SAMPLE_RATE)
    signal = np.zeros((int(SAMPLE_RATE*3,)))
    signal[:len(audio)] = audio
    signals.append(pre_emphasis(signal))
    print(f"\rProcessed {i+1}/{len(data)} files", end='')
print('')
signals = np.stack(signals, axis=0)

X = signals
train_ind, test_ind, val_ind = [], [], []
X_train, X_val, X_test = [], [], []
Y_train, Y_val, Y_test = [], [], []
np.random.seed(69)

for emotion in range(1, 6):
    emotion_ind = list(data.loc[data.Emotion == emotion, 'Emotion'].index)
    emotion_ind = np.random.permutation(emotion_ind)
    m = len(emotion_ind)
    ind_train = emotion_ind[:int(0.9*m)]
    ind_val = emotion_ind[int(0.9*m):int(0.95*m)]
    ind_test = emotion_ind[int(0.95*m):]
    X_train.append(X[ind_train, :])
    Y_train.append(np.array([emotion]*len(ind_train), dtype=np.int32))
    X_val.append(X[ind_val, :])
    Y_val.append(np.array([emotion]*len(ind_val), dtype=np.int32))
    X_test.append(X[ind_test, :])
    Y_test.append(np.array([emotion]*len(ind_test), dtype=np.int32))
    train_ind.append(ind_train)
    test_ind.append(ind_test)
    val_ind.append(ind_val)

X_train = np.concatenate(X_train, 0)
X_val = np.concatenate(X_val, 0)
X_test = np.concatenate(X_test, 0)
Y_train = np.concatenate(Y_train, 0)
Y_val = np.concatenate(Y_val, 0)
Y_test = np.concatenate(Y_test, 0)

def getMELspectrogram(audio, sample_rate):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=1024,
                                              win_length=512, window='hamming',
                                              hop_length=256, n_mels=128, fmax=sample_rate/2)
    return librosa.power_to_db(mel_spec, ref=np.max)

mel_train = [getMELspectrogram(x, SAMPLE_RATE) for x in X_train]
mel_val = [getMELspectrogram(x, SAMPLE_RATE) for x in X_val]
mel_test = [getMELspectrogram(x, SAMPLE_RATE) for x in X_test]

def splitIntoChunks(mel_spec, win_size, stride):
    t = mel_spec.shape[1]
    chunks = [mel_spec[:, i*stride:i*stride+win_size] for i in range(int(t/stride)) if mel_spec[:, i*stride:i*stride+win_size].shape[1] == win_size]
    return np.stack(chunks, axis=0)

mel_train_chunked = [splitIntoChunks(m, 128, 64) for m in mel_train]
mel_val_chunked = [splitIntoChunks(m, 128, 64) for m in mel_val]
mel_test_chunked = [splitIntoChunks(m, 128, 64) for m in mel_test]

X_train = np.expand_dims(np.stack(mel_train_chunked, axis=0), 2)
X_val = np.expand_dims(np.stack(mel_val_chunked, axis=0), 2)
X_test = np.expand_dims(np.stack(mel_test_chunked, axis=0), 2)

scaler = StandardScaler()
for arr in [X_train, X_val, X_test]:
    b, t, c, h, w = arr.shape
    arr[:] = np.reshape(scaler.fit_transform(np.reshape(arr, (b, -1))), (b, t, c, h, w))

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        elif len(x.size()) == 4:
            x_reshape = x.contiguous().view(-1, x.size(2), x.size(3))
        else:
            x_reshape = x.contiguous().view(-1, x.size(2), x.size(3), x.size(4))
        y = self.module(x_reshape)
        if len(x.size()) == 4:
            return y.contiguous().view(x.size(0), -1, y.size(1), y.size(2))
        else:
            return y.contiguous().view(x.size(0), -1, y.size(1), y.size(2), y.size(3))

class HybridModel(nn.Module):
    def __init__(self, num_emotions):
        super().__init__()
        self.conv2Dblock = nn.Sequential(
            TimeDistributed(nn.Conv2d(1, 16, 3, 1, 1)),
            TimeDistributed(nn.BatchNorm2d(16)),
            TimeDistributed(nn.ReLU()),
            TimeDistributed(nn.MaxPool2d(2, 2)),
            TimeDistributed(nn.Dropout(0.3)),
            TimeDistributed(nn.Conv2d(16, 32, 3, 1, 1)),
            TimeDistributed(nn.BatchNorm2d(32)),
            TimeDistributed(nn.ReLU()),
            TimeDistributed(nn.MaxPool2d(4, 4)),
            TimeDistributed(nn.Dropout(0.3)),
            TimeDistributed(nn.Conv2d(32, 64, 3, 1, 1)),
            TimeDistributed(nn.BatchNorm2d(64)),
            TimeDistributed(nn.ReLU()),
            TimeDistributed(nn.MaxPool2d(4, 4)),
            TimeDistributed(nn.Dropout(0.3))
        )
        hidden_size = 20
        self.lstm = nn.LSTM(1024, hidden_size, bidirectional=True, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.4)
        self.attention_linear = nn.Linear(2*hidden_size, 1)
        self.out_linear = nn.Linear(2*hidden_size, num_emotions)
    def forward(self, x):
        conv_embedding = self.conv2Dblock(x)
        conv_embedding = torch.flatten(conv_embedding, start_dim=2)
        lstm_embedding, _ = self.lstm(conv_embedding)
        lstm_embedding = self.dropout_lstm(lstm_embedding)
        attention_weights = torch.softmax(self.attention_linear(lstm_embedding).squeeze(-1), dim=-1)
        attention_output = torch.bmm(attention_weights.unsqueeze(1), lstm_embedding).squeeze(1)
        logits = self.out_linear(attention_output)
        return logits, torch.softmax(logits, dim=1)

def loss_fnc(pred, target): return nn.CrossEntropyLoss()(pred, target)

def make_train_step(model, loss_fnc, optimizer):
    def train_step(X, Y):
        model.train()
        logits, probs = model(X)
        preds = torch.argmax(probs, dim=1)
        loss = loss_fnc(logits, Y)
        acc = torch.sum(Y == preds).float() / len(Y)
        loss.backward()
        optimizer.step(); optimizer.zero_grad()
        return loss.item(), acc.item()*100
    return train_step

def make_validate_fnc(model, loss_fnc):
    def validate(X, Y):
        with torch.no_grad():
            model.eval()
            logits, probs = model(X)
            preds = torch.argmax(probs, dim=1)
            loss = loss_fnc(logits, Y)
            acc = torch.sum(Y == preds).float() / len(Y)
        return loss.item(), acc.item()*100, preds
    return validate

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = HybridModel(num_emotions=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_step = make_train_step(model, loss_fnc, optimizer)
validate = make_validate_fnc(model, loss_fnc)

EPOCHS, BATCH_SIZE = 50, 16
losses, val_losses = [], []
for epoch in range(EPOCHS):
    ind = np.random.permutation(len(X_train))
    X_train, Y_train = X_train[ind], Y_train[ind]
    epoch_loss, epoch_acc = 0, 0
    for i in range(0, len(X_train), BATCH_SIZE):
        X_batch = torch.tensor(X_train[i:i+BATCH_SIZE], device=device).float()
        Y_batch = torch.tensor(Y_train[i:i+BATCH_SIZE], device=device).long() - 1
        loss, acc = train_step(X_batch, Y_batch)
        epoch_loss += loss; epoch_acc += acc
        print(f"\rEpoch {epoch+1}/{EPOCHS} Iter {i//BATCH_SIZE+1}", end='')
    X_val_tensor = torch.tensor(X_val, device=device).float()
    Y_val_tensor = torch.tensor(Y_val, device=device).long() - 1
    val_loss, val_acc, _ = validate(X_val_tensor, Y_val_tensor)
    losses.append(epoch_loss); val_losses.append(val_loss)
    print(f"\nEpoch {epoch+1}: loss={epoch_loss:.4f}, acc={epoch_acc/len(X_train)*BATCH_SIZE:.2f}%, val_acc={val_acc:.2f}%")

X_test_tensor = torch.tensor(X_test, device=device).float()
Y_test_tensor = torch.tensor(Y_test, device=device).long() - 1
test_loss, test_acc, predictions = validate(X_test_tensor, Y_test_tensor)
print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.2f}%")

cm = confusion_matrix(Y_test_tensor.cpu(), predictions.cpu())
names = [EMOTIONS[i] for i in range(1, 6)]
df_cm = pd.DataFrame(cm, index=names, columns=names)
sn.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
plt.xlabel('Predicted'); plt.ylabel('True'); plt.show()
print(classification_report(Y_test_tensor.cpu(), predictions.cpu(), target_names=names))
