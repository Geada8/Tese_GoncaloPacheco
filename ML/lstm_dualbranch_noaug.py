"""
LSTM dual-branch 4-class cross-speed benchmark.

"""

import numpy as np, os, glob, time
from scipy.signal import resample as sci_resample
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

BASE = r"c:\Users\ggspa\Desktop\teste wheel\Data2"
N, REF = 2500, 1000.0
FFT_LO, FFT_HI = 1, 251
LF_BINS = 3
ALPHA_FIXED = 11.3
_hann = np.hanning(N)

LABEL_MAP = {"Normal":0,"Mass1":1,"Mass2":2,"Mass3":3,
             "1.2x_Normal":0,"1.2x_Mass1":1,"1.2x_Mass2":2,"1.2x_Mass3":3}
RPM_MAP   = {"Normal":1004,"Mass1":1013,"Mass2":1010,"Mass3":1008,
             "1.2x_Normal":1204,"1.2x_Mass1":1214,"1.2x_Mass2":1208,"1.2x_Mass3":1206}
TRAIN_FOLDERS = ["Normal","Mass1","Mass2","Mass3"]
TEST_FOLDERS  = ["1.2x_Normal","1.2x_Mass1","1.2x_Mass2","1.2x_Mass3"]
CLASS_NAMES   = ["normal","mass1","mass2","mass3"]
N_CLASSES, SPEC_LEN, N_FREQ_BINS, N_CH = 4, 1500, 250, 6

EPOCHS, BATCH_SIZE = 120, 32


def extract_spectrum(raw, actual_rpm):
    nr = int(round(N * actual_rpm / REF))
    rs = sci_resample(raw.astype(np.float64), nr, axis=0)[:N, :]
    w  = rs * _hann[:, np.newaxis]
    mag = np.abs(np.fft.rfft(w, axis=0))
    L  = np.log1p(mag[FFT_LO:FFT_HI, :]).astype(np.float32)
    log_r = np.log(actual_rpm / REF)
    L -= 2.0 * log_r
    L[:LF_BINS, :] -= (ALPHA_FIXED - 2.0) * log_r
    return L


def load_csv(f):
    try:
        raw = np.loadtxt(f, delimiter=',', skiprows=1)
        if raw.ndim < 2 or raw.shape[1] < 7:
            return None
        raw = raw[:, 1:]
    except Exception:
        return None
    return raw if raw.shape == (N, 6) else None


def load_set(folders):
    X, y = [], []
    for folder in folders:
        lab, rpm = LABEL_MAP[folder], RPM_MAP[folder]
        for f in sorted(glob.glob(os.path.join(BASE, folder, "*.csv"))):
            raw = load_csv(f)
            if raw is None:
                continue
            X.append(extract_spectrum(raw, rpm))
            y.append(lab)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


print("Loading + preprocessing training data (1000 RPM)...")
X_all, y_all = load_set(TRAIN_FOLDERS)
print(f"  {len(X_all)} samples  dist={np.bincount(y_all)}")

print("Loading + preprocessing test data (1.2x)...")
X_te, y_te = load_set(TEST_FOLDERS)
print(f"  {len(X_te)} samples  dist={np.bincount(y_te)}")

sss = StratifiedShuffleSplit(1, test_size=0.2, random_state=42)
tr_idx, va_idx = next(sss.split(X_all, y_all))
X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
X_va, y_va = X_all[va_idx], y_all[va_idx]
print(f"  train={len(X_tr)}  val={len(X_va)}")

flat = X_tr.reshape(-1, SPEC_LEN).astype(np.float64)
mean = flat.mean(0).astype(np.float32)
std  = (flat.std(0) + 1e-8).astype(np.float32)
def std_fn(arr):
    return ((arr.reshape(-1, SPEC_LEN) - mean) / std).astype(np.float32)

X_tr_s = std_fn(X_tr)
X_va_s = std_fn(X_va)
X_te_s = std_fn(X_te)

y_tr_oh = np.eye(N_CLASSES)[y_tr].astype(np.float32)
y_va_oh = np.eye(N_CLASSES)[y_va].astype(np.float32)

# ── Dual-branch LSTM ──────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Reshape, LSTM, Dense, Dropout,
                                      Cropping1D, Flatten, Concatenate)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tempfile

tf.keras.backend.clear_session()
inp  = Input(shape=(SPEC_LEN,), dtype='float32')
spec = Reshape((N_FREQ_BINS, N_CH))(inp)
lf   = Cropping1D((0, N_FREQ_BINS - LF_BINS))(spec)
lf   = Flatten()(lf)
lf   = Dense(32, activation='relu')(lf)
x    = LSTM(128, return_sequences=True)(spec)
x    = LSTM(64)(x)
z    = Concatenate()([x, lf])
z    = Dense(128, activation='relu')(z); z = Dropout(0.35)(z)
out  = Dense(N_CLASSES, activation='softmax')(z)
model = Model(inp, out)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='categorical_crossentropy', metrics=['accuracy'])
print(f"\nModel params: {model.count_params():,}")

tmp = tempfile.NamedTemporaryFile(suffix='.keras', delete=False); tmp.close()
cbs = [
    EarlyStopping(monitor='val_accuracy', patience=30,
                  restore_best_weights=True, mode='max', verbose=0),
    ModelCheckpoint(tmp.name, monitor='val_accuracy', save_best_only=True,
                    save_weights_only=False, verbose=0),
]

print("Training (no augmentation)...")
t0 = time.perf_counter()
hist = model.fit(X_tr_s, y_tr_oh, validation_data=(X_va_s, y_va_oh),
                 epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cbs, verbose=0)
train_time = time.perf_counter() - t0
best_val = max(hist.history['val_accuracy'])
n_epochs  = len(hist.history['loss'])
print(f"  epochs={n_epochs}  best_val_acc={best_val:.4f}  train_time={train_time:.1f}s")

_ = model.predict(X_te_s[:32], verbose=0)
t0 = time.perf_counter()
y_prob = model.predict(X_te_s, verbose=0)
infer_total = time.perf_counter() - t0
y_pred = np.argmax(y_prob, axis=1)

single = []
for i in range(min(50, len(X_te_s))):
    s = time.perf_counter()
    model.predict(X_te_s[i:i+1], verbose=0)
    single.append((time.perf_counter() - s) * 1000.0)
single_ms = float(np.median(single))

acc = accuracy_score(y_te, y_pred)
cm  = confusion_matrix(y_te, y_pred)
rep = classification_report(y_te, y_pred, target_names=CLASS_NAMES, digits=4)

print(f"\n{'='*56}")
print(f"LSTM DUAL-BRANCH — 4-CLASS CROSS-SPEED (no augmentation)")
print(f"{'='*56}")
print(f"params={model.count_params():,}  epochs={n_epochs}  train_time={train_time:.1f}s")
print(f"best_val_acc={best_val:.4f}  TEST_acc={acc:.4f} ({acc*100:.2f}%)")
print(f"Inference: {infer_total*1000:.1f} ms / "
      f"{infer_total/len(X_te_s)*1000:.3f} ms/sample batched")
print(f"Single-sample median: {single_ms:.2f} ms")
print(f"\nConfusion matrix (rows=true, cols=pred) {CLASS_NAMES}:\n{cm}")
print(f"\n{rep}")

try: os.unlink(tmp.name)
except: pass
