"""
Random Forest benchmark — binary (normal vs fault) and 4-class.
Loads preprocessed data from Data_processed\ (saved by kmeans_anomaly.py).
"""

import numpy as np, time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

OUT = r"c:\Users\ggspa\Desktop\teste wheel\Data_processed"

print("Loading preprocessed data...")
X_train = np.load(f"{OUT}/X_train.npy")
y_train = np.load(f"{OUT}/y_train.npy")
X_test  = np.load(f"{OUT}/X_test.npy")
y_test  = np.load(f"{OUT}/y_test.npy")
print(f"  X_train {X_train.shape}, X_test {X_test.shape}")


def benchmark_rf(y_tr, y_te, class_names, task_name):
    print(f"\n{'='*56}")
    print(f"RANDOM FOREST — {task_name}")
    print(f"{'='*56}")

    clf = RandomForestClassifier(n_estimators=500, max_features='sqrt',
                                  n_jobs=-1, random_state=42)
    t0 = time.perf_counter()
    clf.fit(X_train, y_tr)
    fit_time = time.perf_counter() - t0

    # batched inference
    t0 = time.perf_counter()
    y_pred = clf.predict(X_test)
    infer_total = time.perf_counter() - t0

    # single-sample latency
    single = []
    for i in range(min(50, len(X_test))):
        s = time.perf_counter()
        clf.predict(X_test[i:i+1])
        single.append((time.perf_counter() - s) * 1000.0)
    single_ms = float(np.median(single))

    acc = accuracy_score(y_te, y_pred)
    cm  = confusion_matrix(y_te, y_pred)
    rep = classification_report(y_te, y_pred, target_names=class_names, digits=4)

    print(f"n_estimators=500  fit_time={fit_time:.2f}s")
    print(f"Test accuracy: {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Inference: {infer_total*1000:.1f} ms total / "
          f"{infer_total/len(X_test)*1000:.3f} ms/sample batched")
    print(f"Single-sample median: {single_ms:.2f} ms")
    print(f"\nConfusion matrix (rows=true, cols=pred) {class_names}:\n{cm}")
    print(f"\n{rep}")


# ── Binary ────────────────────────────────────────────────────────────────
y_tr_bin = (y_train != 0).astype(int)
y_te_bin = (y_test  != 0).astype(int)
benchmark_rf(y_tr_bin, y_te_bin, ["normal", "fault"], "binary (normal vs fault)")

# ── 4-class ───────────────────────────────────────────────────────────────
benchmark_rf(y_train, y_test, ["normal", "mass1", "mass2", "mass3"], "4-class")
