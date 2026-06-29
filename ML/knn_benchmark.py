"""
KNN benchmark — binary and 4-class.

"""

import numpy as np, time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

OUT = r"c:\Users\ggspa\Desktop\teste wheel\Data_processed"

print("Loading preprocessed data...")
X_train = np.load(f"{OUT}/X_train.npy")
y_train = np.load(f"{OUT}/y_train.npy")
X_test  = np.load(f"{OUT}/X_test.npy")
y_test  = np.load(f"{OUT}/y_test.npy")
print(f"  X_train {X_train.shape}, X_test {X_test.shape}")

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train)
X_te_s = scaler.transform(X_test)


def benchmark_knn(y_tr, y_te, class_names, task_name, k=5):
    print(f"\n{'='*56}")
    print(f"KNN (k={k}) — {task_name}")
    print(f"{'='*56}")

    clf = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1)
    t0 = time.perf_counter()
    clf.fit(X_tr_s, y_tr)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = clf.predict(X_te_s)
    infer_total = time.perf_counter() - t0

    single = []
    for i in range(min(50, len(X_te_s))):
        s = time.perf_counter()
        clf.predict(X_te_s[i:i+1])
        single.append((time.perf_counter() - s) * 1000.0)
    single_ms = float(np.median(single))

    acc = accuracy_score(y_te, y_pred)
    cm  = confusion_matrix(y_te, y_pred)
    rep = classification_report(y_te, y_pred, target_names=class_names, digits=4)

    print(f"k={k}  metric=euclidean  fit_time={fit_time:.3f}s")
    print(f"Test accuracy: {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Inference: {infer_total*1000:.1f} ms total / "
          f"{infer_total/len(X_te_s)*1000:.3f} ms/sample batched")
    print(f"Single-sample median: {single_ms:.2f} ms")
    print(f"\nConfusion matrix (rows=true, cols=pred) {class_names}:\n{cm}")
    print(f"\n{rep}")


y_tr_bin = (y_train != 0).astype(int)
y_te_bin = (y_test  != 0).astype(int)

benchmark_knn(y_tr_bin, y_te_bin, ["normal", "fault"], "binary (normal vs fault)", k=5)
benchmark_knn(y_train,  y_test,   ["normal", "mass1", "mass2", "mass3"], "4-class", k=5)
