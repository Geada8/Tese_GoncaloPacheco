"""
K-Means benchmark — binary (normal vs fault) and 4-class.
Loads preprocessed data from Data_processed\.

K-Means is an unsupervised clustering algorithm; to use it for classification
the clusters are labelled by a majority vote over the training labels assigned
to each cluster, and a test sample is classified by the label of its nearest
cluster centroid.
"""

import numpy as np, time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

OUT = r"c:\Users\ggspa\Desktop\teste wheel\Data_processed"

N_CLUSTERS = 64
N_INIT     = 20
RANDOM_STATE = 42

print("Loading preprocessed data...")
X_train = np.load(f"{OUT}/X_train.npy")
y_train = np.load(f"{OUT}/y_train.npy")
X_test  = np.load(f"{OUT}/X_test.npy")
y_test  = np.load(f"{OUT}/y_test.npy")
print(f"  X_train {X_train.shape}, X_test {X_test.shape}")


def benchmark_kmeans(y_tr, y_te, n_classes, class_names, task_name):
    print(f"\n{'='*56}")
    print(f"K-MEANS — {task_name}")
    print(f"{'='*56}")

    # Standardize features (zero mean, unit variance)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    # Fit clustering on the training data
    t0 = time.perf_counter()
    km = KMeans(n_clusters=N_CLUSTERS, n_init=N_INIT, random_state=RANDOM_STATE)
    km.fit(X_tr_s)
    fit_time = time.perf_counter() - t0

    # Label each cluster by majority class vote over its training members
    cluster_label = np.zeros(N_CLUSTERS, dtype=int)
    for c in range(N_CLUSTERS):
        mask = km.labels_ == c
        if mask.sum() > 0:
            cluster_label[c] = np.bincount(y_tr[mask], minlength=n_classes).argmax()

    # Predict: assign each test sample the label of its nearest centroid's cluster
    t0 = time.perf_counter()
    y_pred = cluster_label[km.predict(X_te_s)]
    infer_total = time.perf_counter() - t0

    # single-sample latency
    single = []
    for i in range(min(50, len(X_te_s))):
        s = time.perf_counter()
        cluster_label[km.predict(X_te_s[i:i+1])]
        single.append((time.perf_counter() - s) * 1000.0)
    single_ms = float(np.median(single))

    acc = accuracy_score(y_te, y_pred)
    cm  = confusion_matrix(y_te, y_pred)
    rep = classification_report(y_te, y_pred, target_names=class_names, digits=4)

    print(f"n_clusters={N_CLUSTERS}  n_init={N_INIT}  fit_time={fit_time:.2f}s")
    print(f"Test accuracy: {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Inference: {infer_total*1000:.1f} ms total / "
          f"{infer_total/len(X_te_s)*1000:.3f} ms/sample batched")
    print(f"Single-sample median: {single_ms:.2f} ms")
    print(f"\nConfusion matrix (rows=true, cols=pred) {class_names}:\n{cm}")
    print(f"\n{rep}")


# ── Binary ────────────────────────────────────────────────────────────────
y_tr_bin = (y_train != 0).astype(int)
y_te_bin = (y_test  != 0).astype(int)
benchmark_kmeans(y_tr_bin, y_te_bin, 2, ["normal", "fault"], "binary (normal vs fault)")

# ── 4-class ───────────────────────────────────────────────────────────────
benchmark_kmeans(y_train, y_test, 4, ["normal", "mass1", "mass2", "mass3"], "4-class")
