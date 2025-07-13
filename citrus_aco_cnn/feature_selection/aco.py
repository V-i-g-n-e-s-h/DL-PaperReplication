from __future__ import annotations
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from citrus_aco_cnn.config import ACO_ANTS, ACO_ITERS, ACO_EVAP, SEED, NUM_CLASSES

def aco_select(X, y, *, ants=ACO_ANTS, iters=ACO_ITERS, evap=ACO_EVAP, seed=SEED):
    rng = np.random.default_rng(seed)
    n_feat = X.shape[1]
    pher = np.ones(n_feat, dtype=np.float32)
    best_mask, best_score = None, 0.0
    oh_y = tf.one_hot(y, NUM_CLASSES).numpy()

    for _ in range(iters):
        masks = rng.random((ants, n_feat)) < (pher / pher.max())
        scores = np.zeros(ants, dtype=np.float32)
        for k, m in enumerate(masks):
            if not m.any():
                m[rng.integers(n_feat)] = True
            W = np.linalg.lstsq(X[:, m], oh_y, rcond=None)[0]
            y_hat = np.argmax(X[:, m] @ W, axis=1)
            scores[k] = accuracy_score(y, y_hat)
        pher *= 1.0 - evap
        for m, s in zip(masks, scores):
            pher[m] += s
        idx = scores.argmax()
        if scores[idx] > best_score:
            best_score, best_mask = scores[idx], masks[idx]
    return best_mask
