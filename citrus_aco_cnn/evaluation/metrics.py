from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return acc, prec, rec, f1, cm

def summarize(fold_metrics):
    accs, precs, recs, f1s, cms = zip(*fold_metrics)
    return {
        'accuracy_mean': float(np.mean(accs)),
        'accuracy_std': float(np.std(accs)),
        'precision_mean': float(np.mean(precs)),
        'recall_mean': float(np.mean(recs)),
        'f1_mean': float(np.mean(f1s)),
        'confusion_matrix_total': np.sum(cms, axis=0),
    }
