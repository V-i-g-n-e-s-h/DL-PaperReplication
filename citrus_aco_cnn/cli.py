"""Command-line entry point – `python -m citrus_aco_cnn`."""
from __future__ import annotations
from citrus_aco_cnn.data import load_numpy_data
from citrus_aco_cnn.training import cross_validate
from citrus_aco_cnn.evaluation import summarize

def main():
    images, labels = load_numpy_data()
    print(f"Loaded {len(labels)} images.")
    fold_metrics = cross_validate(images, labels)
    summary = summarize(fold_metrics)
    print("\n===== 10-Fold Summary =====")
    print(f"Accuracy  : {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
    print(f"Precision : {summary['precision_mean']:.4f}")
    print(f"Recall    : {summary['recall_mean']:.4f}")
    print(f"F1-score  : {summary['f1_mean']:.4f}")
    print("Confusion Matrix (summed across folds):\n", summary['confusion_matrix_total'])

if __name__ == '__main__':
    main()
