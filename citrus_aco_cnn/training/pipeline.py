from __future__ import annotations
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import callbacks

from citrus_aco_cnn.config import IMG_SIZE, EPOCHS_CNN, EPOCHS_HEAD, SEED
from citrus_aco_cnn.data.pipeline import make_dataset
from citrus_aco_cnn.models import build_paper_cnn, build_head
from citrus_aco_cnn.feature_selection import aco_select
from citrus_aco_cnn.evaluation import compute_metrics

def cross_validate(images, labels, *, folds=10):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(images, labels), 1):
        print(f"\n===== Fold {fold}/{folds} =====")
        x_train, y_train = images[train_idx], labels[train_idx]
        x_test, y_test = images[test_idx], labels[test_idx]

        train_ds = make_dataset(x_train, y_train, training=True)
        test_ds = make_dataset(x_test, y_test, training=False)

        cnn = build_paper_cnn((IMG_SIZE, IMG_SIZE, 3))
        cnn.fit(train_ds, validation_data=test_ds, epochs=EPOCHS_CNN,
                callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
                verbose=0)

        gap_layer = cnn.get_layer('gap_features').output
        feat_model = tf.keras.Model(cnn.input, gap_layer)
        deep_train = feat_model.predict(train_ds, verbose=0)
        deep_test = feat_model.predict(test_ds, verbose=0)

        mask = aco_select(deep_train, y_train)
        print(f"  ACO selected {mask.sum()} / {mask.size} dims.")

        head = build_head(mask.sum())
        head.fit(deep_train[:, mask], y_train, epochs=EPOCHS_HEAD, batch_size=32,
                 validation_split=0.1,
                 callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
                 verbose=0)

        y_pred = np.argmax(head.predict(deep_test[:, mask], verbose=0), axis=1)
        metrics = compute_metrics(y_test, y_pred)
        fold_metrics.append(metrics)
        print(f"  Fold accuracy = {metrics[0]:.4f}  F1 = {metrics[3]:.4f}")
    return fold_metrics
