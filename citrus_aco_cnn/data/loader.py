from __future__ import annotations

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from citrus_aco_cnn.config import SEED

_KEEP_IDS = tf.constant([0,1,2,3], tf.int64)

def _keep_four(_: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
    return tf.reduce_any(tf.equal(label, _KEEP_IDS))

def load_numpy_data():
    ds = tfds.load("citrus_leaves", split="train", as_supervised=True, shuffle_files=True)
    ds = ds.filter(_keep_four)
    images, labels = [], []
    for img,lbl in tfds.as_numpy(ds):
        images.append(img)
        labels.append(lbl)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(labels))
    return np.array(images)[idx], np.array(labels, np.int64)[idx]
