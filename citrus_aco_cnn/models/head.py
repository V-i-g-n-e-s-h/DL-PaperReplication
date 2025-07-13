from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from citrus_aco_cnn.config import NUM_CLASSES

def build_head(input_dim):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dropout(0.3)(inputs)
    x = layers.Dense(128,activation='relu')(x)
    outputs = layers.Dense(NUM_CLASSES,activation='softmax')(x)
    model = models.Model(inputs, outputs, name='aco_head')
    model.compile(optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
