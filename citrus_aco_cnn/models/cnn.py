from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from citrus_aco_cnn.config import NUM_CLASSES

def build_paper_cnn(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32,3,padding='same',activation='relu')(inputs)
    x = layers.Conv2D(32,3,padding='same',activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64,3,padding='same',activation='relu')(x)
    x = layers.Conv2D(64,3,padding='same',activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128,3,padding='same',activation='relu')(x)
    x = layers.Conv2D(128,3,padding='same',activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256,activation='relu')(x)
    gap = layers.Dense(128,activation='relu',name='gap_features')(x)
    outputs = layers.Dense(NUM_CLASSES,activation='softmax')(gap)

    model = models.Model(inputs, outputs, name='paper_cnn')
    model.compile(optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
