from __future__ import annotations
import tensorflow as tf

def augment(image: tf.Tensor, label: tf.Tensor):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.rot90(image, k=tf.random.uniform([],0,4,tf.int32))
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8,1.2)
    return image, label
