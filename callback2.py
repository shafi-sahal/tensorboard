import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow.keras import layers
from utils import plot_to_image, image_grid

(ds_train, ds_test), ds_info = tfds.load(
    "cifar10",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


def normalize_img(image, label):
    """Normalizes images"""
    return tf.cast(image, tf.float32) / 255.0, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32


def augment(image, label):
    if tf.random.uniform((), minval=0, maxval=1) < 0.1:
        image = tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3])

    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_flip_left_right(image)
    image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)

    return image, label


# Setup for train dataset
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.map(augment)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

# Setup for test Dataset
ds_test = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_train.batch(BATCH_SIZE)
ds_test = ds_train.prefetch(AUTOTUNE)

class_names = [
    "Airplane",
    "Autmobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
]


def get_model():
    model = keras.Sequential(
        [
            layers.Input((32, 32, 3)),
            layers.Conv2D(8, 3, padding="same", activation="relu"),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(10),
        ]
    )

    return model


model = get_model()
num_epochs = 1
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(lr=0.001)
acc_metric = keras.metrics.SparseCategoricalAccuracy()
writer = tf.summary.create_file_writer("logs/train/")
step = 0


for epoch in range(num_epochs):
    for batch_idx, (x, y) in enumerate(ds_train):
        figure = image_grid(x, y, class_names)

        with writer.as_default():
            tf.summary.image(
                'Visualize Images', plot_to_image(figure), step=step
            )
            step += 1

        # Reset accuracy in between epochs (and for testing and test)


        # Iterate through test set
        '''for batch_idx, (x, y) in enumerate(ds_test):
            y_pred = model(x, training=False)
            loss = loss_fn(y, y_pred)
            acc_metric.update_state(y, y_pred)

            with test_writer.as_default():
                tf.summary.scalar("Loss", loss, step=test_step)
                tf.summary.scalar(
                    "Accuracy", acc_metric.result(), step=test_step,
                )
                test_step += 1

        acc_metric.reset_states()

    # Reset accuracy in between epochs (and for testing and test)
    acc_metric.reset_states()'''