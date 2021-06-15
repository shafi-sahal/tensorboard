import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import io
import sklearn.metrics
from tensorboard.plugins import projector
import cv2
import os
import shutil


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


def image_grid(data, labels, class_names):
    assert data.ndim == 4

    figure = plt.figure(figsize=(10, 10))
    num_images = data.shape[0]
    size = int(np.ceil(np.sqrt(num_images)))

    for i in range(data.shape[0]):
        plt.subplot(size, size, i+1, title=class_names[labels[i]])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        if data.shape[3] == 1:
            plt.imshow(data[i], cmap=plt.cm.binary)
        else:
            plt.imshow(data[i])
    return figure
