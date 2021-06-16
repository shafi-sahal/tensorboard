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


def get_confusion_matrix(y_labels, logits, class_names):
    preds = np.argmax(logits, axis=1)
    cm = sklearn.metrics.confusion_matrix(
        y_labels, preds, labels=np.arange(len(class_names))
    )
    return cm


def plot_confusion_matrix(cm, class_names):
    size = len(class_names)
    figure = plt.figure(figsize=(size, size))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')

    indices = np.arange(len(class_names))
    plt.xticks(indices, class_names, rotation=45)
    plt.yticks(indices, class_names)

    cm = np.around(
        cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=3
    )

    threshold = cm.max() / 2.0
    for i in range(size):
        for j in range(size):
            color = 'white' if cm[i, j] > threshold else 'black'
            plt.text(
                i, j, cm[i, j], horizontalalignment='center', color=color
            )

        plt.tight_layout()
        plt.xlabel('True Label')
        plt.ylabel('Predicted Label')

        cm_image = plot_to_image(figure)
        return cm_image
