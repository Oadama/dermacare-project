import matplotlib.pyplot as plt
import os
import pandas as pd
from params import *


def plot_loss_accuracy(history, epochs:int):
    # plot loss & accuracy curves of the training results: history

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def plot_images_serie(data_source, num_batch:int=0):
    class_names=data_source.class_names
    plt.figure(figsize=(10, 10))
    for images, labels in data_source.take(num_batch):
        for i in range(BATCH_SIZE):
            nb_raws=round(1+os.getenv('BATCH_SIZE')/3)
            ax = plt.subplot(nb_raws, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
