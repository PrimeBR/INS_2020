import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import numpy as np
from PIL import Image

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images / 255.0
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images / 255.0


train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(Dense(10, activation='softmax'))


optimizers = (SGD(),
              SGD(learning_rate=0.01, momentum=0.1),
              SGD(learning_rate=0.01, momentum=0.9),
              Adam(),
              Adam(learning_rate=0.01),
              Adam(learning_rate=0.1),
              RMSprop(),
              RMSprop(learning_rate=0.01),
              RMSprop(momentum=0.1))


def upload_image(filepath):
    img = Image.open(fp=filepath)
    img = np.asarray(img)
    img = img.resize((28, 28))
    k = np.array([[[0.2989, 0.587, 0.114]]])
    img = np.sum(img * k, axis=2).reshape((1, 28 * 28)) / 255.0
    return img


def explore_effect(optimizers):
    for optimizer in optimizers:
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        history = model.fit(
            train_images, train_labels, epochs=5, batch_size=128,
            validation_data=(test_images, test_labels), verbose=0
        )

        history_dict = history.history
        loss_values = history_dict["loss"]
        val_loss_values = history_dict["val_loss"]
        epochs = range(1, len(loss_values) + 1)
        plt.plot(epochs, loss_values, "r", label="Training loss")
        plt.plot(epochs, val_loss_values, "b", label="Validation loss")
        plt.title(f"Training and validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        plt.clf()
        acc_values = history_dict["acc"]
        val_acc_values = history_dict["val_acc"]
        plt.plot(epochs, acc_values, "r", label="Training acc")
        plt.plot(epochs, val_acc_values, "b", label="Validation acc")
        plt.title(f"Training and validation acc")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
        print(f"val_acc: {max(val_acc_values)}, val_loss: {min(val_loss_values)}")

explore_effect(optimizers)
