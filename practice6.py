from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from var5 import gen_data

size = 5000
dataset, labels = gen_data(size)
dataset = np.asarray(dataset)
labels = np.asarray(labels)

encoder = LabelEncoder()
encoder.fit(labels)
labels = encoder.transform(labels)

count = size // 5
dataset_train = dataset[count:]
dataset_test = dataset[:count]
labels_train = labels[count:]
labels_test = labels[:count]
dataset_train = dataset_train.reshape(dataset_train.shape[0], 50, 50, 1)
dataset_test = dataset_test.reshape(dataset_test.shape[0], 50, 50, 1)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',
                        input_shape=(50, 50, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
history = model.fit(dataset_train, labels_train, epochs=12, batch_size=128,
              validation_data=(dataset_test, labels_test))


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
