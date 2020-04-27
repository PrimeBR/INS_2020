from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from var5 import gen_sequence


def gen_data_from_sequence(seq_len=1000, lookback=10):
    seq = gen_sequence(seq_len)
    past = np.array([[[seq[j]] for j in range(i, i + lookback)] for i in range(len(seq) - lookback)])
    future = np.array([[seq[i]] for i in range(lookback, len(seq))])
    return past, future


data, res = gen_data_from_sequence()

dataset_size = len(data)
train_size = (dataset_size // 10) * 7
val_size = (dataset_size - train_size) // 2

train_data, train_res = data[:train_size], res[:train_size]
val_data, val_res = (data[train_size : train_size + val_size], res[train_size : train_size + val_size])
test_data, test_res = data[train_size + val_size :], res[train_size + val_size :]

model = Sequential()
model.add(layers.GRU(128, recurrent_activation="sigmoid", input_shape=(None, 1), return_sequences=True))
model.add(layers.LSTM(64, activation="relu", input_shape=(None, 1), return_sequences=True, dropout=0.4))
model.add(layers.GRU(64, input_shape=(None, 1)))
model.add(layers.Dense(1))

model.compile(optimizer="nadam", loss="mse")
history = model.fit(train_data, train_res, epochs=50, validation_data=(val_data, val_res))

res = model.predict(test_data)
plt.plot(range(len(res)),test_res)
plt.plot(range(len(res)),res)
plt.legend(['test', 'predicted'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
