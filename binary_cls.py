import matplotlib.pyplot as plt
import pandas
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:60].astype(float)
Y = dataset[:, 60]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

best_model = 0
best_val_loss = 1
best_neurons_num = 0
best_dense_num = 0


model = Sequential()
model.add(Dense(60, input_dim=60, kernel_initializer='normal',
                activation='relu'))
model.add(Dense(15, input_dim=60, kernel_initializer='normal',
                activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
mc = ModelCheckpoint(filepath=f'best_model.hdf5', monitor='val_loss',
                     save_best_only=True)
history = model.fit(X, encoded_Y, epochs=100, batch_size=10,
                    validation_split=0.1, callbacks=[mc])
current_model = load_model('best_model.hdf5')
current_values = current_model.evaluate(X[187:], encoded_Y[187:])

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'r', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc_values, 'r', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# for best model finding
for denses in range(1, 3):
    for neurons_num in range(5, 60, 5):
        model = Sequential()
        for dense in range(1, denses):
            model.add(Dense(neurons_num, input_dim=60,
                            kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])
        mc = ModelCheckpoint(filepath=f'best_model.hdf5', monitor='val_loss',
                             save_best_only=True)
        history = model.fit(X, encoded_Y, epochs=100, batch_size=10,
                            validation_split=0.1, callbacks=[mc], verbose=0)
        current_model = load_model(f'best_model.hdf5')
        current_values = current_model.evaluate(X[187:], encoded_Y[187:])
        if current_values[0] < best_val_loss:
            best_val_loss = current_values[0]
            best_model = current_model
            best_dense_num = denses
            best_neurons_num = neurons_num
            print(denses, neurons_num, current_values)
