import numpy as np

import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape)
print(test_data.shape)
print(test_targets)

mean = train_data.mean(axis=0)

train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


best_mae = [1, 0, 0]
best_val_mae = [1, 0, 0]
for k in range(4, 9):
    epochs_scores = []
    mae_scores = []
    val_mae_scores = []
    for num_epochs in range(50, 175, 25):
        num_val_samples = len(train_data) // k
        mae_values = []
        val_mae_values = []
        for i in range(k):
            print('processing fold #', i)
            val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
            val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
            partial_train_data = np.concatenate(
                [train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
                axis=0)
            partial_train_targets = np.concatenate(
                [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
            model = build_model()
            history = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1,
                                validation_data=(val_data, val_targets), verbose=0)
            history_dict = history.history
            loss_values = history_dict['loss']
            val_loss_values = history_dict['val_loss']
            mae_values = history_dict['mean_absolute_error']
            val_mae_values = history_dict['val_mean_absolute_error']
            epochs = range(1, len(mae_values) + 1)
            plt.plot(epochs, loss_values, 'r', label='Training loss')
            plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
            plt.title(f'Training and validation loss\n'
                      f'num_epochs={num_epochs}, k={k},  fold={i}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

            plt.clf()
            plt.plot(epochs, mae_values, 'r', label='Training acc')
            plt.plot(epochs, val_mae_values, 'b', label='Validation acc')
            plt.title('Training and validation accuracy\n'
                      f'num_epochs={num_epochs}, k={k},  fold={i}')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()
        mean_mae = np.mean(mae_values)
        mean_val_mae = np.mean(val_mae_values)
        if mean_mae <= best_mae[0]:
            best_mae[0] = mean_mae
            best_mae[1] = num_epochs
            best_mae[2] = k
            print(best_mae)
        if mean_val_mae <= best_val_mae[0]:
            best_val_mae[0] = mean_val_mae
            best_val_mae[1] = num_epochs
            best_val_mae[2] = k
            print(best_val_mae)
        mae_scores.append(mean_mae)
        val_mae_scores.append(mean_val_mae)
        epochs_scores.append(num_epochs)
    plt.plot(epochs_scores, mae_scores, 'r', label='Mean MAE')
    plt.plot(epochs_scores, val_mae_scores, 'b', label='Mean val MAE')
    plt.title(f'k={k}')
    plt.xlabel('Num of epochs')
    plt.ylabel('Mae')
    plt.legend()
    plt.show()