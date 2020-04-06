import csv

import numpy as np
from keras.layers import Dense, Input
from keras.models import Model, Sequential


def generate_data(nrows, ncols):
    data = np.zeros((nrows, ncols))
    target = np.zeros(nrows)
    for i in range(nrows):
        x = np.random.normal(3, 10)
        e = np.random.normal(0, 0.3)
        data[i, :] = (
            x ** 2 + e,
            np.sin(x / 2) + e,
            np.cos(2 * x) + e,
            x - 3 + e,
            -x + e,
            (x ** 3) / 4 + e,
        )
        target[i] = np.abs(x) + e
    return data, target


def write_to_csv(file_name, fields, mode):
    file = open(file_name, "w")
    out = csv.writer(file, delimiter=",")
    if mode == 1:
        for item in fields:
            out.writerow(item)
    else:
        out.writerows(map(lambda x: [x], fields))
    file.close()


(train_data, train_target) = generate_data(200, 6)
(test_data, test_target) = generate_data(30, 6)

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

encoding_dim = 2
main_input = Input(shape=(6,), name="main_input")
encoded = Dense(64, activation="relu")(main_input)
encoded = Dense(64, activation="relu")(encoded)
encoded = Dense(32, activation="relu")(encoded)
encoded = Dense(encoding_dim, activation="linear")(encoded)

decoded = Dense(64, activation="relu", name="dec1")(encoded)
decoded = Dense(64, activation="relu", name="dec2")(decoded)
decoded = Dense(64, activation="relu", name="dec3")(decoded)
decoded = Dense(6, name="dec4")(decoded)

regression = Dense(64, activation="relu", kernel_initializer="normal")(encoded)
regression = Dense(64, activation="relu")(regression)
regression = Dense(64, activation="relu")(regression)
regression = Dense(1, name="out_regression")(regression)


autoencoder = Model(main_input, decoded)
encoded = Model(main_input, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder = autoencoder.get_layer("dec1")(encoded_input)
decoder = autoencoder.get_layer("dec2")(decoder)
decoder = autoencoder.get_layer("dec3")(decoder)
decoder = autoencoder.get_layer("dec4")(decoder)
decoder = Model(encoded_input, decoder)

autoencoder.compile(optimizer="adam", loss="mse")

autoencoder.fit(
    train_data,
    train_data,
    epochs=120,
    batch_size=5,
    verbose=0,
    validation_data=(test_data, test_data),
)

regression = Model(main_input, regression)
regression.compile(optimizer="adam", loss="mse")
regression.fit(
    train_data,
    train_target,
    epochs=80,
    batch_size=2,
    verbose=0,
    validation_data=(test_data, test_target),
)

encoded_data = encoded.predict(test_data)
decoded_data = decoder.predict(encoded_data)
predicted_data = regression.predict(test_data)

decoder.save("decoder.h5")
encoded.save("encoder.h5")
regression.save("regression.h5")

write_to_csv("train_data.csv", np.round(train_data, 3), 1)
write_to_csv("test_data.csv", np.round(test_data, 3), 1)
write_to_csv("train_target.csv", np.round(train_target, 3), 0)
write_to_csv("test_target.csv", np.round(test_target, 3), 0)
write_to_csv("encoded_data.csv", np.round(encoded_data, 3), 1)
write_to_csv("decoded_data.csv", np.round(decoded_data, 3), 1)
write_to_csv(
    "result.csv",
    np.round(np.column_stack((test_target, predicted_data[:, 0])), 3),
    1,
)
