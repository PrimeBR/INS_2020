from math import exp

import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

train_data = np.array(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]
)


def element_wise_predict(data, weights):
    relu = lambda x: max(x, 0)
    sigmoid = lambda x: 1 / (1 + exp(-x))
    act = [relu for _ in weights]
    act[-1] = sigmoid
    tmp_data = data.copy()
    for i, weight in enumerate(weights):
        matrix = np.zeros((tmp_data.shape[0], weight[0].shape[1]))
        for j in range(tmp_data.shape[0]):
            for k in range(weight[0].shape[1]):
                s = 0
                for n in range(tmp_data.shape[1]):
                    s += tmp_data[j][n] * weight[0][n][k]
                matrix[j][k] = act[i](s + weight[1][k])
        tmp_data = matrix
    return matrix


def tensor_predict(input_data, weights):
    relu = lambda x: np.maximum(x, 0)
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    act = [relu for _ in weights]
    act[-1] = sigmoid
    res = input_data.copy()
    for i, weight in enumerate(weights):
        res = act[i](np.dot(res, weight[0]) + weight[1])
    return res


def print_predicts(model, dataset):
    weights = []
    for layer in model.layers:
        weights.append(layer.get_weights())
    element_wise_res = element_wise_predict(dataset, weights)
    tensor_res = tensor_predict(dataset, weights)
    model_res = model.predict(dataset)
    assert np.isclose(element_wise_res, model_res).all()
    assert np.isclose(tensor_res, model_res).all()
    print(f"Element:\n {element_wise_res}")
    print(f"Tensor: \n {tensor_res}")
    print(f"Model: \n {model_res}")


def logic_func(a, b, c):
    return (a != b) and (b != c)


train_target = np.array([int(logic_func(*x)) for x in train_data])
model = Sequential()
model.add(Dense(16, activation="relu", input_shape=(3,)))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

print_predicts(model, train_data)
model.fit(train_data, train_target, epochs=150, batch_size=1)
print_predicts(model, train_data)
