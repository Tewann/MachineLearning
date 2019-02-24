import numpy as np


def init_variables():
    weight = np.random.normal(size=2)
    bias = 0
    return weight, bias


def get_dataset():
    row_per_class = 5
    sick = np.random.randn(row_per_class, 2) + np.array([-2, -2])
    healthy = np.random.randn(row_per_class, 2) + np.array([2, 2])

    features = np.vstack([sick, healthy])
    targets = np.concatenate(
        (np.zeros(row_per_class), np.zeros(row_per_class) + 1))

    return features, targets


def pre_activation(features, weights, biais):
    return np.dot(features, weights) + bias


def activation(z):
    return 1 / (1 + np.exp(-z))


if __name__ == '__main__':
    features, targets = get_dataset()
    weights, bias = init_variables()
    z = pre_activation(features, weights, bias)
    a = activation(z)
    print(targets)
    print(a)
    pass
