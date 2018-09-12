import numpy as np


def load_glove_weights(path, encoding='utf-8'):
    with open(path, 'r', encoding=encoding) as file:
        weights = [[float(v) for v in line.strip().split(' ')[1:]] for line in file]
        weights = np.asarray(weights)
        return weights
