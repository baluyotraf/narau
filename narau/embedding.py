import numpy as np


def load_glove_weights(path, encoding='utf-8'):
    """
    Loads the weights from a glove formatted file
    :param path: path to the glove formatted file
    :param encoding: file encoding
    :return: numpy array containing the embeddings from the file
    """
    with open(path, 'r', encoding=encoding) as file:
        weights = [[float(v) for v in line.strip().split(' ')[1:]] for line in file]
        weights = np.asarray(weights)
        return weights
