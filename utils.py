import numpy as np

TINY = 1e-8


def get_dim(tensor):
    dim = 1

    for d in tensor.get_shape()[1:].as_list():
        dim *= d

    return dim


def to_categorical(y, nb_classes=None):
    y = np.asarray(y, dtype='int32')

    if not nb_classes:
        nb_classes = np.max(y) + 1

    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.

    return Y
