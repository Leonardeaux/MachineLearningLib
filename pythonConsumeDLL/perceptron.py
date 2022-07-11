import numpy as np
import MLLib as ML
import ctypes
from dataset_processing import get_dataset

my_lib = ML.MLLib(ctypes.CDLL(
    "C:/Users/enzol/Documents/MachineLearningLib/DLL/cmake-build-debug/DLL.dll"
))


def to_categorical(y, num_classes, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)

    return categorical


if __name__ == '__main__':
    (xtr, ytr) = get_dataset("Train")
    (xte, yte) = get_dataset("Test")

    # ytr_one_hot = to_categorical(ytr, 2)
    # yte_one_hot = to_categorical(yte, 2)

    mini_xtr = np.array(xtr, dtype=np.float64)
    mini_ytr = ytr

    print(mini_ytr)

    model = my_lib.create_mlp([len(mini_xtr[0]), 64, 1])
    print("Before : --------------------------------------------------------------------")
    for sample_input in mini_xtr:
        print(my_lib.predict_mlp(model, sample_input))

    my_lib.train_mlp(model, mini_xtr, mini_ytr, 0.01, True, 100000)

    print("After : --------------------------------------------------------------------")

    for sample_input in mini_xtr:
        print(my_lib.predict_mlp(model, sample_input))