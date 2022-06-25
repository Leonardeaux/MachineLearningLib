import MLLib
import ctypes
import numpy as np

if __name__ == '__main__':
    my_lib = MLLib.MLLib(ctypes.CDLL(
        "C:/Users/enzol/CLionProjects/DLL/cmake-build-debug/DLL.dll"
    ))
    # print(my_lib.test_mlp(p))
    # my_lib.test_print_array()
    # my_lib.test_print_matrix()

    X = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])

    Y = np.array([
        [-1.0],
        [1.0],
        [1.0],
        [1.0],
    ])

    X2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ])

    Y2 = np.array([
        [8.0],
        [4.0],
        [2.0],
        [3.0]
    ])

    model = my_lib.create_mlp([2, 3, 1])
    print("Before : ")
    for sample_input in X:
        print(my_lib.predict_mlp(model, sample_input))

    my_lib.train_mlp(model, X, Y)

    print("After : ")

    for sample_input in X:
        print(my_lib.predict_mlp(model, sample_input))

    print("For Regression : ")

    model2 = my_lib.create_mlp([2, 3, 1])
    print("Before : ")
    for sample_input in X2:
        print(my_lib.predict_mlp(model2, sample_input, False))

    my_lib.train_mlp(model2, X2, Y2, 0.01, False, 1000)

    print("After : ")

    for sample_input in X2:
        print(my_lib.predict_mlp(model2, sample_input, False))
