import MLLib
import ctypes
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

if __name__ == '__main__':
    my_lib = MLLib.MLLib(ctypes.CDLL(
        "C:/Users/enzol/Documents/MachineLearningLib/DLL/cmake-build-debug/DLL.dll"
    ))

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
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0]
    ])

    Y2 = np.array([
        [1.0],
        [2.0],
        [3.0]
    ])

    data = np.array([
        [9, 9, 9],
        [1, 1, 5],
        [-1, -1, 6],
        [3, 3, 4],
        [10, 10, 9],
        [-2, -2, 0],
        [7, 8, 8],
        [0.2, 0, 5],
        [-1, 0, 8],
        [6, 10, 11]
    ])

    data2 = np.array([
        [9, 9],
        [1, 1],
        [-1, -1],
        [3, 3],
        [10, 10],
        [-2, -2],
        [7, 8],
        [0.2, 0],
        [-1, 0],
        [6, 10]
    ])

    X3 = np.random.random((500, 2)) * 2.0 - 1.0
    Y3 = np.array([[1, 0, 0] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else
                  [0, 1, 0] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else
                  [0, 0, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else
                  [0, 0, 0] for p in X3])

    X3 = X3[[not np.all(arr == [0, 0, 0]) for arr in Y3]]
    Y3 = Y3[[not np.all(arr == [0, 0, 0]) for arr in Y3]]
    Y3 = Y3.astype(np.float64)
    # model = my_lib.create_mlp([2, 3, 1])
    # print("Before : ")
    # for sample_input in X:
    #     print(my_lib.predict_mlp(model, sample_input))
    #
    # my_lib.train_mlp(model, X, Y)
    #
    # print("After : ")
    #
    # for sample_input in X:
    #     print(my_lib.predict_mlp(model, sample_input))


    #
    # print("For Regression : ")
    #
    # model2 = my_lib.create_mlp([2, 3, 1])
    # print("Before : ")
    # for sample_input in X2:
    #     print(my_lib.predict_mlp(model2, sample_input, False))
    #
    # my_lib.train_mlp(model2, X2, Y2, 0.01, False, 1000)
    #
    # print("After : ")
    #
    # for sample_input in X2:
    #     print(my_lib.predict_mlp(model2, sample_input, False))

    nb_iter = 100000
    model = my_lib.create_mlp([2, 3])
    print("Before : ")
    for sample_input in X3:
        print(my_lib.predict_mlp(model, sample_input))

    my_lib.train_mlp(model, X3, Y3, 0.01, True, nb_iter)

    print("After : ")

    for sample_input in X3:
        print(my_lib.predict_mlp(model, sample_input))


    # print(my_lib.kmeans_centroids(data2, 2, ))
