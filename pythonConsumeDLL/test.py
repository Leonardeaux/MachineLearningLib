import ctypes

import numpy as np


def test(my_dll):
    my_dll.return_42.argtypes = []
    my_dll.return_42.restype = ctypes.c_int32

    my_dll.my_add.argtypes = [ctypes.c_int32, ctypes.c_int32]
    my_dll.my_add.restype = ctypes.c_int32

    npl = [2, 3, 1]

    arr_type_npl = ctypes.c_int32 * len(npl)

    my_dll.create_mlp_model.argtypes = [arr_type_npl, ctypes.c_int32]
    my_dll.create_mlp_model.restype = ctypes.c_void_p

    inputs = [2.0, 3.0, 1.0]
    arr_type_inputs = ctypes.c_float * len(inputs)

    my_dll.predict_mlp_model.argtypes = [ctypes.c_void_p, arr_type_inputs, ctypes.c_int32, ctypes.c_int32]
    # my_dll.predict_mlp_model.restype = ctypes.POINTER(ctypes.c_float)
    my_dll.predict_mlp_model.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_float,
                                                              shape=(2,))

    my_dll.destroy_float_array.argtypes = [ctypes.POINTER(ctypes.c_float)]
    my_dll.destroy_float_array.restype = None

    my_dll.destroy_mlp_model.argtypes = [ctypes.c_void_p]
    my_dll.destroy_mlp_model.restype = None

    print(f"Return 42 : {my_dll.return_42()}")
    print(f"My Add : {my_dll.my_add(42, 51)}")

    model = my_dll.create_mlp_model(arr_type_npl(*npl), len(npl))

    native_result = my_dll.predict_mlp_model(model, arr_type_inputs(*inputs), len(npl), 0)
    result = np.ctypeslib.as_array(native_result)

    print(f"My result : {result}, {len(result)}")
    # my_dll.destroy_float_array(native_result)
    #
    # my_dll.destroy_mlp_model(model)


if __name__ == '__main__':
    my_cpp_dll = ctypes.CDLL(
        "C:/Users/enzol/CLionProjects/DLLFromCourse/cmake-build-debug/DLLFromCourse.dll")

    print("In Cpp :")
    test(my_cpp_dll)
