import ctypes
from typing import List, Any, Dict
import numpy as np


class MLLib:
    def __init__(self, dll):
        self.dll = dll

    def create_mlp(self, npl: List[int]) -> ctypes.c_void_p:
        arr_type_npl = ctypes.c_int32 * len(npl)

        self.dll.create_mlp_model.argtypes = [arr_type_npl, ctypes.c_int32]
        self.dll.create_mlp_model.restype = ctypes.c_void_p

        return self.dll.create_mlp_model(arr_type_npl(*npl), len(npl))

    def test_mlp(self, model: ctypes.c_void_p):
        self.dll.test_mlp_model.argtypes = [ctypes.c_void_p]
        self.dll.test_mlp_model.restype = ctypes.c_int32

        return self.dll.test_mlp_model(model)

    def predict_mlp(self, model: ctypes.c_void_p, inputs: np.ndarray, is_classification: bool = True)\
            -> ctypes.POINTER(ctypes.c_double):
        nd_pointer_1 = np.ctypeslib.ndpointer(dtype=np.float64,
                                              ndim=1,
                                              flags="C")

        self.dll.predict_mlp_model.argtypes = [ctypes.c_void_p, nd_pointer_1, ctypes.c_size_t, ctypes.c_int32]
        self.dll.predict_mlp_model.restype = ctypes.POINTER(ctypes.c_double)

        native_result = self.dll.predict_mlp_model(model, inputs, inputs.size, is_classification)
        return np.ctypeslib.as_array(native_result, (1,))

    def train_mlp(self,
                  model: ctypes.c_void_p,
                  all_inputs: np.ndarray,
                  all_outputs: np.ndarray,
                  learning_rate: float = 0.01,
                  is_classification: bool = True,
                  nb_iter: int = 10000):
        nd_pointer_2 = np.ctypeslib.ndpointer(dtype=np.float64,
                                              ndim=2,
                                              flags="C")

        self.dll.train_mlp_model.argtypes = [ctypes.c_void_p,
                                             nd_pointer_2,
                                             ctypes.c_size_t,
                                             ctypes.c_size_t,
                                             nd_pointer_2,
                                             ctypes.c_size_t,
                                             ctypes.c_size_t,
                                             ctypes.c_float,
                                             ctypes.c_int32,
                                             ctypes.c_int32]
        self.dll.train_mlp_model.restype = None
        all_inputs = all_inputs.reshape((all_inputs.shape[0], all_inputs.shape[1]), order="C")
        all_outputs = all_outputs.reshape((all_outputs.shape[0], all_outputs.shape[1]), order="C")
        self.dll.train_mlp_model(model,
                                 all_inputs,
                                 all_inputs.shape[0],
                                 all_inputs.shape[1],
                                 all_outputs,
                                 all_outputs.shape[0],
                                 all_outputs.shape[1],
                                 learning_rate,
                                 is_classification,
                                 nb_iter)

    def test_print_array(self):
        ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=np.float64,
                                              ndim=1,
                                              flags="C")
        self.dll.print_array.argtypes = [ND_POINTER_1, ctypes.c_size_t]
        self.dll.print_array.restype = None
        X = np.ones(5)
        self.dll.print_array(X, X.size)

    def test_print_matrix(self):
        ND_POINTER_2 = np.ctypeslib.ndpointer(dtype=np.float64,
                                              ndim=2,
                                              flags="C")

        self.dll.print_matrix.argtypes = [ND_POINTER_2, ctypes.c_size_t, ctypes.c_size_t]
        self.dll.print_array.restype = None
        M = np.arange(1, 10, 1, dtype=np.float64).reshape((3, 3), order="C")
        print(M)
        print(M.shape)
        self.dll.print_matrix(M, M.shape[0], M.shape[1])
