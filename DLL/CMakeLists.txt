cmake_minimum_required (VERSION 3.22)
project (DLL)

set(CMAKE_CXX_STANDARD 14)

set(EIGEN_DIR "C:/Users/enzol/Documents/Library/eigen-3.4.0")

include_directories(${EIGEN_DIR})

add_library(DLL SHARED library.cpp library.h Perceptron.cpp Perceptron.h)
target_link_libraries (DLL)