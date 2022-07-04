cmake_minimum_required (VERSION 3.22)
project (DLLExec)

set(CMAKE_CXX_STANDARD 14)

set(EIGEN_DIR "C:/Users/enzol/Documents/Library/eigen-3.4.0")

include_directories(${EIGEN_DIR})

add_executable(DLLExec main.cpp Perceptron.cpp Perceptron.h Point.cpp Point.h Cluster.cpp Cluster.h KMeans.cpp KMeans.h)

target_link_libraries (DLLExec)