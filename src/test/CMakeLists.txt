### CMakeLists.txt for the tests

cmake_minimum_required(VERSION 2.8)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)
set (CMAKE_CXX_STANDARD 11)

add_executable(
    test_regular
    test_main.cpp
    test_neighbor.cpp
    test_random.cpp
    test_weighting.cpp
    )
add_executable(
    test_neighbor_10
    test_main.cpp
    test_neighbor_10.cpp
    )
add_executable(
    test_neighbor_r3
    test_main.cpp
    test_neighbor_r3.cpp
    )
add_executable(
    test_paramsum
    test_main.cpp
    test_paramsum.cpp
    )
add_executable(
    test_dtlz1
    test_main.cpp
    test_dtlz1.cpp
    )
add_executable(
    test_dtlz2
    test_main.cpp
    test_dtlz2.cpp
    )
add_executable(
    test_dtlz3
    test_main.cpp
    test_dtlz3.cpp
    )
add_executable(
    test_dtlz4
    test_main.cpp
    test_dtlz4.cpp
    )
add_executable(
    test_dtlz5
    test_main.cpp
    test_dtlz5.cpp
    )
add_executable(
    test_dtlz6
    test_main.cpp
    test_dtlz6.cpp
    )
add_executable(
    test_dtlz7
    test_main.cpp
    test_dtlz7.cpp
    )
