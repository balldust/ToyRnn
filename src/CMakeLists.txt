cmake_minimum_required(VERSION 3.23)

project(ToyRnn)
set(CMAKE_CXX_STANDARD 20)

include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake)
enable_testing()
add_subdirectory(model)

add_executable(ToyRnn simple_example.cpp)

target_link_libraries(ToyRnn PRIVATE ModelLib)
