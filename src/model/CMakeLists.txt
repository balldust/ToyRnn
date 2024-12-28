add_library(ModelLib InputLayer.cxx)

find_package(Eigen3 REQUIRED)

target_link_libraries(ModelLib Eigen3::Eigen)

include_directories(${Eigen3_INCLUDE_DIR})
link_libraries(${Eigen3_LIBRARIES})
target_include_directories(ModelLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

enable_testing()
add_subdirectory(test)
