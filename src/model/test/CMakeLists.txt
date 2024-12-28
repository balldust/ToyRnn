find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(testModel TestInputLayer.cxx)
target_link_libraries(testModel PRIVATE GTest::GTest GTest::gtest_main ModelLib)

include(GoogleTest)
gtest_discover_tests(testModel WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
