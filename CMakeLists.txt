cmake_minimum_required(VERSION 3.5)

project(IALib LANGUAGES C)

add_library(ial STATIC ialib.c ialib.h)

add_executable(tests tests/test.c)

target_link_libraries(ial PUBLIC m)
target_link_libraries(tests PRIVATE ial)
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR})

enable_testing()

add_test(NAME TestVersionWorks COMMAND tests)

add_executable(test_iterative tests/test_iter.c)

target_link_libraries(test_iterative PRIVATE ial)
target_include_directories(test_iterative PRIVATE ${CMAKE_SOURCE_DIR})

add_test(NAME TestIterativeWorks COMMAND test_iterative)