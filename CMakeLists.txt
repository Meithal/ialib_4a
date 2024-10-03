cmake_minimum_required(VERSION 3.5)

project(IALib LANGUAGES C CXX)

add_library(ial STATIC ialib.c ialib.h)

add_executable(tests test.c)

target_link_libraries(tests PRIVATE ial)

enable_testing()

add_test(NAME TestVersionWorks COMMAND tests)