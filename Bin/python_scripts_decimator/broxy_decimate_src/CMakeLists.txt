cmake_minimum_required(VERSION 3.18)
project(broxy-minimal LANGUAGES CXX Fortran)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# M_PI not defined by default on Windows
if(WIN32)
    add_compile_definitions(_USE_MATH_DEFINES)
endif()


# core library #

add_library(broxy_core STATIC
    Decimator/Decimator.cpp
    Decimator/ql0001.f
    Decimator/Types.h
    Decimator/SolveQP.h
    Decimator/SolveQP.cpp
    Common/EigenUtils.h
    Decimator.h
    Decimator.cpp
)

target_include_directories(broxy_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/vcglib
    ${CMAKE_CURRENT_SOURCE_DIR}/eigen
    ${CMAKE_CURRENT_SOURCE_DIR}/Common
    ${CMAKE_CURRENT_SOURCE_DIR}/Decimator
)

# executable #

add_executable(broxy-minimal
    Main.cpp
)
target_link_libraries(broxy-minimal PRIVATE broxy_core)

