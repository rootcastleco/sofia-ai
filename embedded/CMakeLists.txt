# SOFIA ENGINE — embedded reference runtime (CMake)
# Copyright 2026 Rootcastle Engineering & Innovation. Apache-2.0.
#
# Scope: host build and test of the portable C99 runtime. This project does NOT
# cross-compile for STM32/ESP32 or any other MCU. Target-specific builds are
# NOT VERIFIED in this repository (no toolchain or hardware in CI).

cmake_minimum_required(VERSION 3.16)
project(sofia_embedded C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_library(sofia_features
    src/sofia_features.c
    src/sofia_features.h
    src/sofia_fixed.h
)

target_include_directories(sofia_features PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(MSVC)
    target_compile_options(sofia_features PRIVATE /W4 /WX)
else()
    target_compile_options(sofia_features PRIVATE
        -Wall -Wextra -Wpedantic -Werror -Wconversion -Wshadow
        -Wstrict-prototypes -fno-common)
endif()

add_executable(test_sofia_features tests/test_sofia_features.c)
target_link_libraries(test_sofia_features PRIVATE sofia_features)

if(NOT MSVC)
    target_link_libraries(test_sofia_features PRIVATE m)
endif()

enable_testing()
add_test(NAME sofia_embedded_runtime COMMAND test_sofia_features)
