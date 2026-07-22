cmake_minimum_required(VERSION 3.21)
project(myelin_iree C CXX)

# Runtime only — no compiler, no LLVM. Disable GPU drivers/tests/samples we don't use.
set(IREE_BUILD_COMPILER OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_SAMPLES  OFF CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_VULKAN OFF CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_METAL  OFF CACHE BOOL "" FORCE)

# IREE_SRC passed via -DIREE_SRC=... ; build its tree under ./iree_core
add_subdirectory(${IREE_SRC} iree_core)

add_library(myelin_iree SHARED shim.c)
target_link_libraries(myelin_iree PRIVATE iree::runtime)
