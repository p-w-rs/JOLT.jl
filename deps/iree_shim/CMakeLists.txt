cmake_minimum_required(VERSION 3.21)
project(jolt_iree C CXX)

# Runtime only — no compiler, no LLVM. Disable drivers/tests/samples we don't use.
# CPU "local-task"/"local-sync" is always built; Metal (the Mac GPU, driver
# "metal") is enabled here so `IREE_METAL` sessions can run — it uses the same
# page-aligned mmap arena as CPU (newBufferWithBytesNoCopy needs a vm/mmap ptr).
set(IREE_BUILD_COMPILER OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_SAMPLES  OFF CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_VULKAN OFF CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_METAL  ON  CACHE BOOL "" FORCE)

# IREE_SRC passed via -DIREE_SRC=... ; build its tree under ./iree_core
add_subdirectory(${IREE_SRC} iree_core)

add_library(jolt_iree SHARED shim.c)
# iree::runtime is the core meta-target; the io_parameters weight path lives in
# separate libs it doesn't transitively pull in (only iree::io::file_handle is).
target_link_libraries(jolt_iree PRIVATE
  iree::runtime
  iree::io::parameter_index
  iree::io::parameter_index_provider
  iree::io::parameter_provider
  iree::io::scope_map
  iree::modules::io::parameters)
