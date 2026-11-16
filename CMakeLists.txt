# CMake generates the actual build files (Makefiles, Ninja, ...) for your platform.
# Typical use:
#   cmake -S . -B build        # configure: read this file, write build files into build/
#   cmake --build build        # compile everything
#   ./build/test_sobol         # run the Sobol tests directly
#   ctest --test-dir build     # or run every registered test through CTest

cmake_minimum_required(VERSION 3.18)   # the plan (Â§13) asks for CMake 3.18+
project(SobolStochasticMultiplier CXX) # project name; CXX = this is a C++ project

# Use C++17 and fail if the compiler can't provide it.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Writes build/compile_commands.json, which lets editors/clangd find the include paths
# (this removes the "'sobol_generator.hpp' file not found" errors in the editor).
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# A library target holding the generator, so every later program and test can link it
# instead of recompiling the same source. More src/*.cpp files get added here in later phases.
add_library(sobol
    src/sobol_generator.cpp
    src/scrambling.cpp
    src/sc_metrics.cpp
    src/seed_selector.cpp
    src/avx2_kernels.cpp
    src/sc_image.cpp)
# PUBLIC: anything that links 'sobol' also gets include/ on its header search path.
target_include_directories(sobol PUBLIC include)

# AVX2 kernel (Task 1): compile avx2_kernels.cpp with -mavx2 -mpopcnt only if the compiler
# accepts them. The CPU is re-checked at run time (avx2Supported()), so the binary stays safe.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" SC_COMPILER_HAS_AVX2)
if(SC_COMPILER_HAS_AVX2 AND CMAKE_SYSTEM_PROCESSOR MATCHES "([xX]86|AMD64|amd64|x64)")
  set_source_files_properties(src/avx2_kernels.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mpopcnt")
  target_compile_definitions(sobol PUBLIC SC_HAVE_AVX2=1)
  message(STATUS "AVX2 kernel: enabled (-mavx2 -mpopcnt on src/avx2_kernels.cpp)")
else()
  message(STATUS "AVX2 kernel: not available for this compiler/target; scalar fallback only")
endif()

# Turns on CTest support so `ctest` can discover the tests registered below.
enable_testing()

# The Phase 1 test program: build it, link the generator, register it as a test.
add_executable(test_sobol tests/test_sobol.cpp)
target_link_libraries(test_sobol PRIVATE sobol)
add_test(NAME test_sobol COMMAND test_sobol)

# Gap 2 tests: direction numbers vs scipy, scramblers keep the net property, metric maths.
add_executable(test_correlation tests/test_correlation.cpp)
target_include_directories(test_correlation PRIVATE tests)
target_link_libraries(test_correlation PRIVATE sobol)
add_test(NAME test_correlation COMMAND test_correlation)

# Gap 2 experiment harness (stages 1-5): writes results/*.csv and prints summary tables.
add_executable(run_experiments src/experiments.cpp)
target_link_libraries(run_experiments PRIVATE sobol)

# Release flags so the generation-latency numbers in the experiments mean something.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Phase 6: stochastic RGB image multiplication demo (reads PPMs from python/prepare_dataset.py).
add_executable(image_demo src/image_demo.cpp)
target_link_libraries(image_demo PRIVATE sobol)
# Build description recorded in the timing CSV (compiler, flags, build type).
target_compile_definitions(image_demo PRIVATE
    SC_BUILD_FLAGS="${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} avx2kernel=${SC_COMPILER_HAS_AVX2}"
    SC_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
# Optional OpenCV baseline (cv::multiply) — only if OpenCV is installed.
find_package(OpenCV QUIET COMPONENTS core)
if(OpenCV_FOUND)
  message(STATUS "OpenCV ${OpenCV_VERSION} found: enabling cv::multiply baseline")
  target_compile_definitions(image_demo PRIVATE SC_HAVE_OPENCV=1)
  target_include_directories(image_demo PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(image_demo PRIVATE ${OpenCV_LIBS})
else()
  message(STATUS "OpenCV not found: baseline skipped (optional)")
endif()

add_executable(test_avx2_kernel tests/test_avx2_kernel.cpp)
target_link_libraries(test_avx2_kernel PRIVATE sobol)
add_test(NAME test_avx2_kernel COMMAND test_avx2_kernel)

add_executable(test_image_path tests/test_image_path.cpp)
target_link_libraries(test_image_path PRIVATE sobol)
add_test(NAME test_image_path COMMAND test_image_path)
