cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(gpu_tigramite LANGUAGES CXX CUDA VERSION 1.0.0)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable verbose output for debugging
set(CMAKE_VERBOSE_MAKEFILE ON)

# Platform-specific settings
if(WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# Find CUDA Toolkit (12.x preferred, 11.x compatible)
find_package(CUDAToolkit 11.0 REQUIRED)
message(STATUS "Found CUDA ${CUDAToolkit_VERSION}")

# Find Python and NumPy
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
message(STATUS "Found Python ${Python3_VERSION}")
message(STATUS "Python include: ${Python3_INCLUDE_DIRS}")
message(STATUS "NumPy include: ${Python3_NumPy_INCLUDE_DIRS}")

# pybind11 (header-only, automatically fetched)
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# CUDA architecture detection (auto-detect or manual)
# IMPORTANT: Double-precision atomicAdd() requires compute capability >= 6.0
set(MIN_CUDA_ARCH 60)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Auto-detect GPU architecture
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    
    # Filter out architectures below minimum (required for double atomics)
    set(FILTERED_ARCH_LIST "")
    foreach(ARCH ${CUDA_ARCH_LIST})
        if(ARCH GREATER_EQUAL ${MIN_CUDA_ARCH})
            list(APPEND FILTERED_ARCH_LIST ${ARCH})
        else()
            message(WARNING "Ignoring compute_${ARCH} (< compute_${MIN_CUDA_ARCH}, no double atomicAdd support)")
        endif()
    endforeach()
    
    set(CMAKE_CUDA_ARCHITECTURES ${FILTERED_ARCH_LIST})
    if(CMAKE_CUDA_ARCHITECTURES)
        message(STATUS "Auto-detected CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
else()
    message(STATUS "Using specified CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# If detection failed or all GPUs too old, use V100+ defaults
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
    message(WARNING "Auto-detection failed or GPUs too old for double precision")
    message(STATUS "Using default CUDA architectures (V100+): ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "  70 = V100 (minimum for double atomics)")
    message(STATUS "  75 = T4, RTX 20xx")
    message(STATUS "  80 = A100")
    message(STATUS "  86 = RTX 30xx")
    message(STATUS "  89 = L4, RTX 40xx")
    message(STATUS "  90 = H100")
endif()

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")

# Optimization flags
if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
    message(STATUS "Building in DEBUG mode")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DNDEBUG")
    message(STATUS "Building in RELEASE mode")
endif()

# Source files
set(CUDA_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_tigramite/cuda/gpucmiknn_cuda.cu)
set(CPP_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_tigramite/cuda/gpucmiknn.cpp)

# Replicate original Makefile's 3-step CUDA build EXACTLY:
# Step 1: nvcc -dc gpucmiknn_cuda.cu -o gpucmiknn_cuda.o
# Step 2: nvcc -dlink gpucmiknn_cuda.o -o gpucmiknn_cuda_link.o
# Step 3: g++ gpucmiknn.o gpucmiknn_cuda.o gpucmiknn_cuda_link.o -o gpucmiknn.so

# Step 1: Compile CUDA source with separable compilation (-dc)
add_library(cuda_kernels OBJECT ${CUDA_SOURCE})
set_target_properties(cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF  # We'll do -dlink manually
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(cuda_kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_tigramite/cuda
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Step 2: Explicit device linking step (nvcc -dlink)
# This creates the device-linked object that contains __cudaRegisterLinkedBinary
set(CUDA_DEVICE_LINK_OBJ ${CMAKE_CURRENT_BINARY_DIR}/gpucmiknn_cuda_link.o)

add_custom_command(
    OUTPUT ${CUDA_DEVICE_LINK_OBJ}
    COMMAND ${CMAKE_CUDA_COMPILER}
        -dlink
        $<TARGET_OBJECTS:cuda_kernels>
        -o ${CUDA_DEVICE_LINK_OBJ}
        --generate-code=arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=[compute_${CMAKE_CUDA_ARCHITECTURES},sm_${CMAKE_CUDA_ARCHITECTURES}]
        -Xcompiler -fPIC
    DEPENDS cuda_kernels
    COMMENT "Device linking CUDA objects with nvcc -dlink"
)

add_custom_target(cuda_device_link DEPENDS ${CUDA_DEVICE_LINK_OBJ})

# Step 3: Create Python module linking everything together
pybind11_add_module(gpucmiknn MODULE
    ${CPP_SOURCE}
    $<TARGET_OBJECTS:cuda_kernels>
    ${CUDA_DEVICE_LINK_OBJ}
)

add_dependencies(gpucmiknn cuda_device_link)

# Target properties
set_target_properties(gpucmiknn PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Include directories
target_include_directories(gpucmiknn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_tigramite/cuda
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Find CUDA device runtime library
find_library(CUDA_DEVICE_RUNTIME_LIBRARY cudadevrt
    HINTS ${CUDAToolkit_LIBRARY_DIR}
    PATHS ${CUDAToolkit_LIBRARY_DIR}
    NO_DEFAULT_PATH
)

if(NOT CUDA_DEVICE_RUNTIME_LIBRARY)
    message(FATAL_ERROR "Could not find libcudadevrt in ${CUDAToolkit_LIBRARY_DIR}")
endif()

message(STATUS "Found cudadevrt: ${CUDA_DEVICE_RUNTIME_LIBRARY}")

# Link libraries (including device runtime and device-linked object)
target_link_libraries(gpucmiknn PRIVATE
    pybind11::module
    pybind11::lto
    Python3::Python
    Python3::NumPy
    CUDA::cudart
    CUDA::cuda_driver
    ${CUDA_DEVICE_RUNTIME_LIBRARY}
)

# Platform-specific linking
if(UNIX AND NOT APPLE)
    target_link_libraries(gpucmiknn PRIVATE stdc++)
endif()

# NO INSTALL TARGET - setuptools handles installation
# The module is built to CMAKE_LIBRARY_OUTPUT_DIRECTORY which setuptools copies
# to the correct site-packages location. An install() directive interferes with this.

# Print build summary
message(STATUS "")
message(STATUS "========== GPU-Tigramite Build Configuration ==========")
message(STATUS "  CMake version:        ${CMAKE_VERSION}")
message(STATUS "  System:               ${CMAKE_SYSTEM_NAME}")
message(STATUS "  C++ compiler:         ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  CUDA compiler:        ${CMAKE_CUDA_COMPILER}")
message(STATUS "  CUDA version:         ${CUDAToolkit_VERSION}")
message(STATUS "  CUDA architectures:   ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Python version:       ${Python3_VERSION}")
message(STATUS "  Build type:           ${CMAKE_BUILD_TYPE}")
message(STATUS "  Install prefix:       ${CMAKE_INSTALL_PREFIX}")
message(STATUS "========================================================")
message(STATUS "")