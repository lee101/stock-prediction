cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(market_simulator LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set LibTorch path (must be before find_package)
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/../external/libtorch/libtorch")

# Use system CUDA instead of LibTorch bundled version
set(ENV{CUDA_HOME} "/usr/local/cuda-12")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12")
set(CUDAToolkit_ROOT "/usr/local/cuda-12")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12/bin/nvcc")

# Download CPU version would be simpler, but we'll work with CUDA version
# Tell Caffe2 where to find CUDA libs
set(CAFFE2_USE_CUDNN OFF)
set(USE_CUDA ON)

# Set CUDA architecture BEFORE finding Torch
set(TORCH_CUDA_ARCH_LIST "8.9")  # RTX 5090 = Ada (8.9)

# Find PyTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add LibTorch's lib directory to linker path
link_directories("${CMAKE_PREFIX_PATH}/lib")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${TORCH_INCLUDE_DIRS})

# PufferLib path for V-trace CUDA kernels
set(PUFFERLIB_PATH "${CMAKE_SOURCE_DIR}/../external/pufferlib-3.0.0/pufferlib")
include_directories(${PUFFERLIB_PATH}/extensions)

# Link PufferLib CUDA extension
link_directories(${PUFFERLIB_PATH})

# C++ Source files
set(CPP_SOURCES
    src/csv_loader.cpp
    src/market_state.cpp
    src/portfolio.cpp
    src/pnl_logger.cpp
    src/market_env.cpp
    src/execution_policy.cpp
)

# CUDA Source files
set(CUDA_SOURCES
    src/cuda_kernels.cu
)

# Combined sources
set(SOURCES ${CPP_SOURCES} ${CUDA_SOURCES})

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(TORCH_CUDA_ARCH_LIST "8.9")  # RTX 5090 = Ada (8.9)
set(CMAKE_CUDA_ARCHITECTURES "89")  # RTX 5090 = Ada (8.9)

# Create library
add_library(market_sim SHARED ${SOURCES})

# Set CUDA properties for the library
set_target_properties(market_sim PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link libraries (LibTorch only - don't link pufferlib directly, call it from Python)
target_link_libraries(market_sim
    "${TORCH_LIBRARIES}"
)

# Set RPATH for finding libtorch and pufferlib at runtime
set_target_properties(market_sim PROPERTIES
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib:${PUFFERLIB_PATH}"
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib:${PUFFERLIB_PATH}"
)

# Training executable
add_executable(train_market src/main_train.cpp)
target_link_libraries(train_market market_sim "${TORCH_LIBRARIES}")

set_target_properties(train_market PROPERTIES
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

# Test executable
add_executable(test_market src/main_test.cpp)
target_link_libraries(test_market market_sim "${TORCH_LIBRARIES}")

set_target_properties(test_market PROPERTIES
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib:${PUFFERLIB_PATH}"
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib:${PUFFERLIB_PATH}"
)

# Fast training executable (with CUDA kernels + learned execution)
add_executable(train_market_fast src/main_train_fast.cpp)
target_link_libraries(train_market_fast market_sim "${TORCH_LIBRARIES}")

set_target_properties(train_market_fast PROPERTIES
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib:${PUFFERLIB_PATH}"
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib:${PUFFERLIB_PATH}"
)

# Enable warnings
if(MSVC)
    target_compile_options(market_sim PRIVATE /W4)
else()
    target_compile_options(market_sim PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Installation
install(TARGETS market_sim train_market test_market
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
