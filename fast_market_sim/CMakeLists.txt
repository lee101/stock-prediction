cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(fast_market_sim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -arch=sm_86")

# Find LibTorch
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../external/libtorch/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/forecaster.cpp
    src/market_env_fast.cpp
)

# Create shared library
add_library(fast_market_sim SHARED ${SOURCES})

target_link_libraries(fast_market_sim
    ${TORCH_LIBRARIES}
)

# Set properties
set_target_properties(fast_market_sim PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)

# Example executables
add_executable(benchmark_forecaster examples/benchmark_forecaster.cpp)
target_link_libraries(benchmark_forecaster fast_market_sim ${TORCH_LIBRARIES})

add_executable(train_fast examples/train_fast.cpp)
target_link_libraries(train_fast fast_market_sim ${TORCH_LIBRARIES})

add_executable(test_integration examples/test_integration.cpp)
target_link_libraries(test_integration fast_market_sim ${TORCH_LIBRARIES})

# Installation
install(TARGETS fast_market_sim
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include/fast_market_sim
)

# Copy compiled models to build directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../compiled_models_torchscript
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

# Print configuration
message(STATUS "==============================================")
message(STATUS "Fast Market Sim Configuration")
message(STATUS "==============================================")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "LibTorch: ${TORCH_LIBRARIES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "==============================================")
