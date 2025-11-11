cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(market_sim_c LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 17)

# Find required packages
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Set paths
set(LIBTORCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../external/libtorch/libtorch")
set(PUFFERLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../external/pufferlib-3.0.0/pufferlib")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${PUFFERLIB_DIR}
    ${PUFFERLIB_DIR}/ocean
)

# Compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native -ffast-math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -fsanitize=address")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address")
endif()

# Market simulation library
add_library(market_sim SHARED
    src/market_env.c
    src/binding.c
)

target_link_libraries(market_sim
    ${TORCH_LIBRARIES}
    ${Python3_LIBRARIES}
)

# CUDA-accelerated RL kernels (following pufferlib pattern)
if(CUDA_FOUND)
    add_library(market_sim_cuda SHARED
        src/cuda_kernels.cu
    )

    target_link_libraries(market_sim_cuda
        ${TORCH_LIBRARIES}
        ${CUDA_LIBRARIES}
    )

    set_target_properties(market_sim_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# Set output properties
set_target_properties(market_sim PROPERTIES
    PREFIX ""
    OUTPUT_NAME "binding"
    SUFFIX "${Python3_SOABI}.so"
    POSITION_INDEPENDENT_CODE ON
)

# Installation
install(TARGETS market_sim
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/python
)
