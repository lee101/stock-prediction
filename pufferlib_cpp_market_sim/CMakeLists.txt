cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(market_simulator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set LibTorch path (must be before find_package)
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/../external/libtorch/libtorch")

# Set CUDA paths for LibTorch's bundled CUDA
set(ENV{CUDA_HOME} "${CMAKE_PREFIX_PATH}")
set(CUDA_TOOLKIT_ROOT_DIR "${CMAKE_PREFIX_PATH}")
set(CUDAToolkit_ROOT "${CMAKE_PREFIX_PATH}")
set(CMAKE_CUDA_COMPILER_FORCED ON)

# Download CPU version would be simpler, but we'll work with CUDA version
# Tell Caffe2 where to find CUDA libs
set(CAFFE2_USE_CUDNN OFF)
set(USE_CUDA ON)

# Find PyTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add LibTorch's lib directory to linker path
link_directories("${CMAKE_PREFIX_PATH}/lib")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${TORCH_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/csv_loader.cpp
    src/market_state.cpp
    src/portfolio.cpp
    src/pnl_logger.cpp
    src/market_env.cpp
)

# Create library
add_library(market_sim SHARED ${SOURCES})
target_link_libraries(market_sim "${TORCH_LIBRARIES}")

# Set RPATH for finding libtorch at runtime
set_target_properties(market_sim PROPERTIES
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
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
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
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
