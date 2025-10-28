cmake_minimum_required(VERSION 3.20)
project(cppsimulator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED Torch_DIR)
  message(FATAL_ERROR "Torch_DIR is not set. Point it to the libtorch distribution's share/cmake/Torch directory.")
endif()

find_package(Torch REQUIRED)

add_library(market_sim STATIC
  src/market_sim.cpp
  src/forecast.cpp
)

target_include_directories(market_sim
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(market_sim
  PUBLIC
    ${TORCH_LIBRARIES}
)

target_compile_definitions(market_sim
  PRIVATE
    -D_GLIBCXX_USE_CXX11_ABI=1
)

target_compile_options(market_sim
  PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -fopenmp>
)

add_executable(run_sim apps/run_sim.cpp)
target_link_libraries(run_sim PRIVATE market_sim ${TORCH_LIBRARIES})

if(TARGET torch_cuda)
  message(STATUS "LibTorch CUDA target detected.")
endif()

if(NOT DEFINED ENV{TORCH_CUDA_ARCH_LIST})
  message(WARNING "TORCH_CUDA_ARCH_LIST is not set; consider setting it (e.g. 8.9) for optimal binaries.")
endif()
