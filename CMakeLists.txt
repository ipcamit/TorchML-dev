cmake_minimum_required(VERSION 3.10)

# Need these packages on my laptop
# TODO find more agnostic way for it. Discuss with Ryan?
find_package(EnvModules REQUIRED)

if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    message("LOADING DEBUG CONFIG")
    env_module(load kim_api_debug)
    # Uncomment for enabling libasan
    # Also export LD_PRELOAD as `export LD_PRELOAD = $LD_PRELOAD":/lib/x86_64-linux-gnu/libasan.so.5"`
    # env_module(load kim_api_asan)
    # add_compile_options(-fsanitize=address)
    # add_link_options(-fsanitize=address)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RELEASE")
    message("LOADING RELEASE CONFIG")
    env_module(load kim_api)
endif()
env_module(load python39)
env_module(load libtorch110)

list(APPEND CMAKE_PREFIX_PATH $ENV{KIM_API_CMAKE_PREFIX_DIR})

find_package(KIM-API-ITEMS 2.2 REQUIRED CONFIG)

kim_api_items_setup_before_project(ITEM_TYPE "modelDriver")
project(TorchMLModelDriver LANGUAGES CXX)
kim_api_items_setup_after_project(ITEM_TYPE "modelDriver")

# This model driver requires C++11
# But ++14 works fine too, so using that for now
# TODO check strict c++11 compliance
set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)

# Needs these dependencies for TorchGeometric based models
# TODO find more neat way to package them
# Set environment variables TorchScatter_DIR and TorchSparse_DIR
# to install/share/cmake
find_package(TorchScatter CONFIG REQUIRED)
find_package(TorchSparse CONFIG REQUIRED)

include_directories(torch_geometric_dependencies/include)

add_kim_api_model_driver_library(
  NAME                    ${PROJECT_NAME}
  CREATE_ROUTINE_NAME     "model_driver_create"
  CREATE_ROUTINE_LANGUAGE "cpp"
)

include_directories(/home/amit/Projects/COLABFIT/colabfit-kim-model/colabfit-descriptor-library)
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

add_subdirectory(MLModel)


target_sources(${PROJECT_NAME} PRIVATE
        TorchMLModelDriver.cpp
        TorchMLModelDriverImplementation.cpp
)

target_link_libraries("${PROJECT_NAME}" PRIVATE MLModel PRIVATE ${TORCH_LIBRARIES})
target_link_libraries("${PROJECT_NAME}" PRIVATE TorchScatter::TorchScatter)
target_link_libraries("${PROJECT_NAME}" PRIVATE TorchSparse::TorchSparse)

target_include_directories(
        "${PROJECT_NAME}" PRIVATE
        "${PROJECT_BINARY_DIR}"
        "${PROJECT_SOURCE_DIR}/MLModel")

# Suppress annoying Variable not used warnings, parameter still warns :(
add_compile_options(-Wno-unused-variable -Wno-unused-parameter)
find_library(DESC_LIB descriptor HINTS /home/amit/Projects/COLABFIT/colabfit-kim-model/colabfit-descriptor-library/cmake-build-debug-clang12)
target_link_libraries("${PROJECT_NAME}" PRIVATE ${TORCH_LIBRARIES} ${DESC_LIB})
