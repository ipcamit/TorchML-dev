cmake_minimum_required(VERSION 3.10)

# Need these packages on my laptop
# TODO find more agnostic way for it. Discuss with Ryan?
find_package(EnvModules REQUIRED)
env_module(load kim_api)
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

add_kim_api_model_driver_library(
  NAME                    ${PROJECT_NAME}
  CREATE_ROUTINE_NAME     "model_driver_create"
  CREATE_ROUTINE_LANGUAGE "cpp"
)

include_directories(../colabfit-descriptor-library)

# TODO re-enable subdirectory for ML model, as otherwise compilation is lot slower
#add_subdirectory(MLModel)

target_sources(${PROJECT_NAME} PRIVATE
        TorchMLModelDriver.cpp
        MLModel.cpp
#        TorchMLModelImplementation.cpp
)

# Suppress annoying Variable not used warnings, parameter still warns :(
add_compile_options(-Wno-unused-variable -Wno-unused-parameter)

target_link_libraries("${PROJECT_NAME}" PRIVATE ${TORCH_LIBRARIES})
