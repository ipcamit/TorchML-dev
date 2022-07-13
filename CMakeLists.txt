cmake_minimum_required(VERSION 3.10)

find_package(EnvModules REQUIRED)
env_module(load kim_api)
env_module(load libtorch110)

list(APPEND CMAKE_PREFIX_PATH $ENV{KIM_API_CMAKE_PREFIX_DIR})

find_package(KIM-API-ITEMS 2.2 REQUIRED CONFIG)

kim_api_items_setup_before_project(ITEM_TYPE "modelDriver")
project(generic_pytorch LANGUAGES CXX)
kim_api_items_setup_after_project(ITEM_TYPE "modelDriver")

# This model driver requires C++11
set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)

add_kim_api_model_driver_library(
  NAME                    ${PROJECT_NAME}
  CREATE_ROUTINE_NAME     "model_drive_create"
  CREATE_ROUTINE_LANGUAGE "cpp"
)

#add_subdirectory(MLModel)

target_sources(${PROJECT_NAME} PRIVATE
        TorchMLModel.cpp
        TorchMLModelImplementation.cpp
)

#target_link_libraries("${PROJECT_NAME}" PRIVATE MLModel PRIVATE ${TORCH_LIBRARIES})
#
#target_include_directories("${PROJECT_NAME}" PRIVATE
#  "${PROJECT_BINARY_DIR}"
#  "${PROJECT_SOURCE_DIR}/MLModel"
#  )

