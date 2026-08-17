#cmake_minimum_required(VERSION 3.10)
#
## This model driver requires C++17
#
#set(CMAKE_CXX_STANDARD 17)
#cmake_policy(SET CMP0074 NEW)
#cmake_policy(SET CMP0144 NEW)
#STRING(COMPARE EQUAL "${CMAKE_BUILD_TYPE}" "" flg)
#if(flg)
#    set(CMAKE_BUILD_TYPE Release)
#endif()
#message("Build Type: ${CMAKE_BUILD_TYPE}")
#
## TORCH CMAKE FILES-------------------------
## Not needed, using CMP0144 policy now. User must set Torch_ROOT variable
#
## print all files -----------------------------------------
#message("KIM PREFIX DIR: ${KIM_ROOT}")
#message("TORCH PREFIX DIR: ${TORCH_ROOT}")
#
## Append to PREFIX PATH -----------------------------------------
## No scatter sparce now
#set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${KIM_ROOT} ${TORCH_ROOT})
##list(APPEND CMAKE_PREFIX_PATH ${KIM_ROOT} ${TORCH_ROOT} ${TorchScatter_ROOT} ${TorchSparse_ROOT})
#find_package(KIM-API-ITEMS 2.2 REQUIRED CONFIG)
#
## KIM-API setup --------------------------------------------------
#kim_api_items_setup_before_project(ITEM_TYPE "modelDriver")
#project(TorchML__MD_173118614730_001 LANGUAGES CXX)
#kim_api_items_setup_after_project(ITEM_TYPE "modelDriver")
#
## Find Torch (env Torch_ROOT)
#find_package(Torch REQUIRED)
#
#add_kim_api_model_driver_library(
#  NAME                    ${PROJECT_NAME}
#  CREATE_ROUTINE_NAME     "model_driver_create"
#  CREATE_ROUTINE_LANGUAGE "cpp"
#)
#
## Link to libdescriptor (Not needed for just-gnn branch)
## Check if libdescriptor is installed in LIBDESCRIPTOR_ROOT env variable or /usr/local/lib, if yes,
## add USE_LIBDESC flag to the compiler.
#if (DEFINED LIBDESCRIPTOR_ROOT)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_LIBDESC")
#    include_directories("${LIBDESCRIPTOR_ROOT}/include")
#    message("libdescriptor location: ${LIBDESCRIPTOR_ROOT}")
#    set(ENABLE_LIBDESC ON)
#elseif(DEFINED ENV{LIBDESCRIPTOR_ROOT})
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_LIBDESC")
#    set(LIBDESCRIPTOR_ROOT $ENV{LIBDESCRIPTOR_ROOT})
#    include_directories("${LIBDESCRIPTOR_ROOT}/include")
#    set(ENABLE_LIBDESC ON)
#    message("libdescriptor location: ${LIBDESCRIPTOR_ROOT}")
#elseif(EXISTS "/usr/local/lib/libdescriptor.so")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_LIBDESC")
#    set(LIBDESCRIPTOR_ROOT "/usr/local")
#    include_directories("${LIBDESCRIPTOR_ROOT}/include")
#    message("libdescriptor location: ${LIBDESCRIPTOR_ROOT}")
#    set(ENABLE_LIBDESC ON)
#else()
#    message("libdescriptor not found, skipping...")
#    set(ENABLE_LIBDESC OFF)
#endif()
#
## Check if User needs MPI aware version of the model driver
#if (DEFINED ENV{KIM_MODEL_MPI_AWARE})
#    STRING(COMPARE EQUAL $ENV{KIM_MODEL_MPI_AWARE} "yes" flg)
#    if(flg)
#    # Find MPI
#    find_package(MPI)
#        if(MPI_FOUND)
#            message("MPI found, using MPI compilers to compile")
#            include_directories(SYSTEM ${MPI_INCLUDE_PATH})
#            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_MPI")
#        else()
#            message(FATAL_ERROR "MPI not found, you requested MPI aware build, either provide MPI LIB or set KIM_MODEL_MPI_AWARE to no")
#        endif()
#    endif()
#endif()
#
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g") # this is not debug, but just include symbol table
#
#SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
#
## Add ML model directory
#add_subdirectory(MLModel)
#
#target_sources(${PROJECT_NAME} PRIVATE
#        TorchMLModelDriver.cpp
#        TorchMLModelDriverImplementation.cpp
#)
#
## Explicitly set CXX version 17, for some reason above setting does not cut it
#set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
#
## Link against torch geometric dependencies
#target_link_libraries("${PROJECT_NAME}" PRIVATE MLModel PRIVATE ${TORCH_LIBRARIES})
#
#target_include_directories(
#        "${PROJECT_NAME}" PRIVATE
#        "${PROJECT_BINARY_DIR}"
#        "${PROJECT_SOURCE_DIR}/MLModel")
#
## Link against libdescriptor if ENABLE_LIBDESC is ON
#if(ENABLE_LIBDESC)
#    target_link_libraries("${PROJECT_NAME}" PRIVATE "${LIBDESCRIPTOR_ROOT}/lib/libdescriptor.so")
#endif()
#
## Link against MPI if needed
#if (DEFINED ENV{KIM_MODEL_MPI_AWARE})
#    STRING(COMPARE EQUAL $ENV{KIM_MODEL_MPI_AWARE} "yes" flg)
#    if(flg)
#        target_link_libraries(${PROJECT_NAME} PRIVATE ${MPI_CXX_LIBRARIES})
#    endif()
#endif()
cmake_minimum_required(VERSION 3.10)

# --------------------------------------------------
# Policies and C++ standard
# --------------------------------------------------
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0144 NEW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------------------------------------
# Build type
# --------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

list(APPEND CMAKE_PREFIX_PATH ${KIM_ROOT})

# --------------------------------------------------
# Find KIM first
# --------------------------------------------------
find_package(KIM-API-ITEMS 2.2 REQUIRED CONFIG)

# IMPORTANT: keep this order
kim_api_items_setup_before_project(ITEM_TYPE "modelDriver")
project(TorchML_MD_173118614730_001 LANGUAGES CXX)
kim_api_items_setup_after_project(ITEM_TYPE "modelDriver")

# --------------------------------------------------
# Create target
# --------------------------------------------------
add_kim_api_model_driver_library(
        NAME                    ${PROJECT_NAME}
        CREATE_ROUTINE_NAME     "model_driver_create"
        CREATE_ROUTINE_LANGUAGE "cpp"
)

# --------------------------------------------------
# Optional: libdescriptor
# --------------------------------------------------
set(ENABLE_LIBDESC OFF)

if(LIBDESCRIPTOR_ROOT)
    set(ENABLE_LIBDESC ON)
elseif(DEFINED ENV{LIBDESCRIPTOR_ROOT})
    set(LIBDESCRIPTOR_ROOT $ENV{LIBDESCRIPTOR_ROOT})
    set(ENABLE_LIBDESC ON)
elseif(EXISTS "/usr/local/lib/libdescriptor.so")
    set(LIBDESCRIPTOR_ROOT "/usr/local")
    set(ENABLE_LIBDESC ON)
endif()

if(ENABLE_LIBDESC)
    message(STATUS "libdescriptor: ${LIBDESCRIPTOR_ROOT}")
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_LIBDESC)
    target_include_directories(${PROJECT_NAME} PRIVATE "${LIBDESCRIPTOR_ROOT}/include")
    target_link_libraries(${PROJECT_NAME} PRIVATE "${LIBDESCRIPTOR_ROOT}/lib/libdescriptor.so")
else()
    message(STATUS "libdescriptor not found")
endif()

# --------------------------------------------------
# Optional: MPI
# --------------------------------------------------
if(DEFINED ENV{KIM_MODEL_MPI_AWARE} AND "$ENV{KIM_MODEL_MPI_AWARE}" STREQUAL "yes")
    find_package(MPI REQUIRED)

    message(STATUS "MPI enabled")

    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_MPI)
    target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${MPI_INCLUDE_PATH})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${MPI_CXX_LIBRARIES})
endif()

# --------------------------------------------------
# Optional: Disable graph (GNN) preprocessing
# --------------------------------------------------
if(DEFINED ENV{KIM_MODEL_DISABLE_GRAPH})
    message(STATUS "Graph preprocessing disabled via KIM_MODEL_DISABLE_GRAPH")
    target_compile_definitions(${PROJECT_NAME} PRIVATE KIM_MODEL_DISABLE_GRAPH)
endif()

# --------------------------------------------------
# Subdirectory
# --------------------------------------------------
add_subdirectory(MLModel)

# --------------------------------------------------
# Sources
# --------------------------------------------------
target_sources(${PROJECT_NAME} PRIVATE
        TorchMLModelDriver.cpp
        TorchMLModelDriverImplementation.cpp
)

# --------------------------------------------------
# Includes and linking
# --------------------------------------------------
target_include_directories(${PROJECT_NAME} PRIVATE
        ${PROJECT_BINARY_DIR}
        ${PROJECT_SOURCE_DIR}/MLModel
)

target_link_libraries(${PROJECT_NAME} PRIVATE
        MLModel
        ${TORCH_LIBRARIES}
)

# --------------------------------------------------
# Debug symbols better crash reports
# --------------------------------------------------
target_compile_options(${PROJECT_NAME} PRIVATE -g)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
