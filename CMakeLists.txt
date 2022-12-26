cmake_minimum_required(VERSION 3.10)

# This model driver requires C++11
# But ++14 works fine too, so using that for now
# TODO check strict c++11 compliance

project(TorchMLModelDriver LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
cmake_policy(SET CMP0074 NEW)
STRING(COMPARE EQUAL "${CMAKE_BUILD_TYPE}" "" flg)
if(flg)
    set(CMAKE_BUILD_TYPE Release)
endif()
message("Build Type: ${CMAKE_BUILD_TYPE}")

# KIM-API TORCH TorchScatter TorchSparse CMAKE FILES-------------------------
if (NOT DEFINED KIM_ROOT)
    if (DEFINED ENV{KIM_API_CMAKE_PREFIX_DIR})
        set(KIM_ROOT $ENV{KIM_API_CMAKE_PREFIX_DIR})
    else()
        set(KIM_ROOT "/usr/local")
    endif()
endif()
if (NOT DEFINED TORCH_ROOT)
    if (DEFINED ENV{TORCH_ROOT})
        set(TORCH_ROOT $ENV{TORCH_ROOT})
    else()
        set(TORCH_ROOT "/opt/libtorch")
    endif()
endif()
if (NOT DEFINED TorchScatter_ROOT)
    if (DEFINED ENV{TorchScatter_ROOT})
        set(TorchScatter_ROOT $ENV{TorchScatter_ROOT})
    else()
        set(TorchScatter_ROOT "/usr/local")
    endif()
endif()
if (NOT DEFINED TorchSparse_ROOT)
    if (DEFINED ENV{TorchSparse_ROOT})
        set(TorchSparse_ROOT $ENV{TorchSparse_ROOT})
    else()
        set(TorchSparse_ROOT "/usr/local")
    endif()
endif()

# print all files -----------------------------------------
message("KIM PREFIX DIR: ${KIM_ROOT}")
message("TORCH PREFIX DIR: ${TORCH_ROOT}")
message("TorchScatter PREFIX DIR: ${TorchScatter_ROOT}")
message("TorchSparse PREFIX DIR: ${TorchSparse_ROOT}")

# Append to PREFIX PATH -----------------------------------------
list(APPEND CMAKE_PREFIX_PATH ${KIM_ROOT} ${TORCH_ROOT} ${TorchScatter_ROOT} ${TorchSparse_ROOT})
find_package(KIM-API-ITEMS 2.2 REQUIRED CONFIG)

# KIM-API setup --------------------------------------------------
kim_api_items_setup_before_project(ITEM_TYPE "modelDriver")
kim_api_items_setup_after_project(ITEM_TYPE "modelDriver")

# Find Torch (env TORCH_ROOT) and Torch Geometric dependencies
find_package(Torch REQUIRED)
find_package(TorchScatter CONFIG REQUIRED)
find_package(TorchSparse CONFIG REQUIRED)

include_directories(torch_geometric_dependencies/include)

add_kim_api_model_driver_library(
  NAME                    ${PROJECT_NAME}
  CREATE_ROUTINE_NAME     "model_driver_create"
  CREATE_ROUTINE_LANGUAGE "cpp"
)

# Link to libdescriptor (Not needed for just-gnn branch)
# Check if libdescriptor is installed in LIBDESCRIPTOR_ROOT env variable or /usr/local/lib, if yes,
# add USE_LIBDESC flag to the compiler.
if (DEFINED LIBDESCRIPTOR_ROOT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_LIBDESC")
    include_directories("${LIBDESCRIPTOR_ROOT}/include")
    message("libdescriptor location: ${LIBDESCRIPTOR_ROOT}")
    set(ENABLE_LIBDESC ON)
elseif(DEFINED ENV{LIBDESCRIPTOR_ROOT})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_LIBDESC")
    set(LIBDESCRIPTOR_ROOT $ENV{LIBDESCRIPTOR_ROOT})
    include_directories("${LIBDESCRIPTOR_ROOT}/include")
    set(ENABLE_LIBDESC ON)
    message("libdescriptor location: ${LIBDESCRIPTOR_ROOT}")
elseif(EXISTS "/usr/local/lib/libdescriptor.so")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_LIBDESC")
    set(LIBDESCRIPTOR_ROOT "/usr/local")
    include_directories("${LIBDESCRIPTOR_ROOT}/include")
    message("libdescriptor location: ${LIBDESCRIPTOR_ROOT}")
    set(ENABLE_LIBDESC ON)
else()
    message("libdescriptor not found, skipping...")
    set(ENABLE_LIBDESC OFF)
endif()

SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# Add ML model directory
add_subdirectory(MLModel)

target_sources(${PROJECT_NAME} PRIVATE
        TorchMLModelDriver.cpp
        TorchMLModelDriverImplementation.cpp
)

# Explicitly set CXX version 14, for some reason above setting does not cut it
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)

# Link against torch geometric dependencies
target_link_libraries("${PROJECT_NAME}" PRIVATE MLModel PRIVATE ${TORCH_LIBRARIES})
target_link_libraries("${PROJECT_NAME}" PRIVATE TorchScatter::TorchScatter)
target_link_libraries("${PROJECT_NAME}" PRIVATE TorchSparse::TorchSparse)

target_include_directories(
        "${PROJECT_NAME}" PRIVATE
        "${PROJECT_BINARY_DIR}"
        "${PROJECT_SOURCE_DIR}/MLModel")

# Suppress annoying Variable not used warnings, parameter still warns :(
add_compile_options(-Wno-unused-variable -Wno-unused-parameter)
# Link against libdescriptor if ENABLE_LIBDESC is ON
if(ENABLE_LIBDESC)
    target_link_libraries("${PROJECT_NAME}" PRIVATE "${LIBDESCRIPTOR_ROOT}/lib/libdescriptor.so")
#    find_library(DESC_LIB descriptor HINTS "${LIBDESCRIPTOR_ROOT}/lib")
#    target_link_libraries("${PROJECT_NAME}" PRIVATE ${TORCH_LIBRARIES} ${DESC_LIB})
endif()

# Documentation and stuff ----------------------------------------------------------------
find_package(Doxygen OPTIONAL_COMPONENTS dot)
if (DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/docs/Doxyfile.out)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message("Doxygen build started")

    # Note: do not put "ALL" - this builds docs together with application EVERY TIME!
    add_custom_target(docs COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
else (DOXYGEN_FOUND)
  message("Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)

