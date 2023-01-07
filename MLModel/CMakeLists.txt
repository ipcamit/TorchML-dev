set(ML_RUNTIME_TARGET_LIB MLModel)

list(APPEND CMAKE_PREFIX_PATH $ENV{TORCH_ROOT})
find_package(Torch REQUIRED)

add_library(${ML_RUNTIME_TARGET_LIB} STATIC MLModel.cpp)

set_property(TARGET ${ML_RUNTIME_TARGET_LIB} PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET ${ML_RUNTIME_TARGET_LIB} PROPERTY CXX_STANDARD 14)

target_link_libraries(${ML_RUNTIME_TARGET_LIB} PRIVATE ${TORCH_LIBRARIES})

# TODO: Set CXX_STANDARD target property here?

# TODO: Copy DLLs if on windows?  See
# https://pytorch.org/cppdocs/installing.html#minimal-example
