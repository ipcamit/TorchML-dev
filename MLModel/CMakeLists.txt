set(ML_RUNTIME_TARGET_LIB MLModel)

env_module(load python39)
env_module(load libtorch110)
find_package(Torch REQUIRED)

add_library(${ML_RUNTIME_TARGET_LIB} STATIC MLModel.cpp)

set_property(TARGET ${ML_RUNTIME_TARGET_LIB} PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(${ML_RUNTIME_TARGET_LIB} PRIVATE ${TORCH_LIBRARIES})# PRIVATE /home/amit/Projects/COLABFIT/misc/cutom_op/libdescriptor/cmake-build-debugclang/libdescriptor.so)

# TODO: Set CXX_STANDARD target property here?

# TODO: Copy DLLs if on windows?  See
# https://pytorch.org/cppdocs/installing.html#minimal-example
