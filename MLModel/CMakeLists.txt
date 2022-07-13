# TODO: Do we need to impose a minimum cmake version here?

set(ML_RUNTIME_TARGET_LIB MLModel)

#include_directories(/home/amit/Projects/COLABFIT/misc/cutom_op/libdescriptor)


#link_directories(/home/amit/Projects/COLABFIT/misc/cutom_op/libdescriptor/cmake-build-debugclang)
#find_library(descriptor_static /home/amit/Projects/COLABFIT/misc/cutom_op/libdescriptor/cmake-build-debugclang)
#add_library(descriptor_static STATIC IMPORTED)
#set_target_properties(descriptor_static PROPERTIES IMPORTED_LOCATION /home/amit/Projects/COLABFIT/misc/cutom_op/libdescriptor/cmake-build-debugclang/libdescriptor_static.a)


add_library(${ML_RUNTIME_TARGET_LIB} STATIC ml_model.cpp)

set_property(TARGET ${ML_RUNTIME_TARGET_LIB} PROPERTY POSITION_INDEPENDENT_CODE ON)

# FIXME: This is a temporary hack so I don't have to formally install torch
# into system directories.  Delete this later.
#list(APPEND CMAKE_PREFIX_PATH "/opt/libtorch/")
#list(APPEND CMAKE_PREFIX_PATH "/root/cuda/")

find_package(Torch REQUIRED)
target_link_libraries(${ML_RUNTIME_TARGET_LIB} PRIVATE ${TORCH_LIBRARIES})
#target_link_libraries(${ML_RUNTIME_TARGET_LIB} PRIVATE ${TORCH_LIBRARIES} PRIVATE /home/amit/Projects/COLABFIT/misc/cutom_op/libdescriptor/cmake-build-debugclang/libdescriptor.so)

# TODO: Set CXX_STANDARD target property here?

# TODO: Copy DLLs if on windows?  See
# https://pytorch.org/cppdocs/installing.html#minimal-example
