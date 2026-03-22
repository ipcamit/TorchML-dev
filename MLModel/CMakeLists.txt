set(ML_RUNTIME_TARGET_LIB MLModel)

list(APPEND CMAKE_PREFIX_PATH $ENV{TORCH_ROOT})
find_package(Torch REQUIRED)

add_library(${ML_RUNTIME_TARGET_LIB} STATIC
  TorchScriptModel.cpp
  TorchExportModel.cpp
  TorchUtils.cpp
)

set_property(TARGET ${ML_RUNTIME_TARGET_LIB} PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET ${ML_RUNTIME_TARGET_LIB} PROPERTY CXX_STANDARD 17)

target_link_libraries(${ML_RUNTIME_TARGET_LIB} PRIVATE ${TORCH_LIBRARIES})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)