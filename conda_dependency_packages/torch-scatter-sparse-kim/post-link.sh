#!/usr/bin/env bash
set -euo pipefail

LIBTORCH_TAG="1.13.0"
TORCH_ARCHIVE="https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-1.13.0%2Bcu117.zip"

echo "Downloading LibTorch ${LIBTORCH_TAG} (CUDA 11.7) …"
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

curl -L --retry 3 -o "$TMPDIR/libtorch.zip" "$TORCH_ARCHIVE"
unzip -q "$TMPDIR/libtorch.zip" -d "$TMPDIR"
rsync -a "$TMPDIR/libtorch/" "${PREFIX}/"

echo "Building pytorch-sparse & pytorch-scatter …"
BUILD_DIR=$(mktemp -d)
trap 'rm -rf "$BUILD_DIR"' RETURN
cd "$BUILD_DIR"

# 1. clone sources (same SHAs as your original recipe)
git clone  --recurse-submodules  https://github.com/rusty1s/pytorch_sparse.git sparse
git clone  --recurse-submodules  https://github.com/rusty1s/pytorch_scatter.git scatter

cd sparse
git checkout e55e8331e
cd ..

cd scatter
git checkout  fa4f44295
cd ..


# patch TorchScatter -----------------------------------------------------------
sed -Ei \
  -e 's/file\((GLOB) HEADERS /file(\1 TORCHSCATTER_HEADERS /' \
  -e '/target_include_directories\(\$\{PROJECT_NAME\} INTERFACE/,+2c\
target_include_directories(${PROJECT_NAME}\
  PUBLIC\
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/csrc>\
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)' \
  -e '/install\(TARGETS \$\{PROJECT_NAME\}/ s/\)$/        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})/' \
  -e 's/FILES \$\{HEADERS\}/FILES ${TORCHSCATTER_HEADERS}/' \
  scatter/CMakeLists.txt

# patch TorchSparse ------------------------------------------------------------
sed -Ei \
  -e 's/file\((GLOB) HEADERS /file(\1 TORCHSPARSE_HEADERS /' \
  -e '/target_include_directories\(\$\{PROJECT_NAME\} INTERFACE/,+2c\
target_include_directories(${PROJECT_NAME}\
  PUBLIC\
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/csrc>\
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)' \
  -e '/install\(TARGETS \$\{PROJECT_NAME\}/ s/\)$/        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})/' \
  -e 's/FILES \$\{HEADERS\}/FILES ${TORCHSPARSE_HEADERS}/' \
  sparse/CMakeLists.txt
################################################################################

export CMAKE_PREFIX_PATH="${PREFIX}"
export CUDA_HOME="${CONDA_PREFIX}"

for PKG in sparse scatter; do
  mkdir -p "$PKG/build" && cd "$PKG/build"
  cmake -G Ninja \
        -DCMAKE_PREFIX_PATH="$PREFIX" \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DCMAKE_BUILD_TYPE=Release \
        ..
  cmake --build . -j"$(nproc)"
  cmake --install .
  cd ../..
done

echo " torch-scatter-sparse-kim installation completed."
