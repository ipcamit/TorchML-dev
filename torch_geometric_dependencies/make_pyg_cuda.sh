#!/bin/bash

if [[ -z "${TORCH_ROOT}" ]]; then
  echo "---------------------------------------------------------------------"
  echo -e "TORCH_ROOT env variable not found;\n Please first provide with libtorch install location"
  echo "---------------------------------------------------------------------"
  exit 2
fi

if [[ -z "${CUDNN_LIBRARY_PATH}" ]]; then
  echo "---------------------------------------------------------------------"
  echo -e "CUDNN location not found."
  echo -e "If your CUDA installation does not include cuDNN, please download and untar it, \nand provide its location in CUDNN_INCLUDE_PATH and CUDNN_LIBRARY_PATH env variable"
  echo "---------------------------------------------------------------------"
  exit 2
fi



# Clone Src
git clone --recurse-submodules https://github.com/rusty1s/pytorch_sparse
git clone --recurse-submodules https://github.com/rusty1s/pytorch_scatter

mkdir install
mkdir build_sparse build_scatter

cd build_sparse
cmake ../pytorch_sparse -DCMAKE_INSTALL_PREFIX="" -DCMAKE_PREFIX_PATH="${TORCH_ROOT}" -DCUDNN_INCLUDE_PATH="${CUDNN_INCLUDE_PATH}" -DCUDNN_LIBRARY_PATH="${CUDNN_LIBRARY_PATH}"
make DESTDIR=../install install
cd ../

cd build_scatter 
cmake ../pytorch_scatter  -DCMAKE_INSTALL_PREFIX="" -DCMAKE_PREFIX_PATH="${TORCH_ROOT}" -DCUDNN_INCLUDE_PATH="${CUDNN_INCLUDE_PATH}" -DCUDNN_LIBRARY_PATH="${CUDNN_LIBRARY_PATH}"

make DESTDIR=../install install
cd ../

INSTALL_DIR=`pwd`/install

echo "Please include these lines in .bashrc file"
echo "export INCLUDE=\"\${INCLUDE}:${INSTALL_DIR}/include\""
echo "export LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH}:${INSTALL_DIR}/lib\""
echo "export TorchScatter_DIR=\"${INSTALL_DIR}/share/cmake\""
echo "export TorchSparse_DIR=\"${INSTALL_DIR}/share/cmake\""
echo "export TorchScatter_ROOT=\"${INSTALL_DIR}\""
echo "export TorchSparse_ROOT=\"${INSTALL_DIR}\""
