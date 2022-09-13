#!/bin/bash

# Clone Src
git clone --recurse-submodules https://github.com/rusty1s/pytorch_sparse
git clone --recurse-submodules https://github.com/rusty1s/pytorch_scatter

mkdir install
mkdir build_sparse build_scatter

cd build_sparse
cmake ../pytorch_sparse -DCMAKE_INSTALL_PREFIX=""
make DESTDIR=../install install
cd ../

cd build_scatter 
cmake ../pytorch_scatter  -DCMAKE_INSTALL_PREFIX=""
make DESTDIR=../install install
cd ../

INSTALL_DIR=`pwd`/install

echo "Please include these lines in .bashrc file"
echo "export INCLUDE=\"\${INCLUDE}:${INSTALL_DIR}/include\""
echo "export LD_LIBRRAY_PATH=\"\${LD_LIBRARY_PATH}:${INSTALL_DIR}/lib\""
echo "export TorchScatter_DIR=\"${INSTALL_DIR}/share/cmake\""
echo "export TorchSparse_DIR=\"${INSTALL_DIR}/share/cmake\""
