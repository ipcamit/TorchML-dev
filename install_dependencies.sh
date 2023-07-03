#!/bin/bash
# This script is meant to simplify installing the dependencies of KIM Torch Model Driver
# It will detect if cuda is available and install the dependencies accordingly
# all installations will be done in the current directory


echo "---------------------------------------------------------------------"
echo "This script will install the dependencies for KIM Torch Model Driver"
echo "---------------------------------------------------------------------"

echo "Any installations would be done in the current directory"
echo "System dependencies: cmake <= 1.18, git, gcc, g++, wget, unzip, make"
echo "Following env variables/executables will be searched for:"
echo "1. CUDNN_ROOT (if cuda is available)"
echo "2. TORCH_ROOT, for libtorch"
echo "3. kim-api-collections-management, for KIM-API"
echo "4. TorchScatter_ROOT, for pytorch_scatter"
echo "5. TorchSparse_ROOT, for pytorch_sparse"
echo "Do you want to continue? (y/n Type y for yes, n for no)"

read -r response
if [[ "$response" =~ ^([yY][eE][sS])+$ ]]
then
    echo "Continuing with installation, will not ask for confirmation again"
else
    echo "Exiting"
    exit
fi

current_dir=$(pwd)

# Check for cuda
if ! command -v nvcc &> /dev/null
then
    echo "Info: cuda not found, installing cpu version of dependencies"
    is_cuda_available=0
    exit
else
    echo "Info: cuda found, installing gpu version of dependencies"
    # is cudnn available, check for CUDNN_ROOT
    is_cuda_available=1
    if [[ -z "${CUDNN_ROOT}" ]]; then
        echo "---------------------------------------------------------------------"
        echo -e "CUDNN_ROOT env variable not found;\n THIS STEP CANNOT BE AUTOMATED \n Please first provide with cudnn install location"
        echo -e "CUDNN can be downloaded from: https://developer.nvidia.com/rdp/cudnn-archive \n after registering"
        echo "---------------------------------------------------------------------"
        exit 2
    else
        echo "Info: Located CUDNN at ${CUDNN_ROOT}"
    fi
fi

# env file
touch env.sh
echo "# KIM Torch Model Driver env file" > env.sh

# Check if cmake version is less than 3.19
# check if cmake is installed
if ! command -v cmake &> /dev/null
then
    echo "Error: cmake could not be found, please install cmake version 3.18 or lower"
    exit
fi
cmake_version=$(cmake --version | head -n 1 | cut -d " " -f 3)
cmake_version_major=$(echo "$cmake_version" | cut -d "." -f 1)
cmake_version_minor=$(echo "$cmake_version" | cut -d "." -f 2)

if [ "$cmake_version_major" -ne 3 ] || [ $cmake_version_minor -gt 19 ]
then
    echo "Error: cmake version is not 3.18 or lower"
    exit
fi

if ! command -v kim-api-collections-management &> /dev/null
then
    echo "Installing KIM-API"
    # Install KIM-API
    git clone https://github.com/openkim/kim-api
    cd kim-api
    mkdir build && cd build && cmake .. && make
    make DESTDIR="${current_dir}"/kim-api/install install
    # Add KIM-API to path
    echo "# KIM-API " >> env.sh
    echo "export PATH=\"\${PATH}:${current_dir}/kim-api/install/usr/local/bin\"" >> env.sh
    echo "export LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH}:${current_dir}/kim-api/install/usr/local/lib\"" >> env.sh
    echo "export INCLUDE=\"\${INCLUDE}:${current_dir}/kim-api/install/usr/local/include\"" >> env.sh
else
    echo "---------------------------------------------------------------------"
    echo -e "kim-api-collections-management found"
    echo "---------------------------------------------------------------------"
fi

# install torch 1.13, if TORCH_ROOT is not set
if [[ -z "${TORCH_ROOT}" ]]; then
    echo "---------------------------------------------------------------------"
    echo -e "TORCH_ROOT env variable not found"
    echo "---------------------------------------------------------------------"
    echo "Installing libtorch"
    # Install libtorch
    if [[ $is_cuda_available -eq 1 ]]; then
        wget https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-1.13.0%2Bcu117.zip
        tar -xvf libtorch-cxx11-abi-shared-with-deps-1.13.0+cu117.zip
        rm libtorch-cxx11-abi-shared-with-deps-1.9.0+cu111.zip
    else
        wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.0%2Bcpu.zip
        unzip libtorch-cxx11-abi-shared-with-deps-1.9.0+cpu.zip
        rm libtorch-cxx11-abi-shared-with-deps-1.9.0+cpu.zip
    fi
    # Add libtorch to path
    echo "# libtorch " >> env.sh
    echo "export TORCH_ROOT=\"${current_dir}/libtorch\"" >> env.sh
    echo "export LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH}:${current_dir}/libtorch/lib\"" >> env.sh
    echo "export INCLUDE=\"\${INCLUDE}:${current_dir}/libtorch/include\"" >> env.sh
else
    echo "---------------------------------------------------------------------"
    echo -e "TORCH_ROOT env variable found"
    echo "---------------------------------------------------------------------"
fi

# installing TorchScatter and TorchSparse
# variable is named TorchScatter_ROOT, as otherwise cmake gives issues
if [[ -z "${TorchScatter_ROOT}" ]]; then
    echo "---------------------------------------------------------------------"
    echo -e "TorchScatter_ROOT env variable not found"
    echo "---------------------------------------------------------------------"
    echo "Installing TorchScatter"
    # Install TorchScatter
    git clone --recurse-submodules https://github.com/rusty1s/pytorch_scatter
    mkdir install build_scatter
    cd build_scatter
    if [[ $is_cuda_available -eq 1 ]]; then
        cmake -DCMAKE_PREFIX_PATH="${TORCH_ROOT}" -DCMAKE_INSTALL_PREFIX="" -DCUDNN_INCLUDE_PATH="${CUDNN_ROOT}/include" -DCUDNN_LIBRARY_PATH="${CUDNN_ROOT}/lib" -DCMAKE_BUILD_TYPE=Release ../pytorch_scatter
    else
        cmake -DCMAKE_PREFIX_PATH="${TORCH_ROOT}" -DCMAKE_INSTALL_PREFIX="" -DCMAKE_BUILD_TYPE=Release ../pytorch_scatter
    fi
    make install DESTDIR="${current_dir}/pytorch_scatter/install"
    # Add TorchScatter to path
    echo "# TorchScatter " >> env.sh
    echo "export TorchScatter_ROOT=\"${current_dir}/pytorch_scatter/install/usr/local\"" >> env.sh
    echo "export LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH}:${current_dir}/pytorch_scatter/install/usr/local/lib\"" >> env.sh
    echo "export INCLUDE=\"\${INCLUDE}:${current_dir}/pytorch_scatter/install/usr/local/include\"" >> env.sh
    echo "export TorchScatter_DIR=\"${current_dir}/pytorch_scatter/install/usr/local/share/cmake\"" >> env.sh
else
    echo "Info: Located TorchScatter at ${TorchScatter_ROOT}"
fi

if [[ -z "${TorchSparse_ROOT}" ]]; then
    echo "---------------------------------------------------------------------"
    echo -e "TorchSparse_ROOT env variable not found;\n Install TorchSparse?"
    echo "---------------------------------------------------------------------"
    echo "Installing TorchSparse"
    # Install TorchSparse
    git clone --recurse-submodules https://github.com/rusty1s/pytorch_sparse
    mkdir -p install build_sparse
    cd build_sparse
    if [[ $is_cuda_available -eq 1 ]]; then
        cmake -DCMAKE_PREFIX_PATH="${TORCH_ROOT}" -DCMAKE_INSTALL_PREFIX="" -DCUDNN_INCLUDE_PATH="${CUDNN_ROOT}/include" -DCUDNN_LIBRARY_PATH="${CUDNN_ROOT}/lib" -DCMAKE_BUILD_TYPE=Release ../pytorch_sparse
    else
        cmake -DCMAKE_PREFIX_PATH="${TORCH_ROOT}" -DCMAKE_INSTALL_PREFIX="" -DCMAKE_BUILD_TYPE=Release ../pytorch_sparse
    fi
    make install DESTDIR="${current_dir}/pytorch_sparse/install"
    # Add TorchSparse to path
    echo "# TorchSparse " >> env.sh
    echo "export TorchSparse_ROOT=\"${current_dir}/pytorch_sparse/install/usr/local\"" >> env.sh
    echo "export LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH}:${current_dir}/pytorch_sparse/install/usr/local/lib\"" >> env.sh
    echo "export INCLUDE=\"\${INCLUDE}:${current_dir}/pytorch_sparse/install/usr/local/include\"" >> env.sh
    echo "export TorchSparse_DIR=\"${current_dir}/pytorch_sparse/install/usr/local/share/cmake\"" >> env.sh
else
    echo "Info: Located TorchSparse at ${TorchSparse_ROOT}"
fi

echo "DONE!"
echo "Please run the following command to set the environment variables"
echo "source env.sh"
