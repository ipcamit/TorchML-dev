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
if [[ "$response" =~ [yY] ]]
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

# Check if git is installed
if ! command -v git &> /dev/null
then
    echo "Error: git could not be found, please install git"
    exit
fi

# Check if unzip is installed
if ! command -v unzip &> /dev/null
then
    echo "Error: unzip could not be found, please install unzip"
    exit
fi

if ! command -v kim-api-collections-management &> /dev/null
then
    echo "Installing KIM-API"
    # Install KIM-API
    git clone https://github.com/openkim/kim-api
    cd kim-api || exit
    mkdir build && cd build && cmake .. && make
    make DESTDIR="${current_dir}"/kim-api/install install
    cd ${current_dir} || exit
    # Add KIM-API to path
    echo "# KIM-API " >> env.sh
    # KIM has to be installed with user prefix of /usr/local as otherwise CMAKE gives errors
    KIM_PATH="${current_dir}"/kim-api/install/usr/local/bin
    KIM_LIB="${current_dir}"/kim-api/install/usr/local/lib
    KIM_INCLUDE="${current_dir}"/kim-api/install/usr/local/include

    echo "export PATH=$KIM_PATH:\$PATH" >> env.sh
    echo "export LD_LIBRARY_PATH=$KIM_LIB:\$LD_LIBRARY_PATH" >> env.sh
    echo "export INCLUDE=$KIM_INCLUDE:\$INCLUDE" >> env.sh
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
        unzip libtorch-cxx11-abi-shared-with-deps-1.13.0+cu117.zip || exit
        # rm libtorch-cxx11-abi-shared-with-deps-1.9.0+cu111.zip
    else
        wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.0%2Bcpu.zip
        unzip libtorch-cxx11-abi-shared-with-deps-1.9.0+cpu.zip || exit
        # rm libtorch-cxx11-abi-shared-with-deps-1.9.0+cpu.zip
    fi
    # Add libtorch to path
    TORCH_ROOT="${current_dir}"/libtorch
    TORCH_LIB="${current_dir}"/libtorch/lib
    TORCH_INCLUDE="${current_dir}"/libtorch/include

    echo "# libtorch " >> env.sh
    echo "export TORCH_ROOT=$TORCH_ROOT" >> env.sh
    echo "export LD_LIBRARY_PATH=$TORCH_LIB:\$LD_LIBRARY_PATH" >> env.sh
    echo "export INCLUDE=$TORCH_INCLUDE:\$INCLUDE" >> env.sh
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
    git clone --recurse-submodules https://github.com/rusty1s/pytorch_scatter || exit
    mkdir -p build_scatter
    cd build_scatter || exit
    if [[ $is_cuda_available -eq 1 ]]; then
        cmake -DCMAKE_PREFIX_PATH="${TORCH_ROOT}" -DCMAKE_INSTALL_PREFIX="" -DCUDNN_INCLUDE_PATH="${CUDNN_ROOT}/include" -DCUDNN_LIBRARY_PATH="${CUDNN_ROOT}/lib" -DCMAKE_BUILD_TYPE=Release ../pytorch_scatter || exit
    else
        cmake -DCMAKE_PREFIX_PATH="${TORCH_ROOT}" -DCMAKE_INSTALL_PREFIX="" -DCMAKE_BUILD_TYPE=Release ../pytorch_scatter || exit
    fi
    make install DESTDIR="${current_dir}/pytorch_scatter/install" || exit
    # Add TorchScatter to path
    TorchScatter_ROOT="${current_dir}"/pytorch_scatter/install
    TorchScatter_LIB="${current_dir}"/pytorch_scatter/install/lib
    TorchScatter_INCLUDE="${current_dir}"/pytorch_scatter/install/include
    TorchScatter_DIR="${current_dir}"/pytorch_scatter/install/lib/cmake

    cd ${current_dir} || exit

    echo "# TorchScatter " >> env.sh
    echo "export TorchScatter_ROOT=$TorchScatter_ROOT" >> env.sh
    echo "export LD_LIBRARY_PATH=$TorchScatter_LIB:\$LD_LIBRARY_PATH" >> env.sh
    echo "export INCLUDE=$TorchScatter_INCLUDE:\$INCLUDE" >> env.sh
    echo "export TorchScatter_DIR=$TorchScatter_DIR" >> env.sh
else
    echo "Info: Located TorchScatter at ${TorchScatter_ROOT}"
fi

if [[ -z "${TorchSparse_ROOT}" ]]; then
    echo "---------------------------------------------------------------------"
    echo -e "TorchSparse_ROOT env variable not found;\n Install TorchSparse?"
    echo "---------------------------------------------------------------------"
    echo "Installing TorchSparse"
    # Install TorchSparse
    git clone --recurse-submodules https://github.com/rusty1s/pytorch_sparse || exit
    mkdir -p build_sparse
    cd build_sparse || exit
    if [[ $is_cuda_available -eq 1 ]]; then
        cmake -DCMAKE_PREFIX_PATH="${TORCH_ROOT}" -DCMAKE_INSTALL_PREFIX="" -DCUDNN_INCLUDE_PATH="${CUDNN_ROOT}/include" -DCUDNN_LIBRARY_PATH="${CUDNN_ROOT}/lib" -DCMAKE_BUILD_TYPE=Release ../pytorch_sparse || exit
    else
        cmake -DCMAKE_PREFIX_PATH="${TORCH_ROOT}" -DCMAKE_INSTALL_PREFIX="" -DCMAKE_BUILD_TYPE=Release ../pytorch_sparse || exit
    fi
    make install DESTDIR="${current_dir}/pytorch_sparse/install" || exit
    # Add TorchSparse to path
    TorchSparse_ROOT="${current_dir}"/pytorch_sparse/install
    TorchSparse_LIB="${current_dir}"/pytorch_sparse/install/lib
    TorchSparse_INCLUDE="${current_dir}"/pytorch_sparse/install/include
    TorchSparse_DIR="${current_dir}"/pytorch_sparse/install/lib/cmake

    cd ${current_dir} || exit

    echo "# TorchSparse " >> env.sh
    echo "export TorchSparse_ROOT=$TorchSparse_ROOT" >> env.sh
    echo "export LD_LIBRARY_PATH=$TorchSparse_LIB:\$LD_LIBRARY_PATH" >> env.sh
    echo "export INCLUDE=$TorchSparse_INCLUDE:\$INCLUDE" >> env.sh
    echo "export TorchSparse_DIR=$TorchSparse_DIR" >> env.sh

else
    echo "Info: Located TorchSparse at ${TorchSparse_ROOT}"
fi

echo "DONE!"
echo "---------------------------------------------------------------------"
echo "Please run the following command to set the environment variables"
echo "source env.sh"
echo "---------------------------------------------------------------------"