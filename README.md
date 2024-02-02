# Torch ML Model Driver

 `TorchMLModelDriver` is the base driver needed for machine learning based portable models. 
This driver provides the interface between KIM-API and TorchScript models. It works with
any arbitrary model as long as the models are TorchScript compatible and follow one of the following
calling conventions in their `forward` method`:

1. `forward(self, species, coords, n_neigh, nlist, contributing)`
2. `forward(self, descriptor)`
3. `forward(self, species, coords, graph_layer1, graph_layer2,..., contributing)`

Pattern 1 is for more conventional models which require the raw information about the system, namely
species, coordinates, number of neighbors, neighbor list and contributing atoms. Pattern 2 is for
descriptor based models, where the descriptor is computed by the KIM-API and passed to the model. 
The descriptor computation is done by the `libdescriptor` library, which is a C++ library for computing
descriptor and their gradients. Pattern 3 is for graph neural networks, where the model takes the
species, coordinates and the graph layers as input. The graph layers are computed by the model driver itself
and uses the staged graphs approach for parallelization. In the GNN models on the Pytorch Geometric library is
supported, as DGL does not support TorchScript yet.

The models are supposed to provide either energy as output, or energy and forces as output. The model driver
will compute the forces from the energy, if the model does not provide the forces. If the energy tensor is not
a scalar, the model driver will sum the energy tensor to get the total energy, and assign the per atom energy
to all contributing atoms.

Below is the diagram showing the flow of information in the model driver.

<img src="modelDriververticle.svg" width="800">

## Dependencies
Model driver depends on several libraries for it to functions seamlessly. And it expects them to be
provided by the user at runtime. The core requirement of ML model driver is `libtorch` library that provides the interface
between the KIM model driver's C++ API and the TorchScript models. For Graph Neural Networks, the Torch model
shall use [Pytorch Geometric Library](https://github.com/pyg-team/pytorch_geometric). The C++ API of Pytorch Geometric lib
depends upon `torch-scatter` and `torch-sparse` libraries. The `libdescriptor` library is used for descriptor based models.

Summary of dependencies:
- libtorch (CXX11 ABI, v1.13)
- KIM-API (v2.3)
- libdescriptor (0.0.7)
- Enzyme AD (0.0.94)
- libtorchscatter
- libtorchsparse


> If your compute environment does not contain these dependencies, you can install above dependencies using the `install_dependencies.sh` script provided with the model driver source. 

This script shall install all the dependencies in the current directory, and give a `env.sh` file that can be sourced for
setting the environment variables.
For more detailed instructions on installing dependencies, see below.

## Install
If all dependencies are met, installation should be as simple as calling the appropriate `kim-api-collections-management install` command. 
Your shell environment should provide required variables for dependency resolution namely,
1. `TORCH_ROOT` 
2. `TorchScatter_ROOT` 
3. `TorchSparse_ROOT`
4. `LIBDESCRIPTOR_ROOT`

`libtorch` is simple to install, just download them libtorch binaries from
PyTorch website, and put them at appropriate system paths.
For GPU support, you need to download the CUDA enabled binaries for libtorch, along with the CUDNN library, which libtorch
depends upon. For CUDNN library please register and download them from [NVIDIA website](https://developer.nvidia.com/rdp/cudnn-archive).

## Environment Variables
The model driver provides several environment variables for enhanced functionality. The following environment variables
are used by the model driver:

### Compile Time Variables
1. `KIM_MODEL_MPI_AWARE` -  If set to `yes` (*case-sensitive*), during driver installation the model driver will be built
with MPI support and will require a valid MPI environment to be present at installation time. This ensures a more hardware 
agnostic allocation of GPU resources. Specifically, this will set up `n` GPUs on each node to be used for `m` ranks on 
the same node in a round-robin fashion (i.e. rank `m` will receive GPU number [`m` mod `n`])
2. `KIM_MODEL_DISBALE_GRAPH` - If this environment variable is defined (irrespective of value), the model driver will be
built without graph support. This means during build time it will not try to find and link against `libtorchscatter` and
`libtorchsparse` libraries, and will not support models with pattern 3.

### Runtime Variables
1. `KIM_MODEL_ELEMENTS_MAP` - If set to any value during runtime, will enable mapping of elements to their atomic numbers.
2. `KIM_MODEL_EXECUTION_DEVICE` - If set to `cuda` during runtime, will enable evaluation of the Torch Model on GPU.
```shell
export KIM_MODEL_EXECUTION_DEVICE="cuda"

# Set visible devices if needed
export CUDA_VISIBLE_DEVICES=0,1,2
```
Because KIM model driver is inherently compatible with LAMMPS domain decomposition, enabling distributed
GPU support is as simple as just running LAMMPS with multiple ranks.
Also, at present Torch model resides on GPU, independent of the LAMMPS, so following points shall be kept in mind
1. You need not compile LAMMPS with GPU enabled, model driver only interacts with LAMMPS via KIM, which is CPU only
2. As every evaluation needs copying data from CPU to GPU and vice versa, so to see benefits of GPU you might need 
system of substantial size.

## Known Installation Issues

During installation, you might encounter following error messages

### 1. `Could not locate pthreads/ Threads.cmake` etc
Usually any comparatively modern Linux installation would come with a valid posix threads library. 
In some HPC with minimal or older linux installations, you might get above error because the default compiler
might be some minimal CC wrapper that cannot detect it. Easiest way forward is to simply provide
a valid C and C++ compilers as,
```shell
CC=mpicc CXX=mpic++ bash install_dependencies.sh
# or
CC=gcc CXX=g++ cmake ..
# etc. etc. 
```

### 2. `libcuda.so.1: cannot open shared object file`
Cuda installations come with two set of libraries, `libcudart.so`, which is the actual cuda
implementation, and old `libcuda.so` which are stubs for legacy purposes. Easiest workaround
is 
1. to either compile your code on execution node, or
2. symlink libcudart as libcuda at a local location for compiling purposes,
3. setup cuda env properly, the stubs are kept at `$CUDA_ROOT/lib64/stubs`, add it to `LD_LIBRARY_PATH`

### 3. Compilation runs in infinite loop
cmake > 3.18 have bug that runs in infinite compilation loops in some cases, please use cmake <= 3.18
