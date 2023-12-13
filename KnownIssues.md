Known Issues
============

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

If everything else fails, you can try manually compiling it as a last resort. 
You can do that by making a build folder inside the repository and just compiling `cmake .. && make`.
Following which keep the compiled `libkim-api-model-driver.so` object in 
`~/.kim-api/2.3.0+v*/model-drivers-dir/TorchMLMD__MD_000000000000_000` folder.
Again please note that this must be kept as a last resort option, and one above solutions should work.


### 3. Compilation runs in infinite loop
cmake > 3.18 have bug that runs in infinite compilation loops in some cases, please use cmake <= 3.18