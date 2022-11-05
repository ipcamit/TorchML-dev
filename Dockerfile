# base image
FROM ghcr.io/openkim/developer-platform

# set root user
USER root

# Install clang-12 (taken from enzyme dev-container Dockerfile)
RUN apt-get -q update \
    && apt-get install -y --no-install-recommends ca-certificates software-properties-common curl gnupg2 git\
    && curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key|apt-key add - \
    && apt-add-repository "deb http://apt.llvm.org/`lsb_release -cs`/ llvm-toolchain-`lsb_release -cs`-12 main" || true \
    && apt-get -q update \
    && apt-get install -y --no-install-recommends zlib1g-dev lldb ninja-build llvm-12-dev clang-format clang-12 libclang-12-dev libomp-12-dev lld-12 unzip \
    && python -m pip install --upgrade pip setuptools \
    && python -m pip install lit==12.0.1 pathlib2 \
    && touch /usr/lib/llvm-12/bin/yaml-bench

# Install Enzyme
WORKDIR /opt/enzyme
RUN git clone https://github.com/enzymead/enzyme \
    && cd enzyme/enzyme \
    && mkdir build \
    && cd build \
    && CC=clang-12 CXX=clang++-12 cmake .. -DLLVM_DIR=/usr/lib/llvm-12/cmake -DLLVM_EXTERNAL_LIT=/usr/local/bin/lit \
    && make \
    && make install


# Install libtorch
WORKDIR /opt
RUN curl https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.0%2Bcpu.zip --output libtorch.zip
RUN unzip libtorch.zip
ENV TORCH_ROOT="/opt/libtorch"

# Install TorchScatter/Sparse
RUN git clone --recurse-submodules https://github.com/rusty1s/pytorch_sparse
RUN git clone --recurse-submodules https://github.com/rusty1s/pytorch_scatter
RUN mkdir build_sparse build_scatter
RUN cd build_sparse \
    && cmake ../pytorch_sparse -DCMAKE_PREFIX_PATH=/opt/libtorch \
    && make install \
    && cd ../
RUN cd build_scatter \
    && cmake ../pytorch_scatter -DCMAKE_PREFIX_PATH=/opt/libtorch \
    && make install \
    && cd ../
# Ensure that correct "INTERFACE_INCLUDE_DIRECTORIES" var is set in TorchSparse TorchScatter
# For some reason it hardcode the initial folder
RUN sed -i "s#/opt/pytorch_sparse#/usr/local#g" /usr/local/share/cmake/TorchSparse/TorchSparseTargets.cmake
RUN sed -i "s#/opt/pytorch_scatter#/usr/local#g" /usr/local/share/cmake/TorchScatter/TorchScatterTargets.cmake
ENV TorchScatter_ROOT="/usr/local"
ENV TorchSparse_ROOT="/usr/local"

# Cleanup
RUN rm /opt/libtorch.zip
RUN rm -rf /opt/enzyme
RUN rm -rf pytorch_sparse pytorch_scatter build_scatter build_sparse

# Switch back to openkim env
WORKDIR /home/openkim
USER openkim

# Download and compile libdescriptor
# providing sudo password from commandline, as anyway it is out in open!
# using build-type release, as enzyme fails with debug
RUN git clone https://github.com/ipcamit/colabfit-descriptor-library \
    && cd colabfit-descriptor-library \
    && mkdir build && cd build \
    && CC=clang-12 CXX=clang++-12 cmake .. -DENZYME_LIB=/usr/local/lib -DCMAKE_BUILD_TYPE=Release \
    && make \
    && echo openkim | sudo -S make install
ENV LIBDESCRIPTOR_ROOT="/usr/local"

# Install TorchMLModelDriver
RUN git clone https://github.com/ipcamit/colabfit-model-driver \
    && /usr/local/bin/kim-api-collections-management install user colabfit-model-driver
