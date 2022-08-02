## COLABFIT Model Driver

This repository contains the `TorchMLModelDriver` which is the base driver needed for COLABFIT portable models. It preprocesses the descriptors from a predefined library if needed (colabfit-descriptor-library), and computes the gradients for the same using high-performance Enzyme automatic differentiation. 

**WORK IN PROGRESS** and several breaking changes are expected.

Current dependencies
- libtorch
- libdescriptor [COLABFIT]
- KIM API 2.3

Current CMake file is bit tightly coupled with my dev environment. Will make more general CMake file soon. 