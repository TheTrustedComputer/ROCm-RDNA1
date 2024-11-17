# ROCm-RDNA1

ROCm build scripts and patches for PyTorch and ONNX Runtime targeting the RDNA1 instruction set. Note that RDNA1 GPUs are **NOT** supported. The only official way is to compile ROCm for the architecture and hope for the best. Theoretically, this is not guaranteed to work. However, in practice, most machine learning frameworks function perfectly despite lacking official support.

The latest point release has been tested on Arch Linux in a Docker container using AMD's fork of LLVM/Clang; only GCC is used to build their LLVM fork. As for ROCm 5.4, it was tested with Debian 12 Bookworm utilizing similar techniques.

Since I have two 8GB AMD Radeon RX 5500 XTs for ML purposes, I must set the GPU target to `gfx1012` to ensure compatibility without hacky workarounds with `HSA_OVERRIDE_GFX_VERSION` and so on. The 5600 XT and 5700 XT correspond to `gfx1011` and `gfx1010` respectively, so choose the matching target with your AMD GPU.

## Docker Installation

I recommend installing Docker on your Linux machine for easy reproducibility and testability. Consult the package manager for your distribution on how to install it. For example, on Debian-based distributions:
```
sudo apt install docker.io docker-compose
```

Arch-based distributions:
```
sudo pacman -S docker docker-buildx docker-compose
```
You may need to enable any Docker-related services if the distribution disables it by default, as is the case with Arch.

## Build Instructions
TBA.

## Shoutouts

I would like to thank GitHub contributors [xuhuisheng](https://github.com/xuhuisheng) for providing the initial ROCm 5.4 build scripts, [lamikr](https://github.com/lamikr) for introducing several, novel patches to get multiple cards working at once, and the [Gentoo Linux](https://gentoo.org) team for making hipBLASLt a dummy library for unsupported architectures. Without them, this repository would not have been possible.

## Required Components
### PyTorch
 - comgr
 - hip
 - hipBLAS
 - hipBLASLt (starting 2.3 with ROCm 5.7)
 - hipCUB
 - hipFFT
 - hipRAND
 - hipSOLVER
 - hipSPARSE
 - HSA Runtime
 - MIOpen
 - rocBLAS
 - rocPRIM
 - rocRAND
 - rocThrust
 - roctracer
 - rccl

### ONNX Runtime
 - hip
 - hipFFT
 - hipRAND
 - MIGraphX (alternative execution provider)
 - MIOpen
 - rocBLAS

### MAGMA + OpenBLAS:
 - hip
 - hipBLAS
 - hipSPARSE (optional)
 - OpenBLAS

## Notes
 - These libraries are built without testing enabled. Navigate the verify directory to test their basic functionality.
 - If some libraries don't seem to build, we've provided known fixes in the patch directory to ensure they do.
 - The default CMake build type is set to Release and have "-march=native -s" appended to the compiler flags.
 - hipBLASLt is created as a dummy library, as it's not yet supported on this architecture while still permitting linkage.
 - LLVM's Fortran compiler, Flang, only accepts the Fortran 2018 standard whereas GFortran allows older revisions.
 - Composable kernels won't compile for RDNA1 GPUs out of the box but will when patched to include them (extremely slow).
 - Protobuf 3.20.x is the latest version that doesn't throw undefined reference errors when building MIGraphX.
 - Interestingly, adding -DMIGRAPHX_ENABLE_MLIR=OFF to MIGraphX CMake results in linker errors for some reason.
 - The entire process can take several hours, especially on composable kernels. On our machine, it took nearly 20 hours.
