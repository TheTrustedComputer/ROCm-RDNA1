# ROCm-RDNA1

ROCm build scripts and patches for PyTorch and ONNX Runtime targeting the RDNA1 instruction set. Other ROCm components aren't necessary for either of these, so they're excluded from the script to streamline the operation.

In our setup, we have two 8GB AMD Radeon RX 5500 XTs for machine learning purposes, not the standard 4GB model. We must set the GPU target to `gfx1012` to ensure interoperability without broken workarounds with `HSA_OVERRIDE_GFX_VERSION` that may lead to unpredictable behavior from incompatible instruction sets. The 5600 XT and 5700 XT correspond to `gfx1010` respectively, so choose the matching target with your AMD GPU.

The main branch contains the build script, patches, and Dockerfiles for the most recent version of ROCm. After the next major release, it'll be moved to a branch dedicated to it.

## Disclaimers

### Unofficial Support

It's important to understand that RDNA1 GPUs are **NOT** supported by AMD for ROCm; this is a community-driven effort. The only real option for users is to compile it for this architecture and hope for the best. Theoretically, this isn't guaranteed to work, and it's at the **user's discretion** to run untested code. However, in practice, most machine learning frameworks function perfectly despite lacking official support. Some unit tests mayn't pass, and performance may be suboptimal, which is to be expected on unsupported hardware unless appropriate patches are applied.

### By Enthusiasts, For Enthusiasts

We're not experts in architectures, ROCm, or GPGPU computing in general. We're simply enthusiasts like you who are interested in machine learning applications. As such, we're always open to talented and skilled developers who want to improve the performance and correctness of ROCm on RDNA1 hardware. We also encourage pull requests to the upstream repository to enhance the user experience, as the internal ROCm team actually *welcomes* such requests, as long it does not lead to a regression. Getting them merged **will take some time** as they do not have the appropriate hardware for testing. Therefore, the original submitter is responsible for documenting changes in depth and fixing any reported issues before integration.

## System Requirements

- A discrete RDNA1 AMD GPU (RX 5500/5600/5700 XT)
- A decent computer running a Linux distribution
- About 50GB of free disk space for the full build

## Docker Installation

We recommend installing Docker and Docker Compose on your Linux machine for easy reproducibility and testability, so everyone is on the same page in case of problems. Consult the package manager for your distribution on how to install it. For example, on Debian-based distributions, including Ubuntu and Mint:

```sh
sudo apt install docker.io docker-compose
```

Red Hat-based distributions (RHEL, Fedora):

```sh
sudo dnf install docker-cli docker-compose
```

Arch-based distributions (Manjaro, EndeavourOS):

```sh
sudo pacman -S docker docker-compose
```

After installation, verify that Docker is active and your account has the privilege to run containers. You may need to enable any Docker-related services if the distribution disables it by default, as is the case with Arch.

## Build Instructions (WIP)

### Tested Versions

The latest point release has been tested on an Arch Linux Docker image using AMD's fork of LLVM/Clang; only GCC is used to build their LLVM fork. As for ROCm 5.4.3, it was tested with a Debian 12 Bookworm image utilizing similar techniques. PyTorch and ONNX Runtime do just fine on our hardware under various workloads. The script assumes you have all third-party dependencies such as Protobuf (for MIGraphX) and MAGMA (for PyTorch). If not, download them as it **WILL** fail midway through.

### Build Compatibility

ROCm 5.4.3 (released February 2023) is build-compatible with PyTorch 2.2.2 (March 2024) and ONNX Runtime 1.16.3 (November 2023), all of which are quite old nowadays. ROCm <= 5.3.3 won't compile against PyTorch >= 2.0.0 because the latter uses the newer API. The current version can use the most up-to-date revision of these frameworks. As for other versions, we don't offer build scripts unless requested for specific reasons.

### Clone Repositories

Make a local copy of this repository on your computer. You may do this via a `git clone` of the web address or directly on GitHub.

Download the sources for ROCm, PyTorch, and ONNX Runtime. Use `repo` to retrieve all the ROCm components into a separate directory. Here's how to do it for an ROCm release. Substitute `A` and `B` with a number representing the major and minor revisions of interest.

```sh
mkdir rocm-src
cd rocm-src
repo init -u https://github.com/ROCm/ROCm -b roc-A.B.x
repo sync
```

If for whatever reason `repo` does not work, which can occur with new releases carrying breaking changes, we have scripts that do the old-fashioned `git clone` for each needed library. Here is how you can do it for ROCm 5.4.x.

```sh
bash download-5.4-sources
```

Regarding PyTorch and ONNX Runtime, a recursive `git clone` will suffice, where `RELEASE` is the newest point release tag of that framework, prefixed with `v`. Thus, if you want PyTorch 2.2.2, its `RELEASE` tag will be `v2.2.2`.

```sh
git clone --recursive https://github.com/pytorch/pytorch -b RELEASE
git clone --recursive https://github.com/microsoft/onnxruntime -b RELEASE
```

### Build Dependencies

#### Protobuf/MAGMA

Ensure you have fulfilled all the necessary build dependencies, including Protobuf and MAGMA. The following is for Protobuf 3.20.3 and MAGMA 2.8.0. If you selected ROCm 5.4.3, use MAGMA 2.6.2.

```sh
git clone https://github.com/protocolbuffers/protobuf -b v3.20.3
git clone https://bitbucket.org/icl/magma -b v2.8.0
```

#### Boost

Debian 12 ships with an outdated version of Boost (1.74) for ROCm 5.4.3's MIOpen library. AMD suggests 1.79 for maximum stability. Head over to the Boost home page to download and extract it via a web browser and file manager. A simple Google search for "boost 1.79 download" may also work as well. If you prefer to do it from the command line:

```sh
wget https://archives.boost.io/release/1.79.0/source/boost_1_79_0.tar.bz2
tar -xf boost_1_79_0.tar.bz2
rm boost_1_79_0.tar.bz2 # to save space
```

This step isn't mandatory if you're building the latest ROCm. Boost from your distribution's repository is enough if it's a rolling release.

### Build Docker Image

We've attached a `docker-compose.yaml` file to build each step of the Docker image at your convenience.

## Unit Test Results

| Target | Version | OS | HIP | BLAS | hipBLASLt | CUB | FFT | RAND | SOLVER | SPARSE | RCCL | PRIM | Thrust | Tracer | CK | MIOpen | MIGraphX | 
| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | -
| gfx1012 | 5.4.x | Debian 12 | 🔵 | 🟢 | 🔴 | 🟢 | 🔵 | 🔵 | 🟡 | 🟢 | 🟢 | 🔵 | 🔵 | 🟢 | 🔴 | 🟢 | 🟢 
| gfx1012 | 6.2.x | Ubuntu 24.04 | 🔵 | 🔵 | 🔴 | 🟢 | 🟢 | 🔵 | 🔵 | 🟢 | 🟡 | 🟢 |  🔵 | 🔵  | 🟢 | 🟢 | 🔵
| gfx1012 | 6.3.x | Arch Linux | 🔵 | ? | 🔴 | ? | ? | ? | ? | ? | ? | ? | ? | 🔵 | ? | ? | ?

🔵 Passes all unit tests\
🟢 Passes most unit tests\
🟡 Untested or cannot be tested\
🔴 Not applicable or supported

Some failures resulted in allocating too much VRAM to the device, which we excluded in the test results. We do not consider these to be bugs, but rather a consequence of having older, now unsupported hardware with less memory in a fast-paced ML ecosystem.

## Notes

 - These libraries are built without testing enabled. Navigate the verify directory to test their basic functionality.
 - If some libraries don't seem to build, we've provided known fixes in the patch directory to ensure they do.
 - The default CMake build type is set to Release and has `-march=native -O3 -s -w` appended to the compiler flags.
 - hipBLASLt will be created as a dummy library, as it's not yet supported on this architecture while permitting linkage.
 - Composable kernels won't compile out of the box but will when patched to include them (extremely slow).
 - PyTorch is compiled with no flash or memory-efficient attention due to Triton limitations of this GPU family.
 - Protobuf 3.20.x is the latest version that doesn't throw undefined reference errors when building MIGraphX.
 - Interestingly, adding `-DMIGRAPHX_ENABLE_MLIR=OFF` to MIGraphX CMake results in linker errors for some reason.
 - The entire process may take several hours, especially on composable kernels. On our machine, it took nearly 20 hours.

## Acknowledgements

We would like to thank [xuhuisheng](https://github.com/xuhuisheng) for providing the initial ROCm 5.4 build scripts, [lamikr](https://github.com/lamikr) for introducing several, novel patches to get multiple discrete and integrated AMD graphics cards working at once, the Gentoo Linux team for making hipBLASLt a dummy library for unsupported architectures, and AMD themselves for open-sourcing and developing ROCm in the first place. Without their contributions, this repository wouldn't have been possible.
