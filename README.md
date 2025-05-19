# ROCm-RDNA1

ROCm build scripts and patches for PyTorch and ONNX Runtime targeting the RDNA1 instruction set.

In our setup, we have two 8 GB AMD Radeon RX 5500 XTs for machine learning purposes, not the standard 4 GB model. We set the GPU target to `gfx1012` to ensure interoperability without broken workarounds with `HSA_OVERRIDE_GFX_VERSION` that may lead to unpredictable behavior from mismatched instruction sets. The 5600 XT and 5700 XT correspond to `gfx1010` respectively, so choose the matching target with your AMD GPU.

## Disclaimers

### Unofficial Support

It's important to understand that RDNA1 GPUs are **NOT** supported by AMD for ROCm; this is a community-driven effort. The only real option for users is to compile it for this architecture and hope for the best. Theoretically, this isn't guaranteed to work, and it's at the **user's discretion** to run untested code. However, in practice, most machine learning frameworks **function perfectly** despite lacking official support. Some unit tests mayn't pass, and performance may be suboptimal, which is to be expected on unsupported hardware unless appropriate patches are applied.

### By Enthusiasts, For Enthusiasts

We're not experts in architectures, ROCm, or GPGPU computing. We're simply enthusiasts who are interested in machine learning applications. As such, we're always open to talented and skilled developers who want to improve the performance and correctness of ROCm on RDNA1 hardware. We also encourage pull requests to the original repository to enhance the user experience, as the internal ROCm team actually *welcomes* such requests, as long it does not lead to a regression. Getting them merged **will take some time** as they do not have the appropriate hardware for testing. Therefore, the original submitter is responsible for documenting changes in depth and fixing any reported issues prior to upstream integration.

## System Requirements

- A discrete RDNA1 AMD GPU
    - Radeon RX 5500/5600/5700 XT
- A decent computer running a Linux distribution
    - AMD Ryzen 3000
    - Intel Core 10th Gen
    - 32 GB RAM
- About 30 GB of free disk space for the full build

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

After installation, verify that Docker is active and your user has the privilege to run containers. You may need to enable any Docker-related services if the distribution disables it by default, as is the case with Arch.

## Build Instructions

### 1. Clone this repository to your system

Make a local copy of this repository on your computer. You may do this via a `git clone` of the web address or directly on GitHub.

### 2. Select a volume to mount the software directory in the container

By default, all applications are stored in the ``$HOME/Docker/Apps`` directory under the Docker user ``rdna1_rocm-A.B_container`` for persistence, where ``A.B`` is the ROCm point release number. You can change these by editing the ``.env`` file in the ``SOFTWARE_DIR`` and ``HOME_USER`` fields. As a reminder, don't forget to modify ``AMDGPU_TARGETS`` for your GPU!

### 3. Build the runtime Docker image

We've attached a `build-image` script for each desired version to build the multi-step runtime Docker image at your convenience. If you want ROCm 5.4, all you have to do is run the following commands after cloning:

```sh
cd ROCm-RDNA1/ROCm-5.4
./build-image
```

You're done! Patience is key, as this may take several hours depending on your hardware. As for the other versions, replace 5.4 with another release, and repeat the above lines. When the image is ready, attach the container with:

```sh
docker-compose up runtime
docker attach RDNA1_ROCm54_Runtime # do this on another terminal session or window
```

We're not kidding; it's that simple! The runtime image includes PyTorch, TorchAudio, TorchVision, and ONNX Runtime Python wheels that you can install via pip. You're all set to experiment with the world of AI!

## Offered Configurations

### Debian 12: ROCm 5.4.4 + PyTorch 2.2.2 + ONNX Runtime 1.16.3

Recommended for the most stable and efficient production environment. It's not nearly as fast as the official ROCm 5.2 with ``HSA_OVERRIDE_GFX_VERSION=10.3.0``, but it has the advantage of being a native build.

### Ubuntu 24.04: ROCm 6.3.3 + PyTorch 2.5.1 + ONNX Runtime 1.22.0

Provides newer software and libraries when performance isn't a priority. Choose this if you have conflicting dependencies, erratic program behavior, or runtime errors due to incompatible APIs on the Debian 12 config.

### Arch Linux: Latest ROCm + PyTorch + ONNX Runtime

This distribution tends to have the latest and greatest of just about everything. You might want to give it a shot if you value living on the bleeding edge at the expense of the occasional bug.

## Considerations

 - PyTorch 2.2.2 is recommended for security reasons as older ones contain a critical vulnerability that could lead to remote code execution.
 - ROCm 5.3.3 and earlier are NOT build-compatible with PyTorch 2.x. You'll need PyTorch 1.13.1, or use hacky workarounds.
 - It is possible to build PyTorch 2.3.x against ROCm < 5.7 for RDNA1, but the process is broken after integrating AOTriton. Avoid at all costs.
 - hipBLASLt is created as a dummy library, as it's not yet supported on this architecture while permitting linkage to PyTorch.
 - This PyTorch cannot use flash or memory-efficient attention for transformer models because Triton currently does not implement support for RDNA1.
 - ONNX Runtime 1.17 needs ROCm >= 5.6, and ONNX Runtime 1.20 depends on ROCm >= 6.0 for a successful build.
 - These libraries are built without testing enabled. Copy the verification directory to the container to confirm their basic functionality.
 - The default CMake build type is set to `Release` and has `-march=native -O3 -s -w` appended to the compiler flags.
 - Composable kernels won't compile out of the box but will when patched to include them (extremely slow build time and highly experimental).
 - **URGENT**: The PyTorch ROCm 5.2 precompiled wheels no longer run with glibc 2.41 and later due to changes in shared library execution policies. You must compile ROCm from source code for continued operation on RDNA1. There is no straightforward solution besides downgrading your container to a stable distribution shipping with glibc <= 2.40.

## Acknowledgements

We would like to thank [xuhuisheng](https://github.com/xuhuisheng) for providing the initial ROCm 5.4 build scripts, [lamikr](https://github.com/lamikr) for introducing several, novel patches to get multiple discrete and integrated AMD graphics cards working at once, the Gentoo Linux team for making hipBLASLt a dummy library for unsupported architectures, and AMD themselves for open-sourcing and developing ROCm in the first place. Without their contributions, this repository wouldn't have been possible.
