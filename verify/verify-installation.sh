#! /usr/bin/env bash

set -e

echo "Detecting GPU architectures..."
rocm_agent_enumerator

echo -e "\nChecking HIPCC version..."
hipcc --version

mkdir -p build

echo -e "\nVerifying HIP..."
hipcc -Wall -Wextra -march=native verify_hip.cpp -o build/verify_hip 2> /dev/null
build/verify_hip

echo -e "\nVerifying hipRAND..."
hipcc -Wall -Wextra -march=native verify_hiprand.cpp -o build/verify_hiprand -lhiprand 2> /dev/null
build/verify_hiprand

echo -e "\nVerifying rocRAND..."
hipcc -Wall -Wextra -march=native verify_rocrand.cpp -o build/verify_rocrand -lrocrand 2> /dev/null
build/verify_rocrand

echo -e "\nVerifying hipBLAS..."
hipcc -Wall -Wextra -march=native verify_hipblas.cpp -o build/verify_hipblas -lhipblas 2> /dev/null
build/verify_hipblas

echo -e "\nVerifying rocBLAS..."
hipcc -Wall -Wextra -march=native verify_rocblas.cpp -o build/verify_rocblas -lrocblas 2> /dev/null
build/verify_rocblas

echo -e "\nVerifying hipSPARSE..."
hipcc -Wall -Wextra -march=native verify_hipsparse.cpp -o build/verify_hipsparse -lhipsparse 2> /dev/null
build/verify_hipsparse

echo -e "\nVerifying rocSPARSE..."
hipcc -Wall -Wextra -march=native verify_rocsparse.cpp -o build/verify_rocsparse -lrocsparse 2> /dev/null
build/verify_rocsparse

echo -e "\nVerifying hipSOLVER..."
hipcc -Wall -Wextra -march=native verify_hipsolver.cpp -o build/verify_hipsolver -lhipsolver 2> /dev/null
build/verify_hipsolver

echo -e "\nVerifying rocSOLVER..."
hipcc -Wall -Wextra -march=native verify_rocsolver.cpp -o build/verify_rocsolver -lrocblas -lrocsolver 2> /dev/null
build/verify_rocsolver

echo -e "\nVerifying hipFFT..."
hipcc -Wall -Wextra -march=native verify_hipfft.cpp -o build/verify_hipfft -lhipfft 2> /dev/null
build/verify_hipfft

echo -e "\nVerifying rocFFT..."
hipcc -Wall -Wextra -march=native verify_rocfft.cpp -o build/verify_rocfft -lrocfft 2> /dev/null
build/verify_rocfft

echo -e "\nVerifying MIOpen..."
hipcc -Wall -Wextra -march=native verify_miopen.cpp -o build/verify_miopen -lMIOpen 2> /dev/null
build/verify_miopen

echo -e "\nVerifying MIGraphX..."
hipcc -Wall -Wextra -march=native verify_migraphx.cpp -o build/verify_migraphx -I/opt/rocm/lib/migraphx/include -L/opt/rocm/lib/migraphx/lib -lmigraphx -lmigraphx_gpu 2> /dev/null
build/verify_migraphx

rm -rf build

echo -e "\nAll checks passed!"
