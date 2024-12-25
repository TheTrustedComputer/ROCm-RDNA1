#! /usr/bin/env bash

PATH=/opt/rocm/bin:$PATH

set -e

echo "Detecting GPU architectures..."
rocm_agent_enumerator

echo -e "\nChecking HIPCC version..."
hipcc --version

mkdir -p build

# GPU0: Success
echo -e "\nVerifying HIP..."
hipcc -O3 -w -march=native -I/opt/rocm/include -L/opt/rocm/lib verify_hip.cpp -o build/verify_hip
build/verify_hip

# GPU0: 0.637206 0.499381 0.4742 0.584807 0.120109 0.260578 0.928544 0.471172 0.362893 0.412708 
echo -e "\nVerifying rocRAND..."
hipcc -O3 -w -march=native -I/opt/rocm/include -L/opt/rocm/lib verify_rocrand.cpp -o build/verify_rocrand -lrocrand
build/verify_rocrand

# Matrix A
# [ 1 2 3 4 ]
# [ 5 6 7 8 ]
# [ 9 10 11 12 ]
# [ 13 14 15 16 ]
#
# Matrix B
# [ 16 15 14 13 ]
# [ 12 11 10 9 ]
# [ 8 7 6 5 ]
# [ 4 3 2 1 ]
#
# Matrix AB on GPU0
# [ 80 70 60 50 ]
# [ 240 214 188 162 ]
# [ 400 358 316 274 ]
# [ 560 502 444 386 ]
echo -e "\nVerifying rocBLAS..."
hipcc -O3 -w -march=native -I/opt/rocm/include -L/opt/rocm/lib verify_rocblas.cpp -o build/verify_rocblas -lrocblas
build/verify_rocblas

# Sparse Matrix A
# [ 1 0 2 0 ]
# [ 3 0 0 4 ]
# [ 0 5 6 0 ]
# [ 0 7 0 8 ]
#
# Vector X [ 1 2 3 4 ]
# Vector Y on GPU0 [ 7 19 28 46 ]
echo -e "\nVerifying rocSPARSE..."
hipcc -O3 -w -march=native -I/opt/rocm/include -L/opt/rocm/lib verify_rocsparse.cpp -o build/verify_rocsparse -lrocsparse
build/verify_rocsparse

# Matrix A
# [ 2 3 5 7 ]
# [ 11 13 17 19 ]
# [ 23 29 31 37 ]
# [ 41 43 47 53 ]
#
# Pivot Vector [ -50 -60 -70 -80 ]
# Solution Vector on GPU0 [ -1.09091 14.8182 2.86364 -15.2273 ]
echo -e "\nVerifying rocSOLVER..."
hipcc -O3 -w -march=native -I/opt/rocm/include -L/opt/rocm/lib verify_rocsolver.cpp -o build/verify_rocsolver -lrocblas -lrocsolver
build/verify_rocsolver

# FFT of sine wave at 120 Hz on GPU0
# 0 Hz: 0.998028
# 10 Hz: 1.00478
# 20 Hz: 1.02566
# 30 Hz: 1.06257
# 40 Hz: 1.11921
# 50 Hz: 1.20213
# 60 Hz: 1.32302
# 70 Hz: 1.50378
# 80 Hz: 1.79011
# 90 Hz: 2.29438
# 100 Hz: 3.38486
# 110 Hz: 7.36896
# 120 Hz: 21.9047
# 130 Hz: 3.97039
# 140 Hz: 2.05124
# 150 Hz: 1.31714
# 160 Hz: 0.928637
# 170 Hz: 0.68702
# 180 Hz: 0.520938
# 190 Hz: 0.398368
# 200 Hz: 0.302853
# 210 Hz: 0.225074
# 220 Hz: 0.159697
# 230 Hz: 0.104707
# 240 Hz: 0.0654588
echo -e "\nVerifying rocFFT..."
hipcc -O3 -w -march=native -I/opt/rocm/include -L/opt/rocm/lib verify_rocfft.cpp -o build/verify_rocfft -lrocfft
build/verify_rocfft

# -6 -6 -6
# -6 -6 -6
# -6 -6 -6
echo -e "\nVerifying MIOpen..."
hipcc -O3 -w -march=native -I/opt/rocm/include -L/opt/rocm/lib verify_miopen.cpp -o build/verify_miopen -lMIOpen
build/verify_miopen

# 250 260 270 280 
# 618 644 670 696 
# 986 1028 1070 1112 
# 1354 1412 1470 1528 
echo -e "\nVerifying MIGraphX..."
hipcc -O3 -w -march=native -std=c++17 -I/opt/rocm/include -I/opt/rocm/lib/migraphx/include -L/opt/rocm/lib -L/opt/rocm/lib/migraphx/lib verify_migraphx.cpp -o build/verify_migraphx -lmigraphx -lmigraphx_gpu
build/verify_migraphx

rm -rf build

echo -e "\nAll checks passed!"
