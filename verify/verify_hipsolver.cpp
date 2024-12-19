#include <iostream>
#include <hipsolver/hipsolver.h>

void printMatrix(const char *_STR, const float *_MATRIX, const int _N)
{
    std::cout << _STR << std::endl;
    
    for (int i = 0; i < _N; i++)
    {
        std::cout << "[ ";
        
        for (int j = 0; j < _N; j++)
        {
            std::cout << _MATRIX[j * _N + i] << " ";
        }
        
        std::cout << "]\n";
    }
}

void printVector(const char *_STR, const float *_VECT, const int _SIZE)
{
    std::cout << _STR << " [ ";
    
    for (int i = 0; i < _SIZE; ++i)
    {
        std::cout << _VECT[i] << " ";
    }
    
    std::cout << "]" << std::endl;
}

int main(void)
{
    bool _;
    int devices;
    hipsolverHandle_t handle;
    
    constexpr int N = 4;
    constexpr int N_N = N * N;
    
    float h_A[N_N] = { 2.0f, 11.0f, 23.0f, 41.0f, 3.0f, 13.0f, 29.0f, 43.0f, 5.0f, 17.0f, 31.0f, 47.0f, 7.0f, 19.0f, 37.0f, 53.0f };
    float h_B[N] = { -50.0f, -60.0f, -70.0f, -80.0f };
    float h_C[N];
    int h_info;
    
    float *d_A, *d_B;
    int *d_info, *d_pivot;
    
    int lwork;
    float *d_work;
    
    printMatrix("Matrix A:", h_A, N);
    std::cout << std::endl;
    printVector("Pivot Vector:", h_B, N);
    
    _ = hipGetDeviceCount(&devices);
    
    for (int i = 0; i < devices; i++)
    {
        hipsolverCreate(&handle);
        
        _ = hipMalloc(&d_A, N_N * sizeof(*d_A));
        _ = hipMalloc(&d_B, N * sizeof(*d_B));
        _ = hipMalloc(&d_info, sizeof(*d_info));
        _ = hipMalloc(&d_pivot, N * sizeof(*d_pivot));
        
        _ = hipMemcpy(d_A, h_A, N_N * sizeof(*h_A), hipMemcpyHostToDevice);
        _ = hipMemcpy(d_B, h_B, N * sizeof(*h_B), hipMemcpyHostToDevice);
        
        hipsolverSgetrf_bufferSize(handle, N, N, d_A, N, &lwork);
        
        _ = hipMalloc(&d_work, lwork * sizeof(*d_work));
        
        hipsolverSgetrf(handle, N, N, d_A, N, d_work, lwork, d_pivot, d_info);
        hipsolverSgetrs(handle, HIPSOLVER_OP_N, N, 1, d_A, N, d_pivot, d_B, N, d_work, lwork, d_info);
        
        _ = hipMemcpy(h_C, d_B, N * sizeof(*d_B), hipMemcpyDeviceToHost);
        _ = hipMemcpy(&h_info, d_info, sizeof(*d_info), hipMemcpyHostToDevice);
        
        _ = hipFree(d_A);
        _ = hipFree(d_B);
        _ = hipFree(d_info);
        _ = hipFree(d_pivot);
        _ = hipFree(d_work);
        
        hipsolverDestroy(handle);
    }
    
    printVector("Solution Vector:", h_C, N);
    
    return 0;
}
