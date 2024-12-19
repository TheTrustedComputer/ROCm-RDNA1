#include <iostream>
#include <hip/hip_runtime.h>
#include <rocsolver/rocsolver.h>

void print_matrix(const char *_STR, const float *_MATRIX, const int _N)
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

void print_vector(const char *_STR, const float *_VECT, const int _SIZE)
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
    rocblas_handle handle;
    
    constexpr int N = 4;
    constexpr int N_N = N * N;
    
    float h_A[N_N] = { 2.0f, 11.0f, 23.0f, 41.0f, 3.0f, 13.0f, 29.0f, 43.0f, 5.0f, 17.0f, 31.0f, 47.0f, 7.0f, 19.0f, 37.0f, 53.0f };
    float h_B[N] = { -50.0f, -60.0f, -70.0f, -80.0f };
    float h_C[N];
    
    float *d_A, *d_B;
    int *d_info, *d_pivot;
    
    print_matrix("Matrix A:", h_A, N);
    std::cout << std::endl;
    print_vector("Pivot Vector:", h_B, N);
    
    _ = hipGetDeviceCount(&devices);
    
    for (int i = 0; i < devices; i++)
    {
        _ = hipSetDevice(i);
    
        rocblas_create_handle(&handle);
        
        _ = hipMalloc(&d_A, N_N * sizeof(*d_A));
        _ = hipMalloc(&d_B, N * sizeof(*d_B));
        _ = hipMalloc(&d_info, sizeof(*d_info));
        _ = hipMalloc(&d_pivot, N * sizeof(*d_pivot));
        
        _ = hipMemcpy(d_A, h_A, N_N * sizeof(*h_A), hipMemcpyHostToDevice);
        _ = hipMemcpy(d_B, h_B, N * sizeof(*h_B), hipMemcpyHostToDevice);
        
        rocsolver_sgetrf(handle, N, N, d_A, N, d_pivot, d_info);
        rocsolver_sgetrs(handle, rocblas_operation_none, N, 1, d_A, N, d_pivot, d_B, N);
        
        _ = hipMemcpy(h_C, d_B, N * sizeof(float), hipMemcpyDeviceToHost);
        
        _ = hipFree(d_A);
        _ = hipFree(d_B);
        _ = hipFree(d_info);
        _ = hipFree(d_pivot);
        
        rocblas_destroy_handle(handle);
    }
    
    print_vector("Solution Vector:", h_C, N);
    
    return 0;
}
