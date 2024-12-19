#include <iostream>
#include <hipblas/hipblas.h>

constexpr int N = 4;
constexpr int N_N = N * N;

void printMatrix(const char *_NAME, const float *_MATRIX)
{
    std::cout << _NAME << ":\n";
    
    for (int i = 0; i < N; i++)
    {
        std::cout << "[ ";
        
        for (int j = 0; j < N; j++)
        {
            std::cout << _MATRIX[j * N + i] << " ";
        }
        
        std::cout << "]\n";
    }
}

int main(void)
{
    bool _;
    int devices;
    hipblasHandle_t handle;
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    float h_mat_A[N_N] = { 1.0f, 5.0f, 9.0f, 13.0f, 2.0f, 6.0f, 10.0f, 14.0f, 3.0f, 7.0f, 11.0f, 15.0f, 4.0f, 8.0f, 12.0f, 16.0f };
    float h_mat_B[N_N] = { 16.0f, 12.0f, 8.0f, 4.0f, 15.0f, 11.0f, 7.0f, 3.0f, 14.0f, 10.0f, 6.0f, 2.0f, 13.0f, 9.0f, 5.0f, 1.0f };
    float h_mat_C[N_N] = { 0.0f };
    float *d_mat_A, *d_mat_B, *d_mat_C;
    
    printMatrix("Matrix A", h_mat_A);
    std::cout << "\n";
    printMatrix("Matrix B", h_mat_B);
    std::cout << "\n";
    
    _ = hipGetDeviceCount(&devices);
    
    for (int i = 0; i < devices; i++)
    {
        _ = hipSetDevice(i);
        
        hipblasCreate(&handle);
        
        _ = hipMalloc(&d_mat_A, sizeof(*d_mat_A) * N_N);
        _ = hipMalloc(&d_mat_B, sizeof(*d_mat_B) * N_N);
        _ = hipMalloc(&d_mat_C, sizeof(*d_mat_C) * N_N);
        
        _ = hipMemcpy(d_mat_A, h_mat_A, sizeof(*h_mat_A) * N_N, hipMemcpyHostToDevice);
        _ = hipMemcpy(d_mat_B, h_mat_B, sizeof(*h_mat_B) * N_N, hipMemcpyHostToDevice);
        _ = hipMemcpy(d_mat_C, h_mat_C, sizeof(*h_mat_C) * N_N, hipMemcpyHostToDevice);
        
        hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, N, N, N, &alpha, d_mat_A, N, d_mat_B, N, &beta, d_mat_C, N);
        
        _ = hipMemcpy(h_mat_C, d_mat_C, sizeof(*d_mat_C) * N_N, hipMemcpyDeviceToHost);
        
        _ = hipFree(d_mat_A);
        _ = hipFree(d_mat_B);
        _ = hipFree(d_mat_C);
        
        hipblasDestroy(handle);
    }
    
    printMatrix("Matrix A * B", h_mat_C);
    
    return 0;
}
