#include <iostream>
#include <rocblas/rocblas.h>

constexpr int N = 4;
constexpr int N_N = N * N;

void print_matrix(const char *_NAME, const float *_MATR)
{
    if (_NAME[0])
    {
        std::cout << _NAME << std::endl;
    }
    
    for (int i = 0; i < N; i++)
    {
        std::cout << "[ ";
        
        for (int j = 0; j < N; j++)
        {
            // Column-major over row-major
            std::cout << _MATR[j * N + i] << " ";
        }
        
        std::cout << "]\n";
    }
}

bool verify_matrix(const float *_REF, const float *_COMP)
{
    for (int i = 0; i < N_N; i++)
    {
        if (std::isnan(_COMP[i]))
        {
            std::cerr << "Failure: NaN at index " << i << std::endl;
            return false;
        }
        
        if (std::abs(_REF[i] - _COMP[i]) > 1e-5f)
        {
            std::cerr << "Failure: Mismatch at index " << i << "; expected " << _REF[i] << ", received " << _COMP[i] << std::endl;
            return false;
        }
    }
    
    return true;
}

int main(void)
{
    hipError_t err;
    int devices, failed = 0;
    rocblas_handle handle;
    rocblas_status status;
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    float host_mat_A[N_N] = { 1.0f, 5.0f, 9.0f, 13.0f, 2.0f, 6.0f, 10.0f, 14.0f, 3.0f, 7.0f, 11.0f, 15.0f, 4.0f, 8.0f, 12.0f, 16.0f };
    float host_mat_B[N_N] = { 16.0f, 12.0f, 8.0f, 4.0f, 15.0f, 11.0f, 7.0f, 3.0f, 14.0f, 10.0f, 6.0f, 2.0f, 13.0f, 9.0f, 5.0f, 1.0f };
    float host_mat_C[N_N] = { 0.0f }; // General matrix multiplication of A and B
    float matmul_ref[N_N] = { 80.0f, 240.0f, 400.0f, 560.0f, 70.0f, 214.0f, 358.0f, 502.0f, 60.0f, 188.0f, 316.0f, 444.0f, 50.0f, 162.0f, 274.0f, 386.0f };
    float *dev_mat_A, *dev_mat_B, *dev_mat_C;
    
    if ((err = hipGetDeviceCount(&devices)) != hipSuccess)
    {
        std::cerr << "Couldn't find any HIP devices: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    
    print_matrix("Matrix A", host_mat_A);
    std::cout << "\n";
    print_matrix("Matrix B", host_mat_B);
    std::cout << "\n";
    
    for (int i = 0; i < devices; i++)
    {
        if ((err = hipSetDevice(i)) != hipSuccess)
        {
            std::cerr << "Couldn't set GPU" << i << ": " << hipGetErrorString(err) << std::endl;
            failed++;
            continue;
        }
        
        std::cout << "Matrix AB on GPU" << i << std::endl;
        
        if ((status = rocblas_create_handle(&handle)) != rocblas_status_success)
        {
            std::cerr << "Couldn't create handle on GPU" << i << ": " << rocblas_status_to_string(status) << std::endl;
            failed++;
            continue;
        }
        
        if (hipMalloc(&dev_mat_A, sizeof(*dev_mat_A) * N_N) != hipSuccess ||
            hipMalloc(&dev_mat_B, sizeof(*dev_mat_B) * N_N) != hipSuccess ||
            hipMalloc(&dev_mat_C, sizeof(*dev_mat_C) * N_N) != hipSuccess)
        {
            std::cerr << "Couldn't allocate matrices on GPU" << i << ": " << hipGetErrorString(hipGetLastError()) << std::endl;
            rocblas_destroy_handle(handle);
            failed++;
            continue;
        }
        
        hipMemcpy(dev_mat_A, host_mat_A, sizeof(*host_mat_A) * N_N, hipMemcpyHostToDevice);
        hipMemcpy(dev_mat_B, host_mat_B, sizeof(*host_mat_B) * N_N, hipMemcpyHostToDevice);
        hipMemcpy(dev_mat_C, host_mat_C, sizeof(*host_mat_C) * N_N, hipMemcpyHostToDevice);
        
        if ((status = rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, N, N, N, &alpha, dev_mat_A, N, dev_mat_B, N, &beta, dev_mat_C, N)) != rocblas_status_success)
        {
            std::cerr << "Couldn't run GEMM on GPU" << i << ": " << rocblas_status_to_string(status) << std::endl;
            
            hipFree(dev_mat_A);
            hipFree(dev_mat_B);
            hipFree(dev_mat_C);
            rocblas_destroy_handle(handle);
            failed++;
            continue;
        }
        
        hipDeviceSynchronize();
        hipMemcpy(host_mat_C, dev_mat_C, sizeof(*dev_mat_C) * N_N, hipMemcpyDeviceToHost);
        
        if (verify_matrix(matmul_ref, host_mat_C))
        {
            print_matrix("", host_mat_C);
            
            if (i != devices - 1)
            {
                std::cout << std::endl;
            }
        }
        else
        {
            failed++;
        }
        
        hipFree(dev_mat_A);
        hipFree(dev_mat_B);
        hipFree(dev_mat_C);
        rocblas_destroy_handle(handle);
    }
    
    return failed;
}
