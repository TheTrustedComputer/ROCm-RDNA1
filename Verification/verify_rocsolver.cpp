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
    
    for (int i = 0; i < _SIZE; i++)
    {
        std::cout << _VECT[i] << " ";
    }
    
    std::cout << "]" << std::endl;
}

bool verify_vector(const float *_REF, const float *_COMP, const int _SIZE)
{
    for (int i = 0; i < _SIZE; i++)
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
    
    constexpr int N = 4;
    constexpr int N_N = N * N;
    
    float host_A[N_N] = { 2.0f, 11.0f, 23.0f, 41.0f, 3.0f, 13.0f, 29.0f, 43.0f, 5.0f, 17.0f, 31.0f, 47.0f, 7.0f, 19.0f, 37.0f, 53.0f };
    float host_B[N] = { -50.0f, -60.0f, -70.0f, -80.0f };
    float host_pivot[N] = { 0.0f };
    float pivot_ref[N] = { -12.0f / 11.0f, 163.0f / 11.0f, 63.0f / 22.0f, -335.0f / 22.0f };
    
    float *dev_A, *dev_B;
    int *dev_info, *dev_pivot;
    
    if ((err = hipGetDeviceCount(&devices)) != hipSuccess)
    {
        std::cerr << "Couldn't find any HIP devices: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    
    print_matrix("Matrix A", host_A, N);
    std::cout << std::endl;
    print_vector("Pivot Vector", host_B, N);
    
    for (int i = 0; i < devices; i++)
    {
        if ((err = hipSetDevice(i)) != hipSuccess)
        {
            std::cerr << "Couldn't set GPU" << i << ": " << hipGetErrorString(err) << std::endl;
            failed++;
            continue;
        }
    
        if ((status = rocblas_create_handle(&handle)) != rocblas_status_success)
        {
            std::cerr << "Couldn't create handle on GPU" << i << ": " << rocblas_status_to_string(status) << std::endl;
            failed++;
            continue;
        }
        
        if (hipMalloc(&dev_A, N_N * sizeof(*dev_A)) != hipSuccess ||
            hipMalloc(&dev_B, N * sizeof(*dev_B)) != hipSuccess ||
            hipMalloc(&dev_info, sizeof(*dev_info)) != hipSuccess ||
            hipMalloc(&dev_pivot, N * sizeof(*dev_pivot)) != hipSuccess)
        {
            std::cerr << "Couldn't allocate memory on GPU" << i << ": " << hipGetErrorString(hipGetLastError()) << std::endl;
            rocblas_destroy_handle(handle);
            failed++;
            continue;
        }
        
        hipMemcpy(dev_A, host_A, N_N * sizeof(*host_A), hipMemcpyHostToDevice);
        hipMemcpy(dev_B, host_B, N * sizeof(*host_B), hipMemcpyHostToDevice);
        
        if (rocsolver_sgetrf(handle, N, N, dev_A, N, dev_pivot, dev_info) != rocblas_status_success)
        {
            std::cerr << "Couldn't perform LU factorization on GPU" << i << std::endl;
            
            hipFree(dev_A);
            hipFree(dev_B);
            hipFree(dev_info);
            hipFree(dev_pivot);
            rocblas_destroy_handle(handle);
            failed++;
            continue;
        }

        if (rocsolver_sgetrs(handle, rocblas_operation_none, N, 1, dev_A, N, dev_pivot, dev_B, N) != rocblas_status_success)
        {
            std::cerr << "Couldn't compute solution on GPU" << i << std::endl;
            
            hipFree(dev_A);
            hipFree(dev_B);
            hipFree(dev_info);
            hipFree(dev_pivot);
            rocblas_destroy_handle(handle);
            failed++;
            continue;
        }
        
        hipDeviceSynchronize();
        hipMemcpy(host_pivot, dev_B, N * sizeof(float), hipMemcpyDeviceToHost);
        
        if (verify_vector(pivot_ref, host_pivot, N))
        {
            std::cout << "Solution Vector on GPU" << i;
            print_vector("", host_pivot, N);
        }
        else
        {
            failed++;
        }
        
        hipFree(dev_A);
        hipFree(dev_B);
        hipFree(dev_info);
        hipFree(dev_pivot);
        
        rocblas_destroy_handle(handle);
    }
    
    return failed;
}
