#include <iostream>
#include <rocsparse/rocsparse.h>

void print_vector(const char *_STR, const float *_VEC, const int _SIZE)
{
    std::cout << _STR << "[ ";
    
    for (int i = 0; i < _SIZE; i++)
    {
        std::cout << _VEC[i] << " ";
    }
    
    std::cout << "]\n";
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

void print_sparse_matrix(const char *_STR, const float *_VALS, const int *_COLS, const int *_ROW_OFFSETS, const int _M, const int _N)
{
    std::cout << _STR << "\n";
    
    for (int i = 0; i < _M; i++)
    {
        int row_start = _ROW_OFFSETS[i];
        int row_end = _ROW_OFFSETS[i + 1];
        int val_index = 0;
        
        std::cout << "[ ";
        
        for (int j = 0; j < _N; j++)
        {
            if (val_index < (row_end - row_start) && _COLS[row_start + val_index] == j)
            {
                std::cout << _VALS[row_start + val_index] << " ";
                val_index++;
            }
            else
            {
                std::cout << 0.0f << " ";
            }
        }
        
        std::cout << "]\n";
    }
}

int main(void)
{
    hipError_t err;
    int devices, failed = 0;
    rocsparse_handle handle;
    rocsparse_mat_descr descr;
    
    constexpr int M = 4;
    constexpr int N = 4;
    constexpr int NNZ = 8;
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    float host_A_vals[NNZ] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
    float host_x[N] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float host_y[M] = { 0.0f }; // Sparse matrix-vector multiplication of A and x
    float spmv_ref[M] = { 7.0f, 19.0f, 28.0f, 46.0f };
    
    int host_A_cols[NNZ] = { 0, 2, 0, 3, 1, 2, 1, 3 };
    int host_A_row_offsets[M + 1] = { 0, 2, 4, 6, 8 };
    
    float *dev_A_vals, *dev_x, *dev_y;
    int *dev_A_cols, *dev_A_row_offsets;
    
    if ((err = hipGetDeviceCount(&devices)) != hipSuccess)
    {
        std::cerr << "Couldn't find any HIP devices: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    
    print_sparse_matrix("Sparse Matrix A", host_A_vals, host_A_cols, host_A_row_offsets, M, N);
    std::cout << std::endl;
    print_vector("Vector X ", host_x, N);
    
    for (int i = 0; i < devices; i++)
    {
        if ((err = hipSetDevice(i)) != hipSuccess)
        {
            std::cerr << "Couldn't set GPU" << i << ": " << hipGetErrorString(err) << std::endl;
            failed++;
            continue;
        }
    
        if (rocsparse_create_handle(&handle) != rocsparse_status_success)
        {
            std::cerr << "Couldn't create handle on GPU" << i << "." << std::endl; 
            failed++;
            continue;
        }
        
        if (hipMalloc(&dev_A_vals, NNZ * sizeof(*dev_A_vals)) != hipSuccess ||
            hipMalloc(&dev_A_cols, NNZ * sizeof(*dev_A_cols)) != hipSuccess ||
            hipMalloc(&dev_A_row_offsets, (M + 1) * sizeof(*dev_A_row_offsets)) != hipSuccess ||
            hipMalloc(&dev_x, N * sizeof(*dev_x)) != hipSuccess ||
            hipMalloc(&dev_y, M * sizeof(*dev_y)) != hipSuccess)
        {
            std::cerr << "Couldn't allocate sparse matrices on GPU" << i << std::endl;
            rocsparse_destroy_handle(handle);
            failed++;
            continue;
        }
        
        hipMemcpy(dev_A_vals, host_A_vals, NNZ * sizeof(*dev_A_vals), hipMemcpyHostToDevice);
        hipMemcpy(dev_A_cols, host_A_cols, NNZ * sizeof(*dev_A_cols), hipMemcpyHostToDevice);
        hipMemcpy(dev_A_row_offsets, host_A_row_offsets, (M + 1) * sizeof(*dev_A_row_offsets), hipMemcpyHostToDevice);
        hipMemcpy(dev_x, host_x, N * sizeof(*dev_x), hipMemcpyHostToDevice);
        hipMemcpy(dev_y, host_y, M * sizeof(*dev_y), hipMemcpyHostToDevice);
        
        rocsparse_create_mat_descr(&descr);
        
        if (rocsparse_scsrmv(handle, rocsparse_operation_none, M, N, NNZ, &alpha, descr, dev_A_vals, dev_A_row_offsets, dev_A_cols, nullptr, dev_x, &beta, dev_y) != rocsparse_status_success)
        {
            std::cerr << "Couldn't perform SpMV on GPU" << i << std::endl;
            
            hipFree(dev_A_vals);
            hipFree(dev_A_cols);
            hipFree(dev_A_row_offsets);
            hipFree(dev_x);
            hipFree(dev_y);
            rocsparse_destroy_mat_descr(descr);
            rocsparse_destroy_handle(handle); 
            failed++;
            continue;
        }
        
        hipDeviceSynchronize();
        hipMemcpy(host_y, dev_y, M * sizeof(float), hipMemcpyDeviceToHost);
        
        if (verify_vector(spmv_ref, host_y, M))
        {
            std::cout << "Vector Y on GPU" << i << " ";
            print_vector("", host_y, M);
        }
        else
        {
            failed++;
        }
        
        hipFree(dev_A_vals);
        hipFree(dev_A_cols);
        hipFree(dev_A_row_offsets);
        hipFree(dev_x);
        hipFree(dev_y);

        rocsparse_destroy_mat_descr(descr);
        rocsparse_destroy_handle(handle);
    }

    return failed;
}
