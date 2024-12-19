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
    bool _;
    int devices;
    rocsparse_handle handle;
    rocsparse_mat_descr descr;
    
    constexpr int M = 4;
    constexpr int N = 4;
    constexpr int NNZ = 8;
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    float h_A_vals[NNZ] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
    float h_x[N] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float h_y[M] = { 0.0f };
    
    int h_A_cols[NNZ] = { 0, 2, 0, 3, 1, 2, 1, 3 };
    int h_A_row_offsets[M + 1] = { 0, 2, 4, 6, 8 };
    
    float *d_A_vals, *d_x, *d_y;
    int *d_A_cols, *d_A_row_offsets;
    
    print_sparse_matrix("Sparse Matrix A:", h_A_vals, h_A_cols, h_A_row_offsets, M, N);
    std::cout << std::endl;
    print_vector("Vector x: ", h_x, N);
    
    _ = hipGetDeviceCount(&devices);
    
    for (int i = 0; i < devices; i++)
    {
        _ = hipSetDevice(i);
    
        rocsparse_create_handle(&handle);
        
        _ = hipMalloc(&d_A_vals, NNZ * sizeof(*d_A_vals));
        _ = hipMalloc(&d_A_cols, NNZ * sizeof(*d_A_cols));
        _ = hipMalloc(&d_A_row_offsets, (M + 1) * sizeof(*d_A_row_offsets));
        _ = hipMalloc(&d_x, N * sizeof(*d_x));
        _ = hipMalloc(&d_y, M * sizeof(*d_y));
        
        _ = hipMemcpy(d_A_vals, h_A_vals, NNZ * sizeof(*d_A_vals), hipMemcpyHostToDevice);
        _ = hipMemcpy(d_A_cols, h_A_cols, NNZ * sizeof(*d_A_cols), hipMemcpyHostToDevice);
        _ = hipMemcpy(d_A_row_offsets, h_A_row_offsets, (M + 1) * sizeof(*d_A_row_offsets), hipMemcpyHostToDevice);
        _ = hipMemcpy(d_x, h_x, N * sizeof(*d_x), hipMemcpyHostToDevice);
        _ = hipMemcpy(d_y, h_y, M * sizeof(*d_y), hipMemcpyHostToDevice);
        
        rocsparse_create_mat_descr(&descr);
        rocsparse_scsrmv(handle, rocsparse_operation_none, M, N, NNZ, &alpha, descr, d_A_vals, d_A_row_offsets, d_A_cols, nullptr, d_x, &beta, d_y);
        
        _ = hipMemcpy(h_y, d_y, M * sizeof(float), hipMemcpyDeviceToHost);

        _ = hipFree(d_A_vals);
        _ = hipFree(d_A_cols);
        _ = hipFree(d_A_row_offsets);
        _ = hipFree(d_x);
        _ = hipFree(d_y);

        rocsparse_destroy_mat_descr(descr);
        rocsparse_destroy_handle(handle);
    }
    
    print_vector("Vector y: ", h_y, M);

    return 0;
}
