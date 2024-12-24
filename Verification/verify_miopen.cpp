#include <iostream>
#include <miopen/miopen.h>

int main(void)
{
    constexpr int n = 1; // Batch size
    constexpr int c = 1; // Channels
    constexpr int h = 5; // Height
    constexpr int w = 5; // Width

    // Input data
    float input_data[n * c * h * w] = \
    {
        1, 2, 3, 4, 5,
        1, 2, 3, 4, 5,
        1, 2, 3, 4, 5,
        1, 2, 3, 4, 5,
        1, 2, 3, 4, 5
    };

    // Kernel data
    constexpr int k = 1; // Filters
    constexpr int kh = 3; // Kernel height
    constexpr int kw = 3; // Kernel width
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    float kernel_data[k * c * kh * kw] = \
    {
        1, 0, -1,
        1, 0, -1,
        1, 0, -1
    };
    
    float *d_input;
    float *d_output;
    float *d_kernel;
    
    miopenHandle_t handle;
    miopenTensorDescriptor_t input_desc;
    miopenTensorDescriptor_t output_desc;
    miopenTensorDescriptor_t filter_desc;
    miopenConvolutionDescriptor_t conv_desc;
    miopenConvAlgoPerf_t perf_results;
    int ret_algo_count;
    size_t workspace_size;
    void *d_workspace;
    
    hipError_t err;
    int devices, failed = 0;
    
    if ((err = hipGetDeviceCount(&devices)) != hipSuccess)
    {
        std::cerr << "Couldn't find any HIP devices: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    
    for (int i = 0; i < devices; i++)
    {
        if ((err = hipSetDevice(i)) != hipSuccess)
        {
            std::cerr << "Couldn't set GPU" << i << ": " << hipGetErrorString(err) << std::endl;
            failed++;
            continue;
        }
    }
    
    hipMalloc(&d_input, n * c * h * w * sizeof(*d_input));
    hipMalloc(&d_output, n * k * (h - kh + 1) * (w - kw + 1) * sizeof(*d_output));
    hipMalloc(&d_kernel, k * c * kh * kw * sizeof(*d_kernel));

    hipMemcpy(d_input, input_data, n * c * h * w * sizeof(*input_data), hipMemcpyHostToDevice);
    hipMemcpy(d_kernel, kernel_data, k * c * kh * kw * sizeof(*kernel_data), hipMemcpyHostToDevice);
    
    miopenCreate(&handle);
    
    miopenCreateTensorDescriptor(&input_desc);
    miopenSet4dTensorDescriptor(input_desc, miopenFloat, n, c, h, w);
    
    miopenCreateTensorDescriptor(&output_desc);
    miopenSet4dTensorDescriptor(output_desc, miopenFloat, n, k, h - kh + 1, w - kw + 1);
    
    miopenCreateTensorDescriptor(&filter_desc);
    miopenSet4dTensorDescriptor(filter_desc, miopenFloat, k, c, kh, kw);
    
    miopenCreateConvolutionDescriptor(&conv_desc);
    miopenInitConvolutionDescriptor(conv_desc, miopenConvolution, 0, 0, 1, 1, 1, 1);
    
    miopenConvolutionForwardGetWorkSpaceSize(handle, filter_desc, input_desc, conv_desc, output_desc, &workspace_size);
    
    hipMalloc(&d_workspace, workspace_size);

    miopenFindConvolutionForwardAlgorithm(handle, input_desc, d_input, filter_desc, d_kernel, conv_desc, output_desc, d_output, 1, &ret_algo_count, &perf_results, d_workspace, workspace_size, false);
    
    miopenConvolutionForward(handle, &alpha, input_desc, d_input, filter_desc, d_kernel, conv_desc, perf_results.fwd_algo, &beta, output_desc, d_output, d_workspace, workspace_size);

    // Copy output back to host and print
    float output_data[n * k * (h - kh + 1) * (w - kw + 1)];
    hipMemcpy(output_data, d_output, n * k * (h - kh + 1) * (w - kw + 1) * sizeof(*d_output), hipMemcpyDeviceToHost);
    
    for (int i = 0; i < n * k * (h - kh + 1) * (w - kw + 1); i++)
    {
        std::cout << output_data[i] << " ";
        
        if (!((i + 1) % (w - kw + 1)))
        {
            std::cout << std::endl;
        }
    }

    miopenDestroyTensorDescriptor(input_desc);
    miopenDestroyTensorDescriptor(output_desc);
    miopenDestroyTensorDescriptor(filter_desc);
    miopenDestroyConvolutionDescriptor(conv_desc);
    miopenDestroy(handle);
    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_kernel);
    hipFree(d_workspace);

    return failed;
}
