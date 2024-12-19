#include <iostream>
#include <cmath>
#include <hip/hip_runtime.h>
#include <rocfft/rocfft.h>

typedef struct rocfft_complex
{
    float x, y;
}
rocfft_complex;

void print_fft_result(const rocfft_complex* _RESULT, const int _SIZE, const float _RATE)
{
    for (int i = 0; i < _SIZE; ++i)
    {
        float freq = i * _RATE / (2.0f * _SIZE);
        float magnitude = sqrt(_RESULT[i].x * _RESULT[i].x + _RESULT[i].y * _RESULT[i].y);
        std::cout << freq << " Hz: " << magnitude << "\n";
    }
}

int main(void)
{
    bool _;
    int devices;
    
    constexpr int N = 99;
    constexpr int N2_P1 = N / 2 + 1;
    constexpr float SAMPLE_RATE = 2000.0f;
    
    float *d_signal;
    float h_signal[N];
    rocfft_complex* d_fft_result;
    rocfft_complex h_fft_result[N2_P1];
    rocfft_plan plan;
    size_t work_buffer_size;
    void *work_buffer;
    size_t lengths[1] = { static_cast<size_t>(N) };
   
    for (int i = 0; i < N; i++) // Sine wave at 500 Hz
    {
        h_signal[i] = sinf(2 * M_PI * 500.0f * i / SAMPLE_RATE);
    }
    
    _ = hipGetDeviceCount(&devices);
    
    for (int i = 0; i < devices; i++)
    {
        _ = hipMalloc(&d_signal, N * sizeof(*d_signal));
        _ = hipMalloc(&d_fft_result, N2_P1 * sizeof(*d_fft_result));
        
        _ = hipMemcpy(d_signal, h_signal, N * sizeof(*h_signal), hipMemcpyHostToDevice);
        
        rocfft_plan_create(&plan, rocfft_placement_notinplace, rocfft_transform_type_real_forward, rocfft_precision_single, 1, lengths, 1, nullptr);
        rocfft_plan_get_work_buffer_size(plan, &work_buffer_size);
        
        _ = hipMalloc(&work_buffer, work_buffer_size);
        
        void* in_buffer[1] = { d_signal };
        void* out_buffer[1] = { d_fft_result };
        rocfft_execute(plan, in_buffer, out_buffer, nullptr);
        
        _ = hipMemcpy(h_fft_result, d_fft_result, N2_P1 * sizeof(*h_fft_result), hipMemcpyDeviceToHost);
        
        _ = hipFree(d_signal);
        _ = hipFree(d_fft_result);
        _ = hipFree(work_buffer);
        
        rocfft_plan_destroy(plan);
    }
    
    print_fft_result(h_fft_result, N2_P1, SAMPLE_RATE);

    return 0;
}
