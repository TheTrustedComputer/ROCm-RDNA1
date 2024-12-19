#include <iostream>
#include <cmath>
#include <hipfft/hipfft.h>

void printFFTResult(const hipfftComplex *_RESULT, const int _SIZE, const float _RATE)
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
    hipfftComplex *d_fft_result;
    hipfftComplex h_fft_result[N2_P1];
    hipfftHandle plan;
    
    for (int i = 0; i < N; i++) // Sine wave at 500 Hz
    {
        h_signal[i] = sinf(2.0f * M_PI * 500.0f * i / SAMPLE_RATE);
    }
    
    _ = hipGetDeviceCount(&devices);
    
    for (int i = 0; i < devices; i++)
    {
        _ = hipSetDevice(i);
        
        _ = hipMalloc(&d_signal, N * sizeof(*d_signal));
        _ = hipMalloc(&d_fft_result, N2_P1 * sizeof(hipfftComplex));
        
        _ = hipMemcpy(d_signal, h_signal, N * sizeof(*h_signal), hipMemcpyHostToDevice);
        
        hipfftPlan1d(&plan, N, HIPFFT_R2C, 1);
        hipfftExecR2C(plan, d_signal, d_fft_result);
        
        _ = hipMemcpy(h_fft_result, d_fft_result, N2_P1 * sizeof(*h_fft_result), hipMemcpyDeviceToHost);
        
        hipfftDestroy(plan);
        _ = hipFree(d_signal);
        _ = hipFree(d_fft_result);
    }
    
    printFFTResult(h_fft_result, N2_P1, SAMPLE_RATE);
    
    return 0;
}
