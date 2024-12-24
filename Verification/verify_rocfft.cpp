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
    for (int i = 0; i < _SIZE; i++)
    {
        float freq = i * _RATE / (2.0f * _SIZE);
        float magnitude = sqrt(_RESULT[i].x * _RESULT[i].x + _RESULT[i].y * _RESULT[i].y);
        std::cout << freq << " Hz: " << magnitude << "\n";
    }
}

bool verify_fft_result(const rocfft_complex* _RESULT, const int _SIZE, const float _RATE, const float _FREQ, const float _TOLERANCE = 1.0f)
{
    int peak_index = 0;
    float max_magnitude = 0.0f;
    
    for (int i = 0; i < _SIZE; i++)
    {
        float magnitude = sqrtf(_RESULT[i].x * _RESULT[i].x + _RESULT[i].y * _RESULT[i].y);
        
        if (magnitude > max_magnitude)
        {
            max_magnitude = magnitude;
            peak_index = i;
        }
    }
    
    float peak_frequency = peak_index * _RATE / (2.0f * _SIZE);
    
    return std::fabs(peak_frequency - _FREQ) <= _TOLERANCE;
}

int main(void)
{
    hipError_t err;
    int devices, failed = 0;
    
    constexpr int N = 49;
    constexpr int N2_P1 = N / 2 + 1;
    constexpr float SAMPLE_RATE = 500.0f;
    constexpr float FREQUENCY = 120.0f;
    
    float *dev_signal;
    float host_signal[N];
    rocfft_complex* dev_fft_result;
    rocfft_complex host_fft_result[N2_P1];
    rocfft_plan plan;
    rocfft_execution_info exec_info;
    size_t lengths[1] = { static_cast<size_t>(N) };
    size_t work_buffer_size;
    void *work_buffer;
    
    if ((err = hipGetDeviceCount(&devices)) != hipSuccess)
    {
        std::cerr << "Couldn't find any HIP devices: " << hipGetErrorString(err) << std::endl;
        return -1;
    }
    
    for (int i = 0; i < N; i++)
    {
        host_signal[i] = sinf(2 * M_PI * FREQUENCY * i / SAMPLE_RATE);
    }
    
    for (int i = 0; i < devices; i++)
    {
        if ((err = hipSetDevice(i)) != hipSuccess)
        {
            std::cerr << "Couldn't set GPU" << i << ": " << hipGetErrorString(err) << std::endl;
            failed++;
            continue;
        }
        
        std::cout << "Sine wave at 120 Hz on GPU" << i << std::endl;
        
        if (hipMalloc(&dev_signal, N * sizeof(*dev_signal)) != hipSuccess ||
            hipMalloc(&dev_fft_result, N2_P1 * sizeof(*dev_fft_result)) != hipSuccess)
        {
            std::cerr << "Couldn't allocate memory on GPU" << i << std::endl;
            failed++;
            continue;
        }
        
        hipMemcpy(dev_signal, host_signal, N * sizeof(*host_signal), hipMemcpyHostToDevice);
        
        if (rocfft_plan_create(&plan, rocfft_placement_notinplace, rocfft_transform_type_real_forward, rocfft_precision_single, 1, lengths, 1, nullptr) != rocfft_status_success)
        {
            std::cerr << "Couldn't create plan on GPU" << i << std::endl;
            hipFree(dev_signal);
            hipFree(dev_fft_result);
            failed++;
            continue;
        }
        
        if (rocfft_plan_get_work_buffer_size(plan, &work_buffer_size) != rocfft_status_success)
        {
            std::cerr << "Couldn't get work buffer size on GPU" << i << std::endl;
            rocfft_plan_destroy(plan);
            hipFree(dev_signal);
            hipFree(dev_fft_result);
            failed++;
            continue;
        }
        
        if (hipMalloc(&work_buffer, work_buffer_size) != hipSuccess)
        {
            std::cerr << "Couldn't allocate work buffer on GPU" << i << std::endl;
            rocfft_plan_destroy(plan);
            hipFree(dev_signal);
            hipFree(dev_fft_result);
            failed++;
            continue;
        }
        
        if (rocfft_execution_info_create(&exec_info) != rocfft_status_success)
        {
            std::cerr << "Couldn't create execution info on GPU" << i << std::endl;
            hipFree(dev_signal);
            hipFree(dev_fft_result);
            hipFree(work_buffer);
            rocfft_plan_destroy(plan);
            failed++;
            continue;
        }
        
        if (rocfft_execution_info_set_work_buffer(exec_info, work_buffer, work_buffer_size) != rocfft_status_success)
        {
            std::cerr << "Couldn't set work buffer on GPU " << i << std::endl;
            rocfft_execution_info_destroy(exec_info);
            hipFree(dev_signal);
            hipFree(dev_fft_result);
            hipFree(work_buffer);
            rocfft_plan_destroy(plan);
            failed++;
            continue;
        }
        
        void *in_buffer[1] = { dev_signal };
        void *out_buffer[1] = { dev_fft_result };
        
        if (rocfft_execute(plan, in_buffer, out_buffer, exec_info) != rocfft_status_success)
        {
            std::cerr << "Couldn't execute plan on GPU" << i << std::endl;
            hipFree(dev_signal);
            hipFree(dev_fft_result);
            hipFree(work_buffer);
            rocfft_plan_destroy(plan);
            failed++;
            continue;
        }
        
        hipDeviceSynchronize();
        hipMemcpy(host_fft_result, dev_fft_result, N2_P1 * sizeof(*host_fft_result), hipMemcpyDeviceToHost);
        
        if (verify_fft_result(host_fft_result, N2_P1, SAMPLE_RATE, FREQUENCY))
        {
            print_fft_result(host_fft_result, N2_P1, SAMPLE_RATE);
        }
        else
        {
            float actual_max_freq = 0.0f;
            float actual_max_mag = 0.0f;
            
            std::cerr << "Failure: Peak frequency mismatch on GPU" << i << "; expected " << FREQUENCY << " Hz, received ";
            
            for (int j = 0; j < N2_P1; j++)
            {
                float actual_freq = j * SAMPLE_RATE / (2.0f * N2_P1);
                float actual_mag = sqrtf(host_fft_result[j].x * host_fft_result[j].x + host_fft_result[j].y * host_fft_result[j].y);
                
                if (actual_mag > actual_max_mag)
                {
                    actual_max_freq = actual_freq;
                    actual_max_mag = actual_mag;
                }
                
            }
            
            std::cerr << actual_max_freq << " Hz" << std::endl;
            failed++;
        }
        
        if (i != devices - 1)
        {
           std::cout << std::endl;
        }
        
        hipFree(dev_signal);
        hipFree(dev_fft_result);
        hipFree(work_buffer);
        rocfft_plan_destroy(plan);
    }
    
    return failed;
}
