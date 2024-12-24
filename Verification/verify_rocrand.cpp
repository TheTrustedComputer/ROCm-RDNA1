#include <iostream>
#include <chrono>
#include <rocrand/rocrand.h>

int main(void)
{
    int devices, failed = 0;
    float *dev_out;
    hipError_t err;
    
    constexpr int N = 10;
    size_t dev_memsize = sizeof(*dev_out) * N;
    float host_out[N];
    
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
        
        if ((err = hipMalloc(&dev_out, dev_memsize)) == hipSuccess)
        {
            rocrand_generator rng_gen;
            
            if (rocrand_create_generator(&rng_gen, ROCRAND_RNG_PSEUDO_DEFAULT) != ROCRAND_STATUS_SUCCESS)
            {
                std::cerr << "Couldn't create random number generator on GPU" << i << std::endl;
                hipFree(dev_out);
                failed++;
                continue;
            }
            
            if (rocrand_set_seed(rng_gen, std::chrono::system_clock::now().time_since_epoch().count()) != ROCRAND_STATUS_SUCCESS)
            {
                std::cerr << "Couldn't set the seed on GPU" << i << std::endl;
                rocrand_destroy_generator(rng_gen);
                hipFree(dev_out);
                failed++;
                continue;
            }
            
            if (rocrand_generate_uniform(rng_gen, dev_out, N) != ROCRAND_STATUS_SUCCESS)
            {
                std::cerr << "Couldn't generate random numbers on GPU" << i << std::endl;
                rocrand_destroy_generator(rng_gen);
                hipFree(dev_out);
                failed++;
                continue;
            }
            
            if ((err = hipMemcpy(host_out, dev_out, dev_memsize, hipMemcpyDeviceToHost)) != hipSuccess)
            {
                std::cerr << "Couldn't copy random numbers from GPU" << i << ": " << hipGetErrorString(err) << std::endl;
                rocrand_destroy_generator(rng_gen);
                hipFree(dev_out);
                failed++;
                continue;
            }
            
            bool valid = true; // No zeros or NaNs
            
            for (float num : host_out)
            {
                if (num == 0.0f || std::isnan(num))
                {
                    std::cerr << "Failure: Detected a zero or NaN in GPU" << i << std::endl;
                    valid = false;
                    failed++;
                    break;
                }
            }
            
            if (valid)
            {
                std::cout << "GPU" << i << ": ";
                
                for (float num : host_out)
                {
                    std::cout << num << " ";
                }
                
                std::cout << std::endl;
            }
            
            rocrand_destroy_generator(rng_gen);
            hipFree(dev_out);
        }
        else
        {
            std::cerr << "Couldn't allocate memory to GPU " << i << ": " << hipGetErrorString(err) << std::endl;
            failed++;
            continue;
        }
    }
    
    return failed;
}
