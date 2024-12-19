#include <iostream>
#include <chrono>
#include <rocrand/rocrand.h>

int main(void)
{
    int devices;
    float *d_out;
    bool _;
    
    constexpr int ELEMS = 10;
    size_t size = sizeof(*d_out) * ELEMS;
    float host_out[ELEMS];
    
    _ = hipGetDeviceCount(&devices);
    
    for (int i = 0; i < devices; i++)
    {
        _ = hipSetDevice(i);
        
        if ((_ = hipMalloc(&d_out, size)) == hipSuccess)
        {
            rocrand_generator rng_gen;
            
            rocrand_create_generator(&rng_gen, ROCRAND_RNG_PSEUDO_DEFAULT);
            rocrand_set_seed(rng_gen, std::chrono::system_clock::now().time_since_epoch().count());
            rocrand_generate_uniform(rng_gen, d_out, ELEMS);
            
            _ = hipMemcpy(host_out, d_out, size, hipMemcpyDeviceToHost);
            
            for (float i : host_out)
            {
                std::cout << i << " ";
            }
            
            std::cout << "\n";
            
            rocrand_destroy_generator(rng_gen);
            _ = hipFree(d_out);
        }
    }
    
    return 0;
}
