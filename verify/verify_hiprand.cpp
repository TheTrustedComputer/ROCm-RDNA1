#include <iostream>
#include <chrono>
#include <hiprand/hiprand.hpp>

int main(void)
{
    int devices;
    unsigned *d_out;
    bool _;
    
    constexpr int ELEMS = 10;
    size_t size = sizeof(*d_out) * ELEMS;
    float h_out[ELEMS];
    
    _ = hipGetDeviceCount(&devices);
    
    for (int i = 0; i < devices; i++)
    {
        _ = hipSetDevice(i);
        
        if ((_ = hipMalloc(&d_out, size)) == hipSuccess)
        {
            hiprandGenerator_t reg_gen;
            
            hiprandCreateGenerator(&reg_gen, HIPRAND_RNG_PSEUDO_DEFAULT);
            hiprandSetPseudoRandomGeneratorSeed(reg_gen, std::chrono::system_clock::now().time_since_epoch().count());
            hiprandGenerate(reg_gen, d_out, ELEMS);
            
            _ = hipMemcpy(h_out, d_out, size, hipMemcpyDeviceToHost);
            
            for (float i : h_out)
            {
                std::cout << i << " ";
            }
            
            std::cout << "\n";
            
            hiprandDestroyGenerator(reg_gen); 
            _ = hipFree(d_out);
        }
    }
    
    return 0;
}
