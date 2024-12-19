#include <cstdlib>
#include <hip/hip_runtime.h>

__global__ 
void testHIP(void)
{
    printf("Success\n");
}

int main(void)
{
    hipError_t err;
    int devices;
    bool _;
    
    if ((_ = hipGetDeviceCount(&devices)) != hipSuccess)
    {
        printf("Couldn't find any HIP devices\n");
        return 1;
    }
    
    for (int i = 0; i < devices; i++)
    {
        _ = hipSetDevice(i);
        hipLaunchKernelGGL(testHIP, dim3(1), dim3(1), 0, 0);
    
        if ((err = hipDeviceSynchronize()) != hipSuccess)
        {
            printf("Failure: %s\n", hipGetErrorString(err));
            return 1;
        }
    }
    
    return 0;
}
