#include <cstdlib>
#include <hip/hip_runtime.h>

__global__ 
void testHIP(const int _ID)
{
    printf("GPU%d: Success\n", _ID);
}

int main(void)
{
    hipError_t err;
    int devices, failed = 0;
    
    if ((err = hipGetDeviceCount(&devices)) != hipSuccess)
    {
        printf("Couldn't find any HIP devices: %s\n", hipGetErrorString(err));
        return -1;
    }
    
    for (int i = 0; i < devices; i++)
    {
        if ((err = hipSetDevice(i)) != hipSuccess)
        {
            printf("Couldn't set GPU %d: %s\n", i, hipGetErrorString(err));    
            failed++;
            continue;
        }
        
        hipLaunchKernelGGL(testHIP, dim3(1), dim3(1), 0, 0, i);
        
        if ((err = hipGetLastError()) != hipSuccess)
        {
            printf("Couldn't launch kernel on GPU %d: %s\n", i, hipGetErrorString(err));
            failed++;
            continue;
        }
    
        if ((err = hipDeviceSynchronize()) != hipSuccess)
        {
            printf("Couldn't synchronize GPU %d: %s\n", i, hipGetErrorString(err));
            failed++;
            continue;
        }
    }
    
    if (failed)
    {
        printf("Failure: %s\n", hipGetErrorString(hipGetLastError()));
    }
    
    return failed;
}
