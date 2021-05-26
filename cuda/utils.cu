#include "utils.cuh"

#include <stddef.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <chrono>
#include <thread>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            exit(code);
        } 
    }
}
