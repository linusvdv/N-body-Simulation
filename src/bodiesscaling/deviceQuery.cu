#include <cuda_runtime.h>
#include <iostream>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device name: " << prop.name << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << "\n";
    std::cout << "Shared mem per block: " << prop.sharedMemPerBlock << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    return 0;
}

