#include <stdio.h>
#include <cuda.h>


__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    cuda_hello<<<2,2>>>(); 
    cudaDeviceSynchronize();
    return 0;
}
