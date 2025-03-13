#include <stdio.h>
#include <cuda.h>


__global__ void CudaHello(){
    printf("Hello World from GPU!\n");
}


int main() {
    CudaHello<<<2,2>>>(); 
    cudaDeviceSynchronize();
    return 0;
}
