#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <driver_types.h>
#include <stdlib.h>
#include <iostream>

__device__ int square_fun(int a) {
    return a*a;
}

__global__ void Square(int *cuda_input, int *cuda_output) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    cuda_output[index] = square_fun(cuda_input[index]);
}


int main() {
    const int num_elements = 10;
    int cuda_input[num_elements];
    int cuda_output[num_elements];
    int* d_cuda_input;
    int* d_cuda_output;
    srand(0);
    std::cout << "CPU: ";
    for (int i = 0; i < num_elements; i++) {
        cuda_input[i] = rand()%1000;
        std::cout << cuda_input[i] << " ";
    }
    std::cout << "\n";

    cudaMalloc((void **)&d_cuda_input, sizeof(int)*num_elements);
    cudaMalloc((void **)&d_cuda_output, sizeof(int)*num_elements);

    cudaMemcpy(d_cuda_input, &cuda_input, sizeof(int)*num_elements, cudaMemcpyHostToDevice);

    Square<<<1, num_elements>>>(d_cuda_input, d_cuda_output);
    cudaDeviceSynchronize();

    cudaMemcpy(cuda_output, d_cuda_output, sizeof(int)*num_elements, cudaMemcpyDeviceToHost);

    std::cout << "CPU: ";
    for (int i = 0; i < num_elements; i++) {
        std::cout << cuda_output[i] << " ";
    }
    std::cout << "\n";

    cudaFree(d_cuda_input);
    cudaFree(d_cuda_output);
}
