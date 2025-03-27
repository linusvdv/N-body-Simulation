#include <cuda.h>
#include <stdlib.h>
#include <iostream>


__global__ void Square(const int *cuda_input, int *cuda_output, int num_elements) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int sum = 0;
    for (int i = 0; i < num_elements; i++) {
        sum += (cuda_input[i] - cuda_input[index]) * (cuda_input[i] - cuda_input[index]);
    }
    cuda_output[index] = sum;
}


int main() {
    const int num_elements = 128*1024;
    int* cuda_input = (int *)malloc(sizeof(int)*num_elements);
    int* cuda_output = (int *)malloc(sizeof(int)*num_elements);
    int* d_cuda_input;
    int* d_cuda_output;
    srand(0);
    std::cout << "CPU: ";
    for (int i = 0; i < num_elements; i++) {
        cuda_input[i] = rand()%100;
//        std::cout << cuda_input[i] << " ";
    }
    std::cout << "\n";

    cudaMalloc((void **)&d_cuda_input, sizeof(int)*num_elements);
    cudaMalloc((void **)&d_cuda_output, sizeof(int)*num_elements);

    cudaMemcpy(d_cuda_input, cuda_input, sizeof(int)*num_elements, cudaMemcpyHostToDevice);

    Square<<<num_elements/1024, 1024>>>(d_cuda_input, d_cuda_output, num_elements);
    cudaDeviceSynchronize();

    cudaMemcpy(cuda_output, d_cuda_output, sizeof(int)*num_elements, cudaMemcpyDeviceToHost);

    std::cout << "CPU: ";
    for (int i = 0; i < num_elements; i++) {
//        std::cout << cuda_output[i] << " ";
    }
    std::cout << "\n";

    cudaFree(d_cuda_input);
    cudaFree(d_cuda_output);
}
