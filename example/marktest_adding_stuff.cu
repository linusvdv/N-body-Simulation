#include <stdio.h>
#include <cuda.h>
#include <iostream>
#define int long long
int const n = 32;
__device__ int sz = n;
#define IDX(i, j, N) ((i) * (N) + (j))

__global__ void matrix_multi(int *a, int *b, int *c, int times){

    int row = blockIdx.x;
    int col = threadIdx.x;

    int val = c[IDX(row, col, sz)];
    for(int j = 0; j<times; j++){
    for(int i = 0; i<sz; i++){
        val += a[IDX(row, i, sz)] * b[IDX(i, col, sz)];
    }
    }
    c[IDX(row, col, sz)] = val;
}

signed main(){
    int a[n*n], b[n*n], c[n*n];
    int *d_a, *d_b, *d_c;
    int size = n * n * sizeof(int);
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    for(int i = 0; i<n; i++){
        for(int j = 0; j<n; j++){
            a[IDX(i, j, n)] = 1;
            b[IDX(i, j, n)] = 1;
            c[IDX(i, j, n)] = 3;
        }
    }
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice);

    for(int i = 0 ; i<1; i++){
        matrix_multi<<<n,n>>>(d_a, d_b, d_c, 10000000);
        //cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    long long res = 0;
    for(int i = 0; i < n; i++){
        for(int j = 0; j<n; j++)res += c[IDX(i, j, n)];
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    printf("%lld\n", res);
    return 0;
}