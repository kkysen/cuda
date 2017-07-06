#include <stdio.h>

const int N = 7;
const int blockSize = 7;

__global__ void hello(char *a, int *b) {
    a[threadIdx.x] += b[threadIdx.x];
}

int main() {
    char a[N] = "Hello ";
    int b[N] = {15, 10, 6, 0, -11, 1, 0};

    char *ad;
    int *bd;
    const int cSize = N * sizeof(char);
    const int iSize = N * sizeof(int);

    printf("%s\n", a);

    cudaMalloc(&ad, cSize);
    cudaMalloc(&bd, iSize);

    cudaMemcpy(ad, a, cSize, cudaMemcpyHostToDevice);
    cudaMemcpy(bd, b, iSize, cudaMemcpyHostToDevice);

    dim3 dimBlock(blockSize, 1);
    dim3 dimGrid(1, 1);
    hello<<<dimGrid, dimBlock>>>(ad, bd);

    cudaMemcpy(a, ad, cSize, cudaMemcpyDeviceToHost);
    cudaFree(ad);

    printf("%s\n", a);
    return EXIT_SUCCESS;
}
