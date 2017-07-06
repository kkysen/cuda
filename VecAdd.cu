#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef unsigned int uint;

long parseInt(const char *s) {
    char *rest;
    long i = strtol(s, &rest, 10);
    //free(rest);
    return i;
}

// Device code
__global__ void VecAdd(float* A, float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

void fill(float *a, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = (float) i;
    }
}

void print(float *a, int n) {
    for (int i = 0; i < n; i++) {
        printf("%.3f\n", a[i]);
    }
}

float sum(float *a, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += a[i];
    }
    return sum;
}

// Host code
int main(int argc, char const **argv) {
    if (argc < 2) {
        printf("usage: <N>");
        return -1;
    }
    const int N = parseInt(argv[1]);
    printf("N = %d\n", N);

    const size_t size = N * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    float* h_A = (float*) malloc(size);
    float* h_B = (float*) malloc(size);
    float *h_C = (float *) malloc(size);
    // Initialize input vectors
    fill(h_A, N);
    fill(h_B, N);

    printf("h_A: sum = %.3f\n", sum(h_A, N));

    printf("h_B: sum = %.3f\n", sum(h_B, N));

    // Allocate vectors in device memory
    float *d_A; cudaMalloc(&d_A, size);
    float *d_B; cudaMalloc(&d_B, size);
    float *d_C; cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);

    printf("h_C: sum = %.3f\n", sum(h_C, N));
    free(h_C);
}
