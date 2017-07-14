#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float cpuSum(float *a, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += a[i];
    }
    return sum;
}

int divUp(int a, int b) {
    return a / b + (a % b != 0);
}

__device__ float sumChunk(float *a, const uint start, const uint end) {
    float sum = 0;
    for (uint i = start; i < end; i++) {
        sum += a[i];
    }
    return sum;
}

__global__ sumChunks(float *a, const uint n, const uint chunkSize, float *b) {
    const uint chunk = blockDim.x * blockIdx.x + threadIdx.x;
    const uint start = chunk * chunkSize;
    uint tempEnd = start + chunkSize;
    const uint end = tempEnd < n ? tempEnd : n;
    sums[chunk] = sumChunk(a, start, end);
}

float gpuSum(float *host, uint n, uint numThreads) {
    size_t size;
    size = n * sizeof(float);
    float *a; cudaMalloc(&a, size);
    float *b; cudaMalloc(&b, numThreads * sizeof(float));
    cudaMemcpy(a, host, size, cudaMemcpyHostToDevice);
    float *temp;
    while (n > threshold) {
        uint chunkSize = divUp(n, numThreads);
        sumChunks<<<blocksPerGrid, threadsPerBlock>>>(a, n, chunkSize, b);
        n = numThreads;
        numThreads >>= 4; // magic number
        temp = a;
        a = b;
        b = temp;
    }
    size = n * sizeof(float);
    float *sums = (float *) malloc(size);
    cudaMemcpy(sums, a, size, cudaMemcpyDeviceToHost);
    return cpuSum(sums);
}

int main(const int argc, const char **argv) {

}
