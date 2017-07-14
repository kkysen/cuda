#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>

int divUp(int a, int b) {
    return a / b + (a % b != 0);
}

template <typename T>
void cpuScan(T *a, const uint n, T *b) {
    T prev = 0;
    b[0] = prev;
    for (int i = 1; i < n; i++) {
        prev = b[i] = a[i - 1] + prev;
    }
}

template <typename T>
void fill(T *a, uint n) {
    for (uint i = 0; i < n; i++) {
        a[i] = i + 1;
    }
}

template <typename T>
__global__ void gpuScanKernel(T *a, const uint n, T *b) {
    extern __shared__ T temp[]; // allocated on invocation
    const uint i = threadIdx.x;
    uint aPtr = 1;
    uint bPtr = 2;
    // Load input into shared memory.
    // This is exclusive scan, so shift right by one
    // and set first element to 0
    temp[bPtr * n + i] = i > 0 ? a[i - 1] : 0; // FIXME should be extra 0 to avoid branch
    __syncthreads();
    for (uint j = 1; j < n; j <<= 1) {
        // swap double buffer indices
        bPtr = 1 - bPtr;
        aPtr = 1 - bPtr;
        const uint bI = bPtr * n + i;
        const uint aI = aPtr * n + i;
        if (i >= j) {
            temp[bI] += temp[aI - j];
        } else {
            temp[bI] = temp[aI];
        }
        __syncthreads();
    }
    b[i] = temp[bPtr * n + i]; // write output
}

template <typename T>
void gpuScan(T *a, const uint n, T *b) {
    size_t size = n * sizeof(T);
    T *da; cudaMalloc(&da, size);
    T *db; cudaMalloc(&db, size);
    cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);
    const uint threadsPerBlock = 256; // FIXME
    const uint blocksPerGrid = divUp(n, threadsPerBlock);
    gpuScanKernel<<<blocksPerGrid, threadsPerBlock>>>(da, n, db);
    cudaMemcpy(b, db, size, cudaMemcpyDeviceToHost);
    cudaFree(da);
    cudaFree(db);
}

template <typename T>
void printArray(T *a, uint n) {
    for (uint i = 0; i < n; i++) {
        std::cout << a[i] << std::endl;
    }
}

template <typename T>
void scan(uint n, bool print) {
    size_t size = n * sizeof(T);
    T *a = (T *) malloc(size);
    T *cpuSums = (T *) malloc(size);
    T *gpuSums = (T *) malloc(size);
    fill<T>(a, n);
    cpuScan<T>(a, n, cpuSums);
    gpuScan<T>(a, n, gpuSums);
    std::cout << "gpuSums" << std::endl;
    if (print) {
        printArray(gpuSums, n);
    }
    if (memcmp(cpuSums, gpuSums, size) != 0) {
        std::cout << "\nwrong\n" << std::endl;
        std::cout << "gpuSums" << std::endl;
        if (print) {
            printArray(cpuSums, n);
        }
    }
    free(a);
    free(cpuSums);
    free(gpuSums);
}

long parseInt(const char *s) {
    char *rest;
    long i = strtol(s, &rest, 10);
    //free(rest);
    return i;
}

bool strEquals(const char *a, const char *b) {
    return strcmp(a, b) == 0;
}

bool parseBool(const char *s) {
    if (strEquals("true", s)) {
        return true;
    } else if (strEquals("false", s)) {
        return false;
    } else {
        throw std::runtime_error("invalid bool");
    }
}

int main(const int argc, const char **argv) {
    uint n = argc < 2 ? 10 : parseInt(argv[1]);
    bool print = argc < 3 ? false : parseBool(argv[2]);
    scan<int>(n, print);
}
