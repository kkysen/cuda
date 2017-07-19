#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>

int divUp(int a, int b) {
    return a / b + (a % b != 0);
}

double millis(const clock_t start, const clock_t end) {
    return (double) (end - start) * 1000.0 / CLOCKS_PER_SEC;
}

double millis(const clock_t start) {
    return millis(start, clock());
}

template <typename T>
void fill(T *a, uint n) {
    for (uint i = 0; i < n; i++) {
        a[i] = i + 1;
    }
}

template <typename T>
void printArray(T *a, uint n) {
    for (uint i = 0; i < n; i++) {
        std::cout << a[i] << std::endl;
    }
}

template <typename T>
__device__ void printIntArray(T *a, uint n) {
    for (uint i = 0; i < n; i++) {
        printf("%d\n", a[i]);
    }
}

template <typename T>
double cpuScan(T *a, const uint n, T *b) {
    clock_t start = clock();
    T prev = 0;
    b[0] = prev;
    for (int i = 1; i < n; i++) {
        prev = b[i] = a[i - 1] + prev;
    }
    return millis(start);
}

template <typename T, uint blockSize>
__global__ void gpuScanKernel(T *in, T *out, const uint n) {
    __shared__ T s[blockSize];
    const uint i = threadIdx.x;
    if (i == blockSize - 1) {
        out[0] = 0;
        return;
    }

    s[i] = i < n ? in[i] : 0;
    __syncthreads();

    #pragma unroll
    for (uint d = 1; d < blockSize; d <<= 1) {
        const uint j = i + d;
        if (j < blockSize) {
            s[j] += s[i];
        }
        __syncthreads();
    }
    __syncthreads();

    out[i + 1] = s[i];

    T correct = (i + 1) * (i + 2) / 2;
    bool right = s[i] == correct;
    printf("%s: i = %d, s[i] = %d %s= %d\n", right ? "RIGHT" : "WRONG", i, s[i], right ? "=" : "!", correct);
}

#define BLOCK_SIZE 128

template <typename T>
double gpuScan(T *a, const uint n, T *b) {
    clock_t start = clock();
    size_t size = n * sizeof(T);
    T *da; cudaMalloc(&da, size);
    T *db; cudaMalloc(&db, size);
    cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);
    //const uint threadsPerBlock = 256; // FIXME
    //const uint blocksPerGrid = divUp(n, threadsPerBlock);
    //const uint blockSize = 64;
    gpuScanKernel<T, BLOCK_SIZE><<<1, BLOCK_SIZE>>>(da, db, n);
    cudaMemcpy(b, db, size, cudaMemcpyDeviceToHost);
    cudaFree(da);
    cudaFree(db);
    return millis(start);
}

void time(std::string s, double time) {
    std::cout << s << ": " << time << std::endl;
}

template <typename T>
void scan(uint n, bool print) {
    std::cout << "n = " << n << std::endl;
    size_t size = n * sizeof(T);
    T *a = (T *) malloc(size);
    T *cpuSums = (T *) malloc(size);
    T *gpuSums = (T *) malloc(size);
    fill<T>(a, n);

    time("CPU", cpuScan<T>(a, n, cpuSums));
    time("GPU", gpuScan<T>(a, n, gpuSums));

    if (print) {
        std::cout << "gpuSums" << std::endl;
        printArray(gpuSums, n);
    }

    if (memcmp(cpuSums, gpuSums, size) != 0) {
        std::cout << "\nwrong\n" << std::endl;
        if (print) {
            std::cout << "cpuSums" << std::endl;
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
    uint n = argc < 2 ? 64 : parseInt(argv[1]);
    bool print = argc < 3 ? false : parseBool(argv[2]);
    const char *type = argc < 4 ? "int" : argv[3];
    if (strEquals(type, "int")) {
        scan<int>(n, print);
    } else if (strEquals(type, "float")) {
        scan<float>(n, print);
    } else if (strEquals(type, "char")) {
        scan<char>(n, print);
    } else {
        throw std::runtime_error("invalid numeric type");
    }
}
