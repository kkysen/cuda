#ifndef CUDA_H
#define CUDA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>

int divUp(const int a, const int b) {
    return a / b + (a % b != 0);
}

double millis(const clock_t start, const clock_t end) {
    return (double) (end - start) * 1000.0 / CLOCKS_PER_SEC;
}

double millis(const clock_t start) {
    return millis(start, clock());
}

template <typename T>
void fill(T *a, const uint n) {
    for (uint i = 0; i < n; ++i) {
        a[i] = i + 1;
    }
}

template <typename T>
void printArray(const T *a, const uint n) {
    for (uint i = 0; i < n; ++i) {
        std::cout << i << ": " << a[i] << std::endl;
    }
}

template <typename T>
void printMatrix(const T *a, const uint m, const uint n) {
    for (uint i = 0; i < m; ++i) {
        std::cout << "[";
        for (uint j = 0; j < n;) {
            std::cout << a[i * n + j];
            if (++j != n) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }
}

template <typename T>
__device__ void printIntArray(const T *a, const uint n) {
    for (uint i = 0; i < n; ++i) {
        printf("%d: %d\n", i, a[i]);
    }
}

template <typename T>
__device__ void printFloatArray(const T *a, const uint n) {
    for (uint i = 0; i < n; ++i) {
        printf("%d: %f\n", i, a[i]);
    }
}

#define check() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            const char *errName = cudaGetErrorName(err); \
            std::cerr \
                << "Fatal error: " << errName \
                << " (" << cudaGetErrorString(err) << std::endl \
                << " at " << __FILE__ << ":" << __LINE__ \
                << std::endl; \
            throw std::runtime_error(errName); \
        } \
    } while (0)
//

template <typename T>
T *cuMalloc(const uint length) {
    T *a;
    cudaMalloc(&a, length * sizeof(T));
    check();
    return a;
}

template <typename T>
void cuMemcpy(T *dst, const T *src, const uint length, const cudaMemcpyKind kind) {
    cudaMemcpy(dst, src, length * sizeof(T), kind);
    check();
}

template <typename T>
void cuMemcpyTo(T *dst, const T *src, const uint length) {
    cuMemcpy<T>(dst, src, length, cudaMemcpyHostToDevice);
}

template <typename T>
void cuMemcpyFrom(T *dst, const T *src, const uint length) {
    cuMemcpy<T>(dst, src, length, cudaMemcpyDeviceToHost);
}

template <typename T>
void cuFree(T *a) {
    cudaFree(a);
}

#define SYNC true

void sync() {
    if (SYNC) {
        cudaDeviceSynchronize();
        check();
    }
}

#define MAX_GRID_SIZE 65535

#define blockId() (blockIdx.y * MAX_GRID_SIZE + blockIdx.x)

void time(const std::string s, const double time) {
    std::cout << s << ": " << time << " ms" << std::endl;
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

#endif
