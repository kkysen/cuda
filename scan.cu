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
        std::cout << i << ": " << a[i] << std::endl;
    }
}

template <typename T>
__device__ void printIntArray(T *a, uint n) {
    for (uint i = 0; i < n; i++) {
        printf("%d: %d\n", i, a[i]);
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
T *cudaMalloc(uint length) {
    T *a;
    cudaMalloc(&a, length * sizeof(T));
    check();
    return a;
}

template <typename T>
void cudaMemcopy(T *dst, const T *src, uint length, cudaMemcpyKind kind) {
    cudaMemcpy(dst, src, length * sizeof(T), kind);
    check();
}

template <typename T>
void cudaMemcpyTo(T *dst, const T *src, uint length) {
    cudaMemcopy<T>(dst, src, length, cudaMemcpyHostToDevice);
}

template <typename T>
void cudaMemcpyFrom(T *dst, const T *src, uint length) {
    cudaMemcopy<T>(dst, src, length, cudaMemcpyDeviceToHost);
}

#define SYNC true

void sync() {
    if (SYNC) {
        cudaDeviceSynchronize();
        check();
    }
}

template <typename T>
double cpuScan(T *in, T *out, const uint n) {
    clock_t start = clock();
    T prev = 0;
    out[0] = prev;
    for (int i = 1; i < n; i++) {
        prev = out[i] = in[i - 1] + prev;
    }
    return millis(start);
}

template <typename T, uint blockSize>
__global__ void gpuScanBlockKernelSlow(T *a, const uint n, T *b) {
    __shared__ T s[blockSize];
    const uint i = threadIdx.x;
    if (i == blockSize) {
        b[0] = 0;
        return;
    }

    s[i] = i < n ? a[i] : 0;
    __syncthreads();

    #pragma unroll
    for (uint d = 1; d < blockSize; d <<= 1) {
        const uint j = i + d;
        if (j < blockSize) {
            s[j] += s[i];
            __syncthreads();
        }
    }
    __syncthreads();

    b[i + 1] = s[i];
}

#define MAX_GRID_SIZE 65535

#define blockId() (blockIdx.y * MAX_GRID_SIZE + blockIdx.x)

template <typename T, uint blockSize>
__global__ void scanBlocks(T *in, T *out, T *blockSums, const uint numBlocks) {
    __shared__ T s[blockSize];
    const uint blockId = blockId();
    if (blockId >= numBlocks) {
        printf("blockId too high\n");
        return;
    }
    const uint i = threadIdx.x;
    const uint index = i + (blockId * blockSize);

    s[i] = in[index];
    __syncthreads();

    #pragma unroll
    for (uint d = 1; d < blockSize; d <<= 1) {
        const uint j = i + d;
        if (j < blockSize) {
            s[j] += s[i];
        }
        __syncthreads();
    }

    if (i == blockSize - 1) {
        out[index - i] = 0;
        blockSums[blockId] = s[i];
    } else {
        out[index + 1] = s[i];
    }
}

template <typename T, uint blockSize>
__global__ void scanBlocksFast(T *in, T *out, T *blockSums, const uint numBlocks) {
    __shared__ T s[blockSize];
    const uint blockId = blockId();
    if (blockId >= numBlocks) {
        printf("blockId too high");
        return;
    }
    const uint i = threadIdx.x;
    const uint index = i + (blockId * blockSize);

    s[i] = in[index];
    __syncthreads();

    //#pragma unroll

}

template <typename T, uint blockSize>
void scanBlocks(T *in, T *out, T *blockSums, const uint numBlocks, const dim3 grid) {
    std::cout << "scanning " << numBlocks << " blocks" << std::endl;
    scanBlocks<T, blockSize><<<grid, blockSize>>>(in, out, blockSums, numBlocks);
    check();
    sync();
}

// TODO make blockSums into constant or texture memory
template <typename T, uint blockSize>
__global__ void addBlockSums(T *a, T *blockSums, const uint numBlocks) {
    const uint blockId = blockId();
    if (blockId == 0 || blockId >= numBlocks) {
        return;
    }
    const uint i = blockSize * blockId + threadIdx.x;
    a[i] += blockSums[blockId];
}

template <typename T, uint blockSize>
void addBlockSums(T *a, T *blockSums, const uint numBlocks, const dim3 grid) {
    std::cout << "adding " << numBlocks << " blockSums to "
        << numBlocks << " blocks" << std::endl;
    addBlockSums<T, blockSize><<<grid, blockSize>>>(a, blockSums, numBlocks);
    check();
    sync();
}

template <typename T, uint blockSize>
void gpuScan(T *in, T *out, T *blockSums, const uint numBlocks) {
    const dim3 grid(numBlocks % MAX_GRID_SIZE, divUp(numBlocks, MAX_GRID_SIZE));
    scanBlocks<T, blockSize>(in, out, blockSums, numBlocks, grid);
    if (numBlocks == 1) {
        return;
    } else {
        // recurse
        const uint newNumBlocks = divUp(numBlocks, blockSize);
        T *newBlockSums = cudaMalloc<T>(newNumBlocks);
        printf("scanning blockSums\n");
        gpuScan<T, blockSize>(blockSums, blockSums, newBlockSums, newNumBlocks);
        addBlockSums<T, blockSize>(out, blockSums, numBlocks, grid);
    }
}

template <typename T, uint blockSize>
double gpuScan(T *in, T *out, uint n) {
    clock_t start = clock();
    const uint numBlocks = divUp(n, blockSize);
    const uint paddedN = numBlocks * blockSize;
    T *dIn = cudaMalloc<T>(paddedN);
    T *dOut = cudaMalloc<T>(paddedN);
    T *blockSums = cudaMalloc<T>(numBlocks);
    cudaMemcpyTo<T>(dIn, in, n);
    gpuScan<T, blockSize>(dIn, dOut, blockSums, numBlocks);
    cudaMemcpyFrom<T>(out, dOut, n);
    cudaFree(dIn);
    cudaFree(dOut);
    cudaFree(blockSums);
    return millis(start);
}

void time(std::string s, double time) {
    std::cout << s << ": " << time << " ms" << std::endl;
}

#define BLOCK_SIZE 512

template <typename T>
void scan(uint n, bool print) {
    std::cout << "n = " << n << std::endl;
    size_t size = n * sizeof(T);
    T *a = (T *) malloc(size);
    T *cpuSums = (T *) malloc(size);
    T *gpuSums = (T *) malloc(size);
    fill<T>(a, n);

    time("CPU", cpuScan<T>(a, cpuSums, n));
    time("GPU", gpuScan<T, BLOCK_SIZE>(a, gpuSums, n));

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
