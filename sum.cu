#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <iostream>

// template <uint N>
// struct unroll {
//     template <typename F>
//     static void call(F const &f) {
//         f();
//         unroll<N - 1>::call(f);
//     }
// };
//
// template <>
// struct unroll<0u> {
//     template <typename F>
//     static void call(F const& f) {}
// };
//
// template <uint N>
// struct logUnroll {
//     template <typename F>
//     static void call(F const &f) {
//         f();
//         unroll<N >> 1>::call(f);
//     }
// };
//
// template<>
// struct logUnroll<0u> {
//     template <typename F>
//     static void call(F const &f) {}
// };


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
void fill(T *a, const uint n) {
    for (uint i = 0; i < n; i++) {
        a[i] = i + 1;
    }
}

template <typename T>
void printArray(T *a, const uint n) {
    for (uint i = 0; i < n; i++) {
        std::cout << a[i] << std::endl;
    }
}

template <typename T>
T cpuSum(T *a, const uint n) {
    T sum = 0;
    for (uint i = 0; i < n; ++i) {
        sum += a[i];
    }
    return sum;
}

template <typename T>
double cpuSum(T *a, const uint n, T &sum) {
    clock_t start = clock();
    sum = cpuSum(a, n);
    return millis(start);
}

template <typename T, uint blockSize, uint initialSumSize>
__global__ void gpuSumKernel1(T *a, T *b, const uint n) {
    __shared__ T s[blockSize];
    const uint i = threadIdx.x;
    const uint j = blockIdx.x * blockSize + i;

    s[i] = j < n ? a[j] : 0;
    __syncthreads();

    #pragma unroll
    for (uint d = 1; d < blockSize; d *= 2) {
        if ((i % (d * 2)) == 0) {
            s[i] += s[i + d];
        }
        __syncthreads();
    }

    if (i == 0) {
        b[blockIdx.x] = s[0];
    }
}

template <typename T, uint blockSize, uint initialSumSize>
__global__ void gpuSumKernel2(T *a, T *b, const uint n) {
    __shared__ T s[blockSize];
    const uint i = threadIdx.x;
    const uint j = blockIdx.x * blockSize + i;

    s[i] = j < n ? a[j] : 0;
    __syncthreads();

    #pragma unroll
    for (uint d = 1; d < blockSize; d *= 2) {
        const uint index = d * 2 * i;
        if (index < blockSize) {
            s[index] += s[index + d];
        }
        __syncthreads();
    }

    if (i == 0) {
        b[blockIdx.x] = s[0];
    }
}

template <typename T, uint blockSize, uint initialSumSize>
__global__ void gpuSumKernel3(T *a, T *b, const uint n) {
    __shared__ T s[blockSize];
    const uint i = threadIdx.x;
    const uint j = blockIdx.x * blockSize + i;

    s[i] = j < n ? a[j] : 0;
    __syncthreads();

    #pragma unroll
    for (uint d = blockSize >> 1; d > 0; d >>= 1) {
        if (i < d) {
            s[i] += s[i + d];
        }
        __syncthreads();
    }

    if (i == 0) {
        b[blockIdx.x] = s[0];
    }
}

template <typename T, uint blockSize, uint initialSumSize>
__global__ void gpuSumKernel4(T *a, T *b, const uint n) {
    __shared__ T s[blockSize];
    const uint i = threadIdx.x;
    const uint size = blockSize * initialSumSize;
    const uint j = blockIdx.x * size + i;

    T sum = 0;
    #pragma unroll
    for (uint offset = 0;
        offset < size && j + offset < n;
        offset += blockSize) {
        sum += a[j + offset];
    }
    s[i] = sum;
    __syncthreads();

    #pragma unroll
    for (uint d = blockSize >> 1; d > 0; d >>= 1) {
        if (i < d) {
            s[i] = sum = sum + s[i + d];
        }
        __syncthreads();
    }

    if (i == 0) {
        b[blockIdx.x] = s[0];
    }
}

#define WARP_SIZE 32

template <typename T, uint blockSize, uint initialSumSize>
__global__ void gpuSumKernel(T *a, T *b, const uint n) {
    __shared__ T s[blockSize];
    const uint i = threadIdx.x;
    const uint size = blockSize * initialSumSize;
    const uint j = blockIdx.x * size + i;

    T sum = 0;
    #pragma unroll
    for (uint offset = 0;
        offset < size && j + offset < n;
        offset += blockSize) {
        sum += a[j + offset];
    }
    s[i] = sum;
    __syncthreads();

    #pragma unroll
    for (uint d = blockSize >> 1; d > WARP_SIZE; d >>= 1) {
        if (i < d) {
            s[i] = sum = sum + s[i + d];
        }
        __syncthreads();
    }

    // this gives the wrong answer
    // if (i < 32) {
    //     if (blockSize >= 64) {
    //         sum += s[i + 32];
    //     }
    //     for (int offset = 16; offset > 0; offset >>= 1) {
    //         sum += __shfl_down(sum, offset);
    //     }
    // }

    // unrolled loop is slower
    if (blockSize >= 64 && i < 32) {
        s[i] = sum = sum + s[i + 32];
    }
    __syncthreads();

    if (blockSize >= 32 && i < 16) {
        s[i] = sum = sum + s[i + 16];
    }
    __syncthreads();

    if (blockSize >= 16 && i < 8) {
        s[i] = sum = sum + s[i + 8];
    }
    __syncthreads();

    if (blockSize >= 8 && i < 4) {
        s[i] = sum = sum + s[i + 4];
    }
    __syncthreads();

    if (blockSize >= 4 && i < 2) {
        s[i] = sum = sum + s[i + 2];
    }
    __syncthreads();

    if (blockSize >= 2 && i < 1) {
        s[i] = sum = sum + s[i + 1];
    }
    __syncthreads();

    if (i == 0) {
        b[blockIdx.x] = s[0];
    }
}

// TODO use __shfl_down
template <typename T, uint blockSize, uint warpSize>
__device__ void parallelSum(T *s, uint i, T &sum) {
    #pragma unroll
    for (uint d = blockSize >> 1; d > 0; d >>= 1) {
        // d is the stride distance b/w
        // indices that are supposed to be summed
        if (blockSize >= d * 2) {
            if (i < WARP_SIZE) {
                sum += s[i + d];
            } else if (i < d) {
                s[i] = sum = sum + s[i + d];
                __syncthreads();
            }
        }
    }
}

// TODO bool nIsPow2 template param
template <typename T, uint blockSize, uint initialSumSize>
__global__ void gpuSumKernelNew(T *a, T *b, const uint n) {
    __shared__ T s[blockSize];
    // blockSize is blockDim.x,
    // but passed as a template arg
    // so the looped can be unrolled
    const uint i = threadIdx.x; // i is threadId
    const uint size = blockSize * initialSumSize;
    // j is the index of the element in global memory (a)
    uint j = blockIdx.x * size + i;
    const uint gridSize = gridDim.x * size;

    // copy and sum initialSumSize elems to shared memory (s)
    T sum = 0;
    for (; j < n; j += gridSize) {
        #pragma unroll
        for (uint offset = 0;
            offset < size && j + offset < n;
            offset += blockSize
        ) {
            sum += a[j + offset];
        }
    }
    s[i] = sum;
    __syncthreads();

    parallelSum<T, blockSize, WARP_SIZE>(s, i, sum);

    // write back to global memory
    if (i == 0) b[blockIdx.x] = sum;
}

template <typename T, uint blockSize>
void gpuSum(T *a, T *b, const uint n, const uint gridSize) {
    if (blockSize <= 32) {
        throw std::runtime_error("blockSize must be > 32");
    }
    gpuSumKernel<T, blockSize, 4><<<gridSize, blockSize>>>(a, b, n);
}

template <typename T>
double gpuSum(T *a, const uint n, T &sum) {
    const uint threadsPerBlock = 256; // FIXME
    const uint blocksPerGrid = divUp(n, threadsPerBlock);
    const clock_t start = clock();
    const size_t size = n * sizeof(T);
    const size_t reducedSize = blocksPerGrid * sizeof(T);
    T *da; cudaMalloc(&da, size);
    T *db; cudaMalloc(&db, reducedSize);
    cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);
    gpuSum<T, threadsPerBlock>(da, db, n, blocksPerGrid);
    T *b = (T *) malloc(reducedSize);
    cudaMemcpy(b, db, reducedSize, cudaMemcpyDeviceToHost);
    sum = cpuSum(b, blocksPerGrid);
    cudaFree(da);
    cudaFree(db);
    free(b);
    return millis(start);
}

void time(std::string s, double time) {
    std::cout << s << " time: " << time << " ms" << std::endl;
}

template <typename T>
void sum(const uint n) {
    std::cout << "n = " << n << std::endl;
    size_t size = n * sizeof(T);
    T *a = (T *) malloc(size);
    fill<T>(a, n);
    T cpuResult;
    T gpuResult;
    time("CPU", cpuSum<T>(a, n, cpuResult));
    time("GPU", gpuSum<T>(a, n, gpuResult));
    T correct = (n * (n + 1)) >> 1;

    std::cout << "CPU Sum: " << cpuResult << std::endl;
    std::cout << "GPU Sum: " << gpuResult << std::endl;
    std::cout << "Correct: " << correct << std::endl;

    if (cpuResult != correct) {
        std::cout << "\nCPU wrong\n" << std::endl;
    }

    if (gpuResult != correct) {
        std::cout << "\nGPU wrong\n" << std::endl;
    }

    free(a);
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
    const uint n = argc < 2 ? 10 : parseInt(argv[1]);
    const char *type = argc < 3 ? "int" : argv[2];
    if (strEquals(type, "int")) {
        sum<int>(n);
    } else if (strEquals(type, "float")) {
        sum<float>(n);
    } else if (strEquals(type, "char")) {
        sum<char>(n);
    } else {
        throw std::runtime_error("invalid numeric type");
    }
}
