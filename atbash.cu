#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

const int threadsPerBlock = 256;

double millis(const clock_t start, const clock_t end) {
    return (double) (end - start) * 1000.0 / CLOCKS_PER_SEC;
}

double millis(const clock_t start) {
    return millis(start, clock());
}

long parseInt(const char *s) {
    char *rest;
    long i = strtol(s, &rest, 10);
    //free(rest);
    return i;
}

int write(const char *file, char *src, size_t size) {
    printf("trying to write to %s", file);
    int mode = 0x0777;
    int fdout = open(file, O_RDWR | O_CREAT | O_TRUNC, mode);
    if (fdout < 0) {
        printf("failed to open %s", file);
        return -1;
    }
    char *dst = (char *) mmap(0, size, PROT_READ | PROT_READ, MAP_SHARED, fdout, 0);
    if (dst == (caddr_t) - 1) {
        printf("failed to mmap %s", file);
        return -1;
    }
    memcpy(dst, src, size);
    return 0;
}

void atbashCPU(char const *in, char *out, int n) {
    for (int i = 0; i < n; i++) {
        out[n - 1 - i] = in[i];
    }
}

__global__ void atbashGPU(char const *in, char *out, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        out[n - 1 - i] = in[i];
    }
}

double atbashTime(char const *in, char *out, int n) {
    printf("n = %d : ", n);
    // CPU
    clock_t cpuStart = clock();
    atbashCPU(in, out, n);
    double cpuTime = millis(cpuStart);
    printf("CPU: %.5f, ", cpuTime);

    // GPU
    clock_t gpuStart = clock();
    const size_t size = n * sizeof(char);
    char *dIn; cudaMalloc(&dIn, size);
    char *dOut; cudaMalloc(&dOut, size);
    cudaMemcpy(dIn, in, size, cudaMemcpyHostToDevice);
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    atbashGPU<<<blocksPerGrid, threadsPerBlock>>>(dIn, dOut, n);
    cudaMemcpy(out, dOut, size, cudaMemcpyDeviceToHost);
    cudaFree(dIn);
    cudaFree(dOut);
    double gpuTime = millis(gpuStart);
    printf("GPU: %.5f\n", gpuTime);
    return cpuTime - gpuTime;
}

int main(int argc, char const **argv) {
    if (argc < 2) {
        return -1;
    }

    char const *original = argv[1];
    const int originalLength = strlen(original);
    const size_t originalSize = originalLength * sizeof(char);

    const int n = originalLength * 1024 * 1024 * 128;
    const size_t size = n * sizeof(char);

    char *in = (char *) malloc(size);
    char *out = (char *) malloc(size);

    for (size_t i = 0; i < size; i += originalSize) {
        memcpy(in + i, original, originalSize);
    }

    const int numGPUWins = argc < 3 ? 16 : parseInt(argv[2]);
    int gpuWinCounter = 0;
    for (int i = originalLength; i < n; i <<= 1) {
        if (atbashTime(in, out, i) > 0) {
            gpuWinCounter++;
            if (gpuWinCounter == numGPUWins) {
                break;
            }
        }
    }

    in[originalLength] = '\0';
    out[originalLength] = '\0';
    printf("in: %s\n", in);
    printf("out: %s\n", out);

    // if (write("hw1In.txt", in, size) < 0) {
    //     return -1;
    // }
    // if (write("hw1Out.txt", out, size) < 0) {
    //     return -1;
    // }

    free(in);
    free(out);
    return 0;
}
