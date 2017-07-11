#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void cpuTranspose(float *a, float *b, int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            b[j * m + i] = a[i * n + j];
        }
    }
}

__global__ void gpuTranspose(float *a, float *b, int m, int n) {
    uint i = blockDim.x * blockIdx.x + threadIdx.x;
    uint j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < m && j < n) {
        b[j * m + i] = a[i * n + j];
    }
}

void cpuMatMul(float *a, float *b, float *c, int m, int n, int p) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            float val = 0;
            for (int k = 0; k < n; ++k) {
                val += a[i * n + k] * b[k * p + j];
            }
            c[i * p + j] = val;
        }
    }
}

__global__ void gpuMatMul(float *a, float *b, float *c, int m, int n, int p) {
    uint i = blockDim.x * blockIdx.x + threadIdx.x;
    uint j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < m && j < p) {
        float val = 0;
        for (int k = 0; k < n; ++k) {
            val += a[i * n + k] * b[k * p + j];
        }
        c[i * p + j] = val;
    }
}

void fill(float *a, int n) {
    for (int i = 0; i < n; ++i) {
        a[i] = i;
    }
}

long parseInt(const char *s) {
    char *rest;
    long i = strtol(s, &rest, 10);
    //free(rest);
    return i;
}

double millis(const clock_t start, const clock_t end) {
    return (double) (end - start) * 1000.0 / CLOCKS_PER_SEC;
}

double millis(const clock_t start) {
    return millis(start, clock());
}

void printMat(float *a, int m, int n) {
    for (int i = 0; i < m; ++i) {
        printf("{");
        for (int j = 0; j < n;) {
            printf("%.0f", a[i * n + j]);
            if (++j != n) {
                printf(", ");
            }
        }
        printf("}\n");
    }
}

int divUp(int a, int b) {
    return a / b + (a % b != 0);
}

void transposeTime(float *a, float *b, int m, int n) {
    printf("transpose: m = %d, n = %d\n", m, n);
    const uint blockSize = 32;

    // CPU
    clock_t cpuStart = clock();
    cpuTranspose(a, b, m, n);
    double cpuTime = millis(cpuStart);
    printf("CPU: %.5f, ", cpuTime);

    // GPU
    clock_t gpuStart = clock();
    const size_t size = m * n * sizeof(float);
    float *correct = (float *) malloc(size);
    memcpy(correct, b, size);
    float *dA; cudaMalloc(&dA, size);
    float *dB; cudaMalloc(&dB, size);
    cudaMemcpy(dA, a, size, cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 blocksPerGrid(divUp(m, blockSize), divUp(n, blockSize));
    gpuTranspose<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, m, n);
    cudaMemcpy(b, dB, size, cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    double gpuTime = millis(gpuStart);
    printf("GPU: %.5f\n", gpuTime);

    if (memcmp(b, correct, size) != 0) {
        printf("wrong\n");
        printf("CPU:\n");
        printMat(correct, m, n);
        printf("GPU:\n");
        printMat(b, m, n);
    }
    free(correct);
}

void transposeTest(int m, int n) {
    size_t size = m * n * sizeof(float);
    float *a = (float *) malloc(size);
    float *b = (float *) malloc(size);
    fill(a, m * n);
    transposeTime(a, b, m, n);
    free(a);
    free(b);
}

void matMulTime(float *a, float *b, int m, int n, int p) {
    printf("matMul: (%d, %d) x (%d, %d) = (%d, %d)\n", m, n, n, p, m, p);
    const uint blockSize = 32;

    size_t aSize = m * n * sizeof(float);
    size_t bSize = n * p * sizeof(float);
    size_t cSize = m * p * sizeof(float);

    float *c = (float *) malloc(cSize);

    // CPU
    clock_t cpuStart = clock();
    cpuMatMul(a, b, c, m, n, p);
    double cpuTime = millis(cpuStart);
    printf("CPU: %.5f, ", cpuTime);

    // GPU
    clock_t gpuStart = clock();
    float *correct = (float *) malloc(cSize);
    memcpy(correct, c, cSize);
    float *dA; cudaMalloc(&dA, aSize);
    float *dB; cudaMalloc(&dB, bSize);
    float *dC; cudaMalloc(&dC, cSize);
    cudaMemcpy(dA, a, aSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, b, bSize, cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 blocksPerGrid(divUp(m, blockSize), divUp(n, blockSize));
    gpuMatMul<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, m, n, p);
    cudaMemcpy(c, dC, cSize, cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    double gpuTime = millis(gpuStart);
    printf("GPU: %.5f\n", gpuTime);

    if (memcmp(c, correct, cSize) != 0) {
        printf("wrong\n");
        // printf("CPU:\n");
        // printMat(correct, m, p);
        // printf("GPU:\n");
        // printMat(c, m, p);
    }

    if (m * n * p < 1000) {
        printf("multiplying\n");
        printMat(a, m, n);
        printf("by\n");
        printMat(b, n, p);
        printf("into\n");
        printMat(correct, m, p);
    }
    free(correct);
    free(c);
}

void matMulTest(int m, int n, int p) {
    float *a = (float *) malloc(m * n * sizeof(float));
    float *b = (float *) malloc(n * p * sizeof(float));
    fill(a, m * n);
    fill(b, n * p);
    matMulTime(a, b, m, n, p);
    free(a);
    free(b);
}

int main(int argc, char **argv) {
    // for (int i = 0; i < argc; ++i) {
    //     printf("%d: %s\n", i, argv[i]);
    // }
    if (argc < 2) {
        printf("usage: <m> [<n> [<p>]]");
        return EXIT_FAILURE;
    }

    // int dim = 5 * 2 * blockSize;
    // dim3 dimsA(dim, dim, 1);
    // dim3 dimsB(dim * 2, dim, 1);

    int m = parseInt(argv[1]);
    int n = argc < 3 ? m : parseInt(argv[2]);
    int p = argc < 4 ? m : parseInt(argv[3]);

    //transposeTest(m, n);
    matMulTest(m, n, p);
    return EXIT_SUCCESS;
}
