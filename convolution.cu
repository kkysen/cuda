#include "cuda.cu"

template <typename T, uint blockSize>
__global__ void convolveKernel(const T *a, const uint m, const uint n,
        const T *b, const uint p, const uint q,
        T *c) {
    // TODO
}

template <typename T, uint blockSize>
void convolveKernel(const T *a, const uint m, const uint n,
        const T *b, const uint p, const uint q,
        T *c,
        const dim3 grid) {
    // TODO
    //std::cout << "convolving " << numBlocks << " blocks" << std::endl;
    convolveKernel<T, blockSize><<<grid, blockSize>>>(a, m, n, b, p, q, c);
    check();
    sync();
}

template <typename T, uint blockSize>
double gpuConvolve(const T *a, const uint m, const uint n,
        const T *b, const uint p, const uint q,
        T *c) {
    const clock_t start = clock();

    const uint mn = m * n;
    const uint pq = p * q;

    T *da = cuMalloc<T>(mn);
    T *db = cuMalloc<T>(pq);
    T *dc = cuMalloc<T>(mn);

    cuMemcpyTo<T>(da, a, mn);
    cuMemcpyTo<T>(db, b, pq);

    const dim3 grid(1, 1, 1); // FIXME
    convolveKernel<T, blockSize>(a, m, n, b, p, q, c, grid);

    cuMemcpyFrom<T>(c, dc, mn);

    cuFree(da);
    cuFree(db);
    cuFree(dc);

    return millis(start);
}

#define BLOCK_SIZE 512

#include "convolutionBase.cu"
