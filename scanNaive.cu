#include "cuda.cu"

template <typename T, uint blockSize>
__global__ void scanSingleBlock(T *a, const uint n, T *b) {
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
    T *dIn = cuMalloc<T>(paddedN);
    T *dOut = cuMalloc<T>(paddedN);
    T *blockSums = cuMalloc<T>(numBlocks);
    cuMemcpyTo<T>(dIn, in, n);
    gpuScan<T, blockSize>(dIn, dOut, blockSums, numBlocks);
    cuMemcpyFrom<T>(out, dOut, n);
    cuFree<T>(dIn);
    cuFree<T>(dOut);
    cuFree<T>(blockSums);
    return millis(start);
}

#define BLOCK_SIZE 512

#include "scanBase.cu"
