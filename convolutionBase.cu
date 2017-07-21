#include "cuda.cu"

template <typename T>
void cpuConvolveSlow(const T *a, const uint m, const uint n,
        const T *b, const uint p, const uint q,
        T *c) {
    const uint pMid = p >> 1;
    const uint qMid = q >> 1;

    for (uint i = 0; i < m; ++i) {
        for (uint j = 0; j < n; ++j) {
            T sum = 0;
            for (uint y = 0; y < p; ++y) {
                const uint yy = p - 1 - y;
                for (uint x = 0; x < q; ++x) {
                    // TODO check signedness
                    const uint xx = q - 1 - x;
                    const int r = i + y - pMid;
                    const int c = j + x - qMid;
                    if (r >= 0 && r < m && c >= 0 && c < n) {
                        sum += a[r * n + c] * b[yy * q + xx];
                    }
                }
            }
            c[i * n + j] = sum; // FIXME ??? (unsigned char)((float)fabs(sum) + 0.5f);
        }
    }
}

// 1st macro try
// //#define clearSums() memset(&sums, 0, n * sizeof(T))
//
// #define iLoop(start, end) i = start; i < end; ++i
// #define forI(start, end) for (uint iLoop())
// #define jLoop(start, end) j = start; j < end; ++j
// #define ujLoop(start, end) uint jLoop(start, end)
// #define jqLoop(start, end, qOff) uint qOffset = qOff, jLoop(start, end)
// #define jTmpLoop(start, end) jLoop(start, end), ++tmpPtr
// #define jqTmpLoop(start, end, qOff) uint qOffset = qOff, jTmpLoop(start, end)
// #define forJ(start, end) for (jLoop(start, end))
// #define tmpLoopBody() *tmpPtr += *(aPtr + x) * bX[k];
// #define kyLoop(start) \
//     *tmpPtr = 0;
//     int k = start; \
//     for (uint y = 0; k >= 0; --k, ++y) { \
//         tmpLoopBody() \
//     }
//
// template <typename T>
// void cpuConvolveSeparable(const T *a, const uint m, const uint n,
//         const T *b, const uint p, const uint q,
//         T *c) {
//     T *tmp = new T[m * n];
//     T *sums = new T[n];
//
//     const uint qMid = q >> 1;
//     const uint qEnd = n - qMid;
//
//     T *aPtr = a;
//     T *tmpPtr = tmp;
//
//     forI(0, m) {
//         uint qOffset = 0;
//         for (jLoop(0, qMid), ++tmpPtr, ++qOffset) {
//             kyLoop(qMid + qOffset);
//         }
//         for (jLoop(qMid, qEnd), ++aPtr, ++tmpPtr) {
//             kyLoop(q - 1);
//         }
//         qOffset = 1;
//
//         for (jqLoop(qEnd, n, 1)) {
//             *tmpPtr = 0;
//             int k = q - 1;
//             for (uint x = 0; k >= qOffset; --k; ++j, ++aPtr, ++tmpPtr, ++qOffset) {
//                 tmpLoopBody();
//             }
//         }
//     }
//
//     const uint pMid = p >> 1;
//     const uint pEnd = m - p;
//
//     tmpPtr = tmp;
//     T *tmpPtr2 = tmp;
//
//     for (uint pOffset = 0, i = 0; i < pMid; ++i, ++pOffset, tmpPtr = tmpPtr2) {
//         for (int k = pMid + pOffset; k >= 0; --k) {
//             for (uint j = 0; j < n; ++j, ++tmpPtr) {
//                 sums[j] += *tmpPtr * bY[k];
//             }
//         }
//         for (uint x = 0; x < n; ++x, ++outPtr) {
//             *outPtr = sum[x];
//         }
//     }
//
//     for (uint i = pMid; i < pEnd; ++i, tmpPtr2 += n, tmpPtr = tmpPtr2) {
//         for (int k = p - 1; k >= 0; --k) {
//             for (uint j = 0; j < n; ++j, ++tmpPtr) {
//                 sums[j] += *tmpPtr * bY[k];
//             }
//         }
//         for (uint x = 0; x < n; ++x, ++outPtr) {
//             *outPtr = sum[x];
//         }
//     }
//
//     for (uint pOffset = 1, i = pEnd; i < m; ++i, tmpPtr2 += n; tmpPtr = tmpPtr2, ++pOffset) {
//         for (int k = p - 1; k >= pOffset; --k) {
//             for (uint j = 0; j < n; ++j, ++tmpPtr) {
//                 sum[j] += *tmpPtr * bY[k];
//             }
//         }
//         for (uint x = 0; x < n; ++x, ++outPtr) {
//             *outPtr = sum[n];
//         }
//     }
//
//     delete [] tmp;
//     delete [] sum;
//
// }

// 2nd macro try
// #define clearSums() memset(&sums, 0, n * sizeof(T))
// #define moveSums() \
//     memcpy(outPtr, sums, n * sizeof(T)); \
//     outPtr += n; \
//     clearSums();
//
// #define iLoop(start, end) i = start; i < end; ++i
// #define viLoop(start, end, n) iLoop(start, end), tmpPtr2 += n, tmpPtr = tmpPtr2
// #define forVi(start, end, n) for (uint viLoop(start, end, n))
// #define vipLoop(pOff, start, end, n) uint pOffset = pOff, viLoop(start, end, n), ++pOffset
// #define forVip(pOff, start, end, n) for (vipLoop(pOff, start, end, n))
//
// #define kjLoop(start, end) \
//     for (int k = start; k >= end; --k, tmpPtr += n) { \
//         T byk = bY[k]; \
//         for (uint j = 0; j < n; ++j) { \
//             sums[j] += tmpPtr[j] * byk; \
//         } \
//         moveSums(); \
//     }
//
// template <typename T>
// void cpuConvolveSeparable(const T *a, const uint m, const uint n,
//         const T *b, const uint p, const uint q,
//         T *c) {
//     T *tmp = new T[m * n];
//     T *sums = new T[n];
//
//     const uint qMid = q >> 1;
//     const uint qEnd = n - qMid;
//
//     T *aPtr = a;
//     T *tmpPtr = tmp;
//
//     for (uint i = 0; i < m; ++i) {
//         for (uint qOffset = 0, j = 0; j < qMid; ++j, ++tmpPtr, ++qOffset) {
//             *tmpPtr = 0;
//             int k = qMid + qOffset;
//             for (uint y = 0; k >= 0; --k, ++y) {
//                 *tmpPtr += *(aPtr + y) * bX[k];
//             }
//         }
//         for (uint j = qMid; j < qEnd; ++j, ++aPtr, ++tmpPtr) {
//             *tmpPtr = 0;
//             int k = q - 1;
//             for (uint y = 0; k >= 0; --k, ++y) {
//                 *tmpPtr += *(aPtr + y) * bX[k];
//             }
//         }
//         for (uint qOffset = 1, j = qEnd; j < n; ++j) {
//             *tmpPtr = 0;
//             int k = q - 1;
//             for (uint x = 0; k >= qOffset; --k; ++j, ++aPtr, ++tmpPtr, ++qOffset) {
//                 *tmpPtr += *(aPtr + x) * bX[k];
//             }
//         }
//
//     }
//
//     const uint pMid = p >> 1;
//     const uint pEnd = m - p;
//
//     tmpPtr = tmp;
//     T *tmpPtr2 = tmp;
//
//     clearSums();
//     forVip(0, 0, pMid, 0) {
//         kjLoop(pMid + pOffset, 0)
//     }
//     forVi(pMid, pEnd, n) {
//         kjLoop(p - 1, 0)
//     }
//     forVip(1, pEnd, m, n) {
//         kjLoop(p - 1, pOffset)
//     }
//
//     delete [] tmp;
//     delete [] sums;
//
// }

template <typename T>
void cpuConvolveMedium(const T *a, const uint m, const uint n,
        const T *b, const uint p, const uint q,
        T *c) {
    const uint pMid = p >> 1;
    const uint qMid = q >> 1;

    T *aPtr2 = a + (n * pMid + qMid);
    T *aPtr = aPtr2;
    T *cPtr = c;

    for (uint i = 0; i < m; ++i) {
        const int rowMax = i + pMid;
        const int rowMin = rowMax - m;
        for (uint j = 0; j < n; ++j) {
            T *bPtr = b;
            const int colMax = j + qMid;
            const int colMin = colMax - n;
            T sum = 0;
            for (uint y = 0; y < p; ++y, aPtr -= n) {
                if (y > rowMax || y <= rowMin) {
                    bPtr += q;
                    continue;
                }
                for (uint x = 0; x < q; ++x, ++bPtr) {
                    if (x <= colMax && x > colMin) {
                        sum += *(aPtr - n) * *bPtr;
                    }
                }
            }
            *cPtr++ = sum;
            aPtr = ++aPtr2;
        }
    }
}

template <typename T>
void cpuConvolveSeparable(const T *a, const uint m, const uint n,
        const T *bY, const T *bX, const uint p, const uint q,
        T *c) {
    T *tmp = new T[m * n];
    T *sums = new T[n];

    const uint qMid = q >> 1;
    const uint qEnd = n - qMid;

    T *aPtr = a;
    T *tmpPtr = tmp;

    for (uint i = 0; i < m; ++i) {
        for (uint qOffset = 0, j = 0; j < qMid; ++j, ++tmpPtr, ++qOffset) {
            *tmpPtr = 0;
            int k = qMid + qOffset;
            for (uint y = 0; k >= 0; --k, ++y) {
                *tmpPtr += *(aPtr + y) * bX[k];
            }
        }
        for (uint j = qMid; j < qEnd; ++j, ++aPtr, ++tmpPtr) {
            *tmpPtr = 0;
            int k = q - 1;
            for (uint y = 0; k >= 0; --k, ++y) {
                *tmpPtr += *(aPtr + y) * bX[k];
            }
        }
        for (uint qOffset = 1, j = qEnd; j < n; ++j) {
            *tmpPtr = 0;
            int k = q - 1;
            for (uint x = 0; k >= qOffset; --k; ++j, ++aPtr, ++tmpPtr, ++qOffset) {
                *tmpPtr += *(aPtr + x) * bX[k];
            }
        }

    }

    const uint pMid = p >> 1;
    const uint pEnd = m - p;

    tmpPtr = tmp;
    T *tmpPtr2 = tmp;

    memset(&sums, 0, n * sizeof(T))

    for (uint pOffset = 0, i = 0; i < pMid; ++i, ++pOffset) {
        for (int k = pMid + pOffset; k >= 0; --k) {
            for (uint j = 0; j < n; ++j, ++tmpPtr) {
                sums[j] += *tmpPtr * bY[k];
            }
        }
        for (uint x = 0; x < n; ++x, ++outPtr) {
            *outPtr = sums[x];
        }
        memset(&sums, 0, n * sizeof(T))
        tmpPtr = tmpPtr2;
    }

    for (uint i = pMid; i < pEnd; ++i, tmpPtr2 += n, tmpPtr = tmpPtr2) {
        for (int k = p - 1; k >= 0; --k) {
            for (uint j = 0; j < n; ++j, ++tmpPtr) {
                sums[j] += *tmpPtr * bY[k];
            }
        }
        for (uint x = 0; x < n; ++x, ++outPtr) {
            *outPtr = sums[x];
        }
        memset(&sums, 0, n * sizeof(T))
    }

    for (uint pOffset = 1, i = pEnd; i < m; ++i, tmpPtr2 += n; tmpPtr = tmpPtr2, ++pOffset) {
        for (int k = p - 1; k >= pOffset; --k) {
            for (uint j = 0; j < n; ++j, ++tmpPtr) {
                sums[j] += *tmpPtr * bY[k];
            }
        }
        for (uint x = 0; x < n; ++x, ++outPtr) {
            *outPtr = sums[n];
        }
        memset(&sums, 0, n * sizeof(T))
    }

    delete [] tmp;
    delete [] sums;

}

template <typename T>
void cpuConvolveFast(const T *a, const uint m, const uint n,
        const T *b, const uint p, const uint q,
        T *c) {

}

// bool convolve2DFast(unsigned char* in, unsigned char* out, int dataSizeX, int dataSizeY,
//                     float* kernel, int kernelSizeX, int kernelSizeY)
// {
//     int i, j, m, n, x, y, t;
//     unsigned char **inPtr, *outPtr, *ptr;
//     int kCenterX, kCenterY;
//     int rowEnd, colEnd;                             // ending indice for section divider
//     float sum;                                      // temp accumulation buffer
//     int k, kSize;
//
//     // check validity of params
//     if(!in || !out || !kernel) return false;
//     if(dataSizeX <= 0 || kernelSizeX <= 0) return false;
//
//     // find center position of kernel (half of kernel size)
//     kCenterX = kernelSizeX >> 1;
//     kCenterY = kernelSizeY >> 1;
//     kSize = kernelSizeX * kernelSizeY;              // total kernel size
//
//     // allocate memeory for multi-cursor
//     inPtr = new unsigned char*[kSize];
//     if(!inPtr) return false;                        // allocation error
//
//     // set initial position of multi-cursor, NOTE: it is swapped instead of kernel
//     ptr = in + (dataSizeX * kCenterY + kCenterX); // the first cursor is shifted (kCenterX, kCenterY)
//     for(m=0, t=0; m < kernelSizeY; ++m)
//     {
//         for(n=0; n < kernelSizeX; ++n, ++t)
//         {
//             inPtr[t] = ptr - n;
//         }
//         ptr -= dataSizeX;
//     }
//
//     // init working  pointers
//     outPtr = out;
//
//     rowEnd = dataSizeY - kCenterY;                  // bottom row partition divider
//     colEnd = dataSizeX - kCenterX;                  // right column partition divider
//
//     // convolve rows from index=0 to index=kCenterY-1
//     y = kCenterY;
//     for(i=0; i < kCenterY; ++i)
//     {
//         // partition #1 ***********************************
//         x = kCenterX;
//         for(j=0; j < kCenterX; ++j)                 // column from index=0 to index=kCenterX-1
//         {
//             sum = 0;
//             t = 0;
//             for(m=0; m <= y; ++m)
//             {
//                 for(n=0; n <= x; ++n)
//                 {
//                     sum += *inPtr[t] * kernel[t];
//                     ++t;
//                 }
//                 t += (kernelSizeX - x - 1);         // jump to next row
//             }
//
//             // store output
//             *outPtr = (unsigned char)((float)fabs(sum) + 0.5f);
//             ++outPtr;
//             ++x;
//             for(k=0; k < kSize; ++k) ++inPtr[k];    // move all cursors to next
//         }
//
//         // partition #2 ***********************************
//         for(j=kCenterX; j < colEnd; ++j)            // column from index=kCenterX to index=(dataSizeX-kCenterX-1)
//         {
//             sum = 0;
//             t = 0;
//             for(m=0; m <= y; ++m)
//             {
//                 for(n=0; n < kernelSizeX; ++n)
//                 {
//                     sum += *inPtr[t] * kernel[t];
//                     ++t;
//                 }
//             }
//
//             // store output
//             *outPtr = (unsigned char)((float)fabs(sum) + 0.5f);
//             ++outPtr;
//             ++x;
//             for(k=0; k < kSize; ++k) ++inPtr[k];    // move all cursors to next
//         }
//
//         // partition #3 ***********************************
//         x = 1;
//         for(j=colEnd; j < dataSizeX; ++j)           // column from index=(dataSizeX-kCenter) to index=(dataSizeX-1)
//         {
//             sum = 0;
//             t = x;
//             for(m=0; m <= y; ++m)
//             {
//                 for(n=x; n < kernelSizeX; ++n)
//                 {
//                     sum += *inPtr[t] * kernel[t];
//                     ++t;
//                 }
//                 t += x;                             // jump to next row
//             }
//
//             // store output
//             *outPtr = (unsigned char)((float)fabs(sum) + 0.5f);
//             ++outPtr;
//             ++x;
//             for(k=0; k < kSize; ++k) ++inPtr[k];    // move all cursors to next
//         }
//
//         ++y;                                        // add one more row to convolve for next run
//     }
//
//     // convolve rows from index=kCenterY to index=(dataSizeY-kCenterY-1)
//     for(i= kCenterY; i < rowEnd; ++i)               // number of rows
//     {
//         // partition #4 ***********************************
//         x = kCenterX;
//         for(j=0; j < kCenterX; ++j)                 // column from index=0 to index=kCenterX-1
//         {
//             sum = 0;
//             t = 0;
//             for(m=0; m < kernelSizeY; ++m)
//             {
//                 for(n=0; n <= x; ++n)
//                 {
//                     sum += *inPtr[t] * kernel[t];
//                     ++t;
//                 }
//                 t += (kernelSizeX - x - 1);
//             }
//
//             // store output
//             *outPtr = (unsigned char)((float)fabs(sum) + 0.5f);
//             ++outPtr;
//             ++x;
//             for(k=0; k < kSize; ++k) ++inPtr[k];    // move all cursors to next
//         }
//
//         // partition #5 ***********************************
//         for(j = kCenterX; j < colEnd; ++j)          // column from index=kCenterX to index=(dataSizeX-kCenterX-1)
//         {
//             sum = 0;
//             t = 0;
//             for(m=0; m < kernelSizeY; ++m)
//             {
//                 for(n=0; n < kernelSizeX; ++n)
//                 {
//                     sum += *inPtr[t] * kernel[t];
//                     ++inPtr[t]; // in this partition, all cursors are used to convolve. moving cursors to next is safe here
//                     ++t;
//                 }
//             }
//
//             // store output
//             *outPtr = (unsigned char)((float)fabs(sum) + 0.5f);
//             ++outPtr;
//             ++x;
//         }
//
//         // partition #6 ***********************************
//         x = 1;
//         for(j=colEnd; j < dataSizeX; ++j)           // column from index=(dataSizeX-kCenter) to index=(dataSizeX-1)
//         {
//             sum = 0;
//             t = x;
//             for(m=0; m < kernelSizeY; ++m)
//             {
//                 for(n=x; n < kernelSizeX; ++n)
//                 {
//                     sum += *inPtr[t] * kernel[t];
//                     ++t;
//                 }
//                 t += x;
//             }
//
//             // store output
//             *outPtr = (unsigned char)((float)fabs(sum) + 0.5f);
//             ++outPtr;
//             ++x;
//             for(k=0; k < kSize; ++k) ++inPtr[k];    // move all cursors to next
//         }
//     }
//
//     // convolve rows from index=(dataSizeY-kCenterY) to index=(dataSizeY-1)
//     y = 1;
//     for(i= rowEnd; i < dataSizeY; ++i)               // number of rows
//     {
//         // partition #7 ***********************************
//         x = kCenterX;
//         for(j=0; j < kCenterX; ++j)                 // column from index=0 to index=kCenterX-1
//         {
//             sum = 0;
//             t = kernelSizeX * y;
//
//             for(m=y; m < kernelSizeY; ++m)
//             {
//                 for(n=0; n <= x; ++n)
//                 {
//                     sum += *inPtr[t] * kernel[t];
//                     ++t;
//                 }
//                 t += (kernelSizeX - x - 1);
//             }
//
//             // store output
//             *outPtr = (unsigned char)((float)fabs(sum) + 0.5f);
//             ++outPtr;
//             ++x;
//             for(k=0; k < kSize; ++k) ++inPtr[k];    // move all cursors to next
//         }
//
//         // partition #8 ***********************************
//         for(j=kCenterX; j < colEnd; ++j)            // column from index=kCenterX to index=(dataSizeX-kCenterX-1)
//         {
//             sum = 0;
//             t = kernelSizeX * y;
//             for(m=y; m < kernelSizeY; ++m)
//             {
//                 for(n=0; n < kernelSizeX; ++n)
//                 {
//                     sum += *inPtr[t] * kernel[t];
//                     ++t;
//                 }
//             }
//
//             // store output
//             *outPtr = (unsigned char)((float)fabs(sum) + 0.5f);
//             ++outPtr;
//             ++x;
//             for(k=0; k < kSize; ++k) ++inPtr[k];
//         }
//
//         // partition #9 ***********************************
//         x = 1;
//         for(j=colEnd; j < dataSizeX; ++j)           // column from index=(dataSizeX-kCenter) to index=(dataSizeX-1)
//         {
//             sum = 0;
//             t = kernelSizeX * y + x;
//             for(m=y; m < kernelSizeY; ++m)
//             {
//                 for(n=x; n < kernelSizeX; ++n)
//                 {
//                     sum += *inPtr[t] * kernel[t];
//                     ++t;
//                 }
//                 t += x;
//             }
//
//             // store output
//             *outPtr = (unsigned char)((float)fabs(sum) + 0.5f);
//             ++outPtr;
//             ++x;
//             for(k=0; k < kSize; ++k) ++inPtr[k];    // move all cursors to next
//         }
//
//         ++y;                                        // the starting row index is increased
//     }
//
//     return true;
// }

template <typename T>
double cpuConvolve(const T *a, const uint m, const uint n,
        const T *b, const uint p, const uint q,
        T *c) {
    const clock_t start = clock();
    cpuConvolveMedium<T>(a, m, n, b, p, q, c);
    return millis(start);
}

template <typename T>
void printMatrix(const T *a, const uint m, const uint n, const std::string name) {
    std::cout << name << std::endl;
    printMatrix<T>(a, m, n);
    std::cout << std::endl;
}

template <typename T>
void convolve(const T *a, const uint m, const uint n,
        const T *b, const uint p, const uint q) {
    std::cout << "convolving [" << m << ", " << n << "] by [" << p << ", " << q << "]" << std::endl;
    const size_t size = m * n * sizeof(T);
    T *cpuC = (T *) malloc(size);
    T *gpuC = (T *) malloc(size);

    time("CPU", cpuConvolve<T>(a, m, n, b, p, q, cpuC));
    time("GPU", gpuConvolve<T, BLOCK_SIZE>(a, m, n, b, p, q, gpuC));

    bool print = true;
    if (print) {
        printMatrix<T>(a, m, n, "In");
        printMatrix<T>(b, p, q, "Kernel");
        printMatrix<T>(cpuC, m, n, "CPU Result");
        printMatrix<T>(gpuC, m, n, "GPU Result");
    }

    if (memcmp(cpuC, gpuC, size) != 0) {
        std::cerr << "\nwrong\n" << std::endl;
        throw std::runtime_error("gpu convolution is wrong");
    }

    free(cpuC);
    free(gpuC);
}

template <typename T>
void convolve(const uint m, const uint n, const uint p, const uint q) {
    T *a = (T *) malloc(m * n * sizeof(T));
    T *b = (T *) malloc(p * q * sizeof(T));
    fill<T>(a, m * n);
    fill<T>(b, p * q);
    convolve<T>(a, m, n, b, p, q);
    free(a);
    free(b);
}

template <typename T>
int convolveCaught(const uint m, const uint n, const uint p, const uint q) {
    try {
        convolve<T>(m, n, p, q);
        return 0;
    } catch (const std::runtime_error &) {
        return -1;
    }
}

extern "C" {
    int convolveInt(const uint m, const uint n, const uint p, const uint q) {
        return convolveCaught<int>(m, n, p, q);
    }
}

extern "C" {
    int convolveFloat(const uint m, const uint n, const uint p, const uint q) {
        return convolveCaught<float>(m, n, p, q);
    }
}

extern "C" {
    int convolveChar(const uint m, const uint n, const uint p, const uint q) {
        return convolveCaught<char>(m, n, p, q);
    }
}

int main(const int argc, const char **argv) {
    const uint m = argc < 2 ? 16 : parseInt(argv[1]);
    const uint n = m;
    const uint p = 3;
    const uint q = 3;
    const char *type = argc < 4 ? "int" : argv[2];
    if (strEquals(type, "int")) {
        convolve<int>(m, n, p, q);
    } else if (strEquals(type, "float")) {
        convolve<float>(m, n, p, q);
    } else if (strEquals(type, "char")) {
        convolve<char>(m, n, p, q);
    } else {
        throw std::runtime_error("invalid numeric type");
    }
}
