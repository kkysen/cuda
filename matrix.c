#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void printMat(float *a, uint m, uint n) {
    for (uint i = 0; i < m; ++i) {
        printf("{");
        for (uint j = 0; j < n;) {
            printf("%.0f", a[i * n + j]);
            if (++j != n) {
                printf(", ");
            }
        }
        printf("}\n");
    }
}

void transpose(float *a, float *b, uint n) {
    for (uint i = 0; i < n; ++i) {
        for (uint j = 0; j < n; ++j) {
            b[j * n + i] = a[i * n + j];
        }
    }
}

float * transposeOut(float *a, uint n) {
    float *b = (float *) malloc(n * n * sizeof(float));
    transpose(a, b, n);
    return b;
}

void matMul(float *a, float *b, float *c, uint m, uint n, uint p) {
    printf("multiplying\n");
    printMat(a, m, n);
    printf("by\n");
    printMat(b, n, p);
    printf("into\n");
    printMat(c, m, p);
    for (uint i = 0; i < m; ++i) {
        for (uint j = 0; j < p; ++j) {
            float val = 0;
            for (uint k = 0; k < n; ++k) {
                val += a[i * n + k] * b[k * p + j];
            }
            c[i * p  +j] = val;
        }
    }
    printf("result\n");
    printMat(c, m, p);
}

float * matMulOut(float *a, float *b, uint m, uint n, uint p) {
    float *c = (float *) malloc(m * p * sizeof(float));
    matMul(a, b, c, m, n, p);
    return c;
}

int main() {
    const uint n = 3;
    size_t size = n * n * sizeof(float);
    float *a = malloc(size);
    float *b = malloc(size);
    for (uint i = 0; i < n * n; ++i) {
        a[i] = i + 1;
        b[i] = a[i] * 2;
    }
    float *c = matMulOut(a, b, n, n, n);
}
