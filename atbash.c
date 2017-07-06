#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef unsigned int uint;

void atbash(char const *in, char *out, uint n) {
    for (uint i = 0; i < n; i++) {
        out[n - 1 - i] = in[i];
    }
}

int main(int argc, char const *argv[]) {
    if (argc < 2) {
        return -1;
    }
    char const *in = argv[1];
    uint n = strlen(in);
    char *out = (char *) malloc(n);
    atbash(in, out, n);
    printf("in: %s\n", in);
    printf("out: %s\n", out);
    free(out);
    return 0;
}
