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

template <typename T>
void scan(const uint n, const bool print) {
    std::cout << "n = " << n << std::endl;
    if (n == 0) {
        return;
    }
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
        std::cerr << "\nwrong\n" << std::endl;
        if (print) {
            std::cout << "cpuSums" << std::endl;
            printArray(cpuSums, n);
        }
        throw std::runtime_error("gpu scan is wrong");
    }

    free(a);
    free(cpuSums);
    free(gpuSums);
}

template <typename T>
int scanCaught(const uint n, const bool print) {
    try {
        scan<T>(n, print);
        return 0;
    } catch (const std::runtime_error &) {
        return -1;
    }
}

extern "C" {
    int scanInt(const uint n, const bool print) {
        return scanCaught<int>(n, print);
    }
}

extern "C" {
    int scanFloat(const uint n, const bool print) {
        return scanCaught<float>(n, print);
    }
}

extern "C" {
    int scanChar(const uint n, const bool print) {
        return scanCaught<char>(n, print);
    }
}

int main(const int argc, const char **argv) {
    const uint n = argc < 2 ? 64 : parseInt(argv[1]);
    const bool print = argc < 3 ? false : parseBool(argv[2]);
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
