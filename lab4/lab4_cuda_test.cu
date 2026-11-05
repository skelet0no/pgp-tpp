#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cstdint>
#include <string>
#include <chrono>

#define CUDA_CHECK(call)                                                \
    do {                                                               \
        cudaError_t err = (call);                                      \
        if (err != cudaSuccess) {                                      \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)     \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            std::exit(EXIT_FAILURE);                                   \
        }                                                              \
    } while (0)

struct AbsoluteLess {
    __host__ __device__
    bool operator()(double lhs, double rhs) const
    {
        return fabs(lhs) < fabs(rhs);
    }
};

__global__
void swapRowsKernel(double* matrix, int n, int pivot, int bestRow)
{
    const int offset = blockDim.x * gridDim.x;

    for (int column = pivot + blockIdx.x * blockDim.x + threadIdx.x;
         column <= n; column += offset) {
        const size_t base = static_cast<size_t>(column) * n;
        const double value = matrix[base + pivot];
        matrix[base + pivot] = matrix[base + bestRow];
        matrix[base + bestRow] = value;
    }
}

__global__
void eliminateKernel(double* matrix, int n, int pivot)
{
    const int firstRow = pivot + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    const int firstColumn = pivot + 1 + blockIdx.y * blockDim.y + threadIdx.y;
    const int rowOffset = blockDim.x * gridDim.x;
    const int columnOffset = blockDim.y * gridDim.y;
    const size_t pivotBase = static_cast<size_t>(pivot) * n;
    const double diagonal = matrix[pivotBase + pivot];

    for (int row = firstRow; row < n; row += rowOffset) {
        const double factor = matrix[pivotBase + row] / diagonal;

        for (int column = firstColumn; column <= n; column += columnOffset) {
            const size_t base = static_cast<size_t>(column) * n;
            matrix[base + row] -= factor * matrix[base + pivot];
        }
    }
}

__global__
void backSubstituteKernel(double* matrix, double* solution, int n, int pivot)
{
    const size_t rhsBase = static_cast<size_t>(n) * n;
    const size_t pivotBase = static_cast<size_t>(pivot) * n;
    const double value = matrix[rhsBase + pivot] / matrix[pivotBase + pivot];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int offset = blockDim.x * gridDim.x;

    if (idx == 0) solution[pivot] = value;

    for (int row = idx; row < pivot; row += offset)
        matrix[rhsBase + row] -= matrix[pivotBase + row] * value;
}

// The matrix is column-major; the last column is the right-hand side.
bool solveGPU(double* deviceMatrix, double* deviceSolution, int n,
                     dim3 block = dim3(32, 8), dim3 gridLimit = dim3(32, 32))
{
    constexpr int THREADS = 256;
    for (int pivot = 0; pivot < n; ++pivot) {
        auto column = thrust::device_pointer_cast(
            deviceMatrix + static_cast<size_t>(pivot) * n);
        auto best = thrust::max_element(column + pivot, column + n, AbsoluteLess{});
        const int bestRow = static_cast<int>(best - column);
        double pivotValue;
        CUDA_CHECK(cudaMemcpy(&pivotValue, deviceMatrix + static_cast<size_t>(pivot) * n + bestRow,
                              sizeof(double), cudaMemcpyDeviceToHost));

        if (pivotValue == 0.0) {
            return false;
        }

        if (bestRow != pivot) {
            swapRowsKernel<<<(n - pivot + THREADS) / THREADS, THREADS>>>(
                deviceMatrix, n, pivot, bestRow);
            CUDA_CHECK(cudaGetLastError());
        }

        if (pivot + 1 < n) {
            const dim3 grid(
                std::min((n - pivot - 1 + static_cast<int>(block.x) - 1) / static_cast<int>(block.x), static_cast<int>(gridLimit.x)),
                std::min((n - pivot + static_cast<int>(block.y) - 1) / static_cast<int>(block.y), static_cast<int>(gridLimit.y)));
            eliminateKernel<<<grid, block>>>(deviceMatrix, n, pivot);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    for (int pivot = n - 1; pivot >= 0; --pivot) {
        const int blocks = std::max(1, (pivot + THREADS - 1) / THREADS);
        backSubstituteKernel<<<blocks, THREADS>>>(deviceMatrix, deviceSolution, n, pivot);
        CUDA_CHECK(cudaGetLastError());
    }

    return true;
}

bool solveCPU(std::vector<double>& matrix, std::vector<double>& solution, int n)
{
    for (int pivot = 0; pivot < n; ++pivot) {
        const size_t pivotBase = static_cast<size_t>(pivot) * n;
        int bestRow = pivot;
        for (int row = pivot + 1; row < n; ++row)
            if (std::abs(matrix[pivotBase + row]) > std::abs(matrix[pivotBase + bestRow]))
                bestRow = row;
        if (matrix[pivotBase + bestRow] == 0.0) return false;

        if (bestRow != pivot)
            for (int column = pivot; column <= n; ++column)
                std::swap(matrix[static_cast<size_t>(column) * n + pivot],
                          matrix[static_cast<size_t>(column) * n + bestRow]);

        for (int row = pivot + 1; row < n; ++row)
            matrix[pivotBase + row] /= matrix[pivotBase + pivot];
        for (int column = pivot + 1; column <= n; ++column) {
            const size_t base = static_cast<size_t>(column) * n;
            const double value = matrix[base + pivot];
            for (int row = pivot + 1; row < n; ++row)
                matrix[base + row] -= matrix[pivotBase + row] * value;
        }
    }

    solution.resize(n);
    const size_t rhsBase = static_cast<size_t>(n) * n;
    for (int pivot = n - 1; pivot >= 0; --pivot) {
        const size_t base = static_cast<size_t>(pivot) * n;
        solution[pivot] = matrix[rhsBase + pivot] / matrix[base + pivot];
        for (int row = 0; row < pivot; ++row)
            matrix[rhsBase + row] -= matrix[base + row] * solution[pivot];
    }
    return true;
}

struct SystemCase {
    std::string name;
    int n;
    std::vector<double> matrix;
    std::vector<double> solution;
    bool unique = true;
};

SystemCase makeSystem(int n, bool permute = true)
{
    SystemCase test{"generated", n, std::vector<double>(static_cast<size_t>(n) * (n + 1)),
                    std::vector<double>(n), true};
    uint32_t state = 12345;
    for (int i = 0; i < n; ++i) test.solution[i] = (i % 17 - 8) / 4.0;
    for (int row = 0; row < n; ++row) {
        double sum = 0.0;
        for (int column = 0; column < n; ++column) {
            state = state * 1664525u + 1013904223u;
            const double value = (static_cast<int>((state >> 16) % 201) - 100) / 100.0;
            test.matrix[static_cast<size_t>(column) * n + row] = value;
            sum += std::abs(value);
        }
        test.matrix[static_cast<size_t>(row) * n + row] += sum + 1.0;
        long double rhs = 0.0;
        for (int column = 0; column < n; ++column)
            rhs += static_cast<long double>(test.matrix[static_cast<size_t>(column) * n + row])
                   * test.solution[column];
        test.matrix[static_cast<size_t>(n) * n + row] = static_cast<double>(rhs);
    }
    if (permute)
        for (int column = 0; column <= n; ++column)
            for (int row = 0; row < n / 2; ++row)
                std::swap(test.matrix[static_cast<size_t>(column) * n + row],
                          test.matrix[static_cast<size_t>(column) * n + n - 1 - row]);
    return test;
}

std::vector<SystemCase> correctnessCases()
{
    std::vector<SystemCase> cases = {
        {"sample", 2, {1, 3, 2, 4, 5, 6}, {-4, 4.5}, true},
        {"scalar", 1, {-2, 6}, {-3}, true},
        {"zero leading pivot", 2, {0, 2, 1, 3, 2, 8}, {1, 2}, true},
        {"negative largest pivot", 2, {1, -4, 2, 3, -1, -11}, {1.7272727272727273, -1.3636363636363636}, true},
        {"tiny pivot", 1, {1e-200, 2e-200}, {2}, true},
        {"zero solution", 2, {2, 0, 0, 3, 0, 0}, {0, 0}, true},
        {"singular", 2, {1, 2, 2, 4, 3, 6}, {}, false},
        {"inconsistent", 2, {1, 2, 2, 4, 3, 7}, {}, false}
    };
    for (int n : {3, 17, 31, 32, 33, 65, 129, 257}) cases.push_back(makeSystem(n));
    return cases;
}

bool checkSolution(const SystemCase& test, const std::vector<double>& actual)
{
    if (actual.size() != test.solution.size()) return false;
    constexpr double tolerance = 1e-10;
    double xNorm = 0.0;
    for (int i = 0; i < test.n; ++i) {
        if (!std::isfinite(actual[i]) ||
            std::abs(actual[i] - test.solution[i]) > tolerance * std::max(1.0, std::abs(test.solution[i]))) {
            std::cerr << "Solution mismatch in " << test.name << " at " << i << '\n';
            return false;
        }
        xNorm = std::max(xNorm, std::abs(actual[i]));
    }
    long double residual = 0.0, aNorm = 0.0, bNorm = 0.0;
    for (int row = 0; row < test.n; ++row) {
        long double sum = 0.0, rowNorm = 0.0;
        for (int column = 0; column < test.n; ++column) {
            const double a = test.matrix[static_cast<size_t>(column) * test.n + row];
            sum += static_cast<long double>(a) * actual[column];
            rowNorm += std::abs(a);
        }
        const long double rhs = test.matrix[static_cast<size_t>(test.n) * test.n + row];
        residual = std::max(residual, std::abs(sum - rhs));
        aNorm = std::max(aNorm, rowNorm);
        bNorm = std::max(bNorm, std::abs(rhs));
    }
    const long double scale = aNorm * xNorm + bNorm;
    return scale == 0.0 ? residual == 0.0 : residual / scale <= tolerance;
}

struct GPUResult {
    bool unique;
    double solveMs;
    double totalMs;
    std::vector<double> solution;
};

class GPUWorkspace {
    double* matrix = nullptr;
    double* solution = nullptr;
    int n;
public:
    explicit GPUWorkspace(int size) : n(size)
    {
        CUDA_CHECK(cudaMalloc(&matrix, static_cast<size_t>(n) * (n + 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&solution, n * sizeof(double)));
    }
    ~GPUWorkspace()
    {
        CUDA_CHECK(cudaFree(solution));
        CUDA_CHECK(cudaFree(matrix));
    }
    GPUWorkspace(const GPUWorkspace&) = delete;
    GPUWorkspace& operator=(const GPUWorkspace&) = delete;

    GPUResult run(const SystemCase& test, dim3 block = dim3(32, 8), dim3 grid = dim3(32, 32))
    {
        GPUResult result{false, 0.0, 0.0, std::vector<double>(n)};
        const auto totalStart = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaMemcpy(matrix, test.matrix.data(), test.matrix.size() * sizeof(double),
                              cudaMemcpyHostToDevice));
        const auto solveStart = std::chrono::steady_clock::now();
        result.unique = solveGPU(matrix, solution, n, block, grid);
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto solveStop = std::chrono::steady_clock::now();
        if (result.unique)
            CUDA_CHECK(cudaMemcpy(result.solution.data(), solution, n * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        const auto totalStop = std::chrono::steady_clock::now();
        result.solveMs = std::chrono::duration<double, std::milli>(solveStop - solveStart).count();
        result.totalMs = std::chrono::duration<double, std::milli>(totalStop - totalStart).count();
        return result;
    }
};

bool supportsConfiguration(const cudaDeviceProp& prop, dim3 block, dim3 grid)
{
    return block.x * block.y <= static_cast<unsigned>(prop.maxThreadsPerBlock) &&
           block.x <= static_cast<unsigned>(prop.maxThreadsDim[0]) &&
           block.y <= static_cast<unsigned>(prop.maxThreadsDim[1]) &&
           grid.x <= static_cast<unsigned>(prop.maxGridSize[0]) &&
           grid.y <= static_cast<unsigned>(prop.maxGridSize[1]);
}

int main()
{
    constexpr int repeats = 3;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "\nCompute capability: "
              << prop.major << '.' << prop.minor << '\n'
              << "Average solve wall time over " << repeats << " runs, ms.\n";
    bool correct = true;
    const auto cases = correctnessCases();
    for (const dim3 block : {dim3(16, 16), dim3(32, 4), dim3(32, 8), dim3(32, 16)}) {
        for (const dim3 grid : {dim3(1, 1), dim3(8, 8), dim3(32, 32)}) {
            if (!supportsConfiguration(prop, block, grid)) continue;
            std::cout << "\nBlock: " << block.x << 'x' << block.y
                      << ", grid limit: " << grid.x << 'x' << grid.y << '\n';
            for (const auto& test : cases) {
                auto cpuMatrix = test.matrix;
                std::vector<double> cpuSolution;
                const bool cpuUnique = solveCPU(cpuMatrix, cpuSolution, test.n);
                bool matches = cpuUnique == test.unique && (!cpuUnique || checkSolution(test, cpuSolution));
                GPUWorkspace workspace(test.n);
                const auto warmup = workspace.run(test, block, grid);
                matches = (warmup.unique == test.unique &&
                           (!warmup.unique || checkSolution(test, warmup.solution))) && matches;
                double total = 0.0;
                for (int i = 0; i < repeats; ++i) {
                    const auto result = workspace.run(test, block, grid);
                    total += result.solveMs;
                    matches = (result.unique == test.unique &&
                               (!result.unique || checkSolution(test, result.solution))) && matches;
                }
                correct = matches && correct;
                std::cout << "n=" << std::setw(4) << test.n << ' ' << std::setw(24) << test.name
                          << ' ' << std::fixed << std::setprecision(6) << total / repeats
                          << " ms  " << (matches ? "OK" : "ERROR") << '\n';
            }
        }
    }
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
