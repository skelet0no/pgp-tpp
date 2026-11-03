#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

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

int main()
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n;
    if (!(std::cin >> n) || n < 1 || n > 10000) {
        std::cerr << "Invalid matrix size\n";
        return EXIT_FAILURE;
    }

    const size_t elementCount = static_cast<size_t>(n) * (n + 1);
    std::vector<double> matrix(elementCount);

    for (int row = 0; row < n; ++row)
        for (int column = 0; column < n; ++column)
            std::cin >> matrix[static_cast<size_t>(column) * n + row];

    for (int row = 0; row < n; ++row)
        std::cin >> matrix[static_cast<size_t>(n) * n + row];

    if (!std::cin) {
        std::cerr << "Can not read system coefficients\n";
        return EXIT_FAILURE;
    }

    double* deviceMatrix = nullptr;
    double* deviceSolution = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceMatrix, elementCount * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&deviceSolution, n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(deviceMatrix, matrix.data(),
                          elementCount * sizeof(double), cudaMemcpyHostToDevice));

    if (!solveGPU(deviceMatrix, deviceSolution, n)) {
        std::cerr << "System has no unique solution\n";
        CUDA_CHECK(cudaFree(deviceSolution));
        CUDA_CHECK(cudaFree(deviceMatrix));
        return EXIT_FAILURE;
    }

    std::vector<double> solution(n);
    CUDA_CHECK(cudaMemcpy(solution.data(), deviceSolution,
                          n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(deviceSolution));
    CUDA_CHECK(cudaFree(deviceMatrix));

    std::cout << std::scientific << std::setprecision(10);
    for (int i = 0; i < n; ++i)
        std::cout << solution[i] << (i + 1 == n ? '\n' : ' ');

    return EXIT_SUCCESS;
}
