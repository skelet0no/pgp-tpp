#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <iostream>

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

int main()
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    int n;
    if (!(std::cin >> n) || n < 1 || n > 10000) {
        std::cerr << "Invalid matrix size\n";
        return EXIT_FAILURE;
    }
    std::vector<double> matrix(static_cast<size_t>(n) * (n + 1));
    for (int row = 0; row < n; ++row)
        for (int column = 0; column < n; ++column)
            std::cin >> matrix[static_cast<size_t>(column) * n + row];
    for (int row = 0; row < n; ++row)
        std::cin >> matrix[static_cast<size_t>(n) * n + row];
    if (!std::cin) {
        std::cerr << "Cannot read system coefficients\n";
        return EXIT_FAILURE;
    }
    std::vector<double> solution;
    if (!solveCPU(matrix, solution, n)) {
        std::cerr << "System has no unique solution\n";
        return EXIT_FAILURE;
    }
    std::cout << std::scientific << std::setprecision(10);
    for (int i = 0; i < n; ++i)
        std::cout << solution[i] << (i + 1 == n ? '\n' : ' ');
    return EXIT_SUCCESS;
}
