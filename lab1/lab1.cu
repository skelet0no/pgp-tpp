#include <iostream>
#include <vector>

__global__ void maxKernel(const double* vector1, const double* vector2,
                          double* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    while (idx < n) {
        result[idx] = vector1[idx] > vector2[idx] ? vector1[idx] : vector2[idx];
        idx += offset;
    }
}

int main() {
    int n;
    std::cin >> n;

    std::vector<double> vector1(n);
    std::vector<double> vector2(n);
    std::vector<double> result(n);

    for (double& x : vector1)
        std::cin >> x;

    for (double& x : vector2)
        std::cin >> x;

    double* dev_vector1;
    double* dev_vector2;
    double* dev_result;

    cudaMalloc(&dev_vector1, n * sizeof(double));
    cudaMalloc(&dev_vector2, n * sizeof(double));
    cudaMalloc(&dev_result, n * sizeof(double));

    cudaMemcpy(dev_vector1, vector1.data(),
               n * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(dev_vector2, vector2.data(),
               n * sizeof(double), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    maxKernel<<<blocks, threads>>>(
        dev_vector1,
        dev_vector2,
        dev_result,
        n
    );

    cudaDeviceSynchronize();

    cudaMemcpy(result.data(), dev_result,
               n * sizeof(double), cudaMemcpyDeviceToHost);

    for (double x : result)
        std::cout << std::scientific << x << ' ';

    std::cout << '\n';

    cudaFree(dev_vector1);
    cudaFree(dev_vector2);
    cudaFree(dev_result);

    return 0;
}