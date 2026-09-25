#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

__global__ void maxKernel(const double* vector1,
                          const double* vector2,
                          double* result,
                          int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    while (idx < n) {
        result[idx] = vector1[idx] > vector2[idx]
                          ? vector1[idx]
                          : vector2[idx];

        idx += offset;
    }
}

int main() {
    int choice;

    std::cout << "Choose input size:\n";
    std::cout << "1 - 1000\n";
    std::cout << "2 - 100000\n";
    std::cout << "3 - 1000000\n";
    std::cout << "4 - 10000000\n";
    std::cout << "Choice: ";
    std::cin >> choice;

    int n;

    switch (choice) {
        case 1:
            n = 1000;
            break;
        case 2:
            n = 100000;
            break;
        case 3:
            n = 1000000;
            break;
        case 4:
            n = 10000000;
            break;
        default:
            std::cout << "Invalid choice\n";
            return 1;
    }

    std::vector<double> vector1(n);
    std::vector<double> vector2(n);

    for (int i = 0; i < n; ++i) {
        vector1[i] = i;
        vector2[i] = n - i;
    }


    std::vector<double> cpu_result(n);

    auto cpu_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n; ++i) {
        cpu_result[i] = vector1[i] > vector2[i]
                            ? vector1[i]
                            : vector2[i];
    }

    auto cpu_end = std::chrono::high_resolution_clock::now();

    double cpu_time =
        std::chrono::duration<double, std::milli>(
            cpu_end - cpu_start
        ).count();


    std::vector<double> gpu_result(n);

    double* dev_vector1;
    double* dev_vector2;
    double* dev_result;

    cudaMalloc(&dev_vector1, n * sizeof(double));
    cudaMalloc(&dev_vector2, n * sizeof(double));
    cudaMalloc(&dev_result, n * sizeof(double));

    cudaMemcpy(dev_vector1, vector1.data(),
               n * sizeof(double),
               cudaMemcpyHostToDevice);

    cudaMemcpy(dev_vector2, vector2.data(),
               n * sizeof(double),
               cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    maxKernel<<<blocks, threads>>>(
        dev_vector1,
        dev_vector2,
        dev_result,
        n
    );

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    maxKernel<<<blocks, threads>>>(
        dev_vector1,
        dev_vector2,
        dev_result,
        n
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time;

    cudaEventElapsedTime(
        &gpu_time,
        start,
        stop
    );

    cudaMemcpy(gpu_result.data(), dev_result,
               n * sizeof(double),
               cudaMemcpyDeviceToHost);


    bool correct = true;

    for (int i = 0; i < n; ++i) {
        if (cpu_result[i] != gpu_result[i]) {
            correct = false;
            break;
        }
    }


    std::cout << "\n";
    std::cout << "====================================\n";
    std::cout << "n = " << n << "\n";
    std::cout << "CPU time: " << cpu_time << " ms\n";
    std::cout << "GPU time: " << gpu_time << " ms\n";
    std::cout << "Blocks: " << blocks << "\n";
    std::cout << "Threads: " << threads << "\n";
    std::cout << "Results: "
              << (correct ? "OK" : "ERROR")
              << "\n";
    std::cout << "====================================\n";


    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(dev_vector1);
    cudaFree(dev_vector2);
    cudaFree(dev_result);

    return 0;
}