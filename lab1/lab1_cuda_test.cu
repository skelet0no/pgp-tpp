#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>


__global__ void maxKernel(const double* vector1,
                          const double* vector2,
                          double* result,
                          int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    while (idx < n) {
        result[idx] = vector1[idx] > vector2[idx] ? vector1[idx] : vector2[idx];
        idx += offset;
    }
}


void checkCuda(cudaError_t error)
{
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: "
                  << cudaGetErrorString(error)
                  << std::endl;
        exit(1);
    }
}


int main()
{
    std::vector<int> sizes = {
        1000,
        1000000,
        10000000
    };

    const int repeats = 10;

    std::vector<int> blocks = {
        1, 2, 4, 8, 16, 32,
        64, 128, 256, 512, 1024
    };

    std::vector<int> threads = {
        32, 64, 128, 256, 512, 1024
    };


    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, 0));

    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Compute capability: "
              << prop.major << "." << prop.minor << "\n";
    std::cout << "Max threads per block: "
              << prop.maxThreadsPerBlock << "\n\n";


    if (prop.maxThreadsPerBlock < 1024) {
        std::cerr << "GPU does not support 1024 threads per block."
                  << std::endl;
        return 1;
    }


    for (int n : sizes) {

        std::cout << "\n";
        std::cout << "============================================================\n";
        std::cout << "n = " << n << "\n";
        std::cout << "Time: milliseconds (ms)\n";
        std::cout << "============================================================\n\n";


        std::vector<double> vector1(n);
        std::vector<double> vector2(n);
        std::vector<double> result(n);


        for (int i = 0; i < n; i++) {
            vector1[i] = i;
            vector2[i] = n - i;
        }


        double* dev_vector1;
        double* dev_vector2;
        double* dev_result;


        checkCuda(cudaMalloc(
            &dev_vector1,
            n * sizeof(double)
        ));

        checkCuda(cudaMalloc(
            &dev_vector2,
            n * sizeof(double)
        ));

        checkCuda(cudaMalloc(
            &dev_result,
            n * sizeof(double)
        ));


        checkCuda(cudaMemcpy(
            dev_vector1,
            vector1.data(),
            n * sizeof(double),
            cudaMemcpyHostToDevice
        ));

        checkCuda(cudaMemcpy(
            dev_vector2,
            vector2.data(),
            n * sizeof(double),
            cudaMemcpyHostToDevice
        ));


        cudaEvent_t start;
        cudaEvent_t stop;

        checkCuda(cudaEventCreate(&start));
        checkCuda(cudaEventCreate(&stop));


        std::cout
            << std::setw(10) << "Blocks"
            << std::setw(12) << "Threads"
            << std::setw(18) << "Time (ms)"
            << "\n";

        std::cout
            << std::string(40, '-')
            << "\n";


        for (int blockCount : blocks) {

            for (int threadCount : threads) {

                if (threadCount > prop.maxThreadsPerBlock)
                    continue;


                maxKernel<<<blockCount, threadCount>>>(
                    dev_vector1,
                    dev_vector2,
                    dev_result,
                    n
                );

                checkCuda(cudaGetLastError());
                checkCuda(cudaDeviceSynchronize());


                float totalTime = 0.0f;


                for (int i = 0; i < repeats; i++) {

                    checkCuda(cudaEventRecord(start));

                    maxKernel<<<blockCount, threadCount>>>(
                        dev_vector1,
                        dev_vector2,
                        dev_result,
                        n
                    );

                    checkCuda(cudaEventRecord(stop));

                    checkCuda(cudaEventSynchronize(stop));


                    float milliseconds;

                    checkCuda(cudaEventElapsedTime(
                        &milliseconds,
                        start,
                        stop
                    ));

                    totalTime += milliseconds;
                }


                float averageTime = totalTime / repeats;


                std::cout
                    << std::setw(10) << blockCount
                    << std::setw(12) << threadCount
                    << std::setw(18)
                    << std::fixed
                    << std::setprecision(6)
                    << averageTime
                    << "\n";
            }
        }


        checkCuda(cudaEventDestroy(start));
        checkCuda(cudaEventDestroy(stop));


        checkCuda(cudaMemcpy(
            result.data(),
            dev_result,
            n * sizeof(double),
            cudaMemcpyDeviceToHost
        ));


        bool correct = true;

        for (int i = 0; i < std::min(n, 10); i++) {

            double expected =
                vector1[i] > vector2[i]
                    ? vector1[i]
                    : vector2[i];

            if (result[i] != expected) {
                correct = false;
                break;
            }
        }


        std::cout << "\nResult check: "
                  << (correct ? "OK" : "ERROR")
                  << "\n";


        checkCuda(cudaFree(dev_vector1));
        checkCuda(cudaFree(dev_vector2));
        checkCuda(cudaFree(dev_result));
    }


    return 0;
}