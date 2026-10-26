#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)      \
                      << " (" << __FILE__ << ":" << __LINE__ << ")";  \
            std::exit(EXIT_FAILURE);                                    \
        }                                                               \
    } while (0)

constexpr int MAX_CLASSES = 32;

__constant__ double3 classAverages[MAX_CLASSES];

__global__
void classifyKernel(uchar4* pixels, size_t pixelCount, int classCount)
{
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t offset = static_cast<size_t>(blockDim.x) * gridDim.x;

    while (idx < pixelCount) {
        uchar4 pixel = pixels[idx];
        double minDistance = 0.0;
        int bestClass = 0;

        for (int j = 0; j < classCount; ++j) {
            const double3 average = classAverages[j];
            const double dr = pixel.x - average.x;
            const double dg = pixel.y - average.y;
            const double db = pixel.z - average.z;
            const double distance = dr * dr + dg * dg + db * db;

            if (j == 0 || distance < minDistance) {
                minDistance = distance;
                bestClass = j;
            }
        }

        pixel.w = static_cast<unsigned char>(bestClass);
        pixels[idx] = pixel;
        idx += offset;
    }
}


void classifyCPU(std::vector<uchar4>& pixels, const std::vector<double3>& averages)
{
    for (uchar4& pixel : pixels) {
        double minDistance = 0.0;
        int bestClass = 0;

        for (size_t j = 0; j < averages.size(); ++j) {
            const double3 average = averages[j];
            const double dr = pixel.x - average.x;
            const double dg = pixel.y - average.y;
            const double db = pixel.z - average.z;
            const double distance = dr * dr + dg * dg + db * db;

            if (j == 0 || distance < minDistance) {
                minDistance = distance;
                bestClass = static_cast<int>(j);
            }
        }

        pixel.w = static_cast<unsigned char>(bestClass);
    }
}

void fillImage(std::vector<uchar4>& pixels)
{
    uint32_t state = 12345;
    for (uchar4& p : pixels) {
        state = state * 1664525u + 1013904223u;
        p = make_uchar4(state & 255, (state >> 8) & 255,
                       (state >> 16) & 255, (state >> 24) & 255);
    }
}

std::vector<double3> makeAverages(const std::vector<uchar4>& pixels, int classCount)
{
    constexpr int sampleCount = 7;
    std::vector<double3> averages(classCount);
    for (int j = 0; j < classCount; ++j) {
        double3 sum = make_double3(0.0, 0.0, 0.0);
        for (int i = 0; i < sampleCount; ++i) {
            const uchar4 pixel = pixels[(static_cast<size_t>(j) * 97 + i * 13) % pixels.size()];
            sum.x += pixel.x;
            sum.y += pixel.y;
            sum.z += pixel.z;
        }
        averages[j] = make_double3(sum.x / sampleCount, sum.y / sampleCount,
                                   sum.z / sampleCount);
    }
    return averages;
}

bool checkResult(const std::vector<uchar4>& expected,
                 const std::vector<uchar4>& actual)
{
    if (expected.size() != actual.size()) return false;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i].x != actual[i].x || expected[i].y != actual[i].y ||
            expected[i].z != actual[i].z || expected[i].w != actual[i].w) {
            std::cerr << "Mismatch at pixel " << i << "\n";
            return false;
        }
    }
    return true;
}

int main()
{
    constexpr int repeats = 10;
    bool correct = true;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "\n"
              << "Compute capability: " << prop.major << "." << prop.minor << "\n"
              << "Max threads per block: " << prop.maxThreadsPerBlock << "\n"
              << "Times are averages over " << repeats << " runs, in ms.\n"
              << "GPU timing excludes allocation and memory transfers.\n";

    struct TestCase {
        int width, height, classCount;
    };
    const std::vector<TestCase> cases = {
        {1, 1, 1}, {1, 257, 2}, {257, 1, 32}, {33, 17, 8},
        {256, 256, 8}, {1024, 1024, 16}, {4096, 4096, 32}
    };
    for (const auto& test : cases) {
        const size_t pixelCount = static_cast<size_t>(test.width) * test.height;
        const int classCount = test.classCount;
        std::vector<uchar4> hostInput(pixelCount);
        std::vector<uchar4> hostOutput(pixelCount);
        fillImage(hostInput);
        const std::vector<double3> averages = makeAverages(hostInput, classCount);
        std::vector<uchar4> expected = hostInput;
        classifyCPU(expected, averages);
        std::cout << "\nImage: " << test.width << " x " << test.height
                  << ", classes: " << classCount << "\n";

        CUDA_CHECK(cudaMemcpyToSymbol(classAverages, averages.data(),
                                      classCount * sizeof(double3)));
        uchar4* devicePixels = nullptr;
        CUDA_CHECK(cudaMalloc(&devicePixels, pixelCount * sizeof(uchar4)));
        CUDA_CHECK(cudaMemcpy(devicePixels, hostInput.data(),
                              pixelCount * sizeof(uchar4), cudaMemcpyHostToDevice));
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        std::cout << "Blocks  Threads  Time (ms)  Result\n";
        for (int blockCount : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}) {
            for (int threadCount : {32, 64, 128, 256, 512, 1024}) {
                if (threadCount > prop.maxThreadsPerBlock ||
                    threadCount > prop.maxThreadsDim[0] ||
                    blockCount > prop.maxGridSize[0]) continue;
                classifyKernel<<<blockCount, threadCount>>>(devicePixels, pixelCount, classCount);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                float totalTime = 0.0f;
                for (int i = 0; i < repeats; ++i) {
                    CUDA_CHECK(cudaEventRecord(start));
                    classifyKernel<<<blockCount, threadCount>>>(devicePixels, pixelCount, classCount);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaEventRecord(stop));
                    CUDA_CHECK(cudaEventSynchronize(stop));
                    float milliseconds;
                    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
                    totalTime += milliseconds;
                }
                CUDA_CHECK(cudaMemcpy(hostOutput.data(), devicePixels,
                                      pixelCount * sizeof(uchar4), cudaMemcpyDeviceToHost));
                const bool matches = checkResult(expected, hostOutput);
                correct = matches && correct;
                std::cout << std::setw(6) << blockCount << std::setw(9) << threadCount
                          << std::setw(12) << std::fixed << std::setprecision(6)
                          << totalTime / repeats << "  " << (matches ? "OK" : "ERROR") << "\n";
            }
        }
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(devicePixels));
    }
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
