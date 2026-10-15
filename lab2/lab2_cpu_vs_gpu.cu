#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <utility>
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

__host__ __device__ __forceinline__
float grayscale(uchar4 p)
{
    return 0.299f * p.x + 0.587f * p.y + 0.114f * p.z;
}

__global__
void robertsKernel(cudaTextureObject_t texture, uchar4* output, int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const float fx = static_cast<float>(x) + 0.5f;
    const float fy = static_cast<float>(y) + 0.5f;

    const uchar4 a = tex2D<uchar4>(texture, fx, fy);
    const uchar4 b = tex2D<uchar4>(texture, fx + 1.0f, fy);
    const uchar4 c = tex2D<uchar4>(texture, fx, fy + 1.0f);
    const uchar4 d = tex2D<uchar4>(texture, fx + 1.0f, fy + 1.0f);

    const float ga = grayscale(a);
    const float gb = grayscale(b);
    const float gc = grayscale(c);
    const float gd = grayscale(d);

    const float dx = ga - gd;
    const float dy = gb - gc;

    float value = sqrtf(dx * dx + dy * dy);

    value = fminf(value, 255.0f);

    const uint8_t result = static_cast<uint8_t>(value);

    output[y * width + x] = make_uchar4(result, result, result, 0);
}


void robertsCPU(const std::vector<uchar4>& input, std::vector<uchar4>& output,
                   int width, int height)
{
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int nextX = std::min(x + 1, width - 1);
            const int nextY = std::min(y + 1, height - 1);

            const uchar4 a = input[y * width + x];
            const uchar4 b = input[y * width + nextX];
            const uchar4 c = input[nextY * width + x];
            const uchar4 d = input[nextY * width + nextX];

            const float ga = grayscale(a);
            const float gb = grayscale(b);
            const float gc = grayscale(c);
            const float gd = grayscale(d);

            const float dx = ga - gd;
            const float dy = gb - gc;

            float value = sqrtf(dx * dx + dy * dy);
            value = fminf(value, 255.0f);

            const uint8_t result = static_cast<uint8_t>(value);
            output[y * width + x] = make_uchar4(result, result, result, 0);
        }
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

bool checkResult(const std::vector<uchar4>& expected,
                 const std::vector<uchar4>& actual)
{
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(int(expected[i].x) - int(actual[i].x)) > 1 ||
            actual[i].x != actual[i].y || actual[i].x != actual[i].z ||
            actual[i].w != 0) {
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
    const std::vector<std::pair<int, int>> sizes = {
        {1, 1}, {1, 257}, {257, 1}, {33, 17}, {256, 256}, {1024, 1024}, {4096, 4096}
    };
    for (const auto& size : sizes) {
        const int width = size.first;
        const int height = size.second;
        const size_t pixelCount = static_cast<size_t>(width) * height;
        std::vector<uchar4> hostInput(pixelCount);
        std::vector<uchar4> hostOutput(pixelCount);
        std::vector<uchar4> expected(pixelCount);
        fillImage(hostInput);
        robertsCPU(hostInput, expected, width, height);
        std::cout << "\nImage: " << width << " x " << height << "\n";
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();

    cudaArray_t inputArray = nullptr;

    CUDA_CHECK(cudaMallocArray(
        &inputArray,
        &channelDesc,
        width,
        height
    ));

    CUDA_CHECK(cudaMemcpy2DToArray(
        inputArray,
        0,
        0,
        hostInput.data(),
        width * sizeof(uchar4),
        width * sizeof(uchar4),
        height,
        cudaMemcpyHostToDevice
    ));

    cudaResourceDesc resourceDesc{};
    resourceDesc.resType = cudaResourceTypeArray;
    resourceDesc.res.array.array = inputArray;

    cudaTextureDesc textureDesc{};

    textureDesc.normalizedCoords = false;

    textureDesc.addressMode[0] = cudaAddressModeClamp;
    textureDesc.addressMode[1] = cudaAddressModeClamp;

    textureDesc.filterMode = cudaFilterModePoint;

    textureDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t texture = 0;

    CUDA_CHECK(cudaCreateTextureObject(
        &texture,
        &resourceDesc,
        &textureDesc,
        nullptr
    ));

    uchar4* deviceOutput = nullptr;

    CUDA_CHECK(cudaMalloc(
        &deviceOutput,
        pixelCount * sizeof(uchar4)
    ));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        const dim3 block(16, 16);
        const dim3 grid((width + block.x - 1) / block.x,
                        (height + block.y - 1) / block.y);
        double cpuTime = 0.0;
        for (int i = 0; i < repeats; ++i) {
            const auto cpuStart = std::chrono::steady_clock::now();
            robertsCPU(hostInput, expected, width, height);
            const auto cpuStop = std::chrono::steady_clock::now();
            cpuTime += std::chrono::duration<double, std::milli>(cpuStop - cpuStart).count();
        }
            robertsKernel<<<grid, block>>>(texture, deviceOutput, width, height);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            float totalTime = 0.0f;
            for (int i = 0; i < repeats; ++i) {
                CUDA_CHECK(cudaEventRecord(start));
                robertsKernel<<<grid, block>>>(texture, deviceOutput, width, height);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                float milliseconds;
                CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
                totalTime += milliseconds;
            }
            CUDA_CHECK(cudaMemcpy(hostOutput.data(), deviceOutput,
                                  pixelCount * sizeof(uchar4), cudaMemcpyDeviceToHost));
            const bool matches = checkResult(expected, hostOutput);
            correct = matches && correct;
        const double cpuAverage = cpuTime / repeats;
        const double gpuAverage = totalTime / repeats;
        std::cout << std::fixed << std::setprecision(6)
                  << "CPU time: " << cpuAverage << " ms\n"
                  << "GPU kernel time: " << gpuAverage << " ms\n"
                  << "Speedup: " << cpuAverage / gpuAverage << "\n"
                  << "Grid: " << grid.x << " x " << grid.y << "\n"
                  << "Block: " << block.x << " x " << block.y << "\n"
                  << "Results: " << (matches ? "OK" : "ERROR") << "\n";
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaDestroyTextureObject(texture));
    CUDA_CHECK(cudaFree(deviceOutput));
    CUDA_CHECK(cudaFreeArray(inputArray));
    }
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
