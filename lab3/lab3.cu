#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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


bool readImage(const std::string& filename, int& width, int& height, std::vector<uchar4>& pixels)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Cannot open input file: " << filename;
        return false;
    }

    file.read(reinterpret_cast<char*>(&width), sizeof(int));
    file.read(reinterpret_cast<char*>(&height), sizeof(int));

    if (!file) {
        std::cerr << "Cannot read image dimensions";
        return false;
    }

    const size_t pixelCount = static_cast<size_t>(width) * height;
    pixels.resize(pixelCount);

    file.read(reinterpret_cast<char*>(pixels.data()), pixelCount * sizeof(uchar4));

    if (!file) {
        std::cerr << "Cannot read pixel data";
        return false;
    }

    return true;
}


bool writeImage(const std::string& filename, int width, int height, const std::vector<uchar4>& pixels)
{
    std::ofstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Cannot open output file: " << filename;
        return false;
    }

    file.write(reinterpret_cast<const char*>(&width), sizeof(int));
    file.write(reinterpret_cast<const char*>(&height),sizeof(int));
    file.write(reinterpret_cast<const char*>(pixels.data()), pixels.size() * sizeof(uchar4));

    return static_cast<bool>(file);
}


int main()
{
    std::string inputFile;
    std::string outputFile;

    std::getline(std::cin, inputFile);
    std::getline(std::cin, outputFile);

    int width;
    int height;
    std::vector<uchar4> hostPixels;

    if (!readImage(inputFile, width, height, hostPixels)) return EXIT_FAILURE;

    int classCount;

    std::cin >> classCount;

    std::vector<double3> averages(classCount);

    for (int j = 0; j < classCount; ++j) {
        int sampleCount;

        std::cin >> sampleCount;

        double3 sum = make_double3(0.0, 0.0, 0.0);

        for (int i = 0; i < sampleCount; ++i) {
            int x;
            int y;

            std::cin >> x >> y;

            const uchar4 pixel = hostPixels[static_cast<size_t>(y) * width + x];
            sum.x += pixel.x;
            sum.y += pixel.y;
            sum.z += pixel.z;
        }

        averages[j] = make_double3(
            sum.x / sampleCount,
            sum.y / sampleCount,
            sum.z / sampleCount
        );
    }

    CUDA_CHECK(cudaMemcpyToSymbol(
        classAverages,
        averages.data(),
        classCount * sizeof(double3)
    ));

    const size_t pixelCount = hostPixels.size();
    uchar4* devicePixels = nullptr;

    CUDA_CHECK(cudaMalloc(
        &devicePixels,
        pixelCount * sizeof(uchar4)
    ));

    CUDA_CHECK(cudaMemcpy(
        devicePixels,
        hostPixels.data(),
        pixelCount * sizeof(uchar4),
        cudaMemcpyHostToDevice
    ));

    constexpr int THREADS = 256;
    const int blocks = static_cast<int>(std::min(
        (pixelCount + THREADS - 1) / THREADS,
        static_cast<size_t>(65535)
    ));

    classifyKernel<<<blocks, THREADS>>>(
        devicePixels,
        pixelCount,
        classCount
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        hostPixels.data(),
        devicePixels,
        pixelCount * sizeof(uchar4),
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaFree(devicePixels));

    if (!writeImage(outputFile, width, height, hostPixels)) return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
