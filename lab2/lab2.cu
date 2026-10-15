#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
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

__device__ __forceinline__
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

    std::vector<uchar4> hostInput;

    if (!readImage(inputFile, width, height, hostInput)) return EXIT_FAILURE;

    const size_t pixelCount = static_cast<size_t>(width) * height;

    std::vector<uchar4> hostOutput(pixelCount);

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

    constexpr int BLOCK_X = 16;
    constexpr int BLOCK_Y = 16;

    dim3 block(BLOCK_X, BLOCK_Y);

    dim3 grid(
        (width + BLOCK_X - 1) / BLOCK_X,
        (height + BLOCK_Y - 1) / BLOCK_Y
    );

    robertsKernel<<<grid, block>>>(
        texture,
        deviceOutput,
        width,
        height
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        hostOutput.data(),
        deviceOutput,
        pixelCount * sizeof(uchar4),
        cudaMemcpyDeviceToHost
    ));

    if (!writeImage(outputFile, width, height, hostOutput)) return EXIT_FAILURE;

    CUDA_CHECK(cudaDestroyTextureObject(texture));
    CUDA_CHECK(cudaFree(deviceOutput));
    CUDA_CHECK(cudaFreeArray(inputArray));

    return EXIT_SUCCESS;
}
