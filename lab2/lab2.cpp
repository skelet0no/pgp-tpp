#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include <cstdlib>

struct uchar4 {
    uint8_t x, y, z, w;
};

uchar4 make_uchar4(uint8_t x, uint8_t y, uint8_t z, uint8_t w)
{
    return {x, y, z, w};
}

float grayscale(uchar4 p)
{
    return 0.299f * p.x + 0.587f * p.y + 0.114f * p.z;
}

void robertsKernel(const std::vector<uchar4>& input, std::vector<uchar4>& output,
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

    robertsKernel(hostInput, hostOutput, width, height);

    if (!writeImage(outputFile, width, height, hostOutput)) return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
