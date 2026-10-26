#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct uchar4 {
    uint8_t x, y, z, w;
};

struct double3 {
    double x, y, z;
};

double3 make_double3(double x, double y, double z)
{
    return {x, y, z};
}

void classifyKernel(std::vector<uchar4>& pixels, const std::vector<double3>& averages)
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

        pixel.w = static_cast<uint8_t>(bestClass);
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

    classifyKernel(hostPixels, averages);

    if (!writeImage(outputFile, width, height, hostPixels)) return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
