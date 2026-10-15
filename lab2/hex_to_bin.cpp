#include <fstream>
#include <iostream>
#include <string>

int main() {
    std::ifstream in("in.txt");
    std::ofstream out("in.data", std::ios::binary);

    std::string hex;

    while (in >> hex) {
        if (hex.size() != 8) {
            std::cerr << "Invalid value: " << hex << '\n';
            return 1;
        }

        for (int i = 0; i < 8; i += 2) {
            unsigned int byte =
                std::stoul(hex.substr(i, 2), nullptr, 16);

            char c = static_cast<char>(byte);
            out.write(&c, 1);
        }
    }

    if (!out) {
        std::cerr << "Error writing file\n";
        return 1;
    }

    std::cout << "Created in.data\n";
}