#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <iostream>

int main() {
    int n;
    scanf("%d", &n);

    std::vector<double> vector1(n);
    std::vector<double> vector2(n);

    for (double& x : vector1) {
        std::cin >> x;
    }

    for (double& x : vector2) {
        std::cin >> x;
    }

    for (int i = 0; i < n; ++i) {
        std::cout << std::scientific
                  << (vector1[i] > vector2[i] ? vector1[i] : vector2[i])
                  << ' ';
    }
}
