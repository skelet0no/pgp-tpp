#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

int main() {
    int n;
    scanf("%d", &n);

    std::vector<double> vector1(n);
    std::vector<double> vector2(n);

    for (int i = 0; i < n; ++i) {
        scanf("%lf", &vector1[i]);
    }

    for (int i = 0; i < n; ++i) {
        scanf("%lf", &vector2[i]);
    }

    for (int i = 0; i < n; ++i) {
        printf("%.10e ", std::max(vector1[i], vector2[i]));
    }

    return 0;
}