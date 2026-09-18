#include <cstdio>
#include <cmath>

int main() {
    float a, b, c;
    std::scanf("%f %f %f", &a, &b, &c);

    if (a == 0.0f) {
        if (b == 0.0f) {
            if (c == 0.0f) {
                printf("any");
            } else {
                printf("incorrect");
            }
        } else {
            float x = -c / b;
            printf("%.6f", x);
        }
    } else {
        float D = b * b - 4.0f * a * c;
        if (D > 0.0f) {
            float sqrtD = std::sqrtf(D);
            float x1 = (-b + sqrtD) / (2.0f * a);
            float x2 = (-b - sqrtD) / (2.0f * a);
            printf("%.6f %.6f", x1, x2);
        } else if (D == 0.0f) {
            float x = -b / (2.0f * a);
            if (x == 0.0f) x = 0.0f;
            printf("%.6f", x);
        } else {
            printf("imaginary");
        }
    }

    return 0;
}