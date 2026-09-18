#include <cstdio>
#include <cstdlib>

int main() {
    int n;
    scanf("%d", &n);

    float *arr = (float*) malloc(n * sizeof(float));
    if (arr == NULL) {
        return 1;
    }

    for (int i = 0; i < n; ++i) {
        scanf("%f", &arr[i]);
    }

    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n - 1 - i; ++j) {
            if (arr[j] > arr[j + 1]) {
                float tmp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = tmp;
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        printf("%.6e", arr[i]);
        if (i != n - 1) {
            printf(" ");
        }
    }

    free(arr);
    return 0;
}