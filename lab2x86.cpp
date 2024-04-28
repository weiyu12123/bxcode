#include <iostream>
#include <windows.h>
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <immintrin.h>  // AVX, AVX2
#include <chrono>
using namespace std;

const int N = 2048;
alignas(32) float a[N][N];

void init() {
    srand((unsigned int)time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i][j] = 0.0f;
        }
        a[i][i] = 1.0f;
    }

    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            a[i][j] = static_cast<float>(rand() % 100 + 1);
        }
    }
}

void SequentialAlgorithm() {
    for (int k = 0; k < N; k++) {
        for (int j = k + 1; j < N; j++)
            a[k][j] /= a[k][k];
        a[k][k] = 1.0f;
        for (int i = k + 1; i < N; i++) {
            for (int j = k + 1; j < N; j++)
                a[i][j] -= a[i][k] * a[k][j];
            a[i][k] = 0.0f;
        }
    }
}

void AVXAlgorithmAligned() {
    __m256 vec_k, vec_j, vec_kk, vec_tmp;
    for (int k = 0; k < N; k++) {
        vec_kk = _mm256_set1_ps(a[k][k]);
        for (int j = k + 1; j < N; j += 8) {
            vec_k = _mm256_load_ps(&a[k][j]);
            vec_tmp = _mm256_div_ps(vec_k, vec_kk);
            _mm256_store_ps(&a[k][j], vec_tmp);
        }
        a[k][k] = 1.0f;
        for (int i = k + 1; i < N; i++) {
            vec_k = _mm256_set1_ps(a[i][k]);
            for (int j = k + 1; j < N; j += 8) {
                vec_j = _mm256_load_ps(&a[k][j]);
                vec_tmp = _mm256_load_ps(&a[i][j]);
                vec_tmp = _mm256_sub_ps(vec_tmp, _mm256_mul_ps(vec_k, vec_j));
                _mm256_store_ps(&a[i][j], vec_tmp);
            }
            a[i][k] = 0.0f;
        }
    }
}

void AVXAlgorithmUnaligned() {
    __m256 vec_k, vec_j, vec_kk, vec_tmp;
    for (int k = 0; k < N; k++) {
        vec_kk = _mm256_set1_ps(a[k][k]);
        for (int j = k + 1; j < N; j += 8) {
            vec_k = _mm256_loadu_ps(&a[k][j]);
            vec_tmp = _mm256_div_ps(vec_k, vec_kk);
            _mm256_storeu_ps(&a[k][j], vec_tmp);
        }
        a[k][k] = 1.0f;
        for (int i = k + 1; i < N; i++) {
            vec_k = _mm256_set1_ps(a[i][k]);
            for (int j = k + 1; j < N; j += 8) {
                vec_j = _mm256_loadu_ps(&a[k][j]);
                vec_tmp = _mm256_loadu_ps(&a[i][j]);
                vec_tmp = _mm256_sub_ps(vec_tmp, _mm256_mul_ps(vec_k, vec_j));
                _mm256_storeu_ps(&a[i][j], vec_tmp);
            }
            a[i][k] = 0.0f;
        }
    }
}

void SSEAlgorithmAligned() {
    __m128 vec_k, vec_j, vec_kk, vec_tmp;
    for (int k = 0; k < N; k++) {
        vec_kk = _mm_set1_ps(a[k][k]);
        for (int j = k + 1; j < N; j += 4) {
            vec_k = _mm_load_ps(&a[k][j]);
            vec_tmp = _mm_div_ps(vec_k, vec_kk);
            _mm_store_ps(&a[k][j], vec_tmp);
        }
        a[k][k] = 1.0f;
        for (int i = k + 1; i < N; i++) {
            vec_k = _mm_set1_ps(a[i][k]);
            for (int j = k + 1; j < N; j += 4) {
                vec_j = _mm_load_ps(&a[k][j]);
                vec_tmp = _mm_load_ps(&a[i][j]);
                vec_tmp = _mm_sub_ps(vec_tmp, _mm_mul_ps(vec_k, vec_j));
                _mm_store_ps(&a[i][j], vec_tmp);
            }
            a[i][k] = 0.0f;
        }
    }
}

void SSEAlgorithmUnaligned() {
    __m128 vec_k, vec_j, vec_kk, vec_tmp;
    for (int k = 0; k < N; k++) {
        vec_kk = _mm_set1_ps(a[k][k]);
        for (int j = k + 1; j < N; j += 4) {
            vec_k = _mm_loadu_ps(&a[k][j]);
            vec_tmp = _mm_div_ps(vec_k, vec_kk);
            _mm_storeu_ps(&a[k][j], vec_tmp);
        }
        a[k][k] = 1.0f;
        for (int i = k + 1; i < N; i++) {
            vec_k = _mm_set1_ps(a[i][k]);
            for (int j = k + 1; j < N; j += 4) {
                vec_j = _mm_loadu_ps(&a[k][j]);
                vec_tmp = _mm_loadu_ps(&a[i][j]);
                vec_tmp = _mm_sub_ps(vec_tmp, _mm_mul_ps(vec_k, vec_j));
                _mm_storeu_ps(&a[i][j], vec_tmp);
            }
            a[i][k] = 0.0f;
        }
    }
}

int main() {
    LARGE_INTEGER frequency, start, end;
    double elapsedTime;

    QueryPerformanceFrequency(&frequency);
    init();

    QueryPerformanceCounter(&start);
    SequentialAlgorithm();
    QueryPerformanceCounter(&end);
    elapsedTime = static_cast<double>(end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
    cout << "Sequential: " << elapsedTime << " ms" << endl;

    init();
    QueryPerformanceCounter(&start);
    SSEAlgorithmAligned();
    QueryPerformanceCounter(&end);
    elapsedTime = static_cast<double>(end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
    cout << "SSE Aligned: " << elapsedTime << " ms" << endl;

    init();
    QueryPerformanceCounter(&start);
    SSEAlgorithmUnaligned();
    QueryPerformanceCounter(&end);
    elapsedTime = static_cast<double>(end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
    cout << "SSE Unaligned: " << elapsedTime << " ms" << endl;

    init();
    QueryPerformanceCounter(&start);
    AVXAlgorithmAligned();
    QueryPerformanceCounter(&end);
    elapsedTime = static_cast<double>(end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
    cout << "AVX Aligned: " << elapsedTime << " ms" << endl;

    init();
    QueryPerformanceCounter(&start);
    AVXAlgorithmUnaligned();
    QueryPerformanceCounter(&end);
    elapsedTime = static_cast<double>(end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
    cout << "AVX Unaligned: " << elapsedTime << " ms" << endl;

    return 0;
}
