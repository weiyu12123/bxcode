#include <iostream>
#include <windows.h>
using namespace std;

const int N = 200000;
int a[N];
int LOOP = 10;

void init()
{
    for (int i = 0; i < N; i++)
        a[i] = i;
}

void ordinary()
{
    long long int begin, end, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&begin);
    for (int l = 0; l < LOOP; l++)
    {
        // init();
        int sum = 0;
        for (int i = 0; i < N; i++)
            sum += a[i];
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&end);
    cout << "ordinary:" << (end - begin) * 1000.0 / freq / LOOP << "ms" << endl;
}

void optimize()
{
    long long int begin, end, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&begin);
    for (int l = 0; l < LOOP; l++)
    {
        int sum1 = 0, sum2 = 0;
        for (int i = 0; i < N - 1; i += 2)
            sum1 += a[i], sum2 += a[i + 1];
        int sum = sum1 + sum2;
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&end);
    cout << "optimize:" << (end - begin) * 1000.0 / freq / LOOP << "ms" << endl;
}

int main()
{
    init();
    ordinary();
    optimize();
}