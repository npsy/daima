#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// 线程安全的线性同余随机数生成器
double rnd(unsigned int* seed) {
    *seed = (114514919 * (*seed) + 1515926537) % (1 << 24);
    return ((double)(*seed)) / (1 << 24);
}

int main(int argc, char* argv[]) {
    long long num_shots = 100000000;
    long long count = 0;
    int threadsNum = omp_get_max_threads();

    // 为每个线程准备独立的初始种子
    unsigned int* seeds = new unsigned int[threadsNum];
    for (int thread = 0; thread < threadsNum; ++thread) {
        seeds[thread] = thread + 12345;
    }

    double start = omp_get_wtime();

    // 并行区域：每个线程获取私有种子和局部计数器，用 reduction 安全累加
#pragma omp parallel reduction(+:count)
    {
        int tid = omp_get_thread_num();             // 获取当前线程 ID
        unsigned int my_seed = seeds[tid];          // 私有种子副本
        long long local_count = 0;                  // 局部命中数

#pragma omp for
        for (long long i = 0; i < num_shots; ++i) {
            double x = rnd(&my_seed);
            double y = rnd(&my_seed);
            if (x * x + y * y <= 1.0) {
                ++local_count;
            }
        }

        count += local_count;   // 归约到全局 count（由 reduction 自动合并）
    }

    double end = omp_get_wtime();
    double pi = 4.0 * count / num_shots;

    std::cout << "Estimated pi = " << pi << std::endl;
    std::cout << "Elapsed time: " << end - start << " seconds\n";

    delete[] seeds;
    return 0;
}