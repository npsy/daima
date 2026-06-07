#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cstdlib>

void multiply_block(const std::vector<double>& A, const std::vector<double>& B,
    std::vector<double>& C, int N, int start_row, int end_row) {
    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    const int N = 4000;
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N, 0.0);

    // 初始化
    for (int i = 0; i < N * N; ++i) {
        A[i] = rand() % 100 / 100.0;
        B[i] = rand() % 100 / 100.0;
    }

    // 确定线程数量
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4; // 默认4线程
    std::cout << "使用 " << num_threads << " 个线程" << std::endl;

    std::vector<std::thread> threads;
    int rows_per_thread = N / num_threads;
    int start_row = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    // 创建线程
    for (unsigned int t = 0; t < num_threads; ++t) {
        int end_row = (t == num_threads - 1) ? N : start_row + rows_per_thread;
        threads.emplace_back(multiply_block, std::ref(A), std::ref(B), std::ref(C),
            N, start_row, end_row);
        start_row = end_row;
    }

    // 等待所有线程结束
    for (auto& th : threads) {
        th.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "std::thread 矩阵乘法 (" << N << "x" << N << ") 耗时: "
        << elapsed.count() << " 秒" << std::endl;

    std::cout << "C[0][0] = " << C[0] << ", C[1999][1999] = " << C[1999 * N + 1999] << std::endl;

    return 0;
}

