#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <omp.h>
#include <windows.h>

int main() {
    // 使用系统默认代码页（中文 Windows 控制台基本不乱码）
    SetConsoleOutputCP(GetACP());

    // ===============================
    // 参数设置
    // ===============================
    const long long N = 1LL << 24;   // 16777216
    const int runs = 5;

    std::cout << "OpenMP Vector Add Performance Test\n";
    std::cout << "---------------------------------\n";
    std::cout << "Array Size        : " << N << "\n";
    std::cout << "Memory Per Array  : "
        << std::fixed << std::setprecision(2)
        << (N * sizeof(double)) / 1024.0 / 1024.0
        << " MB\n\n";

    std::vector<double> A(N), B(N), C(N);

    // ===============================
    // 初始化
    // ===============================
    std::cout << "Initializing arrays...\n";

#pragma omp parallel for schedule(static)
    for (long long i = 0; i < N; ++i) {
        A[i] = 1.0;
        B[i] = 2.0;
        C[i] = 0.0;
    }

    // ===============================
    // 串行测试
    // ===============================
    double serial_total = 0.0;

    for (int r = 0; r < runs; ++r) {
        double t1 = omp_get_wtime();

        for (long long i = 0; i < N; ++i) {
            C[i] = A[i] + B[i];
        }

        double t2 = omp_get_wtime();
        serial_total += (t2 - t1);
    }

    double t_serial = serial_total / runs;

    double sum_serial = 0.0;
    for (long long i = 0; i < N; ++i)
        sum_serial += C[i];

    std::cout << "\nSerial Avg Time   : "
        << std::fixed << std::setprecision(6)
        << t_serial << " s\n";

    std::cout << "Checksum          : "
        << std::fixed << std::setprecision(0)
        << sum_serial << "\n";

    int max_threads = omp_get_max_threads();

    std::cout << "Max Threads       : "
        << max_threads << "\n\n";

    // ===============================
    // 表头
    // ===============================
    std::cout << std::left
        << std::setw(10) << "Threads"
        << std::setw(16) << "Time(s)"
        << std::setw(14) << "Speedup"
        << std::setw(14) << "Efficiency"
        << "\n";

    std::cout << std::string(54, '-') << "\n";

    // ===============================
    // 并行测试
    // ===============================
    for (int t = 1; t <= max_threads; t = (t == 1 ? 2 : t * 2)) {

        if (t > max_threads) t = max_threads;

        omp_set_num_threads(t);

        // 预热
#pragma omp parallel for schedule(static)
        for (long long i = 0; i < N; ++i)
            C[i] = A[i] + B[i];

        double parallel_total = 0.0;

        for (int r = 0; r < runs; ++r) {

            double t1 = omp_get_wtime();

#pragma omp parallel for schedule(static)
            for (long long i = 0; i < N; ++i)
                C[i] = A[i] + B[i];

            double t2 = omp_get_wtime();
            parallel_total += (t2 - t1);
        }

        double t_parallel = parallel_total / runs;

        double speedup = t_serial / t_parallel;
        double efficiency = speedup / t * 100.0;

        std::cout << std::left
            << std::setw(10) << t
            << std::setw(16) << std::fixed << std::setprecision(6) << t_parallel
            << std::setw(14) << std::setprecision(3) << speedup
            << std::setw(13) << std::setprecision(2) << efficiency << "%"
            << "\n";

        if (t == max_threads) break;
    }

    std::cout << "\nTest Finished.\n";

    return 0;
}