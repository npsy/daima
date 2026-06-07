#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>

//double f(double x) {
//  return 4.0 / (1.0 + x * x);
//}
double f(double x) {
    return 4.0 / (1.0 + x * x);
}
int main() {
    const long long n = 1LL << 20;         // 区间数 (1,048,576)
    const double a = 0.0;
    const double b = 1.0;
    const double h = (b - a) / n;          // 步长

    std::cout << std::fixed << std::setprecision(15);
    std::cout << "Number of intervals: " << n << "\n";

    const double exact = 4*atan(1.0);   // ∫₀¹ sin x dx
    std::cout << "Exact integral (4.0/(1+x^2) from 0 to 1): " << exact << "\n\n";

    // ========== 串行梯形积分 ==========
    double stratTime1 = omp_get_wtime();
    double approx = (f(a) + f(b)) / 2.0;
    for (int i = 1; i <= n - 1; ++i) {   // 或 i < n
        approx += f(a + i * h);
    }
    approx *= h;
    double endTime1 = omp_get_wtime();
    std::cout << "Serial result:   " << approx << "\n";

    // ========== 并行梯形积分（static + reduction）==========
    double approx_parallel = 0.0;
    double stratTime2 = omp_get_wtime();
#pragma omp parallel for schedule(static) reduction(+:approx_parallel)
    for (long long i = 1; i <= n - 1; ++i) {   // 与串行完全相同的迭代
        approx_parallel += f(a + i * h);
    }
    approx_parallel += (f(a) + f(b)) / 2.0;
    approx_parallel *= h;
    double endTime2 = omp_get_wtime();
    std::cout << "Parallel result: " << approx_parallel << "\n\n";

    // ========== 比较差异 ==========
    double diff = std::abs(approx - approx_parallel);
    double difftime1 = std::abs(stratTime1 - endTime1);
    double difftime2 = std::abs(stratTime2 - endTime2);
    std::cout << "Difference (serial - parallel): " << diff << "\n";
    std::cout << "Time (serial): " << difftime1 << "\n";
    std::cout << "Time (parallel): " << difftime2 << "\n";
    std::cout << "Error (serial vs exact):   " << std::abs(approx - exact) << "\n";
    std::cout << "Error (parallel vs exact): " << std::abs(approx_parallel - exact) << "\n";

    return 0;
}