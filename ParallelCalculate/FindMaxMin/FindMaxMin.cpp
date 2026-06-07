#include<iostream>
#include<omp.h>
#include<vector>
#include<random>
#include<cmath>

int main() {
	const long long N = 10'000'000;   // 1000 万
	std::vector<double> a(N);

	std::mt19937 gen(42);             // 随机引擎，种子 42（可改）
	std::uniform_real_distribution<double> dist(0.0, 100.0);   // [0,1) 均匀分布

	for (long long i = 0; i < N; ++i) {
		a[i] = static_cast<double>(gen());
	}
	double max_val = -1e100;
	double min_val = 1e100;
	double t1 = omp_get_wtime();
#pragma omp parallel for reduction(max:max_val) reduction(min:min_val);
	for (int i = 0; i < N; ++i) {
		if (a[i] > max_val) max_val = a[i];
		if (a[i] < min_val) min_val = a[i];
	}
	double t2 = omp_get_wtime();
	double t_parallel = t2 - t1;
	std::cout << "Parallel: min = " << min_val
			  << ", max = " << max_val
			  << "  time: " << t_parallel << " s\n";
}