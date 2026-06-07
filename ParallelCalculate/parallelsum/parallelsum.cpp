// ConsoleApplication1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <omp.h>
#include <cstdlib>
using namespace std;

int main()
{
	const int num_core = 8;
	const int total_data = 24;

	int data[total_data];
	for (int i = 0; i < total_data;i++)
		data[i] = rand() % 10 + 1;//随机1-10

	int id_value[num_core] = { 0 };
	omp_set_num_threads(num_core);

	#pragma omp parallel
	{
		int id = omp_get_thread_num();

		int local_sum = 0;
		for (int i = 0; i < 3; i++)
			local_sum += data[id * 3 + i];

		id_value[id] = local_sum;

		#pragma omp barrier
		
		int coredifference = 1;
		int divisor = 2;
		while (coredifference < num_core) {
			
			if (id % divisor == 0) {
				int partner = id + coredifference;
				if (partner < num_core) {
					id_value[id]  +=  id_value[partner];
				}
				else {
					id_value[id] = id_value[id];
				}
			}
			else {
				id_value[id] = id_value[id];
			}
			#pragma omp barrier
			coredifference = coredifference * 2;
			divisor = divisor * 2;
		}
			
	}
	cout << "Final sum =" << id_value[0] << endl;
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
