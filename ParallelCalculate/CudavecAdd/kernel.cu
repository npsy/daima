#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// CUDA Kernel（GPU函数）
__global__
void vectorAdd(const float* A,
    const float* B,
    float* C,
    int N)
{
    // 全局线程编号
    int idx =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    // 防止越界
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    // 向量大小
    int N = 1024;

    // 内存大小（字节）
    size_t size = N * sizeof(float);

    // CPU端数组
    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];

    // 初始化
    for (int i = 0; i < N; i++)
    {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    // GPU端数组
    float* d_A;
    float* d_B;
    float* d_C;

    // GPU申请内存
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // CPU -> GPU
    cudaMemcpy(d_A, h_A,
        size,
        cudaMemcpyHostToDevice);

    cudaMemcpy(d_B, h_B,
        size,
        cudaMemcpyHostToDevice);

    // 每个block线程数
    int threadsPerBlock = 256;

    // block数量
    int blocksPerGrid =
        (N + threadsPerBlock - 1)
        / threadsPerBlock;

    // 启动Kernel
    vectorAdd << <blocksPerGrid,
        threadsPerBlock >> >
        (d_A, d_B, d_C, N);

    // GPU -> CPU
    cudaMemcpy(h_C, d_C,
        size,
        cudaMemcpyDeviceToHost);

    // 输出前10个结果
    for (int i = 0; i < 10; i++)
    {
        cout
            << h_A[i]
            << " + "
            << h_B[i]
            << " = "
            << h_C[i]
            << endl;
    }

    // 释放GPU内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 释放CPU内存
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}