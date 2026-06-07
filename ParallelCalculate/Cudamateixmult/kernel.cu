#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>

#include <omp.h>
#include <cuda_runtime.h>

using namespace std;

// ======================================
// Matrix Size
// ======================================

#define M 1000
#define K 1000
#define N 1000

// ======================================
// Tile Size
// ======================================

#define TILE_SIZE 16

// ======================================
// CUDA Error Check
// ======================================

#define CHECK(call)                                      \
{                                                        \
    const cudaError_t error = call;                      \
                                                         \
    if (error != cudaSuccess)                            \
    {                                                    \
        cout << "CUDA Error: "                           \
             << cudaGetErrorString(error)                \
             << " at line " << __LINE__ << endl;         \
                                                         \
        exit(1);                                         \
    }                                                    \
}

// ======================================
// CUDA Normal Kernel
// ======================================

__global__ void MatrixMulCUDA(
    float* A,
    float* B,
    float* C)
{
    int row =
        blockIdx.y * blockDim.y + threadIdx.y;

    int col =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;

        for (int i = 0; i < K; i++)
        {
            sum +=
                A[row * K + i] *
                B[i * N + col];
        }

        C[row * N + col] = sum;
    }
}

// ======================================
// CUDA Shared Memory Kernel
// ======================================

__global__ void MatrixMulCUDASharedMemory(
    const float* A,
    const float* B,
    float* C,
    int m,
    int n,
    int k)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row =
        blockIdx.y * TILE_SIZE + threadIdx.y;

    int col =
        blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int kk = 0; kk < K; kk += TILE_SIZE)
    {
        if (row < M &&
            (kk + threadIdx.x) < K)
        {
            As[threadIdx.y][threadIdx.x] =
                A[row * K + (kk + threadIdx.x)];
        }
        else
        {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if ((kk + threadIdx.y) < K &&
            col < N)
        {
            Bs[threadIdx.y][threadIdx.x] =
                B[(kk + threadIdx.y) * N + col];
        }
        else
        {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
        {
            sum +=
                As[threadIdx.y][k] *
                Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
    {
        C[row * N + col] = sum;
    }
}
// ======================================
// CPU Serial
// ======================================

void MatrixMulCPU(
    const vector<float>& A,
    const vector<float>& B,
    vector<float>& C)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;

            for (int k = 0; k < K; k++)
            {
                sum +=
                    A[i * K + k] *
                    B[k * N + j];
            }

            C[i * N + j] = sum;
        }
    }
}

// ======================================
// OpenMP
// ======================================

void MatrixMulOpenMP(
    const vector<float>& A,
    const vector<float>& B,
    vector<float>& C)
{
#pragma omp parallel for
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;

            for (int k = 0; k < K; k++)
            {
                sum +=
                    A[i * K + k] *
                    B[k * N + j];
            }

            C[i * N + j] = sum;
        }
    }
}

// ======================================
// Initialize Matrix
// ======================================

void initMatrix(vector<float>& mat)
{
    random_device rd;

    mt19937 gen(rd());

    uniform_real_distribution<float>
        dis(0.0f, 5.0f);

    for (size_t i = 0;
        i < mat.size();
        i++)
    {
        mat[i] = dis(gen);
    }
}

// ======================================
// Check Result
// ======================================

bool checkResult(
    const vector<float>& A,
    const vector<float>& B)
{
    for (size_t i = 0;
        i < A.size();
        i++)
    {
        if (fabs(A[i] - B[i]) > 1e-2)
        {
            cout << "Mismatch at "
                << i
                << endl;

            return false;
        }
    }

    return true;
}

// ======================================
// Main
// ======================================

int main()
{
    vector<float> h_A(M * K);

    vector<float> h_B(K * N);

    vector<float> h_C_CPU(M * N, 0.0f);

    vector<float> h_C_OPENMP(M * N, 0.0f);

    vector<float> h_C_CUDA(M * N, 0.0f);

    vector<float> h_C_CUDAShared(M * N, 0.0f);

    initMatrix(h_A);

    initMatrix(h_B);

    // ======================================
    // CPU
    // ======================================

    auto cpu_start =
        chrono::high_resolution_clock::now();

    MatrixMulCPU(
        h_A,
        h_B,
        h_C_CPU);

    auto cpu_end =
        chrono::high_resolution_clock::now();

    chrono::duration<double> cpu_time =
        cpu_end - cpu_start;

    cout << "CPU Time: "
        << cpu_time.count()
        << " s"
        << endl;

    // ======================================
    // OpenMP
    // ======================================

    auto omp_start =
        chrono::high_resolution_clock::now();

    MatrixMulOpenMP(
        h_A,
        h_B,
        h_C_OPENMP);

    auto omp_end =
        chrono::high_resolution_clock::now();

    chrono::duration<double> omp_time =
        omp_end - omp_start;

    cout << "OpenMP Time: "
        << omp_time.count()
        << " s"
        << endl;

    // ======================================
    // CUDA Memory
    // ======================================

    size_t sizeA =
        M * K * sizeof(float);

    size_t sizeB =
        K * N * sizeof(float);

    size_t sizeC =
        M * N * sizeof(float);

    float* d_A;
    float* d_B;
    float* d_C;

    CHECK(cudaMalloc((void**)&d_A, sizeA));
    CHECK(cudaMalloc((void**)&d_B, sizeB));
    CHECK(cudaMalloc((void**)&d_C, sizeC));

    CHECK(cudaMemcpy(
        d_A,
        h_A.data(),
        sizeA,
        cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(
        d_B,
        h_B.data(),
        sizeB,
        cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(
        TILE_SIZE,
        TILE_SIZE);

    dim3 blocksPerGrid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE);

    // ======================================
    // CUDA Normal
    // ======================================

    cudaEvent_t start_cuda, stop_cuda;

    cudaEventCreate(&start_cuda);
    cudaEventCreate(&stop_cuda);

    cudaEventRecord(start_cuda);

    MatrixMulCUDA << <
        blocksPerGrid,
        threadsPerBlock
        >> > (
            d_A,
            d_B,
            d_C);

    CHECK(cudaGetLastError());

    CHECK(cudaDeviceSynchronize());

    cudaEventRecord(stop_cuda);

    cudaEventSynchronize(stop_cuda);

    float milliseconds_cuda = 0.0f;

    cudaEventElapsedTime(
        &milliseconds_cuda,
        start_cuda,
        stop_cuda);

    CHECK(cudaMemcpy(
        h_C_CUDA.data(),
        d_C,
        sizeC,
        cudaMemcpyDeviceToHost));

    cout << "CUDA Time: "
        << milliseconds_cuda / 1000.0
        << " s"
        << endl;

    // ======================================
    // CUDA Shared
    // ======================================

    cudaEvent_t start_shared, stop_shared;

    cudaEventCreate(&start_shared);
    cudaEventCreate(&stop_shared);

    cudaEventRecord(start_shared);

    MatrixMulCUDASharedMemory << <
        blocksPerGrid,
        threadsPerBlock
        >> > (
            d_A,
            d_B,
            d_C,
            M,
            N,
            K);

    CHECK(cudaGetLastError());

    CHECK(cudaDeviceSynchronize());

    cudaEventRecord(stop_shared);

    cudaEventSynchronize(stop_shared);

    float milliseconds_shared = 0.0f;

    cudaEventElapsedTime(
        &milliseconds_shared,
        start_shared,
        stop_shared);

    CHECK(cudaMemcpy(
        h_C_CUDAShared.data(),
        d_C,
        sizeC,
        cudaMemcpyDeviceToHost));

    cout << "CUDA Shared Time: "
        << milliseconds_shared / 1000.0
        << " s"
        << endl;

    // ======================================
    // Verify
    // ======================================

    cout << "OpenMP Check: "
        << (checkResult(
            h_C_CPU,
            h_C_OPENMP)
            ? "Correct"
            : "Wrong")
        << endl;

    cout << "CUDA Check: "
        << (checkResult(
            h_C_CPU,
            h_C_CUDA)
            ? "Correct"
            : "Wrong")
        << endl;

    cout << "CUDA Shared Check: "
        << (checkResult(
            h_C_CPU,
            h_C_CUDAShared)
            ? "Correct"
            : "Wrong")
        << endl;

    // ======================================
    // Speedup
    // ======================================

    double cpu_serial =
        cpu_time.count();

    double omp_parallel =
        omp_time.count();

    double gpu_normal =
        milliseconds_cuda / 1000.0;

    double gpu_shared =
        milliseconds_shared / 1000.0;

    cout << "OpenMP Speedup: "
        << cpu_serial / omp_parallel
        << endl;

    cout << "CUDA Speedup: "
        << cpu_serial / gpu_normal
        << endl;

    cout << "CUDA Shared Speedup: "
        << cpu_serial / gpu_shared
        << endl;

    // ======================================
    // Free
    // ======================================

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start_cuda);
    cudaEventDestroy(stop_cuda);

    cudaEventDestroy(start_shared);
    cudaEventDestroy(stop_shared);

    return 0;
}