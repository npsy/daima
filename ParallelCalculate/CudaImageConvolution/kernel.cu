#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#include <cuda_runtime.h>
#include <omp.h>                     // OpenMP 头文件
#include <cufft.h>

// stb image
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

// ======================================
// 手动指定卷积核大小（奇数，≥3）
// ======================================
#define KERNEL_SIZE 21      // <-- 修改这里改变核大小

// ======================================
// CUDA 线程块大小
// ======================================
#define BLOCK_SIZE 16

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
             << " at line " << __LINE__                  \
             << endl;                                    \
                                                         \
        exit(1);                                         \
    }                                                    \
}

// ======================================
// 生成高斯核
// ======================================
void generateGaussianKernel(float* kernel) {
    int ksize = KERNEL_SIZE;
    int radius = ksize / 2;
    float sigma = ksize / 6.0f;    // sigma 经验值
    float sum = 0.0f;
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            float val = expf(-(x * x + y * y) / (2.0f * sigma * sigma));
            int idx = (y + radius) * ksize + (x + radius);
            kernel[idx] = val;
            sum += val;
        }
    }
    // 归一化
    for (int i = 0; i < ksize * ksize; i++)
        kernel[i] /= sum;
}

// ======================================
// OpenMP 卷积（CPU）
// ======================================
void convolutionCPU(
    unsigned char* input,
    float* kernel,
    unsigned char* output,
    int width,
    int height,
    int ksize)
{
    int radius = ksize / 2;

#pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;

            for (int ky = -radius; ky <= radius; ky++) {
                for (int kx = -radius; kx <= radius; kx++) {
                    int iy = y + ky;
                    int ix = x + kx;

                    if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                        int image_idx = iy * width + ix;
                        int kernel_idx = (ky + radius) * ksize + (kx + radius);
                        sum += input[image_idx] * kernel[kernel_idx];
                    }
                }
            }

            sum = min(max(sum, 0.0f), 255.0f);
            output[y * width + x] = (unsigned char)sum;
        }
    }
}

// ======================================
// CUDA Convolution Kernel (普通)
// ======================================

__global__ void ConvolutionCUDA(
    unsigned char* input,
    float* kernel,
    unsigned char* output,
    int width,
    int height,
    int ksize)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int radius = ksize / 2;

    if (row < height && col < width)
    {
        float sum = 0.0f;

        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int iy = row + ky;
                int ix = col + kx;

                if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                    int image_idx = iy * width + ix;
                    int kernel_idx = (ky + radius) * ksize + (kx + radius);
                    sum += input[image_idx] * kernel[kernel_idx];
                }
            }
        }

        sum = min(max(sum, 0.0f), 255.0f);
        output[row * width + col] = (unsigned char)sum;
    }
}

// ======================================
// CUDA Shared Memory Convolution Kernel
// ======================================

#define TILE_SIZE 16          // 输出块大小（不含 halo）
#define RADIUS (KERNEL_SIZE/2) // 根据核大小自动计算

__global__ void ConvolutionSharedMemoryCUDA(
    unsigned char* input,
    float* kernel,
    unsigned char* output,
    int width,
    int height,
    int ksize)
{
    // 共享内存大小：块大小 + 左右各 halo
    const int SHARED_W = TILE_SIZE + 2 * RADIUS;
    const int SHARED_H = TILE_SIZE + 2 * RADIUS;

    __shared__ unsigned char s_input[SHARED_H][SHARED_W];

    // 当前线程在输出图像中的全局坐标
    int out_col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int out_row = blockIdx.y * TILE_SIZE + threadIdx.y;

    // ---------- 1. 协作加载输入数据到共享内存 ----------
    for (int y = threadIdx.y; y < SHARED_H; y += blockDim.y) {
        for (int x = threadIdx.x; x < SHARED_W; x += blockDim.x) {
            int global_x = (blockIdx.x * TILE_SIZE - RADIUS) + x;
            int global_y = (blockIdx.y * TILE_SIZE - RADIUS) + y;

            if (global_x >= 0 && global_x < width &&
                global_y >= 0 && global_y < height) {
                s_input[y][x] = input[global_y * width + global_x];
            }
            else {
                s_input[y][x] = 0;
            }
        }
    }

    __syncthreads(); // 确保整个块加载完成

    // ---------- 2. 卷积计算 ----------
    if (out_row < height && out_col < width) {
        float sum = 0.0f;
        int radius = ksize / 2;

        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                int s_y = threadIdx.y + radius + ky;
                int s_x = threadIdx.x + radius + kx;
                int kernel_idx = (ky + radius) * ksize + (kx + radius);
                sum += s_input[s_y][s_x] * kernel[kernel_idx];
            }
        }

        sum = fminf(fmaxf(sum, 0.0f), 255.0f);
        output[out_row * width + out_col] = (unsigned char)sum;
    }
}
// ======================================
// 复数乘法和归一化
// ======================================
__global__ void ComplexMultiply(
    cufftComplex* A,
    cufftComplex* B,
    cufftComplex* C,
    int size)
{
    int idx =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (idx < size)
    {
        cufftComplex a = A[idx];
        cufftComplex b = B[idx];

        C[idx].x =
            a.x * b.x - a.y * b.y;

        C[idx].y =
            a.x * b.y + a.y * b.x;
    }
}

__global__ void NormalizeFFT(
    float* data,
    int size,
    float factor)
{
    int idx =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (idx < size)
    {
        data[idx] /= factor;
    }
}

// ======================================
// 统一加载函数（支持图片和 4096x2048 .dat 文件）
// ======================================
unsigned char* loadImageUnified(const char* filename, int* out_width, int* out_height) {
    // 1. 尝试用 stbi 加载常见图片
    int w, h, c;
    unsigned char* stbi_data = stbi_load(filename, &w, &h, &c, 1);
    if (stbi_data) {
        size_t num_pixels = (size_t)w * h;
        unsigned char* img = (unsigned char*)malloc(num_pixels);
        if (!img) {
            stbi_image_free(stbi_data);
            return nullptr;
        }
        memcpy(img, stbi_data, num_pixels);
        stbi_image_free(stbi_data);
        *out_width = w;
        *out_height = h;
        return img;
    }

    // 2. stbi 失败，如果是 .dat 文件，按 4096x2048 浮点原始数据读取
    const char* dot = strrchr(filename, '.');
    bool is_dat = dot && (strcmp(dot, ".dat") == 0 || strcmp(dot, ".DAT") == 0);
    if (is_dat) {
        const int WIDTH = 4096;
        const int HEIGHT = 2048;
        size_t num = (size_t)WIDTH * HEIGHT;

        std::vector<float> fbuf(num);
        FILE* fp = fopen(filename, "rb");
        if (!fp) {
            fprintf(stderr, "无法打开 .dat 文件: %s\n", filename);
            return nullptr;
        }
        size_t rd = fread(fbuf.data(), sizeof(float), num, fp);
        fclose(fp);
        if (rd != num) {
            fprintf(stderr, ".dat 文件读取不完整: 期望 %zu 个 float, 实际 %zu\n", num, rd);
            return nullptr;
        }

        unsigned char* img = (unsigned char*)malloc(num);
        if (!img) return nullptr;

        for (size_t i = 0; i < num; ++i) {
            float val = fbuf[i];
            if (val < 0.0f) val = 0.0f;
            if (val > 1.0f) val = 1.0f;
            img[i] = (unsigned char)(val * 255.0f);
        }

        *out_width = WIDTH;
        *out_height = HEIGHT;
        return img;
    }

    fprintf(stderr, "无法加载文件: %s (格式不支持或损坏)\n", filename);
    return nullptr;
}

void FFTConvolutionCUDA(
    unsigned char* h_input,
    float* h_kernel,
    unsigned char* h_output,
    int width,
    int height)
{
    int FFT_W = width + KERNEL_SIZE - 1;
    int FFT_H = height + KERNEL_SIZE - 1;

    int realSize = FFT_W * FFT_H;
    int complexSize = FFT_H * (FFT_W / 2 + 1);

    //==================================================
    // Host Padding
    //==================================================

    float* h_imgPad = new float[realSize];
    float* h_kernelPad = new float[realSize];

    memset(h_imgPad, 0, realSize * sizeof(float));
    memset(h_kernelPad, 0, realSize * sizeof(float));

    //==================================================
    // Image Padding
    //==================================================

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            h_imgPad[y * FFT_W + x]
                =
                (float)h_input[y * width + x];
        }
    }

    //==================================================
    // Kernel Shift
    //==================================================

    int radius = KERNEL_SIZE / 2;

    for (int y = 0; y < KERNEL_SIZE; y++)
    {
        for (int x = 0; x < KERNEL_SIZE; x++)
        {
            int yy =
                (y - radius + FFT_H) % FFT_H;

            int xx =
                (x - radius + FFT_W) % FFT_W;

            h_kernelPad[yy * FFT_W + xx]
                =
                h_kernel[y * KERNEL_SIZE + x];
        }
    }

    //==================================================
    // Device Memory
    //==================================================

    float* d_img;
    float* d_kernelReal;
    float* d_resultReal;

    cufftComplex* d_imgFFT;
    cufftComplex* d_kernelFFT;
    cufftComplex* d_resultFFT;

    cudaMalloc(&d_img,
        realSize * sizeof(float));

    cudaMalloc(&d_kernelReal,
        realSize * sizeof(float));

    cudaMalloc(&d_resultReal,
        realSize * sizeof(float));

    cudaMalloc(&d_imgFFT,
        complexSize * sizeof(cufftComplex));

    cudaMalloc(&d_kernelFFT,
        complexSize * sizeof(cufftComplex));

    cudaMalloc(&d_resultFFT,
        complexSize * sizeof(cufftComplex));

    cudaMemcpy(
        d_img,
        h_imgPad,
        realSize * sizeof(float),
        cudaMemcpyHostToDevice);

    cudaMemcpy(
        d_kernelReal,
        h_kernelPad,
        realSize * sizeof(float),
        cudaMemcpyHostToDevice);

    //==================================================
    // FFT Plan
    //==================================================

    cufftHandle planR2C;
    cufftHandle planC2R;

    cufftPlan2d(
        &planR2C,
        FFT_H,
        FFT_W,
        CUFFT_R2C);

    cufftPlan2d(
        &planC2R,
        FFT_H,
        FFT_W,
        CUFFT_C2R);

    //==================================================
    // FFT
    //==================================================

    cufftExecR2C(
        planR2C,
        d_img,
        d_imgFFT);

    cufftExecR2C(
        planR2C,
        d_kernelReal,
        d_kernelFFT);

    //==================================================
    // Multiply
    //==================================================

    int threads = 256;

    int blocks =
        (complexSize + threads - 1) / threads;

    ComplexMultiply << <
        blocks,
        threads >> > (
            d_imgFFT,
            d_kernelFFT,
            d_resultFFT,
            complexSize);

    cudaDeviceSynchronize();

    //==================================================
    // IFFT
    //==================================================

    cufftExecC2R(
        planC2R,
        d_resultFFT,
        d_resultReal);

    NormalizeFFT << <
        (realSize + 255) / 256,
        256 >> > (
            d_resultReal,
            realSize,
            (float)realSize);

    cudaDeviceSynchronize();

    //==================================================
    // Copy Back
    //==================================================

    float* h_result =
        new float[realSize];

    cudaMemcpy(
        h_result,
        d_resultReal,
        realSize * sizeof(float),
        cudaMemcpyDeviceToHost);

    //==================================================
    // Crop
    //==================================================

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            float v =
                h_result[y * FFT_W + x];

            v = min(max(v, 0.0f), 255.0f);

            h_output[y * width + x]
                =
                (unsigned char)v;
        }
    }

    //==================================================
    // Free
    //==================================================

    delete[] h_imgPad;
    delete[] h_kernelPad;
    delete[] h_result;

    cudaFree(d_img);
    cudaFree(d_kernelReal);
    cudaFree(d_resultReal);

    cudaFree(d_imgFFT);
    cudaFree(d_kernelFFT);
    cudaFree(d_resultFFT);

    cufftDestroy(planR2C);
    cufftDestroy(planC2R);
}

// ======================================
// Main
// ======================================

int main()
{
    cout << "====================================" << endl;
    cout << "CUDA vs OpenMP Image Convolution" << endl;
    cout << "====================================" << endl;

    // ======================================
    // 加载图像
    // ======================================
    int width, height;
    unsigned char* h_input = loadImageUnified(
        "C:/Users/likaituo/Pictures/Saved Pictures/imgData.dat",
        &width, &height);

    if (!h_input) {
        cerr << "图像加载失败！" << endl;
        return -1;
    }

    cout << "Image Loaded:" << endl;
    cout << "Width  : " << width << endl;
    cout << "Height : " << height << endl;

    // ======================================
    // 卷积核设置
    // ======================================
    const int ksize = KERNEL_SIZE;
    cout << "Convolution kernel size: " << ksize << "x" << ksize << endl;

    float* h_kernel = new float[ksize * ksize];
    generateGaussianKernel(h_kernel);

    // ======================================
    // 分配输出内存（三份）
    // ======================================
    size_t image_size = width * height * sizeof(unsigned char);
    unsigned char* h_output_cuda =
        new unsigned char[width * height];

    unsigned char* h_output_omp =
        new unsigned char[width * height];

    unsigned char* h_output_SMcuda =
        new unsigned char[width * height];

    unsigned char* h_output_fft =
        new unsigned char[width * height];

    // ======================================
    // GPU 内存分配（一次分配，三个 kernel 共用）
    // ======================================
    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;
    float* d_kernel = nullptr;

    CHECK(cudaMalloc(&d_input, image_size));
    CHECK(cudaMalloc(&d_output, image_size));
    CHECK(cudaMalloc(&d_kernel, ksize * ksize * sizeof(float)));

    CHECK(cudaMemcpy(d_input, h_input, image_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_kernel, h_kernel, ksize * ksize * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid(
        (width + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // ======================================
    // 1) CUDA 普通卷积 + 计时
    // ======================================
    cudaEvent_t start_cuda, stop_cuda;
    cudaEventCreate(&start_cuda);
    cudaEventCreate(&stop_cuda);

    cudaEventRecord(start_cuda);
    ConvolutionCUDA << <blocksPerGrid, threadsPerBlock >> > (
        d_input, d_kernel, d_output, width, height, ksize);
    cudaEventRecord(stop_cuda);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    float ms_cuda = 0.0f;
    cudaEventElapsedTime(&ms_cuda, start_cuda, stop_cuda);

    CHECK(cudaMemcpy(h_output_cuda, d_output, image_size, cudaMemcpyDeviceToHost));

    cudaEventDestroy(start_cuda);
    cudaEventDestroy(stop_cuda);

    // ======================================
    // 2) CUDA 共享内存卷积 + 计时
    // ======================================
    cudaEvent_t start_SMcuda, stop_SMcuda;
    cudaEventCreate(&start_SMcuda);
    cudaEventCreate(&stop_SMcuda);

    cudaEventRecord(start_SMcuda);
    ConvolutionSharedMemoryCUDA << <blocksPerGrid, threadsPerBlock >> > (
        d_input, d_kernel, d_output, width, height, ksize);
    cudaEventRecord(stop_SMcuda);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    float ms_SMcuda = 0.0f;
    cudaEventElapsedTime(&ms_SMcuda, start_SMcuda, stop_SMcuda);

    CHECK(cudaMemcpy(h_output_SMcuda, d_output, image_size, cudaMemcpyDeviceToHost));

    cudaEventDestroy(start_SMcuda);
    cudaEventDestroy(stop_SMcuda);

    // 释放 GPU 资源
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    // ======================================
    // 3) FFT 卷积 + 计时
    // ======================================
    auto fft_start =
        chrono::high_resolution_clock::now();

    FFTConvolutionCUDA(
        h_input,
        h_kernel,
        h_output_fft,
        width,
        height);

    auto fft_end =
        chrono::high_resolution_clock::now();

    double ms_fft =
        chrono::duration<double, milli>(
            fft_end - fft_start).count();

    // ======================================
    // 4) OpenMP 卷积 + 计时
    // ======================================
    auto t1 = chrono::high_resolution_clock::now();
    convolutionCPU(h_input, h_kernel, h_output_omp, width, height, ksize);
    auto t2 = chrono::high_resolution_clock::now();
    double ms_omp = chrono::duration<double, milli>(t2 - t1).count();

    // ======================================
    // 保存输出图像
    // ======================================
    stbi_write_png("cuda_output.png", width, height, 1, h_output_cuda, width);
    stbi_write_png("openmp_output.png", width, height, 1, h_output_omp, width);
    stbi_write_png("SMcuda_output.png", width, height, 1, h_output_SMcuda, width);
    stbi_write_png(
        "fft_output.png",
        width,
        height,
        1,
        h_output_fft,
        width);
    // ======================================
    // 显示结果
    // ======================================
    cout << endl;
    cout << endl;

    cout
        << "CUDA Normal : "
        << ms_cuda
        << " ms" << endl;

    cout
        << "CUDA Shared : "
        << ms_SMcuda
        << " ms" << endl;

    cout
        << "CUDA FFT    : "
        << ms_fft
        << " ms" << endl;

    cout
        << "OpenMP      : "
        << ms_omp
        << " ms" << endl;

    cout << endl;

    cout
        << "OpenMP / CUDA Normal = "
        << ms_omp / ms_cuda
        << "x" << endl;

    cout
        << "OpenMP / CUDA Shared = "
        << ms_omp / ms_SMcuda
        << "x" << endl;

    cout
        << "OpenMP / CUDA FFT = "
        << ms_omp / ms_fft
        << "x" << endl;
    cout << endl;
    cout << "Output images saved: cuda_output.png, openmp_output.png, SMcuda_output.png,SMFFTcuda_output.png" << endl;

    // ======================================
    // 释放内存
    // ======================================
    free(h_input);
    delete[] h_output_cuda;
    delete[] h_output_omp;
    delete[] h_output_SMcuda;
    delete[] h_kernel;

    return 0;
}