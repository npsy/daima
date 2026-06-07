#include <iostream>
#include <vector>
#include <omp.h>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    // 强制控制台 UTF-8，避免乱码
    system("chcp 65001 > nul");

    // 1. 读取图像（用原始字符串避免路径转义问题）
    std::string inputPath = R"(C:\Users\likaituo\Pictures\数字图像处理\2-1\Fig0620a.jpg)";
    if (argc > 1) {
        inputPath = argv[1];
    }

    cv::Mat colorImage = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (colorImage.empty()) {
        std::cerr << "无法读取图像，请检查路径是否正确！" << std::endl;
        return -1;
    }

    std::cout << "图像尺寸: " << colorImage.cols << " x " << colorImage.rows << std::endl;

    // 2. 并行彩色转灰度图
    cv::Mat grayImage(colorImage.rows, colorImage.cols, CV_8UC1);
    int rows = colorImage.rows;
    int cols = colorImage.cols;

#pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cv::Vec3b bgr = colorImage.at<cv::Vec3b>(i, j);
            // 灰度公式：0.299*R + 0.587*G + 0.114*B
            uchar gray = static_cast<uchar>(
                0.299 * bgr[2] +
                0.587 * bgr[1] +
                0.114 * bgr[0]
                );
            grayImage.at<uchar>(i, j) = gray;
        }
    }

    // 3. 并行直方图统计（0-255 共256个灰度级）
    std::vector<int> histogram(256, 0);

#pragma omp parallel
    {
        // 每个线程创建自己的局部直方图，避免竞争
        std::vector<int> localHist(256, 0);

#pragma omp for
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                uchar val = grayImage.at<uchar>(i, j);
                localHist[val]++;
            }
        }

        // 合并局部结果到全局直方图
#pragma omp critical
        {
            for (int k = 0; k < 256; ++k) {
                histogram[k] += localHist[k];
            }
        }
    }

    // 4. 打印直方图数据（可选）
    std::cout << "\n灰度直方图（前20级）:" << std::endl;
    for (int k = 0; k < 20; ++k) {
        std::cout << "灰度 " << k << ": " << histogram[k] << " 像素" << std::endl;
    }

    // 5. 绘制直方图图像
    int histSize = 256;
    int histWidth = 512;
    int histHeight = 400;
    cv::Mat histImage(histHeight, histWidth, CV_8UC3, cv::Scalar(255, 255, 255));

    // 找到最大值，方便归一化
    int maxVal = *std::max_element(histogram.begin(), histogram.end());

    // 绘制柱状图
    int binW = cvRound((double)histWidth / histSize);
    for (int i = 0; i < histSize; ++i) {
        int h = cvRound((double)histogram[i] / maxVal * histHeight);
        cv::rectangle(histImage,
            cv::Point(i * binW, histHeight - h),
            cv::Point((i + 1) * binW, histHeight),
            cv::Scalar(0, 0, 255), -1);
    }

    // 6. 显示结果
    cv::imshow("原始图像", colorImage);
    cv::imshow("灰度图像", grayImage);
    cv::imshow("灰度直方图", histImage);

    // 7. 保存结果
    cv::imwrite("gray_output.jpg", grayImage);
    cv::imwrite("histogram_output.jpg", histImage);
    std::cout << "\n灰度图和直方图已保存！" << std::endl;

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
