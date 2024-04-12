#include <iostream>
#include <opencv2/opencv.hpp>
#include "Functions.h"
#include "Exercises.h"

int main()
{

    //cv::Mat src_8uc3_img = cv::imread("C:\\Users\\Lenovo\\source\\repos\\IA_1\\Images\\lena.png", cv::IMREAD_COLOR);

    //Exercise1(src_8uc3_img);

    //cv::Mat binaryImage = cv::imread("C:\\Users\\Lenovo\\source\\repos\\IA_2\\IA_2\\Images\\train.png", cv::IMREAD_GRAYSCALE);

    //Exercise2(binaryImage);

    cv::Mat testImageEthalon = cv::imread("C:\\Users\\Lenovo\\Downloads\\etalon.png");

    if (testImageEthalon.empty()) {
        std::cerr << "Error: Failed to load image." << std::endl;
        return EXIT_FAILURE;
    }

    processImage(testImageEthalon);

    return EXIT_SUCCESS;
}
