#include <iostream>
#include <opencv2/opencv.hpp>
#include "Functions.h"
#include "Exercises.h"

int main()
{

    cv::Mat src_8uc3_img = cv::imread("C:\\Users\\Lenovo\\source\\repos\\IA_1\\Images\\lena.png", cv::IMREAD_COLOR);

    Exercise1(src_8uc3_img);

    cv::Mat binaryImage = cv::imread("C:\\Users\\Lenovo\\source\\repos\\IA_2\\IA_2\\Images\\binary_image.png", cv::IMREAD_GRAYSCALE);

    Exercise2(binaryImage);

    return 0;
}
