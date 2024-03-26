#include "Functions.h"

double computeArea(const cv::Mat& binaryImage)
{
    return cv::countNonZero(binaryImage);
}

cv::Point2d computeCenterOfMass(const cv::Moments& moments)
{
    return cv::Point2d(moments.m10 / moments.m00, moments.m01 / moments.m00);
}

int computeCircumference(const cv::Mat& binaryImage)
{
    int circumference = 0;
    for (int y = 1; y < binaryImage.rows - 1; ++y) {
        for (int x = 1; x < binaryImage.cols - 1; ++x) {
            if (binaryImage.at<uchar>(y, x) > 0) {
                if (binaryImage.at<uchar>(y - 1, x) == 0 ||
                    binaryImage.at<uchar>(y + 1, x) == 0 ||
                    binaryImage.at<uchar>(y, x - 1) == 0 ||
                    binaryImage.at<uchar>(y, x + 1) == 0) {
                    circumference++;
                }
            }
        }
    }
    return circumference;
}

void computeMinMaxMoments(const cv::Moments& moments, double& minMoment, double& maxMoment)
{
    double mu20 = moments.m20 / moments.m00;
    double mu02 = moments.m02 / moments.m00;
    double mu11 = moments.m11 / moments.m00;

    double diff = sqrt(pow(mu20 - mu02, 2) + 4 * pow(mu11, 2));

    maxMoment = 0.5 * (mu20 + mu02 + diff);
    minMoment = 0.5 * (mu20 + mu02 - diff);
}

void computeFeatures(const cv::Mat& binaryImage, int imgIndex)
{
    cv::Moments moments = cv::moments(binaryImage);

    double area = computeArea(binaryImage);
    cv::Point2d centerOfMass = computeCenterOfMass(moments);
    int circumference = computeCircumference(binaryImage);

    double minMoment, maxMoment;
    computeMinMaxMoments(moments, minMoment, maxMoment);

    double F1 = (double)(circumference * circumference) / (100 * area);
    double F2 = minMoment / maxMoment;

    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "Img" << imgIndex << std::endl;
    std::cout << "Area: " << area << std::endl;
    std::cout << "Center of Mass (x_t, y_t): [" << centerOfMass.x << ", " << centerOfMass.y << "]" << std::endl;
    std::cout << "Circumference: " << circumference << std::endl;
    std::cout << "F1: " << F1 << std::endl;
    std::cout << "F2: " << F2 << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
}

void computeEthalons(const std::vector<std::vector<cv::Point>>& contours, const std::vector<int>& labels, std::vector<cv::Point2d>& sqaureEthalon, std::vector<cv::Point2d>& rectangleEthalon, std::vector<cv::Point2d>& triangleEthalon)
{

}

