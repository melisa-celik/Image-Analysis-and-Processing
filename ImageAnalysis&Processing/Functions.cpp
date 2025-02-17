#include "Functions.h"

double computeArea(const cv::Mat& binaryImage)
{
    if (binaryImage.empty()) {
        std::cerr << "Error: Input image is empty." << std::endl;
        return 0.0;
    }

    if (binaryImage.type() != CV_8UC1) {
        std::cerr << "Error: Input image is not a binary image (CV_8UC1 type required)." << std::endl;
        return 0.0;
    }

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

double computeF1(const cv::Mat& binaryImage)
{
    double F1 = (double)(computeCircumference(binaryImage) * computeCircumference(binaryImage)) / (100 * computeArea(binaryImage));

    return F1;
}

double computeF2(const cv::Mat& binaryImage)
{
    // Check if the input image is empty
    if (binaryImage.empty()) {
        std::cerr << "Error: Input image is empty." << std::endl;
        return 0.0;
    }

    // Check if the input image is not of type CV_8UC1
    if (binaryImage.type() != CV_8UC1) {
        std::cerr << "Error: Input image is not a binary image (CV_8UC1 type required)." << std::endl;
        return 0.0;
    }

    cv::Moments moments = cv::moments(binaryImage);

    double minMoment, maxMoment;
    computeMinMaxMoments(moments, minMoment, maxMoment);

    double F2 = minMoment / maxMoment;

    return F2;
}

void computeFeatures(const cv::Mat& binaryImage, int imgIndex, cv::Vec2d& features)
{
    int numCorners = computeNumberOfCorners(binaryImage);
    cv::Moments moments = cv::moments(binaryImage);

    double area = computeArea(binaryImage);
    cv::Point2d centerOfMass = computeCenterOfMass(moments);
    int circumference = computeCircumference(binaryImage);

    double minMoment, maxMoment;
    computeMinMaxMoments(moments, minMoment, maxMoment);

    double F1 = (double)(circumference * circumference) / (100 * area);
    double F2 = minMoment / maxMoment;

    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "OBJ-" << imgIndex << std::endl;
    std::cout << "Number of Corners: " << numCorners << std::endl;
    std::cout << "Area: " << area << std::endl;
    std::cout << "Center of Mass (x_t, y_t): [" << centerOfMass.x << ", " << centerOfMass.y << "]" << std::endl;
    std::cout << "Circumference: " << circumference << std::endl;
    std::cout << "F1: " << F1 << std::endl;
    std::cout << "F2: " << F2 << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    features = cv::Vec2d(numCorners, 0.0); 
}

int computeNumberOfCorners(const cv::Mat& binaryImage)
{
    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryImage.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Calculate number of corners using contour approximation
    int numCorners = 0;
    for (const auto& contour : contours) {
        std::vector<cv::Point> approxCurve;
        cv::approxPolyDP(contour, approxCurve, cv::arcLength(contour, true) * 0.02, true);
        numCorners += approxCurve.size();
    }

    return numCorners;
}
