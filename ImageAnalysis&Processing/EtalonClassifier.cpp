// EtalonClassifier.cpp
#include "EtalonClassifier.h"
#include <fstream>
#include <iostream>

EtalonClassifier::EtalonClassifier() {}

EtalonClassifier::~EtalonClassifier() {}

cv::Vec2d EtalonClassifier::getFeatures(const cv::Mat& binaryImage)
{
    // Compute features using moments of the contour's binary image
    return computeFeatures(binaryImage);
}

void EtalonClassifier::computeEthalons(const std::vector<cv::Mat>& trainingImages, const std::vector<std::string>& labels) {
    for (size_t i = 0; i < trainingImages.size(); ++i) {
        cv::Vec2d features = getFeatures(trainingImages[i]);
        ethalons[labels[i]] += features;
    }

    for (auto& ethalon : ethalons) {
        ethalon.second /= static_cast<double>(std::count(labels.begin(), labels.end(), ethalon.first));
    }
}

void EtalonClassifier::saveEthalons(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        for (const auto& ethalon : ethalons) {
            file.write(reinterpret_cast<const char*>(&ethalon.second), sizeof(cv::Vec2d));
            size_t labelSize = ethalon.first.size();
            file.write(reinterpret_cast<const char*>(&labelSize), sizeof(size_t));
            file.write(ethalon.first.c_str(), labelSize);
        }
        file.close();
    }
    else {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
    }
}

void EtalonClassifier::loadEthalons(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        ethalons.clear();
        while (!file.eof()) {
            cv::Vec2d features;
            size_t labelSize;
            file.read(reinterpret_cast<char*>(&features), sizeof(cv::Vec2d));
            file.read(reinterpret_cast<char*>(&labelSize), sizeof(size_t));
            std::string label(labelSize, '\0');
            file.read(&label[0], labelSize);
            ethalons[label] = features;
        }
        file.close();
    }
    else {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
    }
}

std::string EtalonClassifier::classifyObject(const cv::Mat& testImage) {
    cv::Vec2d testFeatures = getFeatures(testImage);
    double minDistance = std::numeric_limits<double>::max();
    std::string closestLabel;

    for (const auto& ethalon : ethalons) {
        double d = distance(testFeatures, ethalon.second);
        if (d < minDistance) {
            minDistance = d;
            closestLabel = ethalon.first;
        }
    }

    return closestLabel;
}

std::string EtalonClassifier::classifyShape(const cv::Mat& binaryImage)
{
    double F1 = computeF1(binaryImage);
    double F2 = computeF2(binaryImage);

    // Check the aspect ratios to classify the shape
    double aspectRatio = std::min(F1, F2) / std::max(F1, F2);
    if (aspectRatio > 0.9 && aspectRatio < 1.1) {
        return "square";
    }
    else if (aspectRatio > 0.5 && aspectRatio < 2.0) {
        return "rectangle";
    }
    else if (aspectRatio > 0.2 && aspectRatio < 0.5) {
        return "star";
    }
    else {
        return "unknown";
    }
}

double EtalonClassifier::computeArea(const cv::Mat& binaryImage) {
    return cv::countNonZero(binaryImage);
}

double EtalonClassifier::computeF1(const cv::Mat& binaryImage) {
    // Compute F1 as defined in your previous code
    return (double)(computeCircumference(binaryImage) * computeCircumference(binaryImage)) / (100 * computeArea(binaryImage));
}

double EtalonClassifier::computeF2(const cv::Mat& binaryImage) {
    // Compute F2 as defined in your previous code
    cv::Moments moments = cv::moments(binaryImage);
    double minMoment, maxMoment;
    computeMinMaxMoments(moments, minMoment, maxMoment);
    return minMoment / maxMoment;
}

cv::Vec2d EtalonClassifier::computeFeatures(const cv::Mat& binaryImage)
{
    std::cout << "Inside computeFeatures function" << std::endl; // Add this line for debugging
    double F1 = computeF1(binaryImage);
    double F2 = computeF2(binaryImage);
    std::cout << "F1: " << F1 << ", F2: " << F2 << std::endl; // Print features for debugging
    return cv::Vec2d(F1, F2);
}

double EtalonClassifier::distance(const cv::Vec2d& v1, const cv::Vec2d& v2) {
    // Compute the Euclidean distance between two feature vectors
    cv::Vec2d diff = v1 - v2;
    return cv::norm(diff);
}

int EtalonClassifier::computeCircumference(const cv::Mat& binaryImage) {
    // Compute the circumference as defined in your previous code
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

void EtalonClassifier::computeMinMaxMoments(const cv::Moments& moments, double& minMoment, double& maxMoment) {
    // Compute the minimum and maximum moments as defined in your previous code
    double mu20 = moments.m20 / moments.m00;
    double mu02 = moments.m02 / moments.m00;
    double mu11 = moments.m11 / moments.m00;
    double diff = sqrt(pow(mu20 - mu02, 2) + 4 * pow(mu11, 2));
    maxMoment = 0.5 * (mu20 + mu02 + diff);
    minMoment = 0.5 * (mu20 + mu02 - diff);
}
