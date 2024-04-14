// EtalonClassifier.cpp
#include "EtalonClassifier.h"
#include <algorithm>
#include <limits>
#include "Functions.h"
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
    numFeatures = 2; // We have F1 and F2

    // Clear previous ethalons
    ethalons.clear();

    // Compute features for each training image
    std::vector<cv::Vec2d> features;
    for (const auto& image : trainingImages) {
        features.push_back(getFeatures(image));
    }

    // Compute ethalons
    for (size_t i = 0; i < labels.size(); ++i) {
        std::string label = labels[i];
        ethalons[label] += features[i];
    }

    // Normalize ethalons
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

cv::Vec2d EtalonClassifier::computeFeatures(const cv::Mat& binaryImage)
{
    double F1 = computeF1(binaryImage);
    double F2 = computeF2(binaryImage);
    
    // print F1 and F2
    std::cout << "F1: " << F1 << ", F2: " << F2 << std::endl;

    return cv::Vec2d(F1, F2);
}

void EtalonClassifier::computeMinMaxFeatures(const std::vector<cv::Vec2d>& features, cv::Vec2d& minFeatures, cv::Vec2d& maxFeatures) {
    minFeatures = cv::Vec2d(std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
    maxFeatures = cv::Vec2d(-std::numeric_limits<double>::max(), -std::numeric_limits<double>::max());

    for (const auto& feature : features) {
        minFeatures[0] = std::min(minFeatures[0], feature[0]);
        minFeatures[1] = std::min(minFeatures[1], feature[1]);
        maxFeatures[0] = std::max(maxFeatures[0], feature[0]);
        maxFeatures[1] = std::max(maxFeatures[1], feature[1]);
    }
}

double EtalonClassifier::distance(const cv::Vec2d& v1, const cv::Vec2d& v2) {
    // Compute the Euclidean distance between two feature vectors
    cv::Vec2d diff = v1 - v2;
    return cv::norm(diff);
}
