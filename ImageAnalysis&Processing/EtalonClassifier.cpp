#include "EtalonClassifier.h"
#include <algorithm>
#include <limits>
#include "Functions.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric> 

EtalonClassifier::EtalonClassifier() {}

EtalonClassifier::~EtalonClassifier() {}

cv::Vec2d EtalonClassifier::getFeatures(const cv::Mat& binaryImage)
{
    // Compute features using moments of the contour's binary image
    return computeFeatures(binaryImage);
}

void EtalonClassifier::computeEthalons(const std::vector<cv::Vec2d>& features, const std::vector<std::string>& labels) {
    numFeatures = 2; // We have F1 and F2

    // Clear previous ethalons
    ethalons.clear();

    // Compute ethalons
    std::map<std::string, std::vector<cv::Vec2d>> classFeatures;
    for (size_t i = 0; i < labels.size(); ++i) {
        std::string label = labels[i];
        classFeatures[label].push_back(features[i]);
    }

    for (const auto& classFeature : classFeatures) {
        std::string label = classFeature.first;
        cv::Vec2d sum = cv::Vec2d(0.0, 0.0);
        for (const auto& feature : classFeature.second) {
            sum += feature;
        }
        ethalons[label] = sum / static_cast<double>(classFeature.second.size());
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
        return; // Add return statement to exit function
    }
}

std::string EtalonClassifier::classifyObject(const cv::Mat& testImage) {
    // Compute features for the test image
    cv::Vec2d testFeatures = getFeatures(testImage);

    // Initialize variables to keep track of the closest ethalon and its distance
    double minDistance = std::numeric_limits<double>::max(); // Initialize minDistance
    std::string closestLabel;

    // Iterate over each ethalon and find the closest one
    for (const auto& ethalon : ethalons) {
        // Compute the distance between the test features and the ethalon
        double d = distance(testFeatures, ethalon.second);

        // Check if this ethalon is closer than the current closest one
        if (d < minDistance) {
            minDistance = d;
            closestLabel = ethalon.first;
        }
    }

    // Return the label of the closest ethalon
    return closestLabel;
}


//std::string EtalonClassifier::classifyShape(const cv::Mat& binaryImage)
//{
//    double F1 = computeF1(binaryImage);
//    double F2 = computeF2(binaryImage);
//
//    // Check the aspect ratios to classify the shape
//    double aspectRatio = std::min(F1, F2) / std::max(F1, F2);
//    if (aspectRatio >= 0.9 && aspectRatio <= 1.1) {
//        return "square";
//    }
//    else if (aspectRatio >= 0.5 && aspectRatio <= 2.0) {
//        return "rectangle";
//    }
//    else if (aspectRatio >= 0.2 && aspectRatio <= 0.5) {
//        return "star";
//    }
//    else {
//        return "unknown";
//    }
//}

std::string EtalonClassifier::classifyShape(const cv::Mat& binaryImage)
{
    // Compute features for the binary image
    cv::Vec2d features = computeFeatures(binaryImage);

    // Initialize variables to keep track of the closest ethalon and its distance
    double minDistance = std::numeric_limits<double>::max();
    std::string closestLabel;

    // Iterate over each ethalon and find the closest one
    for (const auto& ethalon : ethalons) {
        // Compute the distance between the test features and the ethalon
        double d = distance(features, ethalon.second);

        // Check if this ethalon is closer than the current closest one
        if (d < minDistance) {
            minDistance = d;
            closestLabel = ethalon.first;
        }
    }

    // Return the label of the closest ethalon
    return closestLabel;
}

double EtalonClassifier::distance(const cv::Vec2d& v1, const cv::Vec2d& v2) {
    // Compute the Euclidean distance between two feature vectors
    cv::Vec2d diff = v1 - v2;
    return cv::norm(diff);
}

cv::Vec2d EtalonClassifier::computeFeatures(const cv::Mat& binaryImage, int imgIndex, cv::Vec2d* features)
{
    if (features == nullptr) {
        // Compute features for the binary image
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
        //std::cout << "OBJ-" << imgIndex << std::endl;
        std::cout << "Number of Corners: " << numCorners << std::endl;
        std::cout << "Area: " << area << std::endl;
        std::cout << "Center of Mass (x_t, y_t): [" << centerOfMass.x << ", " << centerOfMass.y << "]" << std::endl;
        std::cout << "Circumference: " << circumference << std::endl;
        std::cout << "F1: " << F1 << std::endl;
        std::cout << "F2: " << F2 << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;

        return cv::Vec2d(F1, F2);
    }
    else {
        // Use the provided features
        return *features;
    }
}