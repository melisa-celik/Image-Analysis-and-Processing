#include "Etalon.h"
#include <numeric> // Include for std::accumulate

Etalon::Etalon()
{
}

Etalon::~Etalon()
{
}

void Etalon::computeEthalons(const std::vector<cv::Mat>& trainingImages, const std::vector<std::string>& labels)
{
    ethalons.clear();

    // Iterate over each class label
    for (size_t i = 0; i < labels.size(); ++i) {
        std::string label = labels[i];

        // Extract features for all images with the current label
        std::vector<cv::Vec2d> classFeatures;
        for (size_t j = 0; j < trainingImages.size(); ++j) {
            if (labels[j] == label) {
                cv::Vec2d features = computeFeatures(trainingImages[j]);
                classFeatures.push_back(features);
            }
        }

        // Compute the ethalon for the current class
        if (!classFeatures.empty()) {
            cv::Vec2d sum = std::accumulate(classFeatures.begin(), classFeatures.end(), cv::Vec2d(0.0, 0.0));
            cv::Vec2d ethalon = sum / static_cast<double>(classFeatures.size());
            ethalons[label] = ethalon;
        }
    }
}

std::string Etalon::classifyObject(const cv::Mat& testImage)
{
    cv::Vec2d testFeatures = computeFeatures(testImage);

    // Initialize variables to track the closest ethalon
    double minDistance = std::numeric_limits<double>::max();
    std::string closestLabel;

    // Find the closest ethalon
    for (const auto& ethalon : ethalons) {
        double distance = calculateDistance(testFeatures, ethalon.second);
        if (distance < minDistance) {
            minDistance = distance;
            closestLabel = ethalon.first;
        }
    }

    return closestLabel;
}

cv::Vec2d Etalon::computeFeatures(const cv::Mat& binaryImage)
{
    // Calculate image moments
    cv::Moments moments = cv::moments(binaryImage);

    // Calculate area (m00) and circumference (perimeter)
    double area = moments.m00;

    // Calculate the circumference by finding contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryImage.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    double perimeter = 0.0;
    if (!contours.empty()) {
        perimeter = cv::arcLength(contours[0], true);
    }

    // Compute features F1 and F2
    double F1 = (perimeter * perimeter) / (100.0 * area);
    double F2_max = (0.5 * (moments.mu20 + moments.mu02)) +
        (0.5 * sqrt(4.0 * (moments.mu11 * moments.mu11) +
            (moments.mu20 - moments.mu02) * (moments.mu20 - moments.mu02)));
    double F2_min = (0.5 * (moments.mu20 + moments.mu02)) -
        (0.5 * sqrt(4.0 * (moments.mu11 * moments.mu11) +
            (moments.mu20 - moments.mu02) * (moments.mu20 - moments.mu02)));
    double F2 = F2_min / F2_max;

    return cv::Vec2d(F1, F2);
}


double Etalon::calculateDistance(const cv::Vec2d& v1, const cv::Vec2d& v2)
{
    return cv::norm(v1 - v2);
}
