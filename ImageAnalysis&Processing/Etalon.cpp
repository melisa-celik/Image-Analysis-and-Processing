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
    cv::Moments moments = cv::moments(binaryImage);

    double area = moments.m00;
    // Assuming you have a function computeCircumference to calculate the circumference
    double circumference = computeCircumference(binaryImage);
    double F1 = (circumference * circumference) / (100 * area);

    double mu20 = moments.m20 / moments.m00;
    double mu02 = moments.m02 / moments.m00;
    double mu11 = moments.m11 / moments.m00;
    double minMoment = 0.5 * (mu20 + mu02 - std::sqrt((mu20 - mu02) * (mu20 - mu02) + 4 * mu11 * mu11));
    double maxMoment = 0.5 * (mu20 + mu02 + std::sqrt((mu20 - mu02) * (mu20 - mu02) + 4 * mu11 * mu11));
    double F2 = minMoment / maxMoment;

    return cv::Vec2d(F1, F2);
}

double Etalon::calculateDistance(const cv::Vec2d& v1, const cv::Vec2d& v2)
{
    return cv::norm(v1 - v2);
}
